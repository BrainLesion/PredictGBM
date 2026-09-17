"""DIRAC-based longitudinal registration of post-operative (follow-up) to pre-operative MRI.

This module bundles everything needed for the "dirac" registration_algorithm used by
norm_ss_coregistration.register_recurrence:

  1. Grid and nifti I/O helpers
  2. Network building blocks (AdaIn conditioning, spatial transforms)
  3. The three-level laplacian registration network (MICCAI 2021 LDR, DIRAC weights)
  4. Model inference, runnable as `python -m predict_gbm.preprocessing.dirac` (spawned as a
     subprocess by run_dirac_inference so GPU memory is released after inference)
  5. Instance optimization refining the predicted displacement fields per case
  6. The pipeline API called by norm_ss_coregistration

Network and inference code originate from the DIRAC BraTSReg submission
(https://github.com/cwmok/DIRAC), trimmed to the inference path.
"""

import glob
import math
import os
import shutil
import subprocess
import sys
from argparse import ArgumentParser
from pathlib import Path
from typing import Any, Literal

from loguru import logger
import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Input shape the registration network operates on; images are resampled to this
# resolution for inference and the predicted fields are resampled back.
IMGSHAPE = (160, 160, 80)
IMGSHAPE_2 = tuple(s // 2 for s in IMGSHAPE)
IMGSHAPE_4 = tuple(s // 4 for s in IMGSHAPE)
RANGE_FLOW = 0.4



# -----------------------------------------------------------------------------
# Grid and nifti I/O helpers
# -----------------------------------------------------------------------------


def generate_grid_unit(imgshape):
    x = (np.arange(imgshape[0]) - ((imgshape[0] - 1) / 2)) / (imgshape[0] - 1) * 2
    y = (np.arange(imgshape[1]) - ((imgshape[1] - 1) / 2)) / (imgshape[1] - 1) * 2
    z = (np.arange(imgshape[2]) - ((imgshape[2] - 1) / 2)) / (imgshape[2] - 1) * 2
    grid = np.rollaxis(np.array(np.meshgrid(z, y, x)), 0, 4)
    grid = np.swapaxes(grid, 0, 2)
    grid = np.swapaxes(grid, 1, 2)
    return grid


def load_4D(name):
    # X = sitk.GetArrayFromImage(sitk.ReadImage(name, sitk.sitkFloat32 ))
    # X = np.reshape(X, (1,)+ X.shape)
    X = nib.load(name)
    X = X.get_fdata()
    X = np.reshape(X, (1,) + X.shape)
    return X


def imgnorm(img):
    max_v = np.max(img)
    min_v = np.min(img)
    norm_img = (img - min_v) / (max_v - min_v)
    return norm_img


def save_img(I_img, savename, header=None, affine=None):
    if header is None or affine is None:
        affine = np.diag([1, 1, 1, 1])
        new_img = nib.nifti1.Nifti1Image(I_img, affine, header=None)
    else:
        new_img = nib.nifti1.Nifti1Image(I_img, affine, header=header)

    nib.save(new_img, savename)


def save_flow(I_img, savename, header=None, affine=None):
    # I2 = sitk.GetImageFromArray(I_img,isVector=True)
    # sitk.WriteImage(I2,savename)
    if header is None or affine is None:
        affine = np.diag([1, 1, 1, 1])
        new_img = nib.nifti1.Nifti1Image(I_img, affine, header=None)
    else:
        new_img = nib.nifti1.Nifti1Image(I_img, affine, header=header)

    nib.save(new_img, savename)


def save_nifti(data, reference_path: Path, output_path: Path):
    reference = nib.load(str(reference_path))
    nib.save(
        nib.Nifti1Image(
            data.astype(np.float32), affine=reference.affine, header=reference.header
        ),
        str(output_path),
    )


class Validation_Brats(Data.Dataset):
    def __init__(
        self, fixed_list, move_list, fixed_label_list, move_label_list, norm=True
    ):
        super(Validation_Brats, self).__init__()
        self.fixed_list = fixed_list
        self.move_list = move_list
        self.fixed_label_list = fixed_label_list
        self.move_label_list = move_label_list
        self.norm = norm

    def __len__(self):
        "Denotes the total number of samples"
        return len(self.move_list)

    def __getitem__(self, index):
        fixed_img = load_4D(self.fixed_list[index])
        moved_img = load_4D(self.move_list[index])

        if self.norm:
            fixed_img = imgnorm(fixed_img)
            moved_img = imgnorm(moved_img)

        fixed_img = torch.from_numpy(fixed_img).float()
        moved_img = torch.from_numpy(moved_img).float()

        return {"fixed": fixed_img, "move": moved_img}


# -----------------------------------------------------------------------------
# Network building blocks
# -----------------------------------------------------------------------------


class AdaptiveInstanceNorm(nn.Module):
    def __init__(self, in_channel, latent_dim=256):
        super().__init__()

        # self.norm = nn.InstanceNorm3d(in_channel)

        # self.style = EqualLinear(style_dim, in_channel * 2)

        self.style = nn.Linear(latent_dim, in_channel * 2)

        # self.style.bias.data[:in_channel] = 1
        self.style.bias.data[:in_channel] = 0
        self.style.bias.data[in_channel:] = 0

    def forward(self, input, latent_code):
        # style [batch_size, in_channels*2] => [batch_size, in_channels*2, 1, 1, 1]
        style = (
            self.style(latent_code)
            .unsqueeze(dim=-1)
            .unsqueeze(dim=-1)
            .unsqueeze(dim=-1)
        )
        gamma, beta = style.chunk(2, dim=1)

        # out = self.norm(input)
        out = input

        out = (1.0 + gamma) * out + beta

        return out


class PreActBlock_AdaIn(nn.Module):
    """Pre-activation version of the BasicBlock."""

    expansion = 1

    def __init__(
        self,
        in_planes,
        planes,
        num_group=4,
        stride=1,
        bias=False,
        latent_dim=64,
        mapping_fmaps=64,
        num_con=1,
    ):
        super(PreActBlock_AdaIn, self).__init__()
        self.ai1 = AdaptiveInstanceNorm(in_planes, latent_dim=latent_dim)
        self.conv1 = nn.Conv3d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=bias
        )
        self.ai2 = AdaptiveInstanceNorm(in_planes, latent_dim=latent_dim)
        self.conv2 = nn.Conv3d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=bias
        )

        self.mapping = nn.Sequential(
            nn.Linear(num_con, mapping_fmaps),
            nn.LeakyReLU(0.2),
            nn.Linear(mapping_fmaps, mapping_fmaps),
            nn.LeakyReLU(0.2),
            nn.Linear(mapping_fmaps, mapping_fmaps),
            nn.LeakyReLU(0.2),
            nn.Linear(mapping_fmaps, latent_dim),
            nn.LeakyReLU(0.2),
        )

        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv3d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=bias,
                )
            )

    def forward(self, x, reg_code):

        latent_fea = self.mapping(reg_code)

        out = F.leaky_relu(self.ai1(x, latent_fea), negative_slope=0.2)

        shortcut = self.shortcut(out) if hasattr(self, "shortcut") else x
        out = self.conv1(out)

        out = self.conv2(F.leaky_relu(self.ai2(out, latent_fea), negative_slope=0.2))

        out += shortcut
        return out


class SpatialTransform_unit(nn.Module):
    def __init__(self):
        super(SpatialTransform_unit, self).__init__()

    def forward(self, x, flow, sample_grid):
        sample_grid = sample_grid + flow
        # size_tensor = sample_grid.size()
        # sample_grid[0, :, :, :, 0] = (sample_grid[0, :, :, :, 0] - (size_tensor[3] / 2)) / size_tensor[3] * 2
        # sample_grid[0, :, :, :, 1] = (sample_grid[0, :, :, :, 1] - (size_tensor[2] / 2)) / size_tensor[2] * 2
        # sample_grid[0, :, :, :, 2] = (sample_grid[0, :, :, :, 2] - (size_tensor[1] / 2)) / size_tensor[1] * 2
        flow = torch.nn.functional.grid_sample(
            x, sample_grid, mode="bilinear", padding_mode="border", align_corners=True
        )

        return flow


class DiffeomorphicTransform_unit(nn.Module):
    def __init__(self, time_step=7):
        super(DiffeomorphicTransform_unit, self).__init__()
        self.time_step = time_step

    def forward(self, velocity, sample_grid):
        flow = velocity / (2.0**self.time_step)
        # size_tensor = sample_grid.size()
        # 0.5 flow
        for _ in range(self.time_step):
            grid = sample_grid + flow.permute(0, 2, 3, 4, 1)
            # grid[0, :, :, :, 0] = (grid[0, :, :, :, 0] - (size_tensor[3] / 2)) / size_tensor[3] * 2
            # grid[0, :, :, :, 1] = (grid[0, :, :, :, 1] - (size_tensor[2] / 2)) / size_tensor[2] * 2
            # grid[0, :, :, :, 2] = (grid[0, :, :, :, 2] - (size_tensor[1] / 2)) / size_tensor[1] * 2
            flow = flow + F.grid_sample(
                flow, grid, mode="bilinear", padding_mode="border", align_corners=True
            )
        return flow


class CompositionTransform_unit(nn.Module):
    def __init__(self):
        super(CompositionTransform_unit, self).__init__()

    def forward(self, flow_1, flow_2, sample_grid):
        # size_tensor = sample_grid.size()
        grid = sample_grid + flow_2.permute(0, 2, 3, 4, 1)
        # grid[0, :, :, :, 0] = (grid[0, :, :, :, 0] - (size_tensor[3] / 2)) / size_tensor[3] * 2
        # grid[0, :, :, :, 1] = (grid[0, :, :, :, 1] - (size_tensor[2] / 2)) / size_tensor[2] * 2
        # grid[0, :, :, :, 2] = (grid[0, :, :, :, 2] - (size_tensor[1] / 2)) / size_tensor[1] * 2
        compos_flow = (
            F.grid_sample(
                flow_1, grid, mode="bilinear", padding_mode="border", align_corners=True
            ) + flow_2
        )
        return compos_flow


# -----------------------------------------------------------------------------
# Registration network (three-level laplacian pyramid)
# -----------------------------------------------------------------------------


class Miccai2021_LDR_laplacian_unit_disp_add_AdaIn_lvl1(nn.Module):
    def __init__(
        self,
        in_channel,
        n_classes,
        start_channel,
        is_train=True,
        imgshape=(160, 192, 144),
        range_flow=0.4,
        num_block=5,
    ):
        super(Miccai2021_LDR_laplacian_unit_disp_add_AdaIn_lvl1, self).__init__()
        self.in_channel = in_channel
        self.n_classes = n_classes
        self.start_channel = start_channel

        self.range_flow = range_flow
        self.is_train = is_train

        self.imgshape = imgshape

        self.grid_1 = generate_grid_unit(self.imgshape)
        self.grid_1 = (
            torch.from_numpy(np.reshape(self.grid_1, (1,) + self.grid_1.shape))
            .cuda()
            .float()
        )

        self.diff_transform = DiffeomorphicTransform_unit(time_step=7).cuda()
        self.transform = SpatialTransform_unit().cuda()
        # self.com_transform = CompositionTransform().cuda()

        bias_opt = False

        self.input_encoder_lvl1 = self.input_feature_extract(
            self.in_channel, self.start_channel * 4, bias=bias_opt
        )

        self.down_conv = nn.Conv3d(
            self.start_channel * 4,
            self.start_channel * 4,
            3,
            stride=2,
            padding=1,
            bias=bias_opt,
        )
        # self.input_encoder_lvl2 = self.input_feature_extract(self.in_channel, self.start_channel * 4, bias=bias_opt)
        # self.input_encoder_lvl3 = self.input_feature_extract(self.in_channel, self.start_channel * 4, bias=bias_opt)

        self.resblock_group_lvl1 = self.resblock_seq(
            self.start_channel * 4, num_block=num_block, bias_opt=bias_opt
        )
        # self.resblock_group_lvl2 = self.resblock_seq(self.start_channel * 4, bias_opt=bias_opt)
        # self.resblock_group_lvl3 = self.resblock_seq(self.start_channel * 4, bias_opt=bias_opt)

        # self.up = torch.nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False)
        self.up = nn.ConvTranspose3d(
            self.start_channel * 4,
            self.start_channel * 4,
            2,
            stride=2,
            padding=0,
            output_padding=0,
            bias=bias_opt,
        )

        self.down_avg = nn.AvgPool3d(
            kernel_size=3, stride=2, padding=1, count_include_pad=False
        )

        self.output_lvl1 = self.outputs(
            self.start_channel * 8,
            self.n_classes,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        # self.output_lvl2 = self.outputs(self.start_channel * 4, self.n_classes, kernel_size=5, stride=1, padding=2,
        #                            bias=False)
        # self.output_lvl3 = self.outputs(self.start_channel * 4, self.n_classes, kernel_size=5, stride=1, padding=2,
        #                            bias=False)

        # for m in self.modules():
        #     if isinstance(m, nn.Conv3d):
        #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def resblock_seq(self, in_channels, num_block, bias_opt=False):
        blocks = []
        for i in range(num_block):
            blocks.append(PreActBlock_AdaIn(in_channels, in_channels, bias=bias_opt))
            blocks.append(nn.LeakyReLU(0.2))

        layer = nn.ModuleList(blocks)
        return layer

    def input_feature_extract(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
        batchnorm=False,
    ):
        if batchnorm:
            layer = nn.Sequential(
                nn.Conv3d(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=stride,
                    padding=padding,
                    bias=bias,
                ),
                nn.BatchNorm3d(out_channels),
                nn.ReLU(),
            )
        else:
            layer = nn.Sequential(
                nn.Conv3d(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=stride,
                    padding=padding,
                    bias=bias,
                ),
                nn.LeakyReLU(0.2),
                nn.Conv3d(
                    out_channels,
                    out_channels,
                    kernel_size,
                    stride=stride,
                    padding=padding,
                    bias=bias,
                ),
            )
        return layer

    def decoder(
        self,
        in_channels,
        out_channels,
        kernel_size=2,
        stride=2,
        padding=0,
        output_padding=0,
        bias=True,
    ):
        layer = nn.Sequential(
            nn.ConvTranspose3d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                output_padding=output_padding,
                bias=bias,
            ),
            nn.ReLU(),
        )
        return layer

    def outputs(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=0,
        bias=False,
        batchnorm=False,
    ):
        if batchnorm:
            layer = nn.Sequential(
                nn.Conv3d(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=stride,
                    padding=padding,
                    bias=bias,
                ),
                nn.BatchNorm3d(out_channels),
                nn.Tanh(),
            )
        else:
            # layer = nn.Sequential(
            #     nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
            #     nn.Tanh())
            layer = nn.Sequential(
                nn.Conv3d(
                    in_channels,
                    int(in_channels / 2),
                    kernel_size,
                    stride=stride,
                    padding=padding,
                    bias=bias,
                ),
                nn.LeakyReLU(0.2),
                nn.Conv3d(
                    int(in_channels / 2),
                    out_channels,
                    kernel_size,
                    stride=stride,
                    padding=padding,
                    bias=bias,
                ),
                nn.Softsign(),
            )
        return layer

    def forward(self, x, y, reg_code):

        cat_input = torch.cat((x, y), 1)
        cat_input = self.down_avg(cat_input)
        cat_input_lvl1 = self.down_avg(cat_input)

        down_y = cat_input_lvl1[:, 1:2, :, :, :]

        fea_e0 = self.input_encoder_lvl1(cat_input_lvl1)
        e0 = self.down_conv(fea_e0)

        # e0 = self.resblock_group_lvl1(e0)
        for i in range(len(self.resblock_group_lvl1)):
            if i % 2 == 0:
                e0 = self.resblock_group_lvl1[i](e0, reg_code)
            else:
                e0 = self.resblock_group_lvl1[i](e0)

        e0 = self.up(e0)
        output_disp_e0_v = (
            self.output_lvl1(torch.cat([e0, fea_e0], dim=1)) * self.range_flow
        )
        # output_disp_e0 = self.diff_transform(output_disp_e0_v, self.grid_1)
        warpped_inputx_lvl1_out = self.transform(
            x, output_disp_e0_v.permute(0, 2, 3, 4, 1), self.grid_1
        )

        if self.is_train is True:
            return (
                output_disp_e0_v,
                warpped_inputx_lvl1_out,
                down_y,
                output_disp_e0_v,
                e0,
            )
        else:
            return output_disp_e0_v


class Miccai2021_LDR_laplacian_unit_disp_add_AdaIn_lvl2(nn.Module):
    def __init__(
        self,
        in_channel,
        n_classes,
        start_channel,
        is_train=True,
        imgshape=(160, 192, 144),
        range_flow=0.4,
        model_lvl1=None,
        num_block=5,
    ):
        super(Miccai2021_LDR_laplacian_unit_disp_add_AdaIn_lvl2, self).__init__()
        self.in_channel = in_channel
        self.n_classes = n_classes
        self.start_channel = start_channel

        self.range_flow = range_flow
        self.is_train = is_train

        self.imgshape = imgshape

        self.model_lvl1 = model_lvl1
        # self.model_lvl1 = [model_lvl1[i] for i in range(len(model_lvl1)-1)]
        # self.model_lvl1 = nn.Sequential(*self.model_lvl1)

        self.grid_1 = generate_grid_unit(self.imgshape)
        self.grid_1 = (
            torch.from_numpy(np.reshape(self.grid_1, (1,) + self.grid_1.shape))
            .cuda()
            .float()
        )

        self.diff_transform = DiffeomorphicTransform_unit(time_step=7).cuda()
        self.transform = SpatialTransform_unit().cuda()
        self.com_transform = CompositionTransform_unit().cuda()

        bias_opt = False

        self.input_encoder_lvl1 = self.input_feature_extract(
            self.in_channel + 3, self.start_channel * 4, bias=bias_opt
        )

        self.down_conv = nn.Conv3d(
            self.start_channel * 4,
            self.start_channel * 4,
            3,
            stride=2,
            padding=1,
            bias=bias_opt,
        )
        # self.input_encoder_lvl2 = self.input_feature_extract(self.in_channel, self.start_channel * 4, bias=bias_opt)
        # self.input_encoder_lvl3 = self.input_feature_extract(self.in_channel, self.start_channel * 4, bias=bias_opt)

        self.resblock_group_lvl1 = self.resblock_seq(
            self.start_channel * 4, num_block=num_block, bias_opt=bias_opt
        )
        # self.resblock_group_lvl2 = self.resblock_seq(self.start_channel * 4, bias_opt=bias_opt)
        # self.resblock_group_lvl3 = self.resblock_seq(self.start_channel * 4, bias_opt=bias_opt)

        self.up_tri = torch.nn.Upsample(
            scale_factor=2, mode="trilinear", align_corners=False
        )
        self.up = nn.ConvTranspose3d(
            self.start_channel * 4,
            self.start_channel * 4,
            2,
            stride=2,
            padding=0,
            output_padding=0,
            bias=bias_opt,
        )

        self.down_avg = nn.AvgPool3d(
            kernel_size=3, stride=2, padding=1, count_include_pad=False
        )

        self.output_lvl1 = self.outputs(
            self.start_channel * 8,
            self.n_classes,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )

        # self.output_lvl2 = self.outputs(self.start_channel * 4, self.n_classes, kernel_size=5, stride=1, padding=2,
        #                            bias=False)
        # self.output_lvl3 = self.outputs(self.start_channel * 4, self.n_classes, kernel_size=5, stride=1, padding=2,
        #                            bias=False)

        # for m in self.modules():
        #     if isinstance(m, nn.Conv3d):
        #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def unfreeze_modellvl1(self):
        # unFreeze model_lvl1 weight
        print("\nunfreeze model_lvl1 parameter")
        for param in self.model_lvl1.parameters():
            param.requires_grad = True

    def resblock_seq(self, in_channels, num_block, bias_opt=False):
        blocks = []
        for i in range(num_block):
            blocks.append(PreActBlock_AdaIn(in_channels, in_channels, bias=bias_opt))
            blocks.append(nn.LeakyReLU(0.2))

        layer = nn.ModuleList(blocks)
        return layer

    def input_feature_extract(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
        batchnorm=False,
    ):
        if batchnorm:
            layer = nn.Sequential(
                nn.Conv3d(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=stride,
                    padding=padding,
                    bias=bias,
                ),
                nn.BatchNorm3d(out_channels),
                nn.ReLU(),
            )
        else:
            layer = nn.Sequential(
                nn.Conv3d(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=stride,
                    padding=padding,
                    bias=bias,
                ),
                nn.LeakyReLU(0.2),
                nn.Conv3d(
                    out_channels,
                    out_channels,
                    kernel_size,
                    stride=stride,
                    padding=padding,
                    bias=bias,
                ),
            )
        return layer

    def decoder(
        self,
        in_channels,
        out_channels,
        kernel_size=2,
        stride=2,
        padding=0,
        output_padding=0,
        bias=True,
    ):
        layer = nn.Sequential(
            nn.ConvTranspose3d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                output_padding=output_padding,
                bias=bias,
            ),
            nn.ReLU(),
        )
        return layer

    def outputs(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=0,
        bias=False,
        batchnorm=False,
    ):
        if batchnorm:
            layer = nn.Sequential(
                nn.Conv3d(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=stride,
                    padding=padding,
                    bias=bias,
                ),
                nn.BatchNorm3d(out_channels),
                nn.Tanh(),
            )
        else:
            # layer = nn.Sequential(
            #     nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
            #     nn.Tanh())
            layer = nn.Sequential(
                nn.Conv3d(
                    in_channels,
                    int(in_channels / 2),
                    kernel_size,
                    stride=stride,
                    padding=padding,
                    bias=bias,
                ),
                nn.LeakyReLU(0.2),
                nn.Conv3d(
                    int(in_channels / 2),
                    out_channels,
                    kernel_size,
                    stride=stride,
                    padding=padding,
                    bias=bias,
                ),
                nn.Softsign(),
            )
        return layer

    def forward(self, x, y, reg_code):
        # output_disp_e0, warpped_inputx_lvl1_out, down_y, output_disp_e0_v, e0
        lvl1_disp, _, _, lvl1_v, lvl1_embedding = self.model_lvl1(x, y, reg_code)

        # lvl1_disp, lvl1_warp, lvl1_y, lvl1_v, lvl1_embedding = self.model_lvl1(x, y, reg_code)
        lvl1_disp_up = self.up_tri(lvl1_disp)

        x_down = self.down_avg(x)
        y_down = self.down_avg(y)

        warpped_x = self.transform(
            x_down, lvl1_disp_up.permute(0, 2, 3, 4, 1), self.grid_1
        )

        cat_input_lvl2 = torch.cat((warpped_x, y_down, lvl1_disp_up), 1)

        fea_e0 = self.input_encoder_lvl1(cat_input_lvl2)
        e0 = self.down_conv(fea_e0)

        e0 = e0 + lvl1_embedding

        # e0 = self.resblock_group_lvl1(e0)
        for i in range(len(self.resblock_group_lvl1)):
            if i % 2 == 0:
                e0 = self.resblock_group_lvl1[i](e0, reg_code)
            else:
                e0 = self.resblock_group_lvl1[i](e0)

        e0 = self.up(e0)
        output_disp_e0_v = (
            self.output_lvl1(torch.cat([e0, fea_e0], dim=1)) * self.range_flow
        )
        # output_disp_e0 = self.diff_transform(output_disp_e0_v, self.grid_1)
        compose_field_e0_lvl1 = lvl1_disp_up + output_disp_e0_v
        warpped_inputx_lvl1_out = self.transform(
            x, compose_field_e0_lvl1.permute(0, 2, 3, 4, 1), self.grid_1
        )

        if self.is_train is True:
            return (
                compose_field_e0_lvl1,
                warpped_inputx_lvl1_out,
                y_down,
                output_disp_e0_v,
                lvl1_v,
                e0,
            )
            # return compose_field_e0_lvl1, warpped_inputx_lvl1_out, y_down, output_disp_e0_v, lvl1_v, e0, lvl1_warp, lvl1_y
        else:
            return compose_field_e0_lvl1


class Miccai2021_LDR_laplacian_unit_disp_add_AdaIn_lvl3(nn.Module):
    def __init__(
        self,
        in_channel,
        n_classes,
        start_channel,
        is_train=True,
        imgshape=(160, 192, 144),
        range_flow=0.4,
        model_lvl2=None,
        num_block=5,
    ):
        super(Miccai2021_LDR_laplacian_unit_disp_add_AdaIn_lvl3, self).__init__()
        self.in_channel = in_channel
        self.n_classes = n_classes
        self.start_channel = start_channel

        self.range_flow = range_flow
        self.is_train = is_train

        self.imgshape = imgshape

        self.model_lvl2 = model_lvl2

        self.grid_1 = generate_grid_unit(self.imgshape)
        self.grid_1 = (
            torch.from_numpy(np.reshape(self.grid_1, (1,) + self.grid_1.shape))
            .cuda()
            .float()
        )

        self.diff_transform = DiffeomorphicTransform_unit(time_step=7).cuda()
        self.transform = SpatialTransform_unit().cuda()
        self.com_transform = CompositionTransform_unit().cuda()

        bias_opt = False

        self.input_encoder_lvl1 = self.input_feature_extract(
            self.in_channel + 3, self.start_channel * 4, bias=bias_opt
        )

        self.down_conv = nn.Conv3d(
            self.start_channel * 4,
            self.start_channel * 4,
            3,
            stride=2,
            padding=1,
            bias=bias_opt,
        )

        self.resblock_group_lvl1 = self.resblock_seq(
            self.start_channel * 4, num_block=num_block, bias_opt=bias_opt
        )

        self.up_tri = torch.nn.Upsample(
            scale_factor=2, mode="trilinear", align_corners=False
        )
        self.up = nn.ConvTranspose3d(
            self.start_channel * 4,
            self.start_channel * 4,
            2,
            stride=2,
            padding=0,
            output_padding=0,
            bias=bias_opt,
        )

        # self.down_avg = nn.AvgPool3d(kernel_size=3, stride=2, padding=1, count_include_pad=False)

        self.output_lvl1 = self.outputs(
            self.start_channel * 8,
            self.n_classes,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )

        # for m in self.modules():
        #     if isinstance(m, nn.Conv3d):
        #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def unfreeze_modellvl2(self):
        # unFreeze model_lvl1 weight
        print("\nunfreeze model_lvl2 parameter")
        for param in self.model_lvl2.parameters():
            param.requires_grad = True

    def resblock_seq(self, in_channels, num_block, bias_opt=False):
        blocks = []
        for i in range(num_block):
            blocks.append(PreActBlock_AdaIn(in_channels, in_channels, bias=bias_opt))
            blocks.append(nn.LeakyReLU(0.2))

        layer = nn.ModuleList(blocks)
        return layer

    def input_feature_extract(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
        batchnorm=False,
    ):
        if batchnorm:
            layer = nn.Sequential(
                nn.Conv3d(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=stride,
                    padding=padding,
                    bias=bias,
                ),
                nn.BatchNorm3d(out_channels),
                nn.ReLU(),
            )
        else:
            layer = nn.Sequential(
                nn.Conv3d(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=stride,
                    padding=padding,
                    bias=bias,
                ),
                nn.LeakyReLU(0.2),
                nn.Conv3d(
                    out_channels,
                    out_channels,
                    kernel_size,
                    stride=stride,
                    padding=padding,
                    bias=bias,
                ),
            )
        return layer

    def decoder(
        self,
        in_channels,
        out_channels,
        kernel_size=2,
        stride=2,
        padding=0,
        output_padding=0,
        bias=True,
    ):
        layer = nn.Sequential(
            nn.ConvTranspose3d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                output_padding=output_padding,
                bias=bias,
            ),
            nn.ReLU(),
        )
        return layer

    def outputs(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=0,
        bias=False,
        batchnorm=False,
    ):
        if batchnorm:
            layer = nn.Sequential(
                nn.Conv3d(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=stride,
                    padding=padding,
                    bias=bias,
                ),
                nn.BatchNorm3d(out_channels),
                nn.Tanh(),
            )
        else:
            # layer = nn.Sequential(
            #     nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
            #     nn.Tanh())
            layer = nn.Sequential(
                nn.Conv3d(
                    in_channels,
                    int(in_channels / 2),
                    kernel_size,
                    stride=stride,
                    padding=padding,
                    bias=bias,
                ),
                nn.LeakyReLU(0.2),
                nn.Conv3d(
                    int(in_channels / 2),
                    out_channels,
                    kernel_size,
                    stride=stride,
                    padding=padding,
                    bias=bias,
                ),
                nn.Softsign(),
            )
        return layer

    def forward(self, x, y, reg_code):
        # compose_field_e0_lvl1, warpped_inputx_lvl1_out, down_y, output_disp_e0_v, lvl1_v, e0
        lvl2_disp, _, _, lvl2_v, lvl1_v, lvl2_embedding = self.model_lvl2(
            x, y, reg_code
        )

        # lvl2_disp, lvl2_warp, lvl2_y, lvl2_v, lvl1_v, lvl2_embedding, lvl1_warp, lvl1_y = self.model_lvl2(x, y, reg_code)

        lvl2_disp_up = self.up_tri(lvl2_disp)
        warpped_x = self.transform(x, lvl2_disp_up.permute(0, 2, 3, 4, 1), self.grid_1)

        cat_input = torch.cat((warpped_x, y, lvl2_disp_up), 1)

        fea_e0 = self.input_encoder_lvl1(cat_input)
        e0 = self.down_conv(fea_e0)

        e0 = e0 + lvl2_embedding

        # e0 = self.resblock_group_lvl1(e0)
        for i in range(len(self.resblock_group_lvl1)):
            if i % 2 == 0:
                e0 = self.resblock_group_lvl1[i](e0, reg_code)
            else:
                e0 = self.resblock_group_lvl1[i](e0)

        e0 = self.up(e0)
        output_disp_e0_v = (
            self.output_lvl1(torch.cat([e0, fea_e0], dim=1)) * self.range_flow
        )
        # output_disp_e0 = self.diff_transform(output_disp_e0_v, self.grid_1)
        compose_field_e0_lvl1 = output_disp_e0_v + lvl2_disp_up

        warpped_inputx_lvl1_out = self.transform(
            x, compose_field_e0_lvl1.permute(0, 2, 3, 4, 1), self.grid_1
        )

        if self.is_train is True:
            return (
                compose_field_e0_lvl1,
                warpped_inputx_lvl1_out,
                y,
                output_disp_e0_v,
                lvl1_v,
                lvl2_disp,
                e0,
            )
            # return compose_field_e0_lvl1, warpped_inputx_lvl1_out, y, output_disp_e0_v, lvl1_v, lvl2_v, e0, lvl1_warp, lvl1_y, lvl2_warp, lvl2_y
        else:
            return compose_field_e0_lvl1


# -----------------------------------------------------------------------------
# Model inference (run via `python -m predict_gbm.preprocessing.dirac`)
# -----------------------------------------------------------------------------


def run_inference(
    model_name,
    datapath,
    start_channel=6,
    num_cblock=5,
    output_seg=True,
    save_transform=True,
):
    """Run DIRAC model inference for each case directory under datapath.

    Expects <datapath>/<case>/t1c_bet_normalized.nii.gz (pre-op, fixed) and
    t1c_bet_normalized_followup.nii.gz (follow-up, moving); writes warped images,
    occlusion segmentations and displacement fields next to them.
    """
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    model_lvl1 = Miccai2021_LDR_laplacian_unit_disp_add_AdaIn_lvl1(
        2,
        3,
        start_channel,
        is_train=True,
        imgshape=IMGSHAPE_4,
        range_flow=RANGE_FLOW,
        num_block=num_cblock,
    ).to(device)
    model_lvl2 = Miccai2021_LDR_laplacian_unit_disp_add_AdaIn_lvl2(
        2,
        3,
        start_channel,
        is_train=True,
        imgshape=IMGSHAPE_2,
        range_flow=RANGE_FLOW,
        model_lvl1=model_lvl1,
        num_block=num_cblock,
    ).to(device)

    model = Miccai2021_LDR_laplacian_unit_disp_add_AdaIn_lvl3(
        2,
        3,
        start_channel,
        is_train=True,
        imgshape=IMGSHAPE,
        range_flow=RANGE_FLOW,
        model_lvl2=model_lvl2,
        num_block=num_cblock,
    ).to(device)

    model_path = model_name
    model.load_state_dict(torch.load(model_path, map_location=device))

    transform = SpatialTransform_unit().to(device)

    for param in transform.parameters():
        param.requires_grad = False
        param.volatile = True

    # Validation
    val_fixed_list = sorted(glob.glob(f"{datapath}/*/t1c_bet_normalized.nii.gz"))
    val_moving_list = sorted(
        glob.glob(f"{datapath}/*/t1c_bet_normalized_followup.nii.gz")
    )

    print("Fixed:", val_fixed_list)
    print("Moving:", val_moving_list)
    print("Datapath:", datapath)

    valid_generator = Data.DataLoader(
        Validation_Brats(val_fixed_list, val_moving_list, None, None, norm=True),
        batch_size=1,
        shuffle=False,
        num_workers=2,
    )

    print("\nValidating...")
    for batch_idx, data in enumerate(valid_generator):
        fixed_image_path = val_fixed_list[batch_idx]
        patient_dir = os.path.dirname(fixed_image_path)
        patient_id = os.path.basename(patient_dir)

        template = nib.load(fixed_image_path)
        header, affine = template.header, template.affine

        Y_ori, X_ori = data["move"].to(device), data["fixed"].to(device)

        ori_img_shape = X_ori.shape[2:]
        h, w, d = ori_img_shape

        X = F.interpolate(X_ori, size=IMGSHAPE, mode="trilinear", align_corners=True)
        Y = F.interpolate(Y_ori, size=IMGSHAPE, mode="trilinear", align_corners=True)

        with torch.no_grad():
            reg_code = torch.tensor([0.3], dtype=X.dtype, device=X.device).unsqueeze(
                dim=0
            )
            F_X_Y, X_Y, Y_4x, F_xy, F_xy_lvl1, F_xy_lvl2, _ = model(X, Y, reg_code)

            F_Y_X, Y_X, X_4x, F_yx, F_yx_lvl1, F_yx_lvl2, _ = model(Y, X, reg_code)

            F_X_Y = F.interpolate(
                F_X_Y, size=ori_img_shape, mode="trilinear", align_corners=True
            )
            F_Y_X = F.interpolate(
                F_Y_X, size=ori_img_shape, mode="trilinear", align_corners=True
            )

            grid_unit = generate_grid_unit(ori_img_shape)
            grid_unit = (
                torch.from_numpy(np.reshape(grid_unit, (1,) + grid_unit.shape))
                .to(device)
                .float()
            )

            if output_seg:
                F_X_Y_warpped = transform(
                    F_X_Y, F_Y_X.permute(0, 2, 3, 4, 1), grid_unit
                )
                F_Y_X_warpped = transform(
                    F_Y_X, F_X_Y.permute(0, 2, 3, 4, 1), grid_unit
                )

                diff_fw = F_X_Y + F_Y_X_warpped  # Y
                diff_bw = F_Y_X + F_X_Y_warpped  # X

                fw_mask = (Y_ori > 0).float()
                bw_mask = (X_ori > 0).float()

                u_diff_fw = torch.sum(
                    torch.norm(diff_fw * fw_mask, dim=1, keepdim=True)
                ) / torch.sum(fw_mask)
                u_diff_bw = torch.sum(
                    torch.norm(diff_bw * bw_mask, dim=1, keepdim=True)
                ) / torch.sum(bw_mask)

                thresh_fw = (u_diff_fw + 0.015) * torch.ones_like(
                    Y_ori, device=Y_ori.device
                )
                thresh_bw = (u_diff_bw + 0.015) * torch.ones_like(
                    X_ori, device=X_ori.device
                )

                # smoothing
                norm_diff_fw = torch.norm(diff_fw, dim=1, keepdim=True)
                norm_diff_bw = torch.norm(diff_bw, dim=1, keepdim=True)

                smo_norm_diff_fw = F.avg_pool3d(
                    F.avg_pool3d(norm_diff_fw, kernel_size=5, stride=1, padding=2),
                    kernel_size=5,
                    stride=1,
                    padding=2,
                )
                smo_norm_diff_bw = F.avg_pool3d(
                    F.avg_pool3d(norm_diff_bw, kernel_size=5, stride=1, padding=2),
                    kernel_size=5,
                    stride=1,
                    padding=2,
                )

                occ_xy = (smo_norm_diff_fw > thresh_fw).float()  # y mask
                occ_yx = (smo_norm_diff_bw > thresh_bw).float()  # x mask

                # mask occ
                occ_xy = occ_xy * fw_mask
                occ_yx = occ_yx * bw_mask

                save_img(
                    occ_xy.cpu().numpy()[0, 0],
                    f"{patient_dir}/{patient_id}_xy_seg.nii.gz",
                    header=header,
                    affine=affine,
                )
                save_img(
                    occ_yx.cpu().numpy()[0, 0],
                    f"{patient_dir}/{patient_id}_yx_seg.nii.gz",
                    header=header,
                    affine=affine,
                )

                save_img(
                    norm_diff_fw.cpu().numpy()[0, 0],
                    f"{patient_dir}/{patient_id}_diff_fw.nii.gz",
                    header=header,
                    affine=affine,
                )
                save_img(
                    norm_diff_bw.cpu().numpy()[0, 0],
                    f"{patient_dir}/{patient_id}_diff_bw.nii.gz",
                    header=header,
                    affine=affine,
                )

            X_Y = transform(X_ori, F_X_Y.permute(0, 2, 3, 4, 1), grid_unit)
            Y_X = transform(Y_ori, F_Y_X.permute(0, 2, 3, 4, 1), grid_unit)

            save_img(
                X_Y.cpu().numpy()[0, 0],
                f"{patient_dir}/{patient_id}_X_Y.nii.gz",
                header=header,
                affine=affine,
            )
            save_img(
                Y_X.cpu().numpy()[0, 0],
                f"{patient_dir}/{patient_id}_Y_X.nii.gz",
                header=header,
                affine=affine,
            )

            if save_transform:
                # DIRAC predicts flow in normalized grid coordinates with channel order (z, y, x).
                # Save both normalized and voxel-space displacement fields for downstream refinement.
                f_x_y_norm = F_X_Y.cpu().numpy()[0].transpose(1, 2, 3, 0)
                f_y_x_norm = F_Y_X.cpu().numpy()[0].transpose(1, 2, 3, 0)

                f_x_y_voxel = np.zeros(F_X_Y.shape, dtype=np.float32)
                f_y_x_voxel = np.zeros(F_Y_X.shape, dtype=np.float32)

                f_x_y_voxel[0, 0] = F_X_Y[0, 2].cpu().numpy() * (h - 1) / 2
                f_x_y_voxel[0, 1] = F_X_Y[0, 1].cpu().numpy() * (w - 1) / 2
                f_x_y_voxel[0, 2] = F_X_Y[0, 0].cpu().numpy() * (d - 1) / 2

                f_y_x_voxel[0, 0] = F_Y_X[0, 2].cpu().numpy() * (h - 1) / 2
                f_y_x_voxel[0, 1] = F_Y_X[0, 1].cpu().numpy() * (w - 1) / 2
                f_y_x_voxel[0, 2] = F_Y_X[0, 0].cpu().numpy() * (d - 1) / 2

                # moving (follow-up) -> fixed (pre-op), useful for warping follow-up tumor labels into pre-op space
                save_flow(
                    f_y_x_norm,
                    f"{patient_dir}/{patient_id}_followup_to_preop_disp_norm.nii.gz",
                    header=header,
                    affine=affine,
                )
                save_flow(
                    f_y_x_voxel[0].transpose(1, 2, 3, 0),
                    f"{patient_dir}/{patient_id}_followup_to_preop_disp_voxel.nii.gz",
                    header=header,
                    affine=affine,
                )

                # fixed (pre-op) -> moving (follow-up), saved for completeness
                save_flow(
                    f_x_y_norm,
                    f"{patient_dir}/{patient_id}_preop_to_followup_disp_norm.nii.gz",
                    header=header,
                    affine=affine,
                )
                save_flow(
                    f_x_y_voxel[0].transpose(1, 2, 3, 0),
                    f"{patient_dir}/{patient_id}_preop_to_followup_disp_voxel.nii.gz",
                    header=header,
                    affine=affine,
                )

    print("Done.")


# -----------------------------------------------------------------------------
# Instance optimization
# -----------------------------------------------------------------------------


def make_identity_grid(d, h, w, device, dtype):
    xs = torch.linspace(-1, 1, w, device=device, dtype=dtype)
    ys = torch.linspace(-1, 1, h, device=device, dtype=dtype)
    zs = torch.linspace(-1, 1, d, device=device, dtype=dtype)
    zz, yy, xx = torch.meshgrid(zs, ys, xs, indexing="ij")
    return torch.stack((xx, yy, zz), dim=-1)[None]


def voxel_disp_to_norm(disp, d, h, w):
    dx, dy, dz = disp[:, 0], disp[:, 1], disp[:, 2]
    sx = 2.0 / max(w - 1, 1)
    sy = 2.0 / max(h - 1, 1)
    sz = 2.0 / max(d - 1, 1)
    return torch.stack((dx * sx, dy * sy, dz * sz), dim=-1)


def warp(img, disp, mode="bilinear"):
    _, _, d, h, w = img.shape
    grid0 = make_identity_grid(d, h, w, img.device, img.dtype)
    grid = grid0 + voxel_disp_to_norm(disp, d, h, w)
    return F.grid_sample(
        img, grid, mode=mode, padding_mode="border", align_corners=True
    )


def warp_field(field, disp):
    _, _, d, h, w = field.shape
    grid0 = make_identity_grid(d, h, w, field.device, field.dtype)
    grid = grid0 + voxel_disp_to_norm(disp, d, h, w)
    return F.grid_sample(
        field, grid, mode="bilinear", padding_mode="border", align_corners=True
    )


def resize_disp_voxel(disp, size):
    _, _, d0, h0, w0 = disp.shape
    d1, h1, w1 = size
    resized = F.interpolate(disp, size=size, mode="trilinear", align_corners=True)
    sx = (w1 - 1) / max(w0 - 1, 1)
    sy = (h1 - 1) / max(h0 - 1, 1)
    sz = (d1 - 1) / max(d0 - 1, 1)
    resized[:, 0] *= sx
    resized[:, 1] *= sy
    resized[:, 2] *= sz
    return resized


def _gaussian_kernel_1d(
    sigma: float, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    radius = int(math.ceil(3.0 * sigma))
    x = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    kernel = torch.exp(-0.5 * (x / sigma) ** 2)
    return kernel / kernel.sum()


def gaussian_blur_for_downsampling(
    x: torch.Tensor, size: tuple[int, int, int]
) -> torch.Tensor:
    """Anti-alias blur applied before resampling a (1,C,D,H,W) tensor down to `size`.

    Separable Gaussian with sigma = (factor - 1) / 2 per axis, where factor is the
    downsampling ratio of that axis under the align_corners=True convention. Axes that
    are not downsampled are left untouched; replicate padding avoids zero bleeding.
    """
    out = x
    channels = x.shape[1]
    for axis, (n_in, n_out) in enumerate(zip(x.shape[2:], size)):
        factor = (n_in - 1) / max(n_out - 1, 1)
        sigma = 0.5 * (factor - 1.0)
        if sigma < 1e-3:
            continue
        kernel = _gaussian_kernel_1d(sigma, x.device, x.dtype)
        shape = [1, 1, 1, 1, 1]
        shape[2 + axis] = kernel.numel()
        weight = kernel.view(shape).repeat(channels, 1, 1, 1, 1)
        # F.pad order is (w0, w1, h0, h1, d0, d1); tensor axis 0 (D) maps to pad[4:6].
        radius = kernel.numel() // 2
        pad = [0] * 6
        pad[2 * (2 - axis)] = radius
        pad[2 * (2 - axis) + 1] = radius
        out = F.conv3d(F.pad(out, pad, mode="replicate"), weight, groups=channels)
    return out


def resample_antialiased(x: torch.Tensor, size: tuple[int, int, int]) -> torch.Tensor:
    """Blur (see gaussian_blur_for_downsampling) then trilinearly resample to `size`."""
    if tuple(x.shape[2:]) == tuple(size):
        return x
    return F.interpolate(
        gaussian_blur_for_downsampling(x, size),
        size=size,
        mode="trilinear",
        align_corners=True,
    )


def resize_disp_voxel_antialiased(
    disp: torch.Tensor, size: tuple[int, int, int]
) -> torch.Tensor:
    """Anti-aliased resize_disp_voxel: blur, resample, rescale voxel magnitudes."""
    if tuple(disp.shape[2:]) == tuple(size):
        return disp
    return resize_disp_voxel(gaussian_blur_for_downsampling(disp, size), size)


def _per_axis_scale(
    disp: torch.Tensor, scale_w: float, scale_h: float, scale_d: float
) -> torch.Tensor:
    scale = torch.tensor(
        [scale_w, scale_h, scale_d], device=disp.device, dtype=disp.dtype
    )
    return disp * scale.view(1, 3, 1, 1, 1)


def voxel_disp_to_norm_units(disp: torch.Tensor) -> torch.Tensor:
    """Scale a (1,3,D,H,W) voxel displacement to normalized [-1,1] units per axis."""
    _, _, d, h, w = disp.shape
    return _per_axis_scale(
        disp, 2.0 / max(w - 1, 1), 2.0 / max(h - 1, 1), 2.0 / max(d - 1, 1)
    )


def voxel_disp_to_mok_units(disp: torch.Tensor) -> torch.Tensor:
    """Scale to the units of Mok's smoothness term: normalized displacement times the
    image dimensions, i.e. a factor 2N/(N-1) (about 2x voxel units) per axis."""
    _, _, d, h, w = disp.shape
    return _per_axis_scale(
        disp,
        2.0 * w / max(w - 1, 1),
        2.0 * h / max(h - 1, 1),
        2.0 * d / max(d - 1, 1),
    )


def _box_sum(x: torch.Tensor, win: int) -> torch.Tensor:
    """Sum over a win^3 box with zero padding, as three separable 1-D convolutions
    (identical to a dense win^3 ones kernel, 3*win instead of win^3 taps)."""
    pad = win // 2
    ones = torch.ones((1, 1, 1, 1, win), device=x.device, dtype=x.dtype)
    x = F.conv3d(x, ones, padding=(0, 0, pad))
    x = F.conv3d(x, ones.view(1, 1, 1, win, 1), padding=(0, pad, 0))
    return F.conv3d(x, ones.view(1, 1, win, 1, 1), padding=(pad, 0, 0))


def ncc_loss(
    i: torch.Tensor,
    j: torch.Tensor,
    weight: torch.Tensor | None = None,
    win: int = 3,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Negative local NCC following Mok's NCC_weight: local statistics over a win^3 box,
    cc = cross^2 / (var_i * var_j + eps), the weight map applied after the local
    statistics, mean over all voxels. Returns a value in [-1, 0]."""
    n_win = float(win**3)

    def conv(x: torch.Tensor) -> torch.Tensor:
        return _box_sum(x, win)

    i_sum, j_sum = conv(i), conv(j)
    i2_sum, j2_sum, ij_sum = conv(i * i), conv(j * j), conv(i * j)
    u_i, u_j = i_sum / n_win, j_sum / n_win
    cross = ij_sum - u_j * i_sum - u_i * j_sum + u_i * u_j * n_win
    i_var = i2_sum - 2 * u_i * i_sum + u_i * u_i * n_win
    j_var = j2_sum - 2 * u_j * j_sum + u_j * u_j * n_win
    cc = cross * cross / (i_var * j_var + eps)
    if weight is not None:
        cc = cc * weight.to(dtype=cc.dtype)
    return -cc.mean()


def multi_scale_ncc_loss(
    i: torch.Tensor,
    j: torch.Tensor,
    weight: torch.Tensor | None = None,
    win: int = 3,
    n_scales: int = 3,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Mok's multi_resolution_NCC_weight: NCC with window win + 2*(n_scales-1-s) at
    scale s over 3x3x3/stride-2 average-pooled images and weights, weighted 1/2^s.
    With the defaults this is windows 7, 5, 3 at scales 0, 1, 2."""
    total: torch.Tensor | None = None
    for s in range(n_scales):
        win_s = win + 2 * (n_scales - 1 - s)
        term = ncc_loss(i, j, weight, win=win_s, eps=eps) / (2**s)
        total = term if total is None else total + term
        if s + 1 < n_scales:
            i = F.avg_pool3d(i, 3, stride=2, padding=1, count_include_pad=False)
            j = F.avg_pool3d(j, 3, stride=2, padding=1, count_include_pad=False)
            if weight is not None:
                weight = F.avg_pool3d(
                    weight, 3, stride=2, padding=1, count_include_pad=False
                )
    assert total is not None
    return total


def smoothness(disp: torch.Tensor) -> torch.Tensor:
    """Mok's smoothloss: mean squared forward difference per axis, averaged over axes.
    Callers pass the field in Mok's units (see voxel_disp_to_mok_units)."""
    dx = disp[:, :, :, :, 1:] - disp[:, :, :, :, :-1]
    dy = disp[:, :, :, 1:, :] - disp[:, :, :, :-1, :]
    dz = disp[:, :, 1:, :, :] - disp[:, :, :-1, :, :]
    return (dx.pow(2).mean() + dy.pow(2).mean() + dz.pow(2).mean()) / 3.0


def inv_consistency(
    d_fwd: torch.Tensor,
    d_bwd: torch.Tensor,
    m_fwd: torch.Tensor | None = None,
    m_bwd: torch.Tensor | None = None,
) -> torch.Tensor:
    """Mok's inverse-consistency term, summed over both directions: the unsquared norm
    of the composition error in normalized coordinates, weighted by (1 - mask), mean
    over all voxels. Fields are in voxel units of their own grid."""
    bwd_warped, fwd_warped = warp_field(d_bwd, d_fwd), warp_field(d_fwd, d_bwd)
    err_fwd = voxel_disp_to_norm_units(d_fwd + bwd_warped).norm(dim=1, keepdim=True)
    err_bwd = voxel_disp_to_norm_units(d_bwd + fwd_warped).norm(dim=1, keepdim=True)
    if m_fwd is not None:
        err_fwd = err_fwd * (1.0 - m_fwd.to(dtype=err_fwd.dtype))
    if m_bwd is not None:
        err_bwd = err_bwd * (1.0 - m_bwd.to(dtype=err_bwd.dtype))
    return err_fwd.mean() + err_bwd.mean()


def instopt_terms(
    B_l: torch.Tensor,
    F_l: torch.Tensor,
    m_fb_l: torch.Tensor,
    m_bf_l: torch.Tensor,
    u_fb: torch.Tensor,
    u_bf: torch.Tensor,
    reg_fb: torch.Tensor,
    reg_bf: torch.Tensor,
    ncc_win: int = 3,
    multi_scale_ncc: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Similarity, smoothness and inverse-consistency terms of one pyramid level.

    B_l / F_l: preop and followup images at this level. m_fb_l / m_bf_l: occlusion
    masks in preop and followup space; (1 - m) weights the similarity and
    inverse-consistency terms. u_fb / u_bf: dense displacement fields in voxel units of
    this level (followup->preop on the preop grid, preop->followup on the followup
    grid). reg_fb / reg_bf: the fields the smoothness term is applied to (the total
    field or the residual only). Returns (l_s, l_r, l_inv), each summed over both
    directions, so l_s is in [-2, 0].
    """
    w_fb, w_bf = 1.0 - m_fb_l, 1.0 - m_bf_l
    if multi_scale_ncc:
        l_s = multi_scale_ncc_loss(
            B_l, warp(F_l, u_fb), w_fb, win=ncc_win
        ) + multi_scale_ncc_loss(F_l, warp(B_l, u_bf), w_bf, win=ncc_win)
    else:
        l_s = ncc_loss(B_l, warp(F_l, u_fb), w_fb, win=ncc_win) + ncc_loss(
            F_l, warp(B_l, u_bf), w_bf, win=ncc_win
        )
    l_r = smoothness(voxel_disp_to_mok_units(reg_fb)) + smoothness(
        voxel_disp_to_mok_units(reg_bf)
    )
    l_inv = inv_consistency(u_fb, u_bf, m_fb_l, m_bf_l)
    return l_s, l_r, l_inv


def _field_magnitude_stats(disp: torch.Tensor) -> dict[str, float]:
    mag = disp.norm(dim=1).flatten()
    k = max(1, int(round(0.95 * mag.numel())))
    return {
        "mean": float(mag.mean()),
        "p95": float(mag.kthvalue(k).values),
        "max": float(mag.max()),
    }


def _format_stats(s: dict[str, float]) -> str:
    return f"{s['mean']:.3f}/{s['p95']:.3f}/{s['max']:.3f}"


def _control_grid_to_voxel_disp(
    c: torch.Tensor, size: tuple[int, int, int]
) -> torch.Tensor:
    """Upsample a control grid in normalized per-axis units to a dense displacement in
    voxel units of a grid of shape `size`."""
    d, h, w = size
    dense = F.interpolate(c, size=size, mode="trilinear", align_corners=True)
    return _per_axis_scale(dense, (w - 1) / 2.0, (h - 1) / 2.0, (d - 1) / 2.0)


def load_image_for_grid_sample(path, device):
    img = nib.load(path).get_fdata().astype(np.float32)
    return torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).unsqueeze(0).to(device)


def load_mask_for_grid_sample(path, device):
    mask = nib.load(path).get_fdata().astype(np.float32)
    mask = (mask > 0.5).astype(np.float32)
    return torch.from_numpy(mask).permute(2, 0, 1).unsqueeze(0).unsqueeze(0).to(device)


def load_dirac_voxel_disp_for_grid_sample(path, device):
    disp = nib.load(path).get_fdata().astype(np.float32)  # (H,W,D,3)

    # DIRAC voxel output components follow image axes: [axis0(H), axis1(W), axis2(D)].
    # grid_sample expects channels [dx(W), dy(H), dz(D)] for tensor shape (1,3,D,H,W).
    comp_axis0 = torch.from_numpy(disp[..., 0]).permute(2, 0, 1)  # (D,H,W)
    comp_axis1 = torch.from_numpy(disp[..., 1]).permute(2, 0, 1)  # (D,H,W)
    comp_axis2 = torch.from_numpy(disp[..., 2]).permute(2, 0, 1)  # (D,H,W)

    disp_grid = torch.stack((comp_axis1, comp_axis0, comp_axis2), dim=0).unsqueeze(0)
    return disp_grid.to(device)


def grid_sample_disp_to_dirac_voxel(disp):
    dx = disp[0, 0].permute(1, 2, 0).cpu().numpy()
    dy = disp[0, 1].permute(1, 2, 0).cpu().numpy()
    dz = disp[0, 2].permute(1, 2, 0).cpu().numpy()
    return np.stack((dy, dx, dz), axis=-1).astype(np.float32)


def dirac_instance_optimization(
    B: torch.Tensor,
    Fup: torch.Tensor,
    disp_fb_init: torch.Tensor,
    disp_bf_init: torch.Tensor,
    m_fb_fixed: torch.Tensor | None = None,
    m_bf_fixed: torch.Tensor | None = None,
    lambdas_reg: tuple[float, ...] = (0.25, 0.3, 0.3, 0.35, 0.35),
    lambdas_inv: tuple[float, ...] = (1.0, 2.0, 4.0, 8.0, 10.0),
    lrs: tuple[float, ...] = (1e-2, 5e-3, 5e-3, 3e-3, 3e-3),
    iters: tuple[int, ...] = (150, 100, 100, 100, 50),
    regularize: Literal["total", "residual"] = "residual",
    multi_scale_ncc: bool = False,
    ncc_win: int = 3,
    coarsest_size: int = 80,
    grid_range: tuple[int, int] = (32, 64),
    stats: dict[str, Any] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Instance optimization of the DIRAC displacement fields (Mok & Chung, Table 1).

    The output field is `u = u_dirac + upsample(c)`: the network field `u_dirac` stays
    frozen at native resolution and a residual control grid `c` (zero-initialized, in
    normalized per-axis coordinates so Table 1's Adam learning rates apply) is optimized
    over a coarse-to-fine pyramid. Levels are built with an anti-aliased resize
    (Gaussian blur + trilinear), aspect-preserving, from `coarsest_size` on the largest
    axis to native resolution with geometric spacing; control grids go linearly from
    grid_range[0]^3 to grid_range[1]^3. Loss per level, with Mok's normalization of each
    term: (1 - lam_reg) * l_s + lam_reg * l_r + lam_inv * l_inv (see instopt_terms).
    `regularize` selects what the smoothness term l_r acts on: "total" = the full field
    u_dirac + upsample(c) (Mok's convention, smooths the network field as well);
    "residual" = the correction upsample(c) only (pipeline default: keeps the correction
    inside the occlusion mask bounded and leaves the network field's detail intact).
    `multi_scale_ncc` replaces the single-window NCC (win=`ncc_win`) by Mok's
    multi-resolution NCC (windows 7/5/3); default off.

    B, Fup: preop and followup images (1,1,D,H,W). disp_*_init: voxel displacement
    fields (1,3,D,H,W) with channels [dx->W, dy->H, dz->D]. m_*_fixed: occlusion masks
    in preop / followup space. If `stats` is given, per-level diagnostics (term values
    and gradient norms at iteration 0, field magnitude before/after) are stored in it.
    Returns (disp_fb, disp_bf, m_fb, m_bf) at native resolution.

    GPU runs are not bitwise reproducible (the grid_sample backward uses atomic adds);
    on 240x240x155 cases two runs differ by about 0.03-0.07 voxels on average and up
    to about 2 voxels at isolated voxel clusters in the brain outside the occlusion
    mask (below 0.8 voxels inside it), so similarity and fold statistics are stable
    to three decimals while |u| statistics are not. CPU runs are deterministic.
    """
    if m_fb_fixed is None:
        m_fb_fixed = torch.zeros_like(B)
    if m_bf_fixed is None:
        m_bf_fixed = torch.zeros_like(B)
    n_levels = len(lrs)
    if not (len(iters) == len(lambdas_reg) == len(lambdas_inv) == n_levels):
        raise ValueError("lrs, iters, lambdas_reg and lambdas_inv must have equal length")
    if regularize not in ("total", "residual"):
        raise ValueError(f"regularize must be 'total' or 'residual', got {regularize!r}")

    native = (int(B.shape[2]), int(B.shape[3]), int(B.shape[4]))
    scale_min = min(coarsest_size / max(native), 1.0)
    level_sizes: list[tuple[int, int, int]] = []
    grid_sizes: list[int] = []
    for lvl in range(n_levels):
        t = lvl / max(n_levels - 1, 1) if n_levels > 1 else 1.0
        s = scale_min ** (1.0 - t)
        level_sizes.append(tuple(max(2, int(round(n * s))) for n in native))  # type: ignore[arg-type]
        grid_sizes.append(int(round(grid_range[0] + (grid_range[1] - grid_range[0]) * t)))

    u_dirac_fb, u_dirac_bf = disp_fb_init.detach(), disp_bf_init.detach()
    g0 = grid_sizes[0]
    c_fb = torch.zeros((1, 3, g0, g0, g0), device=B.device, dtype=B.dtype)
    c_bf = torch.zeros_like(c_fb)

    def total_native(c: torch.Tensor, u_dirac: torch.Tensor) -> torch.Tensor:
        return u_dirac + _control_grid_to_voxel_disp(c, native)

    level_stats: list[dict[str, Any]] = []
    for lvl, (lr, n_iter, lam_reg, lam_inv, size, g) in enumerate(
        zip(lrs, iters, lambdas_reg, lambdas_inv, level_sizes, grid_sizes)
    ):
        if c_fb.shape[2] != g:
            c_fb = F.interpolate(c_fb, size=(g, g, g), mode="trilinear", align_corners=True)
            c_bf = F.interpolate(c_bf, size=(g, g, g), mode="trilinear", align_corners=True)
        c_fb = c_fb.detach().requires_grad_(True)
        c_bf = c_bf.detach().requires_grad_(True)
        with torch.no_grad():
            B_l, F_l = resample_antialiased(B, size), resample_antialiased(Fup, size)
            m_fb_l = resample_antialiased(m_fb_fixed, size).clamp(0.0, 1.0)
            m_bf_l = resample_antialiased(m_bf_fixed, size).clamp(0.0, 1.0)
            u_dirac_fb_l = resize_disp_voxel_antialiased(u_dirac_fb, size)
            u_dirac_bf_l = resize_disp_voxel_antialiased(u_dirac_bf, size)
            mag_before = _field_magnitude_stats(total_native(c_fb, u_dirac_fb))
        # Adam moves each control point by about lr per step; in native voxels that is
        # lr * (N - 1) / 2 per axis (x, y, z).
        step_vox = tuple(lr * (n - 1) / 2.0 for n in (native[2], native[1], native[0]))

        def weighted_terms() -> dict[str, torch.Tensor]:
            r_fb = _control_grid_to_voxel_disp(c_fb, size)
            r_bf = _control_grid_to_voxel_disp(c_bf, size)
            u_fb, u_bf = u_dirac_fb_l + r_fb, u_dirac_bf_l + r_bf
            reg_fb, reg_bf = (u_fb, u_bf) if regularize == "total" else (r_fb, r_bf)
            l_s, l_r, l_inv = instopt_terms(
                B_l, F_l, m_fb_l, m_bf_l, u_fb, u_bf, reg_fb, reg_bf, ncc_win, multi_scale_ncc
            )
            return {
                "similarity": (1.0 - lam_reg) * l_s,
                "smoothness": lam_reg * l_r,
                "inverse_consistency": lam_inv * l_inv,
            }

        # Diagnostics at iteration 0: raw term values and the gradient norm of each
        # weighted term with respect to both control grids.
        init = weighted_terms()
        init_values = {
            "similarity": init["similarity"].item() / (1.0 - lam_reg),
            "smoothness": init["smoothness"].item() / lam_reg if lam_reg else float("nan"),
            "inverse_consistency": init["inverse_consistency"].item() / lam_inv if lam_inv else float("nan"),
        }
        init_grad_norms: dict[str, float] = {}
        for name, term in init.items():
            grads = torch.autograd.grad(term, [c_fb, c_bf], retain_graph=True, allow_unused=True)
            sq = sum(float((gr**2).sum()) for gr in grads if gr is not None)
            init_grad_norms[name] = math.sqrt(sq)
        del init
        logger.debug(
            f"IO level {lvl} init: l_s {init_values['similarity']:.4f}, "
            f"l_r {init_values['smoothness']:.4e}, l_inv {init_values['inverse_consistency']:.4e}; "
            f"|d(weighted term)/dc|: similarity {init_grad_norms['similarity']:.3e}, "
            f"smoothness {init_grad_norms['smoothness']:.3e}, "
            f"inverse_consistency {init_grad_norms['inverse_consistency']:.3e}"
        )

        opt = torch.optim.Adam([c_fb, c_bf], lr=lr)
        for _ in range(n_iter):
            loss = sum(weighted_terms().values())
            opt.zero_grad()
            loss.backward()
            opt.step()

        c_fb, c_bf = c_fb.detach(), c_bf.detach()
        with torch.no_grad():
            mag_after = _field_magnitude_stats(total_native(c_fb, u_dirac_fb))
            residual_mean = float(_control_grid_to_voxel_disp(c_fb, native).norm(dim=1).mean())
        logger.info(
            f"IO level {lvl}: size {size}, grid {g}^3, {n_iter} iters, lr {lr:g} "
            f"(initial step ~ {step_vox[0]:.2f}/{step_vox[1]:.2f}/{step_vox[2]:.2f} voxels x/y/z), "
            f"|u| mean/p95/max {_format_stats(mag_before)} -> {_format_stats(mag_after)}, "
            f"mean |u - u_dirac| {residual_mean:.3f}"
        )
        level_stats.append(
            {
                "level": lvl,
                "size": size,
                "grid": g,
                "iters": n_iter,
                "lr": lr,
                "step_vox": step_vox,
                "init_terms": init_values,
                "init_grad_norms": init_grad_norms,
                "mag_before": mag_before,
                "mag_after": mag_after,
                "residual_mean": residual_mean,
            }
        )

    with torch.no_grad():
        disp_fb = total_native(c_fb, u_dirac_fb)
        disp_bf = total_native(c_bf, u_dirac_bf)
    if stats is not None:
        stats["levels"] = level_stats
        stats["regularize"] = regularize
    return disp_fb, disp_bf, m_fb_fixed.detach(), m_bf_fixed.detach()


# -----------------------------------------------------------------------------
# Pipeline API (used by norm_ss_coregistration.register_recurrence)
# -----------------------------------------------------------------------------


def run_dirac_inference(t1c_pre_file: Path, t1c_post_file: Path, workdir: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    model_path = (
        repo_root
        / "predict_gbm"
        / "data"
        / "models"
        / (
            "Brats_NCC_disp_fea6b5_AdaIn64_t1ce_fbcon_occ01_inv5_a0015_aug_mean_fffixed_github_stagelvl3_64000.pth"
        )
    )
    if not model_path.exists():
        raise FileNotFoundError(f"Missing DIRAC model checkpoint: {model_path}")

    infer_case_dir = workdir / "dirac_infer_case"
    if infer_case_dir.exists():
        shutil.rmtree(infer_case_dir)
    infer_case_dir.mkdir(parents=True, exist_ok=True)

    shutil.copyfile(
        str(t1c_pre_file), str(infer_case_dir / "t1c_bet_normalized.nii.gz")
    )
    shutil.copyfile(
        str(t1c_post_file), str(infer_case_dir / "t1c_bet_normalized_followup.nii.gz")
    )

    cmd = [
        sys.executable,
        "-m",
        "predict_gbm.preprocessing.dirac",
        "--modelname",
        str(model_path),
        "--datapath",
        str(workdir),
        "--output_seg",
        "True",
        "--save_transform",
        "True",
    ]
    subprocess.run(cmd, cwd=str(repo_root), check=True)

    for suffix in [
        "followup_to_preop_disp_voxel",
        "preop_to_followup_disp_voxel",
        "xy_seg",
        "yx_seg",
    ]:
        src = infer_case_dir / f"dirac_infer_case_{suffix}.nii.gz"
        if not src.exists():
            raise FileNotFoundError(f"Expected DIRAC inference output missing: {src}")

        shutil.copyfile(str(src), str(workdir / src.name))

    shutil.rmtree(infer_case_dir)


def resolve_dirac_disp_field(workdir: Path, suffix: str) -> Path:
    candidates = sorted(workdir.glob(f"*_{suffix}.nii.gz"))
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise FileNotFoundError(
            f"Could not find DIRAC displacement field '*_{suffix}.nii.gz' in {workdir}. "
            "Run BRATS_infer_DIRAC.py first or provide the expected file in this directory."
        )

    raise FileNotFoundError(
        f"Expected exactly one '*_{suffix}.nii.gz' in {workdir}, found: {candidates}"
    )


def optimize_warp_field(
    t1c_pre_file: Path,
    t1c_post_file: Path,
    followup_to_preop_disp: Path,
    preop_to_followup_disp: Path,
    optimized_followup_to_preop_disp: Path,
    preop_mask_file: Path | None = None,
    followup_mask_file: Path | None = None,
    device: torch.device | None = None,
):
    device = device or torch.device("cpu" if not torch.cuda.is_available() else "cuda")
    preop = load_image_for_grid_sample(t1c_pre_file, device)
    followup = load_image_for_grid_sample(t1c_post_file, device)
    disp_fb = load_dirac_voxel_disp_for_grid_sample(followup_to_preop_disp, device)
    disp_bf = load_dirac_voxel_disp_for_grid_sample(preop_to_followup_disp, device)
    preop_mask = (
        load_mask_for_grid_sample(preop_mask_file, device)
        if preop_mask_file is not None
        else None
    )
    followup_mask = (
        load_mask_for_grid_sample(followup_mask_file, device)
        if followup_mask_file is not None
        else None
    )
    disp_fb_opt, _, _, _ = dirac_instance_optimization(
        B=preop,
        Fup=followup,
        disp_fb_init=disp_fb,
        disp_bf_init=disp_bf,
        m_fb_fixed=preop_mask,
        m_bf_fixed=followup_mask,
    )
    fb_voxel = grid_sample_disp_to_dirac_voxel(disp_fb_opt)
    save_nifti(fb_voxel, t1c_pre_file, optimized_followup_to_preop_disp)


def warp_image_to_preop(
    image_file: Path,
    reference_file: Path,
    disp_field_file: Path,
    out_file: Path,
    device: torch.device,
    mode: str = "bilinear",
) -> None:
    """Warp a followup-space image into preop space using a DIRAC voxel displacement field."""
    image = load_image_for_grid_sample(image_file, device)
    disp = load_dirac_voxel_disp_for_grid_sample(disp_field_file, device)
    warped = warp(image, disp, mode=mode)
    save_nifti(warped[0, 0].permute(1, 2, 0).cpu().numpy(), reference_file, out_file)


def apply_longitudinal_warp(
    t1c_pre_file: Path,
    t1c_post_file: Path,
    recurrence_seg_file: Path,
    optimized_followup_to_preop_disp: Path,
    warped_post_out: Path,
    recurrence_out: Path,
):
    device = torch.device("cpu" if not torch.cuda.is_available() else "cuda")
    warp_image_to_preop(
        image_file=t1c_post_file,
        reference_file=t1c_pre_file,
        disp_field_file=optimized_followup_to_preop_disp,
        out_file=warped_post_out,
        device=device,
        mode="bilinear",
    )
    warp_image_to_preop(
        image_file=recurrence_seg_file,
        reference_file=t1c_pre_file,
        disp_field_file=optimized_followup_to_preop_disp,
        out_file=recurrence_out,
        device=device,
        mode="nearest",
    )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--modelname",
        type=str,
        dest="modelname",
        default=(
            "Brats_NCC_disp_fea6b5_AdaIn64_t1ce_fbcon_occ01_inv5_a0015_aug_mean_fffixed_github_stagelvl3_64000.pth"
        ),
        help="Model name",
    )
    parser.add_argument(
        "--start_channel",
        type=int,
        dest="start_channel",
        default=6,
        help="number of start channels",
    )
    parser.add_argument(
        "--datapath",
        type=str,
        dest="datapath",
        default="../Dataset/test",
        help="data path for training images",
    )
    parser.add_argument(
        "--num_cblock",
        type=int,
        dest="num_cblock",
        default=5,
        help="Number of conditional block",
    )
    parser.add_argument(
        "--output_seg",
        type=bool,
        dest="output_seg",
        default=True,
        help="True: save segmentation map",
    )
    parser.add_argument(
        "--save_transform",
        type=bool,
        dest="save_transform",
        default=True,
        help="True: save deformation fields for reuse/optimization",
    )
    opt = parser.parse_args()

    print("Running DIRAC inference with %s ..." % opt.modelname)
    run_inference(
        model_name=opt.modelname,
        datapath=opt.datapath,
        start_channel=opt.start_channel,
        num_cblock=opt.num_cblock,
        output_seg=opt.output_seg,
        save_transform=opt.save_transform,
    )
