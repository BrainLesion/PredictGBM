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
import os
import shutil
import subprocess
import sys
from argparse import ArgumentParser
from pathlib import Path

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


def pad_tensor_to_min_size(x, min_size, mode="constant", value=0.0):
    _, _, d, h, w = x.shape
    min_d, min_h, min_w = min_size
    pad_d, pad_h, pad_w = max(min_d - d, 0), max(min_h - h, 0), max(min_w - w, 0)
    pad_d0, pad_d1 = pad_d // 2, pad_d - (pad_d // 2)
    pad_h0, pad_h1 = pad_h // 2, pad_h - (pad_h // 2)
    pad_w0, pad_w1 = pad_w // 2, pad_w - (pad_w // 2)
    pad = (pad_w0, pad_w1, pad_h0, pad_h1, pad_d0, pad_d1)
    x_pad = (
        F.pad(x, pad, mode=mode, value=float(value))
        if mode == "constant"
        else F.pad(x, pad, mode=mode)
    )
    crop_slices = (
        slice(pad_d0, pad_d0 + d),
        slice(pad_h0, pad_h0 + h),
        slice(pad_w0, pad_w0 + w),
    )
    return x_pad, crop_slices


def crop_to_slices(x, crop_slices):
    d_slice, h_slice, w_slice = crop_slices
    return x[:, :, d_slice, h_slice, w_slice]


def ncc_loss(i, j, mask=None, win=3, eps=1e-5):
    pad = win // 2
    filt = torch.ones((1, 1, win, win, win), device=i.device, dtype=i.dtype)

    def conv(x):
        return F.conv3d(x, filt, padding=pad)

    if mask is None:
        mask = torch.ones_like(i)
    mask = mask.to(dtype=i.dtype)
    i2, j2, ij = i * i, j * j, i * j
    w_sum = conv(mask)
    i_sum, j_sum = conv(mask * i), conv(mask * j)
    i2_sum, j2_sum, ij_sum = conv(mask * i2), conv(mask * j2), conv(mask * ij)
    u_i, u_j = i_sum / (w_sum + eps), j_sum / (w_sum + eps)
    cross = ij_sum - u_j * i_sum - u_i * j_sum + u_i * u_j * w_sum
    i_var = i2_sum - 2 * u_i * i_sum + u_i * u_i * w_sum
    j_var = j2_sum - 2 * u_j * j_sum + u_j * u_j * w_sum
    ncc = cross * cross / (i_var * j_var + eps)
    valid = (w_sum > 0).to(dtype=i.dtype)
    return -(ncc * valid).sum()


def smoothness(disp, valid_mask=None):
    dx = disp[:, :, :, :, 1:] - disp[:, :, :, :, :-1]
    dy = disp[:, :, :, 1:, :] - disp[:, :, :, :-1, :]
    dz = disp[:, :, 1:, :, :] - disp[:, :, :-1, :, :]
    if valid_mask is None:
        return dx.pow(2).sum() + dy.pow(2).sum() + dz.pow(2).sum()
    valid_mask = valid_mask.to(dtype=disp.dtype)
    mx = valid_mask[:, :, :, :, 1:] * valid_mask[:, :, :, :, :-1]
    my = valid_mask[:, :, :, 1:, :] * valid_mask[:, :, :, :-1, :]
    mz = valid_mask[:, :, 1:, :, :] * valid_mask[:, :, :-1, :, :]
    return (dx.pow(2) * mx).sum() + (dy.pow(2) * my).sum() + (dz.pow(2) * mz).sum()


def inv_consistency(d_fwd, d_bwd, m_fwd=None, m_bwd=None, valid_mask=None):
    bwd_warped, fwd_warped = warp_field(d_bwd, d_fwd), warp_field(d_fwd, d_bwd)
    err_fwd = ((d_fwd + bwd_warped) ** 2).sum(dim=1, keepdim=True)
    err_bwd = ((d_bwd + fwd_warped) ** 2).sum(dim=1, keepdim=True)
    w_fwd, w_bwd = torch.ones_like(err_fwd), torch.ones_like(err_bwd)
    if m_fwd is not None:
        w_fwd = w_fwd * (1.0 - m_fwd.to(dtype=err_fwd.dtype))
    if m_bwd is not None:
        w_bwd = w_bwd * (1.0 - m_bwd.to(dtype=err_bwd.dtype))
    if valid_mask is not None:
        vm = valid_mask.to(dtype=err_fwd.dtype)
        w_fwd, w_bwd = w_fwd * vm, w_bwd * vm
    return (err_fwd * w_fwd).sum() + (err_bwd * w_bwd).sum()


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
    B,
    Fup,
    disp_fb_init,
    disp_bf_init,
    m_fb_fixed=None,
    m_bf_fixed=None,
    lambdas_reg=(0.25, 0.3, 0.3, 0.35, 0.35),
    lambdas_inv=(1.0, 2.0, 4.0, 8.0, 10.0),
    lrs=(1e-2, 5e-3, 5e-3, 3e-3, 3e-3),
    iters=(150, 100, 100, 100, 50),
):
    if m_fb_fixed is None:
        m_fb_fixed = torch.zeros_like(B)
    if m_bf_fixed is None:
        m_bf_fixed = torch.zeros_like(B)
    _, _, d_full, h_full, w_full = B.shape
    dmin, hmin, wmin, g_min, g_max = 80, 80, 80, 32, 64
    n_levels = len(lrs)
    scale_min = min(dmin / d_full, hmin / h_full, wmin / w_full, 1.0)
    scales = [
        scale_min + (1.0 - scale_min) * i / max(n_levels - 1, 1)
        for i in range(n_levels)
    ]
    pyr_sizes_base = [
        (
            max(1, int(round(d_full * s))),
            max(1, int(round(h_full * s))),
            max(1, int(round(w_full * s))),
        )
        for s in scales
    ]
    pyr_sizes = [
        (max(d, dmin), max(h, hmin), max(w, wmin)) for (d, h, w) in pyr_sizes_base
    ]
    grid_sizes = [
        int(round(g_min + (g_max - g_min) * i / max(n_levels - 1, 1)))
        for i in range(n_levels)
    ]
    disp_fb_full, disp_bf_full = (
        disp_fb_init.clone().detach(),
        disp_bf_init.clone().detach(),
    )

    for lvl, (lr, n_iter, lam_reg, lam_inv) in enumerate(
        zip(lrs, iters, lambdas_reg, lambdas_inv)
    ):
        d_base, h_base, w_base = pyr_sizes_base[lvl]
        d, h, w = pyr_sizes[lvl]
        g = grid_sizes[lvl]
        B_l_base = F.interpolate(
            B, size=(d_base, h_base, w_base), mode="trilinear", align_corners=True
        )
        F_l_base = F.interpolate(
            Fup, size=(d_base, h_base, w_base), mode="trilinear", align_corners=True
        )
        mfb_l_base = F.interpolate(
            m_fb_fixed, size=(d_base, h_base, w_base), mode="nearest"
        )
        mbf_l_base = F.interpolate(
            m_bf_fixed, size=(d_base, h_base, w_base), mode="nearest"
        )
        B_l, crop_slices = pad_tensor_to_min_size(
            B_l_base, min_size=(dmin, hmin, wmin), mode="replicate"
        )
        F_l, _ = pad_tensor_to_min_size(
            F_l_base, min_size=(dmin, hmin, wmin), mode="replicate"
        )
        mfb_l, _ = pad_tensor_to_min_size(
            mfb_l_base, min_size=(dmin, hmin, wmin), mode="constant", value=1.0
        )
        mbf_l, _ = pad_tensor_to_min_size(
            mbf_l_base, min_size=(dmin, hmin, wmin), mode="constant", value=1.0
        )
        valid_mask = torch.zeros_like(B_l)
        d_slice, h_slice, w_slice = crop_slices
        valid_mask[:, :, d_slice, h_slice, w_slice] = 1.0
        disp_fb_l_base = resize_disp_voxel(disp_fb_full, size=(d_base, h_base, w_base))
        disp_bf_l_base = resize_disp_voxel(disp_bf_full, size=(d_base, h_base, w_base))
        disp_fb_l, crop_slices_disp = pad_tensor_to_min_size(
            disp_fb_l_base, min_size=(dmin, hmin, wmin), mode="replicate"
        )
        disp_bf_l, _ = pad_tensor_to_min_size(
            disp_bf_l_base, min_size=(dmin, hmin, wmin), mode="replicate"
        )
        if crop_slices_disp != crop_slices:
            raise RuntimeError("Padding crop mismatch")
        cp_fb = (
            F.interpolate(
                disp_fb_l, size=(g, g, g), mode="trilinear", align_corners=True
            )
            .detach()
            .requires_grad_(True)
        )
        cp_bf = (
            F.interpolate(
                disp_bf_l, size=(g, g, g), mode="trilinear", align_corners=True
            )
            .detach()
            .requires_grad_(True)
        )
        opt = torch.optim.Adam([cp_fb, cp_bf], lr=lr)

        for _ in range(n_iter):
            disp_fb = F.interpolate(
                cp_fb, size=(d, h, w), mode="trilinear", align_corners=True
            )
            disp_bf = F.interpolate(
                cp_bf, size=(d, h, w), mode="trilinear", align_corners=True
            )
            F_warp, B_warp = warp(F_l, disp_fb), warp(B_l, disp_bf)
            Ls = ncc_loss(B_l, F_warp, mask=(1 - mfb_l)) + ncc_loss(
                F_l, B_warp, mask=(1 - mbf_l)
            )
            Lr = smoothness(disp_fb, valid_mask) + smoothness(disp_bf, valid_mask)
            Linv = inv_consistency(disp_fb, disp_bf, mfb_l, mbf_l, valid_mask)
            loss = (1 - lam_reg) * Ls + lam_reg * Lr + lam_inv * Linv
            opt.zero_grad()
            loss.backward()
            opt.step()

        disp_fb_level = F.interpolate(
            cp_fb.detach(), size=(d, h, w), mode="trilinear", align_corners=True
        )
        disp_bf_level = F.interpolate(
            cp_bf.detach(), size=(d, h, w), mode="trilinear", align_corners=True
        )
        disp_fb_base = crop_to_slices(disp_fb_level, crop_slices)
        disp_bf_base = crop_to_slices(disp_bf_level, crop_slices)
        disp_fb_full = resize_disp_voxel(disp_fb_base, size=(d_full, h_full, w_full))
        disp_bf_full = resize_disp_voxel(disp_bf_base, size=(d_full, h_full, w_full))

    return (
        disp_fb_full.detach(),
        disp_bf_full.detach(),
        m_fb_fixed.detach(),
        m_bf_fixed.detach(),
    )


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
