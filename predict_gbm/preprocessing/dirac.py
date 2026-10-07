# Copyright (c) 2022 Tony Mok
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# The four-modality checkpoint from the BraTS-Reg 2022 container is redistributed
# under MIT with the author's written permission (T. Mok, 7 Oct 2026).


"""DIRAC-based longitudinal registration of a follow-up (post-operative) exam to the pre-operative exam.

Port of the inference path of the BraTS-Reg 2022 winning entry (Mok & Chung: conditional
laplacian pyramid network plus instance optimization) for the "dirac" registration_algorithm of
norm_ss_coregistration.register_recurrence:

  1. Nifti loading and orientation helpers. Images are processed in LPS voxel order, the
     orientation of the BraTS-Reg data the challenge container runs on, and results are mapped
     back to the stored orientation.
  2. Network building blocks (AdaIn conditioning, spatial transforms) and the three-level
     laplacian registration network (MICCAI 2021 LDR, cLapIRN weights).
  3. The hand-off from the ANTs affine pre-registration to F.affine_grid sampling grids, the
     network step and the composition of affine and network field.
  4. The container's bidirectional instance optimization: the total field on a control grid,
     occlusion masks recomputed every iteration, NCC on t1c and t2.
  5. run_dirac_registration, the entry point called by norm_ss_coregistration, which returns
     the follow-up-to-pre-op displacement in physical millimetres.

Conventions. Tensors use the container layout (1, C, n0, n1, n2) with the array axes of the
LPS-reoriented image. Fields are in the normalized [-1, 1] coordinates of F.grid_sample with
align_corners=True, channel 0 belonging to the last array axis, until the final conversion.
X is the pre-op (baseline) image and Y the follow-up; F_Y_X is the field on the pre-op grid
that resamples the follow-up onto it, F_X_Y the opposite direction (ported container code
keeps these names). The affine passed in by register_recurrence, affine_pre_to_post_mm, maps
pre-op physical points (LPS mm) to follow-up physical points. This module does not import ANTs.
"""

import os
from pathlib import Path
from typing import Mapping, Sequence

from loguru import logger
import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from predict_gbm.utils.constants import (
    DIRAC_CHECKPOINT_FOUR_MODALITIES,
    DIRAC_CHECKPOINT_T1CE,
)

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Input shape the registration network operates on. The affinely pre-aligned images are sampled
# at this resolution for inference; the predicted fields are composed with the affine here and
# refined from here by the instance optimization.
IMGSHAPE = (160, 160, 80)
IMGSHAPE_2 = tuple(s // 2 for s in IMGSHAPE)
RANGE_FLOW = 0.4
START_CHANNEL = 6
NUM_CBLOCK = 5
# Conditioning value of the network (the smoothness weight it was conditioned on), as used by the
# challenge container for inference; the same value is used with both checkpoints.
REG_CODE = 0.4

# Network input channels per image, in order; the two images are concatenated inside the network.
MODALITIES_FOUR = ("t1", "t1c", "flair", "t2")
MODALITIES_SINGLE = ("t1c",)
# Channels of the instance-optimization similarity term (container: t1ce and t2).
SIMILARITY_MODALITIES_FOUR = ("t1c", "t2")
SIMILARITY_MODALITIES_SINGLE = ("t1c",)

# Grid of the BraTS-Reg data the container runs on. The network sees the images squeezed from
# this grid to IMGSHAPE, so another grid changes the voxel aspect ratio it was trained with.
BRATSREG_GRID = (240, 240, 155)
LPS_AXCODES = ("L", "P", "S")

# Instance-optimization schedule of the container (my_script_pytorch.py:277-327): five levels
# from an 80^3 image to the native grid, control grids from 32^3 to 64^3.
IO_MIN_SIZE = 80
IO_CONTROL_GRID_RANGE = (32, 64)
IO_ITERATIONS = (150, 100, 100, 100, 50)
IO_LEARNING_RATES = (1e-2, 5e-3, 5e-3, 3e-3, 3e-3)
IO_SMOOTH_WEIGHTS = (0.25, 0.3, 0.3, 0.35, 0.35)
IO_INVERSE_WEIGHTS = (1.0, 2.0, 4.0, 8.0, 10.0)
IO_NCC_WIN = 3
IO_OCCLUSION_MARGIN = 0.015


# -----------------------------------------------------------------------------
# Grid helper and intensity normalization (DIRAC Functions.py)
# -----------------------------------------------------------------------------


def generate_grid_unit(imgshape):
    x = (np.arange(imgshape[0]) - ((imgshape[0] - 1) / 2)) / (imgshape[0] - 1) * 2
    y = (np.arange(imgshape[1]) - ((imgshape[1] - 1) / 2)) / (imgshape[1] - 1) * 2
    z = (np.arange(imgshape[2]) - ((imgshape[2] - 1) / 2)) / (imgshape[2] - 1) * 2
    grid = np.rollaxis(np.array(np.meshgrid(z, y, x)), 0, 4)
    grid = np.swapaxes(grid, 0, 2)
    grid = np.swapaxes(grid, 1, 2)
    return grid


def model_grid(imgshape: Sequence[int]) -> torch.Tensor:
    """Identity grid (1, n0, n1, n2, 3) for the network classes as a float32 CUDA tensor.

    Uses the container's numpy construction rather than identity_grid: the two differ by one
    float32 ulp in part of the entries, and only this one reproduces the container's network
    output bit for bit.
    """
    grid = generate_grid_unit(imgshape)
    return torch.from_numpy(np.reshape(grid, (1,) + grid.shape)).cuda().float()


def imgnorm(img):
    max_v = np.max(img)
    min_v = np.min(img)
    norm_img = (img - min_v) / (max_v - min_v)
    return norm_img


# -----------------------------------------------------------------------------
# Nifti loading and orientation helpers
# -----------------------------------------------------------------------------


def load_volume(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load a nifti as a 3-D float32 array and its (RAS, nibabel) affine.

    Trailing singleton dimensions, as in the (240, 240, 155, 1) SRI24 atlas files, are dropped.
    """
    img = nib.load(str(path))
    data = np.asarray(img.get_fdata(dtype=np.float32))
    while data.ndim > 3 and data.shape[-1] == 1:
        data = data[..., 0]
    if data.ndim != 3:
        raise ValueError(f"{path}: expected a 3-D volume, got shape {img.shape}.")
    return data, np.asarray(img.affine, dtype=np.float64)


def reorient_to_lps(
    data: np.ndarray, affine: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Flip and permute the array axes into LPS voxel order.

    Returns the reoriented array, the affine of the reoriented array (RAS convention, as read
    by nibabel) and the orientation transform that maps arrays back (see reorient_from_lps).
    """
    ornt_in = nib.orientations.io_orientation(affine)
    ornt_lps = nib.orientations.axcodes2ornt(LPS_AXCODES)
    to_lps = nib.orientations.ornt_transform(ornt_in, ornt_lps)
    data_lps = np.ascontiguousarray(nib.orientations.apply_orientation(data, to_lps))
    affine_lps = affine @ nib.orientations.inv_ornt_aff(to_lps, data.shape)
    back = nib.orientations.ornt_transform(ornt_lps, ornt_in)
    return data_lps, affine_lps, back


def reorient_from_lps(data_lps: np.ndarray, back: np.ndarray) -> np.ndarray:
    """Inverse of reorient_to_lps. Only the first three (spatial) axes are flipped and permuted;
    trailing axes such as vector components stay as they are."""
    return np.ascontiguousarray(nib.orientations.apply_orientation(data_lps, back))


def index_to_lps_matrix(affine_ras: np.ndarray) -> np.ndarray:
    """Voxel-index-to-physical matrix in LPS millimetres (the ANTs/ITK convention) from a
    nibabel affine, which maps voxel indices to RAS millimetres."""
    return np.diag([-1.0, -1.0, 1.0, 1.0]) @ affine_ras


# -----------------------------------------------------------------------------
# Network building blocks
# -----------------------------------------------------------------------------


class AdaptiveInstanceNorm(nn.Module):
    def __init__(self, in_channel, latent_dim=256):
        super().__init__()

        self.style = nn.Linear(latent_dim, in_channel * 2)

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
        flow = torch.nn.functional.grid_sample(
            x, sample_grid, mode="bilinear", padding_mode="border", align_corners=True
        )

        return flow


class CompositionTransform_unit(nn.Module):
    def __init__(self):
        super(CompositionTransform_unit, self).__init__()

    def forward(self, flow_1, flow_2, sample_grid):
        grid = sample_grid + flow_2.permute(0, 2, 3, 4, 1)
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
        range_flow=0.4,
        num_block=5,
    ):
        super(Miccai2021_LDR_laplacian_unit_disp_add_AdaIn_lvl1, self).__init__()
        self.in_channel = in_channel
        self.n_classes = n_classes
        self.start_channel = start_channel

        self.range_flow = range_flow

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

        self.resblock_group_lvl1 = self.resblock_seq(
            self.start_channel * 4, num_block=num_block, bias_opt=bias_opt
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

        fea_e0 = self.input_encoder_lvl1(cat_input_lvl1)
        e0 = self.down_conv(fea_e0)

        for i in range(len(self.resblock_group_lvl1)):
            if i % 2 == 0:
                e0 = self.resblock_group_lvl1[i](e0, reg_code)
            else:
                e0 = self.resblock_group_lvl1[i](e0)

        e0 = self.up(e0)
        output_disp_e0_v = (
            self.output_lvl1(torch.cat([e0, fea_e0], dim=1)) * self.range_flow
        )
        return output_disp_e0_v, e0


class Miccai2021_LDR_laplacian_unit_disp_add_AdaIn_lvl2(nn.Module):
    def __init__(
        self,
        in_channel,
        n_classes,
        start_channel,
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

        self.imgshape = imgshape

        self.model_lvl1 = model_lvl1

        self.grid_1 = model_grid(self.imgshape)

        self.transform = SpatialTransform_unit().cuda()

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
        lvl1_disp, lvl1_embedding = self.model_lvl1(x, y, reg_code)

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

        for i in range(len(self.resblock_group_lvl1)):
            if i % 2 == 0:
                e0 = self.resblock_group_lvl1[i](e0, reg_code)
            else:
                e0 = self.resblock_group_lvl1[i](e0)

        e0 = self.up(e0)
        output_disp_e0_v = (
            self.output_lvl1(torch.cat([e0, fea_e0], dim=1)) * self.range_flow
        )
        compose_field_e0_lvl1 = lvl1_disp_up + output_disp_e0_v
        return compose_field_e0_lvl1, e0


class Miccai2021_LDR_laplacian_unit_disp_add_AdaIn_lvl3(nn.Module):
    def __init__(
        self,
        in_channel,
        n_classes,
        start_channel,
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

        self.imgshape = imgshape

        self.model_lvl2 = model_lvl2

        self.grid_1 = model_grid(self.imgshape)

        self.transform = SpatialTransform_unit().cuda()

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

        self.output_lvl1 = self.outputs(
            self.start_channel * 8,
            self.n_classes,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )

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
        lvl2_disp, lvl2_embedding = self.model_lvl2(x, y, reg_code)

        lvl2_disp_up = self.up_tri(lvl2_disp)
        warpped_x = self.transform(x, lvl2_disp_up.permute(0, 2, 3, 4, 1), self.grid_1)

        cat_input = torch.cat((warpped_x, y, lvl2_disp_up), 1)

        fea_e0 = self.input_encoder_lvl1(cat_input)
        e0 = self.down_conv(fea_e0)

        e0 = e0 + lvl2_embedding

        for i in range(len(self.resblock_group_lvl1)):
            if i % 2 == 0:
                e0 = self.resblock_group_lvl1[i](e0, reg_code)
            else:
                e0 = self.resblock_group_lvl1[i](e0)

        e0 = self.up(e0)
        output_disp_e0_v = (
            self.output_lvl1(torch.cat([e0, fea_e0], dim=1)) * self.range_flow
        )
        compose_field_e0_lvl1 = output_disp_e0_v + lvl2_disp_up
        return compose_field_e0_lvl1


# -----------------------------------------------------------------------------
# Affine hand-off: ANTs matrix -> F.affine_grid sampling grids
# -----------------------------------------------------------------------------


# F.affine_grid orders coordinates as (x, y, z) = (last, middle, first) array axis. This
# permutation (its own inverse) converts homogeneous vectors between the two orders.
_TORCH_TO_ARRAY_AXES = np.array(
    [[0.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
)


def _norm_to_index(shape: Sequence[int]) -> np.ndarray:
    """4x4 matrix mapping normalized align_corners=True coordinates of each array axis to voxel
    indices, i = (u + 1) (N - 1) / 2."""
    m = np.eye(4)
    for k, n in enumerate(shape):
        m[k, k] = (n - 1) / 2.0
        m[k, 3] = (n - 1) / 2.0
    return m


def sampling_theta(
    output_index_to_mm: np.ndarray,
    output_shape: Sequence[int],
    input_index_to_mm: np.ndarray,
    input_shape: Sequence[int],
    output_mm_to_input_mm: np.ndarray,
) -> torch.Tensor:
    """Normalized 3x4 matrix for F.affine_grid that samples an input image on an output grid.

    With grid = F.affine_grid(theta, size=(1, C) + output_shape, align_corners=True) and
    F.grid_sample(input, grid, align_corners=True), the result at output voxel p holds
    input(T p), where T is output_mm_to_input_mm expressed in normalized coordinates of the
    two grids. output_mm_to_input_mm maps physical (LPS) points of the output grid to the
    physical points of the input image to sample; the *_index_to_mm matrices are the
    index-to-physical (LPS) matrices of the two images.
    """
    m = (
        np.linalg.inv(_norm_to_index(input_shape))
        @ np.linalg.inv(input_index_to_mm)
        @ output_mm_to_input_mm
        @ output_index_to_mm
        @ _norm_to_index(output_shape)
    )
    m = _TORCH_TO_ARRAY_AXES @ m @ _TORCH_TO_ARRAY_AXES
    return torch.from_numpy(m[:3].astype(np.float32)).unsqueeze(0)


def affine_sampling_thetas(
    affine_pre_to_post_mm: np.ndarray,
    pre_index_to_mm: np.ndarray,
    post_index_to_mm: np.ndarray,
    pre_shape: Sequence[int],
    post_shape: Sequence[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """The two sampling matrices derived from the ANTs affine (section 3 step 4 of the plan).

    Returns (theta_sample_post_on_pre_grid, theta_sample_pre_on_post_grid): the first samples
    the follow-up on the pre-op grid and feeds the F_Y_X direction (the container's
    best_affine_theta_yx), the second uses the inverted matrix, samples the pre-op on the
    follow-up grid and feeds the F_X_Y direction (best_affine_theta_xy).
    """
    theta_sample_post_on_pre_grid = sampling_theta(
        pre_index_to_mm, pre_shape, post_index_to_mm, post_shape, affine_pre_to_post_mm
    )
    theta_sample_pre_on_post_grid = sampling_theta(
        post_index_to_mm,
        post_shape,
        pre_index_to_mm,
        pre_shape,
        np.linalg.inv(affine_pre_to_post_mm),
    )
    return theta_sample_post_on_pre_grid, theta_sample_pre_on_post_grid


def identity_grid(shape: Sequence[int], device: torch.device) -> torch.Tensor:
    """Identity sampling grid (1, n0, n1, n2, 3) in normalized coordinates, align_corners=True."""
    theta = torch.eye(4, device=device)[:3].unsqueeze(0)
    return F.affine_grid(theta, size=(1, 1) + tuple(shape), align_corners=True)


# -----------------------------------------------------------------------------
# Network step (container my_script_pytorch.py:214-275)
# -----------------------------------------------------------------------------


def build_network(in_channel: int, device: torch.device) -> nn.Module:
    """The three-level network with the inference configuration of the container."""
    model_lvl1 = Miccai2021_LDR_laplacian_unit_disp_add_AdaIn_lvl1(
        in_channel,
        3,
        START_CHANNEL,
        range_flow=RANGE_FLOW,
        num_block=NUM_CBLOCK,
    ).to(device)
    model_lvl2 = Miccai2021_LDR_laplacian_unit_disp_add_AdaIn_lvl2(
        in_channel,
        3,
        START_CHANNEL,
        imgshape=IMGSHAPE_2,
        range_flow=RANGE_FLOW,
        model_lvl1=model_lvl1,
        num_block=NUM_CBLOCK,
    ).to(device)
    model = Miccai2021_LDR_laplacian_unit_disp_add_AdaIn_lvl3(
        in_channel,
        3,
        START_CHANNEL,
        imgshape=IMGSHAPE,
        range_flow=RANGE_FLOW,
        model_lvl2=model_lvl2,
        num_block=NUM_CBLOCK,
    ).to(device)
    return model


def load_network(single_modality: bool, device: torch.device) -> nn.Module:
    """Build the network for the selected checkpoint and load its weights."""
    checkpoint = DIRAC_CHECKPOINT_T1CE if single_modality else DIRAC_CHECKPOINT_FOUR_MODALITIES
    if not checkpoint.exists():
        raise FileNotFoundError(f"Missing DIRAC model checkpoint: {checkpoint}")
    modalities = MODALITIES_SINGLE if single_modality else MODALITIES_FOUR
    model = build_network(in_channel=2 * len(modalities), device=device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    logger.info(f"Loaded DIRAC checkpoint {checkpoint.name} ({len(modalities)} modalities).")
    return model


def run_network(
    model: nn.Module,
    X_ori: torch.Tensor,
    Y_ori: torch.Tensor,
    theta_sample_pre_on_post_grid: torch.Tensor,
    theta_sample_post_on_pre_grid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Network inference on the affinely pre-aligned images and composition with the affine.

    X_ori, Y_ori: pre-op and follow-up stacks (1, C, n0, n1, n2) at native resolution. The
    image being warped is sampled through its affine grid, which also brings it to IMGSHAPE;
    the other image is resized. Returns (deep_F_X_Y, deep_F_Y_X), the combined fields at
    IMGSHAPE in normalized coordinates (the container's compose_F_X_Y / compose_F_Y_X).
    """
    device = X_ori.device
    size = (1, 1) + IMGSHAPE
    affine_grid_xy = F.affine_grid(
        theta_sample_pre_on_post_grid.to(device), size=size, align_corners=True
    )
    affine_grid_yx = F.affine_grid(
        theta_sample_post_on_pre_grid.to(device), size=size, align_corners=True
    )
    id_grid = identity_grid(IMGSHAPE, device)
    affine_F_X_Y = (affine_grid_xy - id_grid).permute(0, 4, 1, 2, 3)
    affine_F_Y_X = (affine_grid_yx - id_grid).permute(0, 4, 1, 2, 3)

    with torch.no_grad():
        affine_X = F.grid_sample(
            X_ori, affine_grid_xy, mode="bilinear", padding_mode="border", align_corners=True
        )
        affine_Y = F.grid_sample(
            Y_ori, affine_grid_yx, mode="bilinear", padding_mode="border", align_corners=True
        )
        X = F.interpolate(X_ori, size=IMGSHAPE, mode="trilinear", align_corners=False)
        Y = F.interpolate(Y_ori, size=IMGSHAPE, mode="trilinear", align_corners=False)

        reg_code = torch.tensor([REG_CODE], dtype=X.dtype, device=device).unsqueeze(dim=0)
        F_X_Y = model(affine_X, Y, reg_code)
        F_Y_X = model(affine_Y, X, reg_code)

        compose = CompositionTransform_unit()
        deep_F_X_Y = compose(affine_F_X_Y, F_X_Y, id_grid)
        deep_F_Y_X = compose(affine_F_Y_X, F_Y_X, id_grid)
    return deep_F_X_Y.detach(), deep_F_Y_X.detach()


# -----------------------------------------------------------------------------
# Instance optimization (container my_script_pytorch.py:277-433, bratsreg_model_selected.py)
# -----------------------------------------------------------------------------


class NCC_weight(nn.Module):
    """Local (windowed) normalized cross correlation weighted by a voxel map, per channel."""

    def __init__(self, win: int = 5, eps: float = 1e-8, channel: int = 1) -> None:
        super().__init__()
        self.win = win
        self.eps = eps
        self.channel = channel

    def forward(
        self, I: torch.Tensor, J: torch.Tensor, weight_map: torch.Tensor
    ) -> torch.Tensor:
        weight = torch.ones(
            (self.channel, 1, self.win, self.win, self.win), device=I.device, dtype=I.dtype
        )
        pad = int(self.win / 2)

        I2 = I * I
        J2 = J * J
        IJ = I * J

        I_sum = F.conv3d(I, weight, padding=pad, groups=self.channel)
        J_sum = F.conv3d(J, weight, padding=pad, groups=self.channel)
        I2_sum = F.conv3d(I2, weight, padding=pad, groups=self.channel)
        J2_sum = F.conv3d(J2, weight, padding=pad, groups=self.channel)
        IJ_sum = F.conv3d(IJ, weight, padding=pad, groups=self.channel)

        win_size = self.win**3
        u_I = I_sum / win_size
        u_J = J_sum / win_size

        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

        cc = cross * cross / (I_var * J_var + self.eps)
        return -1.0 * torch.mean(cc * weight_map)


def smoothloss(y_pred: torch.Tensor) -> torch.Tensor:
    dy = torch.abs(y_pred[:, :, 1:, :, :] - y_pred[:, :, :-1, :, :])
    dx = torch.abs(y_pred[:, :, :, 1:, :] - y_pred[:, :, :, :-1, :])
    dz = torch.abs(y_pred[:, :, :, :, 1:] - y_pred[:, :, :, :, :-1])
    return (torch.mean(dx * dx) + torch.mean(dy * dy) + torch.mean(dz * dz)) / 3.0


class L2R_Instance_opt_grid_bidirection(nn.Module):
    """Both total fields on a control grid; forward upsamples them to the level's image size
    and warps the two images."""

    def __init__(
        self,
        init_disp_xy: torch.Tensor,
        init_disp_yx: torch.Tensor,
        imgshape: Sequence[int],
    ) -> None:
        super().__init__()
        self.disp_xy = torch.nn.Parameter(init_disp_xy)
        self.disp_yx = torch.nn.Parameter(init_disp_yx)
        self.imgshape = tuple(imgshape)
        self.transform = SpatialTransform_unit()
        self.grid_1 = identity_grid(self.imgshape, init_disp_xy.device)

    def forward(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        flow_xy = F.interpolate(
            self.disp_xy, size=self.imgshape, mode="trilinear", align_corners=True
        )
        flow_yx = F.interpolate(
            self.disp_yx, size=self.imgshape, mode="trilinear", align_corners=True
        )
        warped_x = self.transform(x, flow_xy.permute(0, 2, 3, 4, 1), self.grid_1)
        warped_y = self.transform(y, flow_yx.permute(0, 2, 3, 4, 1), self.grid_1)
        return flow_xy, flow_yx, warped_x, warped_y


def pyramid_sizes(
    min_size: Sequence[int], max_size: Sequence[int], levels: int
) -> list[tuple[int, int, int]]:
    """Per-axis linear spacing from min_size to max_size, truncated to integers like the
    container's np.linspace(..., dtype=np.int16)."""
    axes = [
        np.floor(np.linspace(lo, hi, num=levels)).astype(int)
        for lo, hi in zip(min_size, max_size)
    ]
    return [tuple(int(v) for v in size) for size in zip(*axes)]


def dirac_instance_optimization(
    X: torch.Tensor,
    Y: torch.Tensor,
    deep_F_X_Y: torch.Tensor,
    deep_F_Y_X: torch.Tensor,
    iterations: Sequence[int] = IO_ITERATIONS,
    learning_rates: Sequence[float] = IO_LEARNING_RATES,
    smooth_weights: Sequence[float] = IO_SMOOTH_WEIGHTS,
    inverse_weights: Sequence[float] = IO_INVERSE_WEIGHTS,
    min_size: int = IO_MIN_SIZE,
    control_grid_range: tuple[int, int] = IO_CONTROL_GRID_RANGE,
    ncc_win: int = IO_NCC_WIN,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Bidirectional instance optimization of the combined fields, as in the container.

    X, Y: pre-op and follow-up similarity images (1, C, n0, n1, n2) at native resolution
    (t1c + t2, or t1c alone), min-max normalized; the foreground masks are taken from
    channel 0. deep_F_X_Y, deep_F_Y_X: the initial fields (1, 3, ...) in normalized
    coordinates at any resolution (IMGSHAPE after run_network).

    Per level l (five by default): image size per axis linear from min_size to native,
    control grid from control_grid_range[0]^3 to control_grid_range[1]^3, both truncated to
    integers. The control grids hold the total fields; the first is the initial field
    downsampled, later ones come from the previous level's last iterate. Every iteration
    recomputes the occlusion masks from the current fields (forward-backward error, smoothed
    twice with a 5^3 average, threshold = mean error inside the foreground + 0.015) and
    minimizes (1 - w_smooth) * NCC(win=ncc_win) outside the occlusions + w_smooth * smoothness
    of the total field in voxel units of the level + w_inv * inverse-consistency error outside
    the occlusions, with Adam.

    Returns (F_X_Y, F_Y_X): the dense fields of the lowest-loss iteration of the last level,
    at native resolution, normalized coordinates. Any device; GPU runs are not bitwise
    reproducible.
    """
    levels = len(iterations)
    if not (
        len(learning_rates) == len(smooth_weights) == len(inverse_weights) == levels
    ):
        raise ValueError(
            "iterations, learning_rates, smooth_weights and inverse_weights must have equal length"
        )
    device = X.device
    native = tuple(int(s) for s in X.shape[2:])
    pyramid_size = pyramid_sizes((min_size,) * 3, native, levels)
    theta_size = pyramid_sizes((control_grid_range[0],) * 3, (control_grid_range[1],) * 3, levels)

    transform = SpatialTransform_unit()
    loss_similarity = NCC_weight(win=ncc_win, eps=1e-8, channel=int(X.shape[1]))
    loss_smooth = smoothloss

    X_pyramid = [
        F.interpolate(X, size=size, mode="trilinear", align_corners=False)
        for size in pyramid_size
    ]
    Y_pyramid = [
        F.interpolate(Y, size=size, mode="trilinear", align_corners=False)
        for size in pyramid_size
    ]

    best_theta_xy = best_theta_yx = None
    best_F_X_Y = best_F_Y_X = None
    for level, (X_l, Y_l) in enumerate(zip(X_pyramid, Y_pyramid)):
        if level == 0:
            F_X_Y_theta = F.interpolate(
                deep_F_X_Y, size=theta_size[level], mode="trilinear", align_corners=False
            )
            F_Y_X_theta = F.interpolate(
                deep_F_Y_X, size=theta_size[level], mode="trilinear", align_corners=False
            )
        else:
            F_X_Y_theta = F.interpolate(
                best_theta_xy, size=theta_size[level], mode="trilinear", align_corners=False
            )
            F_Y_X_theta = F.interpolate(
                best_theta_yx, size=theta_size[level], mode="trilinear", align_corners=False
            )

        io_model = L2R_Instance_opt_grid_bidirection(
            F_X_Y_theta.detach(), F_Y_X_theta.detach(), imgshape=X_l.shape[2:]
        ).to(device)
        optimizer = torch.optim.Adam(io_model.parameters(), lr=learning_rates[level])
        grid_unit = identity_grid(X_l.shape[2:], device)

        best_loss = float("inf")
        best_F_X_Y = best_F_Y_X = None
        for _ in range(iterations[level]):
            F_X_Y, F_Y_X, X_Y, Y_X = io_model(X_l, Y_l)

            F_X_Y_warpped = transform(F_X_Y, F_Y_X.permute(0, 2, 3, 4, 1), grid_unit)
            F_Y_X_warpped = transform(F_Y_X, F_X_Y.permute(0, 2, 3, 4, 1), grid_unit)

            diff_fw = F_X_Y + F_Y_X_warpped  # Y
            diff_bw = F_Y_X + F_X_Y_warpped  # X

            fw_mask = (Y_l[:, 0:1] > 0).float()
            bw_mask = (X_l[:, 0:1] > 0).float()

            u_diff_fw = torch.sum(
                torch.norm(diff_fw * fw_mask, dim=1, keepdim=True)
            ) / torch.sum(fw_mask)
            u_diff_bw = torch.sum(
                torch.norm(diff_bw * bw_mask, dim=1, keepdim=True)
            ) / torch.sum(bw_mask)

            thresh_fw = (u_diff_fw + IO_OCCLUSION_MARGIN) * torch.ones_like(Y_l[:, 0:1])
            thresh_bw = (u_diff_bw + IO_OCCLUSION_MARGIN) * torch.ones_like(X_l[:, 0:1])

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

            mask_xy = (occ_xy == 0).float()
            mask_yx = (occ_yx == 0).float()

            loss_sim = loss_similarity(X_Y, Y_l, mask_xy) + loss_similarity(Y_X, X_l, mask_yx)
            loss_inverse = torch.mean(norm_diff_fw * mask_xy) + torch.mean(
                norm_diff_bw * mask_yx
            )
            # (the container's occlusion-area term is weighted 0 and therefore omitted)

            # smoothness of the total fields in voxel units of this level
            _, _, x, y, z = F_X_Y.shape
            norm_vector = torch.zeros((1, 3, 1, 1, 1), dtype=F_X_Y.dtype, device=device)
            norm_vector[0, 0, 0, 0, 0] = (z - 1) / 2
            norm_vector[0, 1, 0, 0, 0] = (y - 1) / 2
            norm_vector[0, 2, 0, 0, 0] = (x - 1) / 2
            loss_regulation = loss_smooth(F_X_Y * norm_vector) + loss_smooth(
                F_Y_X * norm_vector
            )

            total_loss = (
                (1.0 - smooth_weights[level]) * loss_sim
                + smooth_weights[level] * loss_regulation
                + inverse_weights[level] * loss_inverse
            )

            if total_loss <= best_loss:
                best_loss = total_loss.item()
                best_F_X_Y = F_X_Y.detach()
                best_F_Y_X = F_Y_X.detach()

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

        # The container hands the parameter tensors on, which the optimizer kept updating:
        # the next level starts from the last iterate, not from the best one.
        best_theta_xy = io_model.disp_xy.detach()
        best_theta_yx = io_model.disp_yx.detach()
        if best_F_X_Y is None:  # no iteration at this level
            with torch.no_grad():
                best_F_X_Y, best_F_Y_X = io_model(X_l, Y_l)[:2]
        logger.info(
            f"IO level {level}: image {tuple(X_l.shape[2:])}, control grid "
            f"{theta_size[level]}, {iterations[level]} iters, lr {learning_rates[level]:g}, "
            f"best loss {best_loss:.4f}"
        )

    return best_F_X_Y, best_F_Y_X


# -----------------------------------------------------------------------------
# Field conversion and entry point
# -----------------------------------------------------------------------------


def field_to_voxel_displacement(flow: torch.Tensor) -> np.ndarray:
    """(1, 3, n0, n1, n2) normalized field -> (n0, n1, n2, 3) displacement in voxels with the
    components in array-axis order (container my_script_pytorch.py:435-439)."""
    _, _, h, w, d = flow.shape
    f = flow[0].detach().cpu().numpy()
    disp = np.empty((h, w, d, 3), dtype=np.float32)
    disp[..., 0] = f[2] * (h - 1) / 2
    disp[..., 1] = f[1] * (w - 1) / 2
    disp[..., 2] = f[0] * (d - 1) / 2
    return disp


def voxel_to_physical_displacement(
    disp_voxel: np.ndarray, index_to_mm: np.ndarray
) -> np.ndarray:
    """Voxel displacement (..., 3) to physical displacement with the linear part of the
    index-to-physical matrix."""
    return (disp_voxel @ index_to_mm[:3, :3].T).astype(np.float32)


def _load_exam(
    files: Mapping[str, Path], modalities: Sequence[str], label: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """Load, reorient to LPS, min-max normalize and stack the modalities of one exam.

    Returns the stack (C, n0, n1, n2), the LPS-reoriented affine, the orientation transform
    back to the stored layout and the stored axis codes. All modalities must share shape and
    affine.
    """
    stack: list[np.ndarray] = []
    affine_ref: np.ndarray | None = None
    shape_ref: tuple[int, ...] | None = None
    back_ref: np.ndarray | None = None
    axcodes = ""
    for name in modalities:
        data, affine = load_volume(files[name])
        data_lps, affine_lps, back = reorient_to_lps(data, affine)
        if affine_ref is None:
            affine_ref, shape_ref, back_ref = affine_lps, data_lps.shape, back
            axcodes = "".join(nib.aff2axcodes(affine))
        elif data_lps.shape != shape_ref or not np.allclose(affine_lps, affine_ref, atol=1e-3):
            raise ValueError(
                f"{label} {name} image {files[name]} does not share shape and affine with "
                f"{label} {modalities[0]} image {files[modalities[0]]}."
            )
        stack.append(imgnorm(data_lps).astype(np.float32))
    assert affine_ref is not None and back_ref is not None
    return np.stack(stack, axis=0), affine_ref, back_ref, axcodes


def run_dirac_registration(
    pre_files: Mapping[str, Path],
    post_files: Mapping[str, Path],
    affine_pre_to_post_mm: np.ndarray,
    single_modality: bool = False,
) -> np.ndarray:
    """Register the follow-up to the pre-op exam with DIRAC, starting from an affine.

    Parameters:
        pre_files (Mapping[str, Path]): Pre-op modality files keyed by name; needs t1, t1c,
            flair and t2 (t1c alone with single_modality).
        post_files (Mapping[str, Path]): The same for the follow-up exam.
        affine_pre_to_post_mm (np.ndarray): 4x4 matrix in physical LPS millimetres mapping
            pre-op points to follow-up points (the ANTs affine of register_recurrence).
        single_modality (bool): Use the t1c-only checkpoint instead of the four-modality one.

    Returns:
        np.ndarray: disp_pre_to_post_mm, shape (n0, n1, n2, 3) float32 in the pre-op image's
        stored array layout, vectors in physical LPS millimetres. It follows the direction
        convention of affine_pre_to_post_mm: as an ANTs/ITK displacement field on the pre-op
        grid it resamples the follow-up onto the pre-op grid.

    Needs a CUDA device (the network classes are built on the default GPU).
    """
    if not torch.cuda.is_available():
        raise RuntimeError("DIRAC registration needs a CUDA device.")
    device = torch.device("cuda")
    modalities = MODALITIES_SINGLE if single_modality else MODALITIES_FOUR
    similarity_modalities = (
        SIMILARITY_MODALITIES_SINGLE if single_modality else SIMILARITY_MODALITIES_FOUR
    )
    missing = [m for m in modalities if pre_files.get(m) is None or post_files.get(m) is None]
    if missing:
        raise ValueError(
            f"DIRAC needs the modalities {list(modalities)} of both exams; missing: {missing}."
        )
    affine_pre_to_post_mm = np.asarray(affine_pre_to_post_mm, dtype=np.float64)
    if affine_pre_to_post_mm.shape != (4, 4):
        raise ValueError("affine_pre_to_post_mm must be a 4x4 matrix.")

    # Load and reorient (C2); checks of section 3 step 1.
    pre_stack, pre_affine_lps, pre_back, pre_axcodes = _load_exam(
        pre_files, modalities, "pre-op"
    )
    post_stack, post_affine_lps, _, _ = _load_exam(post_files, modalities, "follow-up")
    shape = tuple(pre_stack.shape[1:])
    if tuple(post_stack.shape[1:]) != shape or not np.allclose(
        pre_affine_lps, post_affine_lps, atol=1e-3
    ):
        raise ValueError(
            "Pre-op and follow-up images must share shape and affine for DIRAC registration "
            f"(pre-op {shape}, follow-up {tuple(post_stack.shape[1:])})."
        )
    if shape != BRATSREG_GRID:
        logger.warning(
            f"DIRAC input grid {shape} differs from the {BRATSREG_GRID} BraTS-Reg grid the "
            "network was run on in the challenge; the voxel aspect ratio the network sees "
            "changes accordingly."
        )
    logger.info(
        f"DIRAC registration on grid {shape} with {len(modalities)} modalities "
        f"(stored orientation {pre_axcodes}, processed in LPS voxel order)."
    )

    pre_index_to_mm = index_to_lps_matrix(pre_affine_lps)
    post_index_to_mm = index_to_lps_matrix(post_affine_lps)
    theta_sample_post_on_pre_grid, theta_sample_pre_on_post_grid = affine_sampling_thetas(
        affine_pre_to_post_mm, pre_index_to_mm, post_index_to_mm, shape, shape
    )

    X_ori = torch.from_numpy(pre_stack).unsqueeze(0).to(device)
    Y_ori = torch.from_numpy(post_stack).unsqueeze(0).to(device)

    model = load_network(single_modality, device)
    deep_F_X_Y, deep_F_Y_X = run_network(
        model, X_ori, Y_ori, theta_sample_pre_on_post_grid, theta_sample_post_on_pre_grid
    )
    del model
    logger.info("DIRAC network inference done; starting instance optimization.")

    similarity_channels = [modalities.index(m) for m in similarity_modalities]
    X = X_ori[:, similarity_channels].contiguous()
    Y = Y_ori[:, similarity_channels].contiguous()
    del X_ori, Y_ori
    _, F_Y_X = dirac_instance_optimization(X, Y, deep_F_X_Y, deep_F_Y_X)

    disp_voxel = field_to_voxel_displacement(F_Y_X)
    disp_mm_lps = voxel_to_physical_displacement(disp_voxel, pre_index_to_mm)
    disp_pre_to_post_mm = reorient_from_lps(disp_mm_lps, pre_back)

    del X, Y, F_Y_X, deep_F_X_Y, deep_F_Y_X
    torch.cuda.empty_cache()
    return disp_pre_to_post_mm
