import unittest
import warnings

import torch
import torch.nn.functional as F

from predict_gbm.preprocessing.dirac import (
    SpatialTransform_unit,
    dirac_instance_optimization,
    identity_grid,
)

# Silence third-party warnings that clutter test output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

SHAPE: tuple[int, int, int] = (40, 48, 32)  # (n0, n1, n2), anisotropic
# Small pyramid and short schedule so the test stays fast on CPU; the loss weights and
# control grids are the function defaults.
FAST_SCHEDULE: dict[str, tuple[int, ...] | int] = {
    "iterations": (30, 20, 15, 10, 5),
    "min_size": 16,
}


def voxel_to_normalized(field_voxel: torch.Tensor) -> torch.Tensor:
    """(1,3,n0,n1,n2) field in voxels with channels [axis 2, axis 1, axis 0] to the normalized
    align_corners=True units of F.grid_sample."""
    n0, n1, n2 = field_voxel.shape[2:]
    scale = torch.tensor([2.0 / (n2 - 1), 2.0 / (n1 - 1), 2.0 / (n0 - 1)])
    return field_voxel * scale.view(1, 3, 1, 1, 1)


def smooth_field(
    shape: tuple[int, int, int], amplitude: float, seed: int, control_points: int = 3
) -> torch.Tensor:
    """Smooth random (1,3,n0,n1,n2) field with the given mean magnitude in voxels, returned
    in normalized coordinates."""
    generator = torch.Generator().manual_seed(seed)
    coarse = torch.randn(
        1, 3, control_points, control_points, control_points, generator=generator
    )
    dense = F.interpolate(coarse, size=shape, mode="trilinear", align_corners=True)
    return voxel_to_normalized(dense * (amplitude / dense.norm(dim=1).mean()))


def phantom(shape: tuple[int, int, int], seed: int) -> torch.Tensor:
    """Smooth random (1,1,n0,n1,n2) image with some fine texture, scaled to [0, 1]."""
    generator = torch.Generator().manual_seed(seed)
    coarse = torch.randn(1, 1, *[max(2, s // 3) for s in shape], generator=generator)
    fine = torch.randn(1, 1, *[max(2, s // 2) for s in shape], generator=generator)
    image = F.interpolate(coarse, size=shape, mode="trilinear", align_corners=True)
    image = image + 0.15 * F.interpolate(
        fine, size=shape, mode="trilinear", align_corners=True
    )
    return (image - image.min()) / (image.max() - image.min())


def warp(tensor: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    """Sample an image or field at grid + flow (normalized coordinates)."""
    grid = identity_grid(tensor.shape[2:], tensor.device)
    return SpatialTransform_unit()(tensor, flow.permute(0, 2, 3, 4, 1), grid)


def invert_field(flow: torch.Tensor, n_iter: int = 30) -> torch.Tensor:
    """Fixed-point inverse of a displacement field: v(x) = -u(x + v(x))."""
    inverse = -flow
    for _ in range(n_iter):
        inverse = -warp(flow, inverse)
    return inverse


class TestDiracInstanceOptimization(unittest.TestCase):
    """X is the pre-op image, Y the follow-up; F_Y_X on the pre-op grid samples the follow-up,
    F_X_Y the opposite direction (see dirac.py)."""

    def setUp(self) -> None:
        torch.manual_seed(0)
        self.followup: torch.Tensor = phantom(SHAPE, seed=1)
        self.F_Y_X_true: torch.Tensor = smooth_field(SHAPE, amplitude=1.5, seed=2)
        self.preop: torch.Tensor = warp(self.followup, self.F_Y_X_true)
        self.F_X_Y_true: torch.Tensor = invert_field(self.F_Y_X_true)

    def test_zero_iterations_returns_band_limited_input_fields(self) -> None:
        F_Y_X = self.F_Y_X_true + smooth_field(SHAPE, amplitude=1.0, seed=11)
        F_X_Y = self.F_X_Y_true + smooth_field(SHAPE, amplitude=0.5, seed=12)

        out_F_X_Y, out_F_Y_X = dirac_instance_optimization(
            self.preop,
            self.followup,
            F_X_Y,
            F_Y_X,
            iterations=(0, 0, 0, 0, 0),
            min_size=16,
        )

        # Without iterations the fields only pass through the control grids, which
        # band-limit them: close to the input, not identical.
        self.assertEqual(out_F_Y_X.shape, F_Y_X.shape)
        self.assertEqual(out_F_X_Y.shape, F_X_Y.shape)
        self.assertTrue(torch.isfinite(out_F_Y_X).all())
        self.assertTrue(torch.isfinite(out_F_X_Y).all())
        for out, ref in ((out_F_Y_X, F_Y_X), (out_F_X_Y, F_X_Y)):
            relative_error = float(
                (out - ref).norm(dim=1).mean() / ref.norm(dim=1).mean()
            )
            self.assertLess(relative_error, 0.05, relative_error)

    def test_optimization_improves_and_does_not_collapse(self) -> None:
        F_Y_X_init = self.F_Y_X_true + smooth_field(SHAPE, amplitude=2.0, seed=31)
        F_X_Y_init = invert_field(F_Y_X_init) + smooth_field(
            SHAPE, amplitude=0.3, seed=32
        )
        error_in = float((F_Y_X_init - self.F_Y_X_true).norm(dim=1).mean())
        magnitude_in = float(F_Y_X_init.norm(dim=1).mean())

        _, out_F_Y_X = dirac_instance_optimization(
            self.preop, self.followup, F_X_Y_init, F_Y_X_init, **FAST_SCHEDULE
        )

        self.assertTrue(torch.isfinite(out_F_Y_X).all())
        error_out = float((out_F_Y_X - self.F_Y_X_true).norm(dim=1).mean())
        magnitude_out = float(out_F_Y_X.norm(dim=1).mean())
        self.assertLess(error_out, error_in, (error_in, error_out))
        self.assertGreater(
            magnitude_out, 0.5 * magnitude_in, (magnitude_in, magnitude_out)
        )


if __name__ == "__main__":
    unittest.main()
