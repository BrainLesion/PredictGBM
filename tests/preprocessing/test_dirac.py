import unittest
import warnings

import torch
import torch.nn.functional as F

from predict_gbm.preprocessing.dirac import (
    dirac_instance_optimization,
    warp,
    warp_field,
)

# Silence third-party warnings that clutter test output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

SHAPE: tuple[int, int, int] = (40, 48, 32)  # (D, H, W), anisotropic
# Small pyramid and short schedule so the test stays fast on CPU; the loss and
# regularization settings are the function defaults.
FAST_SCHEDULE: dict[str, tuple[int, ...] | int] = {
    "iters": (30, 20, 15, 10, 5),
    "coarsest_size": 16,
}


def smooth_field(
    shape: tuple[int, int, int], amplitude: float, seed: int, control_points: int = 3
) -> torch.Tensor:
    """Smooth random (1,3,D,H,W) voxel displacement with the given mean magnitude."""
    generator = torch.Generator().manual_seed(seed)
    coarse = torch.randn(
        1, 3, control_points, control_points, control_points, generator=generator
    )
    dense = F.interpolate(coarse, size=shape, mode="trilinear", align_corners=True)
    return dense * (amplitude / dense.norm(dim=1).mean())


def phantom(shape: tuple[int, int, int], seed: int) -> torch.Tensor:
    """Smooth random (1,1,D,H,W) image with some fine texture, scaled to [0, 1]."""
    generator = torch.Generator().manual_seed(seed)
    coarse = torch.randn(1, 1, *[max(2, s // 3) for s in shape], generator=generator)
    fine = torch.randn(1, 1, *[max(2, s // 2) for s in shape], generator=generator)
    image = F.interpolate(coarse, size=shape, mode="trilinear", align_corners=True)
    image = image + 0.15 * F.interpolate(
        fine, size=shape, mode="trilinear", align_corners=True
    )
    return (image - image.min()) / (image.max() - image.min())


def invert_field(disp: torch.Tensor, n_iter: int = 30) -> torch.Tensor:
    """Fixed-point inverse of a displacement field: v(x) = -u(x + v(x))."""
    inverse = -disp
    for _ in range(n_iter):
        inverse = -warp_field(disp, inverse)
    return inverse


class TestDiracInstanceOptimization(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)
        self.followup: torch.Tensor = phantom(SHAPE, seed=1)
        self.u_true: torch.Tensor = smooth_field(SHAPE, amplitude=1.5, seed=2)
        self.preop: torch.Tensor = warp(self.followup, self.u_true)
        self.u_true_inverse: torch.Tensor = invert_field(self.u_true)

    def test_zero_iterations_returns_input_fields(self) -> None:
        u_fb = self.u_true + smooth_field(SHAPE, amplitude=1.0, seed=11)
        u_bf = self.u_true_inverse + smooth_field(SHAPE, amplitude=0.5, seed=12)
        mask_fb = torch.zeros_like(self.preop)
        mask_fb[:, :, 10:18, 12:24, 8:20] = 1.0
        mask_bf = torch.zeros_like(self.preop)
        mask_bf[:, :, 20:26, 30:40, 14:26] = 1.0

        out_fb, out_bf, out_mask_fb, out_mask_bf = dirac_instance_optimization(
            self.preop,
            self.followup,
            u_fb,
            u_bf,
            m_fb_fixed=mask_fb,
            m_bf_fixed=mask_bf,
            iters=(0, 0, 0, 0, 0),
            coarsest_size=16,
        )

        self.assertEqual(out_fb.shape, u_fb.shape)
        self.assertEqual(out_bf.shape, u_bf.shape)
        self.assertTrue(torch.isfinite(out_fb).all())
        self.assertTrue(torch.isfinite(out_bf).all())
        self.assertLessEqual(float((out_fb - u_fb).abs().max()), 1e-5)
        self.assertLessEqual(float((out_bf - u_bf).abs().max()), 1e-5)
        self.assertTrue(torch.equal(out_mask_fb, mask_fb))
        self.assertTrue(torch.equal(out_mask_bf, mask_bf))

    def test_optimization_improves_and_does_not_collapse(self) -> None:
        u_init = self.u_true + smooth_field(SHAPE, amplitude=2.0, seed=31)
        u_init_inverse = invert_field(u_init) + smooth_field(
            SHAPE, amplitude=0.3, seed=32
        )
        error_in = float((u_init - self.u_true).norm(dim=1).mean())
        magnitude_in = float(u_init.norm(dim=1).mean())

        out_fb, _, _, _ = dirac_instance_optimization(
            self.preop, self.followup, u_init, u_init_inverse, **FAST_SCHEDULE
        )

        self.assertTrue(torch.isfinite(out_fb).all())
        error_out = float((out_fb - self.u_true).norm(dim=1).mean())
        magnitude_out = float(out_fb.norm(dim=1).mean())
        self.assertLess(error_out, error_in, (error_in, error_out))
        self.assertGreater(
            magnitude_out, 0.5 * magnitude_in, (magnitude_in, magnitude_out)
        )


if __name__ == "__main__":
    unittest.main()
