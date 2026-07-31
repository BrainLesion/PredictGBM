import tempfile
from unittest.mock import MagicMock, patch
import unittest
import warnings
import nibabel as nib
from pathlib import Path
from tests.helpers import generate_mock_nifti
from predict_gbm.preprocessing.norm_ss_coregistration import (
    normalize,
    initialize_center_modality,
    initialize_moving_modalities,
    register_recurrence,
)
from brainles_preprocessing.modality import Modality, CenterModality
from brainles_preprocessing.normalization.percentile_normalizer import (
    PercentileNormalizer,
)

# Silence third-party warnings that clutter test output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


class TestNormSsCoregistration(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.outdir = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def _save_mock_nifti(self, name: str) -> Path:
        img = generate_mock_nifti()
        file_path = self.outdir / name
        nib.save(img, str(file_path))
        return file_path

    def _get_normalizer(self):
        return PercentileNormalizer(
            lower_percentile=0.1,
            upper_percentile=99.9,
            lower_limit=0,
            upper_limit=1,
        )

    def test_normalize(self):
        img_file = self._save_mock_nifti("img.nii.gz")
        outfile = self.outdir / "norm.nii.gz"

        normalize(img_file, outfile)
        normalized_data = nib.load(outfile).get_fdata()

        self.assertTrue(outfile.exists())
        self.assertTrue(normalized_data.min() >= 0)
        self.assertTrue(normalized_data.max() <= 1)
        self.assertEqual(nib.load(outfile).shape, nib.load(img_file).shape)

    def test_initialize_center_modality_paths(self):
        img_file = self._save_mock_nifti("t1c.nii.gz")
        normalizer = self._get_normalizer()

        center = initialize_center_modality(
            modality_file=img_file,
            modality_name="t1c",
            normalizer=normalizer,
            outdir=self.outdir,
            skull_strip=True,
        )

        center_no_ss = initialize_center_modality(
            modality_file=img_file,
            modality_name="t1c",
            normalizer=normalizer,
            outdir=self.outdir,
            skull_strip=False,
        )

        self.assertTrue(isinstance(center, CenterModality))
        self.assertTrue(isinstance(center_no_ss, CenterModality))

    def test_initialize_moving_modalities_returns_list(self):
        t1 = self._save_mock_nifti("t1.nii.gz")
        t2 = self._save_mock_nifti("t2.nii.gz")
        fl = self._save_mock_nifti("flair.nii.gz")
        normalizer = self._get_normalizer()

        mods = initialize_moving_modalities(
            modality_files=[t1, t2, fl],
            modality_names=["t1", "t2", "flair"],
            normalizer=normalizer,
            outdir=self.outdir,
            skull_strip=False,
        )
        self.assertEqual(len(mods), 3)
        self.assertTrue(all([isinstance(m, Modality) for m in mods]))

    @patch("predict_gbm.preprocessing.norm_ss_coregistration.apply_longitudinal_warp")
    @patch("predict_gbm.preprocessing.norm_ss_coregistration.optimize_warp_field")
    @patch("predict_gbm.preprocessing.norm_ss_coregistration.resolve_dirac_disp_field")
    @patch("predict_gbm.preprocessing.norm_ss_coregistration.run_dirac_inference")
    def test_register_recurrence_runs_three_step_dirac_pipeline(
        self,
        run_dirac_mock,
        resolve_mock,
        optimize_mock,
        apply_mock,
    ):
        pre = self._save_mock_nifti("pre.nii.gz")
        post = self._save_mock_nifti("post.nii.gz")
        seg = self._save_mock_nifti("seg.nii.gz")

        longitudinal_dir = self.outdir / "longitudinal"
        longitudinal_dir.mkdir(parents=True, exist_ok=True)
        initial_fwd = longitudinal_dir / "followup_to_preop_disp_voxel.nii.gz"
        initial_bwd = longitudinal_dir / "preop_to_followup_disp_voxel.nii.gz"
        initial_fwd.write_text("dummy")
        initial_bwd.write_text("dummy")

        optimized = longitudinal_dir / "followup_to_preop_disp_voxel_optimized.nii.gz"

        resolve_mock.side_effect = [
            initial_fwd,
            initial_bwd,
            longitudinal_dir / "dirac_infer_case_yx_seg.nii.gz",
            longitudinal_dir / "dirac_infer_case_xy_seg.nii.gz",
        ]

        def _optimize_side_effect(*args, **kwargs):
            optimized.write_text("optimized")

        optimize_mock.side_effect = _optimize_side_effect

        register_recurrence(pre, post, seg, self.outdir)

        run_dirac_mock.assert_called_once()
        optimize_mock.assert_called_once()
        apply_mock.assert_called_once()
        self.assertTrue(
            (self.outdir / "longitudinal" / "longitudinal_trafo.nii.gz").exists()
        )

    @patch("predict_gbm.preprocessing.norm_ss_coregistration.ants.image_write")
    @patch("predict_gbm.preprocessing.norm_ss_coregistration.ants.apply_transforms")
    @patch("predict_gbm.preprocessing.norm_ss_coregistration.ants.registration")
    @patch("predict_gbm.preprocessing.norm_ss_coregistration.ants.image_read")
    def test_register_recurrence_syn_uses_ants_registration(
        self,
        image_read_mock,
        registration_mock,
        apply_transforms_mock,
        image_write_mock,
    ):
        pre = self._save_mock_nifti("pre.nii.gz")
        post = self._save_mock_nifti("post.nii.gz")
        seg = self._save_mock_nifti("seg.nii.gz")

        pre_img = MagicMock()
        pre_img.clone.return_value = pre_img
        post_img = MagicMock()
        seg_img = MagicMock()
        image_read_mock.side_effect = [pre_img, post_img, seg_img]

        trafo_file = self.outdir / "warp.nii.gz"
        trafo_file.write_text("warp")
        registration_mock.return_value = {
            "fwdtransforms": [str(trafo_file)],
            "warpedmovout": MagicMock(),
        }
        apply_transforms_mock.return_value = MagicMock()

        register_recurrence(
            pre,
            post,
            seg,
            self.outdir,
            registration_algorithm="syn",
        )

        registration_mock.assert_called_once()
        apply_transforms_mock.assert_called_once()
        self.assertTrue(
            (self.outdir / "longitudinal" / "longitudinal_trafo.nii.gz").exists()
        )

    def test_register_recurrence_invalid_algorithm_raises(self):
        pre = self._save_mock_nifti("pre.nii.gz")
        post = self._save_mock_nifti("post.nii.gz")
        seg = self._save_mock_nifti("seg.nii.gz")

        with self.assertRaises(ValueError):
            register_recurrence(
                pre,
                post,
                seg,
                self.outdir,
                registration_algorithm="invalid",
            )


if __name__ == "__main__":
    unittest.main()
