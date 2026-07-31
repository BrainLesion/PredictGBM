import tempfile
import unittest
import warnings
from pathlib import Path
from unittest.mock import patch

import nibabel as nib
import numpy as np

from tests.helpers import generate_mock_nifti
from predict_gbm.evaluation.evaluate import (
    create_standard_plan,
    topk_plan,
    generate_distance_fade_mask,
    generate_distance_fade_mask_no_plateau,
    evaluate_tumor_model,
)

# Silence third-party warnings that clutter test output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


def mock_resample(filepath, resample_params, interp_type=0):
    return nib.load(filepath).get_fdata()


class TestEvaluateUtils(unittest.TestCase):
    def test_create_standard_plan(self):
        core = np.zeros((5, 5, 5), dtype=np.int32)
        core[2, 2, 2] = 1
        plan = create_standard_plan(core, 1)
        self.assertEqual(int(plan.sum()), 7)
        with self.assertRaises(ValueError):
            create_standard_plan(core, 0)

    def test_topk_plan(self):
        scores = np.array([0.1, 0.7, 0.4, 0.9], dtype=np.float32)
        plan = topk_plan(scores, target_voxels=2)
        self.assertEqual(int(plan.sum()), 2)
        self.assertEqual(int(plan[3]), 1)
        self.assertEqual(int(plan[1]), 1)

        mask = np.array([0, 1, 1, 0], dtype=np.int32)
        plan_masked = topk_plan(scores, target_voxels=1, mask=mask)
        self.assertEqual(int(plan_masked.sum()), 1)
        self.assertEqual(int(plan_masked[1]), 1)
        self.assertEqual(int(plan_masked[0]), 0)

    def test_generate_distance_fade_mask(self):
        mask = np.zeros((3, 3, 3), dtype=np.int32)
        mask[1, 1, 1] = 1
        fade = generate_distance_fade_mask(mask)
        self.assertAlmostEqual(float(fade[1, 1, 1]), 1.0)
        self.assertAlmostEqual(float(fade[0, 0, 0]), 0.0)
        with self.assertRaises(ValueError):
            generate_distance_fade_mask(np.array([0, 2]))

    def test_generate_distance_fade_mask_no_plateau(self):
        mask = np.zeros((3, 3, 3), dtype=np.int32)
        mask[1, 1, 1] = 1
        visible_tumor_threshold = 0.5
        fade = generate_distance_fade_mask_no_plateau(
            mask, visible_tumor_threshold=visible_tumor_threshold
        )
        self.assertAlmostEqual(float(fade[1, 1, 1]), 1.0)
        self.assertLess(float(fade[1, 1, 0]), visible_tumor_threshold)
        self.assertLess(float(fade[0, 0, 0]), visible_tumor_threshold)
        with self.assertRaises(ValueError):
            generate_distance_fade_mask_no_plateau(np.array([0, 2]))


class TestEvaluateTumorModel(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.shape = (5, 5, 5)
        base_img = generate_mock_nifti(shape=self.shape)
        affine = base_img.affine

        # t1c with background
        t1c_data = np.ones(self.shape, dtype=np.float32)
        t1c_data[0, 0, 0] = 0
        self.t1c_file = Path(self.temp_dir.name) / "t1c.nii.gz"
        nib.save(nib.Nifti1Image(t1c_data, affine), self.t1c_file)

        # tumor segmentation
        tumor_data = np.zeros(self.shape, dtype=np.int16)
        tumor_data[2, 2, 2] = 3
        self.tumorseg_file = Path(self.temp_dir.name) / "tumorseg.nii.gz"
        nib.save(nib.Nifti1Image(tumor_data, affine), self.tumorseg_file)

        # recurrence segmentation
        rec_data = np.zeros(self.shape, dtype=np.int16)
        rec_data[2, 2, 2] = 3
        self.recurrence_file = Path(self.temp_dir.name) / "recurrence.nii.gz"
        nib.save(nib.Nifti1Image(rec_data, affine), self.recurrence_file)

        # prediction file
        pred_data = (
            np.linspace(0, 1, np.prod(self.shape))
            .reshape(self.shape)
            .astype(np.float32)
        )
        self.pred_file = Path(self.temp_dir.name) / "pred.nii.gz"
        nib.save(nib.Nifti1Image(pred_data, affine), self.pred_file)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_evaluate_tumor_model(self):
        with patch(
            "predict_gbm.evaluation.evaluate.load_and_resample_mri_data",
            side_effect=mock_resample,
        ):
            results, standard_plan, model_plan = evaluate_tumor_model(
                t1c_file=self.t1c_file,
                tumorseg_file=self.tumorseg_file,
                recurrence_file=self.recurrence_file,
                pred_file=self.pred_file,
                ctv_margin=1,
            )
        expected_keys = {
            "recurrence_coverage_standard",
            "recurrence_coverage_standard_all",
            "recurrence_coverage_model",
            "recurrence_coverage_model_all",
        }
        self.assertTrue(expected_keys.issubset(results.keys()))
        self.assertEqual(standard_plan.shape, self.shape)
        self.assertEqual(model_plan.shape, self.shape)


if __name__ == "__main__":
    unittest.main()
