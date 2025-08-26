import tempfile
import unittest
import numpy as np
import nibabel as nib
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace
from predict_gbm.preprocessing import tissue_segmentation as ts
from predict_gbm.utils.constants import (
    PathSchema,
    TISSUE_SEG_SCHEMA,
    TISSUE_SCHEMA,
    TISSUE_PBMAP_SCHEMA,
)


class FakeAntsImage:
    def __init__(self, data):
        self.data = data

    def clone(self, _type=None):
        return self


def mock_image_read(path):
    arr = nib.load(str(path)).get_fdata()
    return FakeAntsImage(arr)


def mock_registration(*args, **kwargs):
    return {"fwdtransforms": ["dummy"]}


def mock_apply_transforms(fixed, moving, transformlist, interpolator):
    return moving


def mock_image_write(image, filename):
    nib.save(nib.Nifti1Image(image.data, np.eye(4)), str(filename))


class TestTissueSegmentation(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.tmp = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_generate_healthy_brain_mask(self):
        brain = np.ones((2, 2, 2), dtype=np.int32)
        brain_file = self.tmp / "brain.nii.gz"
        nib.save(nib.Nifti1Image(brain, np.eye(4)), brain_file)

        tumor = np.zeros((2, 2, 2), dtype=np.int32)
        tumor[0, 0, 0] = 1
        tumor_file = self.tmp / "tumor.nii.gz"
        nib.save(nib.Nifti1Image(tumor, np.eye(4)), tumor_file)

        out_file = self.tmp / "healthy.nii.gz"
        ts.generate_healthy_brain_mask(brain_file, tumor_file, out_file)

        self.assertTrue(out_file.exists())
        result = nib.load(str(out_file)).get_fdata()
        self.assertEqual(result[0, 0, 0], 0)
        self.assertEqual(result[1, 1, 1], 1)

    def test_generate_registration_mask(self):
        tumor = np.zeros((2, 2, 2), dtype=np.int32)
        tumor[0, 0, 0] = 1
        tumor[0, 1, 0] = 2
        tumor_file = self.tmp / "tumor.nii.gz"
        nib.save(nib.Nifti1Image(tumor, np.eye(4)), tumor_file)

        out_file = self.tmp / "mask.nii.gz"
        ts.generate_registration_mask(tumor_file, out_file)

        mask = nib.load(str(out_file)).get_fdata()
        self.assertEqual(mask[0, 0, 0], 0)
        self.assertEqual(mask[0, 1, 0], 1)

    def test_run_tissue_seg_registration_outputs(self):
        t1_file = self.tmp / "t1.nii.gz"
        nib.save(nib.Nifti1Image(np.zeros((2, 2, 2)), np.eye(4)), t1_file)

        atlas_t1 = self.tmp / "atlas_t1.nii.gz"
        nib.save(nib.Nifti1Image(np.zeros((2, 2, 2)), np.eye(4)), atlas_t1)

        tissues = np.zeros((2, 2, 2), dtype=np.int32)
        tissues[0, 0, 0] = 1
        tissues[0, 0, 1] = 2
        tissues[0, 1, 0] = 3
        atlas_tissues = self.tmp / "atlas_tissues.nii.gz"
        nib.save(nib.Nifti1Image(tissues, np.eye(4)), atlas_tissues)

        for tissue, val in {"csf": 0.1, "gm": 0.2, "wm": 0.3}.items():
            pbmap_path = self.tmp / f"{tissue}_pbmap.nii.gz"
            arr = np.full((2, 2, 2), val, dtype=np.float32)
            nib.save(nib.Nifti1Image(arr, np.eye(4)), pbmap_path)

        mask_file = self.tmp / "reg_mask.nii.gz"
        nib.save(nib.Nifti1Image(np.ones((2, 2, 2)), np.eye(4)), mask_file)

        mock_ants = SimpleNamespace(
            image_read=mock_image_read,
            registration=mock_registration,
            apply_transforms=mock_apply_transforms,
            image_write=mock_image_write,
        )

        outdir = self.tmp / "out"

        with patch.object(ts, "ATLAS_T1_DIR", atlas_t1), patch.object(
            ts, "ATLAS_TISSUES_DIR", atlas_tissues
        ), patch.object(
            ts,
            "ATLAS_TISSUE_PBMAPS_DIR",
            PathSchema(self.tmp / "{tissue}_pbmap.nii.gz"),
        ), patch.object(
            ts, "ants", mock_ants
        ):
            ts.run_tissue_seg_registration(t1_file, outdir, mask_file)

        seg_file = TISSUE_SEG_SCHEMA.format(base_dir=outdir)
        gm_mask = TISSUE_SCHEMA.format(base_dir=outdir, tissue="gm")
        gm_pbmap = TISSUE_PBMAP_SCHEMA.format(base_dir=outdir, tissue="gm")

        self.assertTrue(seg_file.exists())
        self.assertTrue(gm_mask.exists())
        self.assertTrue(gm_pbmap.exists())


if __name__ == "__main__":
    unittest.main()
