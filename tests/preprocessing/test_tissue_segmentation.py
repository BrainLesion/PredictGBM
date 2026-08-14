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

    def test_run_tissue_seg_dispatches_to_atropos_n4(self):
        with patch.object(
            ts, "run_tissue_seg_atropos_n4"
        ) as mock_atropos, patch.object(
            ts, "run_tissue_seg_atlas_registration"
        ) as mock_atlas:
            ts.run_tissue_seg(
                t1_file=self.tmp / "t1.nii.gz",
                outdir=self.tmp / "out",
                algorithm="antsAtroposN4",
            )
        mock_atropos.assert_called_once()
        mock_atlas.assert_not_called()

    def test_run_tissue_seg_rejects_unknown_algorithm(self):
        with self.assertRaises(ValueError):
            ts.run_tissue_seg(
                t1_file=self.tmp / "t1.nii.gz",
                outdir=self.tmp / "out",
                algorithm="not_a_real_algorithm",
            )

    def test_run_tissue_seg_atropos_n4(self):
        t1_file = self.tmp / "t1.nii.gz"
        nib.save(nib.Nifti1Image(np.zeros((2, 2, 2)), np.eye(4)), t1_file)

        outdir = self.tmp / "out"

        brain_mask_file = self.tmp / "brain_mask.nii.gz"
        nib.save(nib.Nifti1Image(np.ones((2, 2, 2)), np.eye(4)), brain_mask_file)

        recorded_cmds = []

        def fake_run(cmd, check=False):
            recorded_cmds.append(cmd)
            seg_prefix = Path(cmd[cmd.index("-o") + 1])
            seg = np.zeros((2, 2, 2), dtype=np.int32)
            seg[0, 0, 0] = 1
            seg[0, 0, 1] = 2
            seg[0, 1, 0] = 3
            nib.save(
                nib.Nifti1Image(seg, np.eye(4)),
                f"{seg_prefix}Segmentation.nii.gz",
            )
            for label in (1, 2, 3):
                nib.save(
                    nib.Nifti1Image(
                        np.full((2, 2, 2), 0.1 * label, dtype=np.float32), np.eye(4)
                    ),
                    f"{seg_prefix}SegmentationPosteriors{label}.nii.gz",
                )
            return SimpleNamespace(returncode=0)

        with patch.object(
            ts, "BRAIN_MASK_SCHEMA", PathSchema(str(brain_mask_file))
        ), patch.object(
            ts.subprocess, "run", side_effect=fake_run
        ):
            ts.run_tissue_seg_atropos_n4(t1_file, outdir)

        # No spatial priors: atropos initializes with k-means
        self.assertEqual(len(recorded_cmds), 1)
        self.assertNotIn("-p", recorded_cmds[0])
        self.assertNotIn("-w", recorded_cmds[0])

        gm_pbmap = TISSUE_PBMAP_SCHEMA.format(base_dir=outdir, tissue="gm")

        # antsAtroposN4 no longer keeps the discrete segmentation or per-tissue masks
        self.assertTrue(gm_pbmap.exists())
        self.assertAlmostEqual(
            float(nib.load(str(gm_pbmap)).get_fdata()[0, 0, 1]), 0.2, places=5
        )

    def test_run_tissue_seg_atlas_registration_only_writes_pbmaps(self):
        t1_file = self.tmp / "t1.nii.gz"
        nib.save(nib.Nifti1Image(np.zeros((2, 2, 2)), np.eye(4)), t1_file)

        atlas_t1 = self.tmp / "atlas_t1.nii.gz"
        nib.save(nib.Nifti1Image(np.zeros((2, 2, 2)), np.eye(4)), atlas_t1)

        # Vary the pbmaps per-voxel so they're non-trivial. Since mock_apply_transforms passes
        # the moving image through unchanged, these values become the "warped" pbmaps directly.
        pbmap_values = {
            "csf": {(0, 0, 0): 1.0},
            "gm": {(0, 0, 1): 1.0},
            "wm": {(0, 1, 0): 1.0},
        }
        for tissue, voxels in pbmap_values.items():
            arr = np.zeros((2, 2, 2), dtype=np.float32)
            for voxel, value in voxels.items():
                arr[voxel] = value
            pbmap_path = self.tmp / f"{tissue}_pbmap.nii.gz"
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

        with patch.object(
            ts, "ATLAS_STRIPPED_SCHEMA", PathSchema(str(atlas_t1))
        ), patch.object(
            ts,
            "ATLAS_TISSUE_PBMAP_SCHEMA",
            PathSchema(self.tmp / "{tissue}_pbmap.nii.gz"),
        ), patch.object(
            ts, "ants", mock_ants
        ):
            ts.run_tissue_seg(t1_file, outdir, mask_file)

        gm_pbmap = TISSUE_PBMAP_SCHEMA.format(base_dir=outdir, tissue="gm")

        self.assertTrue(gm_pbmap.exists())
        self.assertEqual(nib.load(str(gm_pbmap)).get_fdata()[0, 0, 1], 1.0)

    def test_derive_tissue_labelmap_from_pbmaps(self):
        pbmap_values = {
            "csf": {(0, 0, 0): 1.0},
            "gm": {(0, 0, 1): 1.0},
            "wm": {(0, 1, 0): 1.0},
        }
        pbmap_paths = {}
        for tissue, voxels in pbmap_values.items():
            arr = np.zeros((2, 2, 2), dtype=np.float32)
            for voxel, value in voxels.items():
                arr[voxel] = value
            pbmap_path = self.tmp / f"{tissue}_pbmap.nii.gz"
            nib.save(nib.Nifti1Image(arr, np.eye(4)), pbmap_path)
            pbmap_paths[tissue] = pbmap_path

        labelmap_nifti = ts.derive_tissue_labelmap_from_pbmaps(
            csf_pbmap_path=pbmap_paths["csf"],
            gm_pbmap_path=pbmap_paths["gm"],
            wm_pbmap_path=pbmap_paths["wm"],
        )

        labelmap = labelmap_nifti.get_fdata()
        self.assertEqual(labelmap[0, 0, 0], 1)  # csf
        self.assertEqual(labelmap[0, 0, 1], 2)  # gm
        self.assertEqual(labelmap[0, 1, 0], 3)  # wm
        self.assertEqual(labelmap[1, 1, 1], 0)  # background, all pbmaps 0

    def test_derive_tissue_labelmap(self):
        outdir = self.tmp / "exam"

        pbmap_values = {
            "csf": {(0, 0, 0): 1.0},
            "gm": {(0, 0, 1): 1.0},
            "wm": {(0, 1, 0): 1.0},
        }
        for tissue, voxels in pbmap_values.items():
            arr = np.zeros((2, 2, 2), dtype=np.float32)
            for voxel, value in voxels.items():
                arr[voxel] = value
            pbmap_path = TISSUE_PBMAP_SCHEMA.format(base_dir=outdir, tissue=tissue)
            pbmap_path.parent.mkdir(parents=True, exist_ok=True)
            nib.save(nib.Nifti1Image(arr, np.eye(4)), pbmap_path)

        labelmap_nifti = ts.derive_tissue_labelmap(outdir)

        labelmap = labelmap_nifti.get_fdata()
        self.assertEqual(labelmap[0, 0, 1], 2)  # gm


if __name__ == "__main__":
    unittest.main()
