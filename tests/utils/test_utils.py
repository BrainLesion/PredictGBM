import os
import stat
import tempfile
import unittest
import warnings
import numpy as np
import nibabel as nib
from pathlib import Path
from unittest.mock import patch
from pypdf import PdfWriter, PdfReader
from tests.helpers import generate_mock_nifti
from predict_gbm.utils.utils import (
    compute_center_of_mass,
    load_mri_data,
    load_and_resample_mri_data,
    load_segmentation,
    make_symlink,
    merge_pdfs,
    is_binary_array,
    temporary_tmpdir,
)

# Silence third-party warnings that clutter test output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


def mock_resample(image, resample_params, use_voxels=True, interp_type=0):
    return image


def mock_write(image, filename):
    nib.save(image, filename)


class TestUtils(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_compute_center_of_mass(self):
        seg = np.zeros((3, 4, 5), dtype=int)
        seg[1, 2, 3] = 1
        mri = np.zeros_like(seg)
        com = compute_center_of_mass(seg, mri, classes=[1])
        self.assertEqual(com, (1, 2, 3))

    def test_compute_center_of_mass_empty(self):
        seg = np.zeros((4, 6, 8), dtype=int)
        mri = np.zeros_like(seg)
        com = compute_center_of_mass(seg, mri)
        self.assertEqual(com, (2, 3, 4))

    def test_load_mri_data(self):
        img = generate_mock_nifti(shape=(2, 2, 2))
        path = self.tmp_path / "img.nii.gz"
        nib.save(img, path)
        data = load_mri_data(path)
        np.testing.assert_array_almost_equal(data, img.get_fdata())

    def test_load_and_resample_mri_data(self):
        # Loading and running ants takes time so we use mock functions
        img = generate_mock_nifti(shape=(2, 2, 2))
        path = self.tmp_path / "img.nii.gz"
        nib.save(img, path)

        def mock_read(_path):
            return img

        with patch("predict_gbm.utils.utils.ants.image_read", mock_read), patch(
            "predict_gbm.utils.utils.ants.resample_image"
        ) as resample_mock, patch(
            "predict_gbm.utils.utils.ants.image_write"
        ) as write_mock:
            resample_mock.side_effect = mock_resample
            write_mock.side_effect = mock_write
            data = load_and_resample_mri_data(path, (1, 1, 1))
            np.testing.assert_array_almost_equal(data, img.get_fdata())
            resample_mock.assert_called_once()
            write_mock.assert_called_once()

    def test_load_segmentation(self):
        affine = generate_mock_nifti(shape=(1, 2, 2)).affine
        data = np.array([[[0, 1.2], [2.8, 1.6]]], dtype=np.float32)
        img = nib.Nifti1Image(data, affine)
        path = self.tmp_path / "seg.nii.gz"
        nib.save(img, path)
        seg = load_segmentation(path)
        self.assertEqual(seg.dtype, np.int32)
        np.testing.assert_array_equal(seg, np.array([[[0, 1], [3, 2]]], dtype=np.int32))

    def test_make_symlink(self):
        src = self.tmp_path / "file.txt"
        src.write_text("data")
        dst = self.tmp_path / "link.txt"
        make_symlink(src, dst)
        self.assertTrue(dst.is_symlink())
        self.assertEqual(os.readlink(dst), str(src.resolve()))
        mode = src.stat().st_mode
        self.assertFalse(mode & stat.S_IWUSR)

    def test_merge_pdfs(self):
        pdf1 = self.tmp_path / "a.pdf"
        pdf2 = self.tmp_path / "b.pdf"
        for p in [pdf1, pdf2]:
            writer = PdfWriter()
            writer.add_blank_page(width=10, height=10)
            with open(p, "wb") as f:
                writer.write(f)
        out = self.tmp_path / "merged.pdf"
        merge_pdfs([pdf1, pdf2], out)
        self.assertTrue(out.exists())
        reader = PdfReader(str(out))
        self.assertEqual(len(reader.pages), 2)

    def test_is_binary_array(self):
        self.assertTrue(is_binary_array(np.array([0, 1, 0, 1])))
        self.assertFalse(is_binary_array(np.array([0, 2])))

    def test_temporary_tmpdir(self):
        base = self.tmp_path / "base"
        old_env = os.environ.get("TMPDIR")
        old_tempdir = tempfile.tempdir
        with temporary_tmpdir(base) as tmpdir:
            self.assertTrue(tmpdir.exists())
            self.assertEqual(os.environ["TMPDIR"], str(tmpdir))
            self.assertEqual(tempfile.tempdir, str(tmpdir))
            (tmpdir / "test.txt").write_text("hi")
        self.assertEqual(os.environ.get("TMPDIR"), old_env)
        self.assertEqual(tempfile.tempdir, old_tempdir)
        self.assertFalse(tmpdir.exists())


if __name__ == "__main__":
    unittest.main()
