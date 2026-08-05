import shutil
import tempfile
import unittest
import warnings
import numpy as np
import nibabel as nib
from pathlib import Path
from tests.helpers import generate_mock_dicom_series
from predict_gbm.preprocessing.conversion import (
    dicom_to_nifti,
    dti_to_adc,
    remove_postfixes,
)

# Silence third-party warnings that clutter test output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


REQUIRED_BINARY = "dcm2niix"
class_skip = unittest.skipUnless(
    shutil.which(REQUIRED_BINARY) is not None,
    reason=f"Skipping: '{REQUIRED_BINARY}' is not installed on this machine.",
)


@class_skip
class TestDicomToNifti(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.dicom_dir = Path(self.temp_dir.name) / "dicoms"
        self.dicom_dir.mkdir()
        self.outdir = Path(self.temp_dir.name) / "converted"
        self.outdir.mkdir()

        generate_mock_dicom_series(outdir=self.dicom_dir)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_dicom_to_nifti_creates_output(self):
        outfile = self.outdir / "test.nii.gz"
        log_file = self.outdir / "test_conversion.log"

        result = dicom_to_nifti(self.dicom_dir, outfile)

        self.assertEqual(result, outfile)
        self.assertTrue(log_file.exists())
        self.assertTrue(outfile.exists())


class TestDicomToNiftiErrors(unittest.TestCase):
    """Error paths that don't need a real dcm2niix binary."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.dicom_dir = Path(self.temp_dir.name) / "dicoms"
        self.dicom_dir.mkdir()
        self.outdir = Path(self.temp_dir.name) / "converted"
        self.outdir.mkdir()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_raises_on_dotted_basename(self):
        outfile = self.outdir / "test.v2.nii.gz"

        with self.assertRaises(ValueError):
            dicom_to_nifti(self.dicom_dir, outfile)

    def test_raises_instead_of_swallowing_failure(self):
        # Previously dicom_to_nifti caught every exception and only logged it; a
        # missing/broken dcm2niix binary would silently no-op. It must now raise.
        outfile = self.outdir / "test.nii.gz"

        with self.assertRaises(Exception):
            dicom_to_nifti(
                self.dicom_dir, outfile, dcm2niix_location="not-a-real-binary-xyz"
            )


class TestRemovePostfixes(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.outdir = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def _touch(self, *names):
        files = []
        for name in names:
            f = self.outdir / name
            f.touch()
            files.append(f)
        return files

    def test_strips_matching_postfix(self):
        files = self._touch("t1_e1.nii.gz")

        remove_postfixes(files, "t1")

        self.assertTrue((self.outdir / "t1.nii.gz").exists())
        self.assertFalse((self.outdir / "t1_e1.nii.gz").exists())

    def test_leaves_non_matching_and_log_files_untouched(self):
        files = self._touch("other_e1.nii.gz", "t1_conversion.log")

        remove_postfixes(files, "t1")

        self.assertTrue((self.outdir / "other_e1.nii.gz").exists())
        self.assertTrue((self.outdir / "t1_conversion.log").exists())

    def test_rejects_dotted_basename(self):
        files = self._touch("t1.v2_e1.nii.gz")

        with self.assertRaises(ValueError):
            remove_postfixes(files, "t1.v2")

    def test_raises_on_collision_instead_of_silently_renaming(self):
        # Previously a collision was resolved by appending '_a' to the new name
        # instead of raising, silently masking multi-echo/multi-series inputs.
        files = self._touch("t1.nii.gz", "t1_e1.nii.gz")

        with self.assertRaises(FileExistsError):
            remove_postfixes(files, "t1")

        # Neither file should have been touched by the failed rename.
        self.assertTrue((self.outdir / "t1.nii.gz").exists())
        self.assertTrue((self.outdir / "t1_e1.nii.gz").exists())


def _synthesize_dwi(shape, bvals, bvecs, tensor, s0=1000.0):
    """Synthesize S = S0 * exp(-b * g^T D g) for a fixed diffusion tensor."""
    bvals = np.asarray(bvals, dtype=np.float64)
    bvecs = np.asarray(bvecs, dtype=np.float64)
    signal = np.empty(shape + (len(bvals),), dtype=np.float32)
    for i, (b, g) in enumerate(zip(bvals, bvecs)):
        atten = np.exp(-b * (g @ tensor @ g))
        signal[..., i] = s0 * atten
    return signal


def _write_bval_bvec(bval_path, bvec_path, bvals, bvecs):
    np.savetxt(bval_path, np.asarray(bvals, dtype=np.float64).reshape(1, -1), fmt="%.1f")
    np.savetxt(bvec_path, np.asarray(bvecs, dtype=np.float64).T, fmt="%.6f")


class TestDtiToAdc(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.workdir = Path(self.temp_dir.name)
        self.shape = (4, 4, 4)
        self.affine = np.diag([2.0, 2.0, 2.0, 1.0])
        # Anisotropic tensor in mm^2/s.
        self.tensor = np.array(
            [
                [1.5e-3, 0.1e-3, 0.0],
                [0.1e-3, 0.8e-3, 0.0],
                [0.0, 0.0, 0.3e-3],
            ]
        )
        self.md_analytic = np.trace(self.tensor) / 3 * 1e6  # 1e-6 mm^2/s units

        rng = np.random.default_rng(0)
        directions = rng.normal(size=(12, 3))
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        self.directions = directions

    def tearDown(self):
        self.temp_dir.cleanup()

    def _write_dwi(self, name, bvals, bvecs, suffix=".nii.gz"):
        signal = _synthesize_dwi(self.shape, bvals, bvecs, self.tensor)
        infile = self.workdir / f"{name}{suffix}"
        nib.save(nib.Nifti1Image(signal, self.affine), infile)
        _write_bval_bvec(
            self.workdir / f"{name}.bval", self.workdir / f"{name}.bvec", bvals, bvecs
        )
        return infile

    def test_recovers_analytic_md(self):
        bvals = np.concatenate([[0.0, 0.0], np.full(12, 1000.0)])
        bvecs = np.vstack([np.zeros((2, 3)), self.directions])
        infile = self._write_dwi("single_shell", bvals, bvecs)
        outfile = self.workdir / "md.nii.gz"

        result = dti_to_adc(infile, outfile)

        self.assertEqual(result, outfile)
        md = nib.load(outfile).get_fdata()
        self.assertTrue(np.allclose(md, self.md_analytic, atol=1e-6))

    def test_multishell_excludes_high_b_shell(self):
        bvals_low = np.concatenate([[0.0, 0.0], np.full(12, 1000.0)])
        bvecs_low = np.vstack([np.zeros((2, 3)), self.directions])
        infile_single = self._write_dwi("single_shell_ref", bvals_low, bvecs_low)
        outfile_single = self.workdir / "md_single.nii.gz"
        dti_to_adc(infile_single, outfile_single)
        md_single = nib.load(outfile_single).get_fdata()

        bvals_multi = np.concatenate([bvals_low, np.full(12, 3000.0)])
        bvecs_multi = np.vstack([bvecs_low, self.directions])
        infile_multi = self._write_dwi("multi_shell", bvals_multi, bvecs_multi)
        outfile_multi = self.workdir / "md_multi.nii.gz"
        dti_to_adc(infile_multi, outfile_multi, bval_max=1200.0)
        md_multi = nib.load(outfile_multi).get_fdata()

        self.assertTrue(np.allclose(md_multi, md_single, atol=1e-6))
        self.assertTrue(np.allclose(md_multi, self.md_analytic, atol=1e-6))

    def test_raises_on_insufficient_directions(self):
        bvals = np.array([0.0, 0.0, 1000.0, 1000.0, 1000.0])
        bvecs = np.vstack([np.zeros((2, 3)), self.directions[:3]])
        infile = self._write_dwi("too_few_directions", bvals, bvecs)
        outfile = self.workdir / "md.nii.gz"

        with self.assertRaises(ValueError):
            dti_to_adc(infile, outfile)

    def test_raises_filenotfound_for_missing_sidecars(self):
        bvals = np.concatenate([[0.0], np.full(6, 1000.0)])
        bvecs = np.vstack([np.zeros((1, 3)), self.directions[:6]])
        signal = _synthesize_dwi(self.shape, bvals, bvecs, self.tensor)
        infile = self.workdir / "no_sidecars.nii.gz"
        nib.save(nib.Nifti1Image(signal, self.affine), infile)
        outfile = self.workdir / "md.nii.gz"

        with self.assertRaises(FileNotFoundError):
            dti_to_adc(infile, outfile)

    def test_nii_and_nii_gz_round_trip(self):
        bvals = np.concatenate([[0.0, 0.0], np.full(12, 1000.0)])
        bvecs = np.vstack([np.zeros((2, 3)), self.directions])
        infile = self._write_dwi("round_trip", bvals, bvecs)

        outfile_gz = self.workdir / "md.nii.gz"
        outfile_plain = self.workdir / "md.nii"
        dti_to_adc(infile, outfile_gz)
        dti_to_adc(infile, outfile_plain)

        self.assertTrue(outfile_gz.exists())
        self.assertTrue(outfile_plain.exists())
        with open(outfile_gz, "rb") as f:
            self.assertEqual(f.read(2), b"\x1f\x8b")  # gzip magic bytes
        md_gz = nib.load(outfile_gz).get_fdata()
        md_plain = nib.load(outfile_plain).get_fdata()
        self.assertTrue(np.allclose(md_gz, md_plain, atol=1e-6))

    def test_output_header_is_fresh(self):
        bvals = np.concatenate([[0.0, 0.0], np.full(12, 1000.0)])
        bvecs = np.vstack([np.zeros((2, 3)), self.directions])
        signal = _synthesize_dwi(self.shape, bvals, bvecs, self.tensor)

        dwi_img = nib.Nifti1Image(signal, self.affine)
        dwi_img.header["cal_max"] = 5000.0  # only valid for the 4D input's own scale
        infile = self.workdir / "header_check.nii.gz"
        nib.save(dwi_img, infile)
        _write_bval_bvec(
            self.workdir / "header_check.bval",
            self.workdir / "header_check.bvec",
            bvals,
            bvecs,
        )

        outfile = self.workdir / "md.nii.gz"
        dti_to_adc(infile, outfile)

        out_nifti = nib.load(outfile)
        self.assertEqual(len(out_nifti.header.get_data_shape()), 3)
        self.assertEqual(out_nifti.header.get_data_shape(), self.shape)
        self.assertEqual(out_nifti.get_data_dtype(), np.dtype(np.float32))
        self.assertTrue(np.allclose(out_nifti.affine, self.affine))
        self.assertEqual(out_nifti.header.get_zooms(), (2.0, 2.0, 2.0))
        self.assertTrue(np.isnan(out_nifti.header["scl_slope"]))
        self.assertTrue(np.isnan(out_nifti.header["scl_inter"]))
        self.assertNotEqual(float(out_nifti.header["cal_max"]), 5000.0)


if __name__ == "__main__":
    unittest.main()
