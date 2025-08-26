import shutil
import tempfile
import unittest
import warnings
from pathlib import Path
from tests.helpers import generate_mock_dicom_series
from predict_gbm.preprocessing.dicom_to_nifti import dicom_to_nifti

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
        outfile = self.outdir / "test"
        outfile_w_suffixes = outfile.with_suffix(".nii.gz")
        log_file = self.outdir / "test_conversion.log"

        dicom_to_nifti(self.dicom_dir, outfile)

        self.assertTrue(log_file.exists())
        self.assertTrue(outfile_w_suffixes.exists())


if __name__ == "__main__":
    unittest.main()
