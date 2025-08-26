import tempfile
import unittest
import warnings
import nibabel as nib
from pathlib import Path
from tests.helpers import generate_mock_nifti
from predict_gbm.preprocessing.norm_ss_coregistration import (
    normalize,
    initialize_center_modality,
    initialize_moving_modalities,
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


if __name__ == "__main__":
    unittest.main()
