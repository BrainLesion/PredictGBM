import tempfile
import unittest
import warnings
import nibabel as nib
from pathlib import Path
from tests.helpers import generate_mock_nifti
from predict_gbm.prediction.growth_model import load_algorithms, TumorGrowthModel
from predict_gbm.utils.constants import GROWTH_MODEL_DIR, PREDICTION_OUTPUT_SCHEMA


# Silence third-party warnings that clutter test output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


class TestTumorGrowthModel(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)
        self.model = TumorGrowthModel(algorithm="test_model")

    def tearDown(self):
        self.temp_dir.cleanup()

    def _save_nifti(self, path: Path) -> None:
        img = generate_mock_nifti(shape=(2, 2, 2))
        nib.save(img, path)

    def test_load_algorithms(self):
        algorithms = load_algorithms(GROWTH_MODEL_DIR)
        self.assertIn("test_model", algorithms)

    def test_load_algorithms_invalid_dir(self):
        with self.assertRaises(ValueError):
            load_algorithms(self.temp_path / "nonexistent")

    def test_invalid_algorithm(self):
        with self.assertRaises(ValueError):
            TumorGrowthModel(algorithm="does_not_exist")

    def test_standardize_input_files(self):
        tmp_data_dir = self.temp_path / "tmp_data"
        tmp_data_dir.mkdir()
        inputs = {}
        for modality in ["t1c", "gm", "wm", "csf", "tumorseg", "pet"]:
            path = self.temp_path / f"{modality}.nii.gz"
            self._save_nifti(path)
            inputs[modality] = path
        self.model._standardize_input_files(tmp_data_dir, 42, inputs)
        subject_dir = tmp_data_dir / "Patient-42"
        for modality in inputs:
            self.assertTrue((subject_dir / f"42-{modality}.nii.gz").exists())

    def test_process_output(self):
        tmp_outdir = self.temp_path / "tmp_out"
        tmp_outdir.mkdir()
        outdir = self.temp_path / "out"
        subject_id = "00000"
        docker_output = tmp_outdir / f"{subject_id}.nii.gz"
        self._save_nifti(docker_output)
        self.model._process_output(tmp_outdir, subject_id, outdir)
        expected = PREDICTION_OUTPUT_SCHEMA.format(
            base_dir=outdir, algo_id="test_model"
        )
        self.assertTrue(expected.exists())
        self.assertFalse(docker_output.exists())


if __name__ == "__main__":
    unittest.main()
