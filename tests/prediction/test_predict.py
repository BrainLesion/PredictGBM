import tempfile
import unittest
import warnings
import nibabel as nib
from pathlib import Path
from unittest.mock import patch
from tests.helpers import generate_mock_nifti
from predict_gbm.prediction.predict import (
    predict_tumor_growth,
    PredictTumorGrowthPipe,
)
from predict_gbm.utils.constants import (
    TUMORSEG_SCHEMA,
    TISSUE_PBMAP_SCHEMA,
)

# Silence third-party warnings that clutter test output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


class TestPredict(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def _save_nifti(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        nib.save(generate_mock_nifti(), path)

    def test_predict_tumor_growth_calls_model(self):
        tumorseg = self.tmp_path / "tumorseg.nii.gz"
        gm = self.tmp_path / "gm.nii.gz"
        wm = self.tmp_path / "wm.nii.gz"
        csf = self.tmp_path / "csf.nii.gz"
        for p in [tumorseg, gm, wm, csf]:
            self._save_nifti(p)
        outdir = self.tmp_path / "out"
        outdir.mkdir()

        with patch("predict_gbm.prediction.predict.TumorGrowthModel") as MockModel:
            instance = MockModel.return_value
            predict_tumor_growth(
                tumorseg,
                gm,
                wm,
                csf,
                model_id="model",
                outdir=outdir,
                cuda_device="cpu",
            )
            MockModel.assert_called_once_with(algorithm="model", cuda_device="cpu")
            instance.predict_single.assert_called_once_with(
                gm=gm,
                wm=wm,
                csf=csf,
                tumorseg=tumorseg,
                outdir=outdir,
            )

    def test_predict_tumor_growth_pipe_constructs_paths(self):
        preop_dir = self.tmp_path
        gm = TISSUE_PBMAP_SCHEMA.format(base_dir=preop_dir, tissue="gm")
        wm = TISSUE_PBMAP_SCHEMA.format(base_dir=preop_dir, tissue="wm")
        csf = TISSUE_PBMAP_SCHEMA.format(base_dir=preop_dir, tissue="csf")
        tumorseg = TUMORSEG_SCHEMA.format(base_dir=preop_dir)
        for p in [gm, wm, csf, tumorseg]:
            self._save_nifti(p)

        with patch("predict_gbm.prediction.predict.TumorGrowthModel") as MockModel:
            instance = MockModel.return_value
            pipe = PredictTumorGrowthPipe(
                preop_dir=preop_dir,
                model_id="model",
                cuda_device="cpu",
            )
            pipe.run()
            MockModel.assert_called_once_with(algorithm="model", cuda_device="cpu")
            instance.predict_single.assert_called_once_with(
                gm=gm,
                wm=wm,
                csf=csf,
                tumorseg=tumorseg,
                outdir=preop_dir,
            )


if __name__ == "__main__":
    unittest.main()
