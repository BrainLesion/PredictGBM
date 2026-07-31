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
    BRAIN_MASK_SCHEMA,
    MODALITY_STRIPPED_SCHEMA,
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
        t1c = self.tmp_path / "t1c.nii.gz"
        flair = self.tmp_path / "flair.nii.gz"
        brain_mask = self.tmp_path / "brain_mask.nii.gz"
        adc = self.tmp_path / "adc.nii.gz"
        for p in [t1c, flair, brain_mask, adc]:
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
                t1c_file=t1c,
                flair_file=flair,
                brain_mask_file=brain_mask,
                adc_file=adc,
            )
            MockModel.assert_called_once_with(algorithm="model", cuda_device="cpu")
            instance.predict_single.assert_called_once_with(
                gm=gm,
                wm=wm,
                csf=csf,
                tumorseg=tumorseg,
                t1c=t1c,
                flair=flair,
                brain_mask=brain_mask,
                adc=adc,
                outdir=outdir,
            )

    def test_predict_tumor_growth_defaults_optional_inputs_to_none(self):
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
            instance.predict_single.assert_called_once_with(
                gm=gm,
                wm=wm,
                csf=csf,
                tumorseg=tumorseg,
                t1c=None,
                flair=None,
                brain_mask=None,
                adc=None,
                outdir=outdir,
            )

    def test_predict_tumor_growth_pipe_constructs_paths(self):
        preop_dir = self.tmp_path
        gm = TISSUE_PBMAP_SCHEMA.format(base_dir=preop_dir, tissue="gm")
        wm = TISSUE_PBMAP_SCHEMA.format(base_dir=preop_dir, tissue="wm")
        csf = TISSUE_PBMAP_SCHEMA.format(base_dir=preop_dir, tissue="csf")
        tumorseg = TUMORSEG_SCHEMA.format(base_dir=preop_dir)
        t1c = MODALITY_STRIPPED_SCHEMA.format(base_dir=preop_dir, modality="t1c")
        flair = MODALITY_STRIPPED_SCHEMA.format(base_dir=preop_dir, modality="flair")
        brain_mask = BRAIN_MASK_SCHEMA.format(base_dir=preop_dir)
        adc = MODALITY_STRIPPED_SCHEMA.format(base_dir=preop_dir, modality="adc")
        for p in [gm, wm, csf, tumorseg, t1c, flair, brain_mask, adc]:
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
                t1c=t1c,
                flair=flair,
                brain_mask=brain_mask,
                adc=adc,
                outdir=preop_dir,
            )


if __name__ == "__main__":
    unittest.main()
