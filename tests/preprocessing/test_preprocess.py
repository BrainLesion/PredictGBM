import tempfile
import unittest
import warnings
from pathlib import Path
from unittest.mock import patch
import nibabel as nib
from tests.helpers import generate_mock_nifti
from predict_gbm.preprocessing.preprocess import (
    DicomPreprocessor,
    NiftiPreprocessor,
    RegisterRecurrencePipe,
)
from predict_gbm.utils.constants import (
    LONGITUDINAL_WARP_SCHEMA,
    MODALITY_STRIPPED_SCHEMA,
    TUMORSEG_SCHEMA,
)

# Silence third-party warnings that clutter test output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


class TestDicomPreprocessor(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.outdir = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    @patch("predict_gbm.preprocessing.preprocess.run_tissue_seg_registration")
    @patch("predict_gbm.preprocessing.preprocess.generate_registration_mask")
    @patch("predict_gbm.preprocessing.preprocess.run_brats")
    @patch("predict_gbm.preprocessing.preprocess.norm_ss_coregister")
    @patch("predict_gbm.preprocessing.preprocess.dicom_to_nifti")
    def test_converts_and_forwards_additional_modalities(
        self,
        dicom_to_nifti_mock,
        norm_ss_coregister_mock,
        run_brats_mock,
        generate_mask_mock,
        tissue_seg_mock,
    ):
        preprocessor = DicomPreprocessor(
            t1_dir=Path("t1dir"),
            t1c_dir=Path("t1cdir"),
            t2_dir=Path("t2dir"),
            flair_dir=Path("flairdir"),
            perform_tissueseg=True,
            outdir=self.outdir,
            additional_modality_dirs={"swi": Path("swidir")},
            additional_quantitative_modality_dirs={"adc": Path("adcdir")},
        )
        preprocessor.run()

        converted_dirs = {
            call.kwargs["input_dir"] for call in dicom_to_nifti_mock.call_args_list
        }
        self.assertEqual(
            converted_dirs,
            {
                Path("t1dir"),
                Path("t1cdir"),
                Path("t2dir"),
                Path("flairdir"),
                Path("swidir"),
                Path("adcdir"),
            },
        )

        norm_kwargs = norm_ss_coregister_mock.call_args.kwargs
        self.assertEqual(set(norm_kwargs["additional_modalities"].keys()), {"swi"})
        self.assertEqual(
            set(norm_kwargs["additional_quantitative_modalities"].keys()), {"adc"}
        )


class TestNiftiPreprocessor(unittest.TestCase):
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

    @patch("predict_gbm.preprocessing.preprocess.generate_registration_mask")
    @patch("predict_gbm.preprocessing.preprocess.run_brats")
    @patch("predict_gbm.preprocessing.preprocess.normalize")
    def test_is_coregistered_skips_normalization_for_quantitative_modality(
        self, normalize_mock, run_brats_mock, generate_mask_mock
    ):
        t1 = self._save_mock_nifti("t1.nii.gz")
        t1c = self._save_mock_nifti("t1c.nii.gz")
        t2 = self._save_mock_nifti("t2.nii.gz")
        fl = self._save_mock_nifti("flair.nii.gz")
        adc = self._save_mock_nifti("adc.nii.gz")

        preprocessor = NiftiPreprocessor(
            t1_file=t1,
            t1c_file=t1c,
            t2_file=t2,
            flair_file=fl,
            perform_tissueseg=False,
            outdir=self.outdir,
            is_coregistered=True,
            is_skull_stripped=True,
            additional_quantitative_modalities={"adc": adc},
        )
        preprocessor.run()

        normalized_files = {
            c.kwargs["img_file"] for c in normalize_mock.call_args_list
        }
        self.assertNotIn(adc, normalized_files)
        self.assertIn(t1c, normalized_files)

        adc_dest = MODALITY_STRIPPED_SCHEMA.format(base_dir=self.outdir, modality="adc")
        self.assertTrue(adc_dest.exists())


class TestRegisterRecurrencePipe(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.outdir = Path(self.temp_dir.name)
        self.preop_dir = self.outdir / "preop"
        self.followup_dir = self.outdir / "followup"
        self.preop_dir.mkdir()
        self.followup_dir.mkdir()

    def tearDown(self):
        self.temp_dir.cleanup()

    def _save_mock_nifti(self, dir_path: Path, name: str) -> Path:
        img = generate_mock_nifti()
        file_path = dir_path / name
        nib.save(img, str(file_path))
        return file_path

    @patch("predict_gbm.preprocessing.preprocess.register_recurrence")
    def test_forwards_additional_modalities_to_register_recurrence(
        self, register_recurrence_mock
    ):
        swi_post = self._save_mock_nifti(self.followup_dir, "swi_post.nii.gz")

        pipe = RegisterRecurrencePipe(
            preop_dir=self.preop_dir,
            followup_dir=self.followup_dir,
            additional_modalities={"swi": swi_post},
        )
        pipe.run()

        register_recurrence_mock.assert_called_once()
        self.assertEqual(
            register_recurrence_mock.call_args.kwargs["additional_modalities"],
            {"swi": swi_post},
        )

    def test_is_coregistered_copies_additional_modalities(self):
        t1c_post_file = MODALITY_STRIPPED_SCHEMA.format(
            base_dir=self.followup_dir, modality="t1c"
        )
        t1c_post_file.parent.mkdir(parents=True, exist_ok=True)
        t1c_post_file.write_text("dummy")

        recurrence_seg_file = TUMORSEG_SCHEMA.format(base_dir=self.followup_dir)
        recurrence_seg_file.parent.mkdir(parents=True, exist_ok=True)
        recurrence_seg_file.write_text("dummy")

        swi_post = self._save_mock_nifti(self.followup_dir, "swi_post.nii.gz")

        pipe = RegisterRecurrencePipe(
            preop_dir=self.preop_dir,
            followup_dir=self.followup_dir,
            is_coregistered=True,
            additional_modalities={"swi": swi_post},
        )
        pipe.run()

        swi_dst = LONGITUDINAL_WARP_SCHEMA.format(
            base_dir=self.followup_dir, modality="swi"
        )
        self.assertTrue(swi_dst.exists())


class TestModalityCollisionValidation(unittest.TestCase):
    """Regression tests: these collisions must be rejected at construction time,
    for every Preprocessor code path, not just when norm_ss_coregister happens to run."""

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

    def test_nifti_preprocessor_is_coregistered_rejects_reserved_name(self):
        t1 = self._save_mock_nifti("t1.nii.gz")
        t1c = self._save_mock_nifti("t1c.nii.gz")
        t2 = self._save_mock_nifti("t2.nii.gz")
        fl = self._save_mock_nifti("flair.nii.gz")
        evil_t2 = self._save_mock_nifti("evil_t2.nii.gz")

        with self.assertRaises(ValueError):
            NiftiPreprocessor(
                t1_file=t1,
                t1c_file=t1c,
                t2_file=t2,
                flair_file=fl,
                perform_tissueseg=False,
                outdir=self.outdir,
                is_coregistered=True,
                is_skull_stripped=True,
                additional_modalities={"t2": evil_t2},
            )

        # Must fail before touching any output files.
        dest = MODALITY_STRIPPED_SCHEMA.format(base_dir=self.outdir, modality="t2")
        self.assertFalse(dest.exists())

    def test_nifti_preprocessor_rejects_overlap_between_modality_dicts(self):
        t1 = self._save_mock_nifti("t1.nii.gz")
        t1c = self._save_mock_nifti("t1c.nii.gz")
        t2 = self._save_mock_nifti("t2.nii.gz")
        fl = self._save_mock_nifti("flair.nii.gz")
        swi_norm = self._save_mock_nifti("swi_norm.nii.gz")
        swi_quant = self._save_mock_nifti("swi_quant.nii.gz")

        with self.assertRaises(ValueError):
            NiftiPreprocessor(
                t1_file=t1,
                t1c_file=t1c,
                t2_file=t2,
                flair_file=fl,
                perform_tissueseg=False,
                outdir=self.outdir,
                is_coregistered=True,
                is_skull_stripped=True,
                additional_modalities={"swi": swi_norm},
                additional_quantitative_modalities={"swi": swi_quant},
            )

    def test_dicom_preprocessor_rejects_reserved_name(self):
        with self.assertRaises(ValueError):
            DicomPreprocessor(
                t1_dir=Path("t1dir"),
                t1c_dir=Path("t1cdir"),
                t2_dir=Path("t2dir"),
                flair_dir=Path("flairdir"),
                perform_tissueseg=True,
                outdir=self.outdir,
                additional_modality_dirs={"flair": Path("evil_flair_dir")},
            )


if __name__ == "__main__":
    unittest.main()
