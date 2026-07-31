import tempfile
from pathlib import Path
from unittest.mock import patch

from predict_gbm.pipeline import DicomProcessor, NiftiProcessor


def _dummy_paths(tmp_path: Path):
    return {
        "t1_preop": tmp_path / "t1_preop",
        "t1c_preop": tmp_path / "t1c_preop",
        "t2_preop": tmp_path / "t2_preop",
        "flair_preop": tmp_path / "flair_preop",
        "t1_followup": tmp_path / "t1_followup",
        "t1c_followup": tmp_path / "t1c_followup",
        "t2_followup": tmp_path / "t2_followup",
        "flair_followup": tmp_path / "flair_followup",
    }


def test_dicom_processor_passes_registration_algorithm_to_register_pipe():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        p = _dummy_paths(tmp_path)

        processor = DicomProcessor(
            patient_id="p1",
            model_id="test_model",
            t1_preop_dir=p["t1_preop"],
            t1c_preop_dir=p["t1c_preop"],
            t2_preop_dir=p["t2_preop"],
            flair_preop_dir=p["flair_preop"],
            t1_followup_dir=p["t1_followup"],
            t1c_followup_dir=p["t1c_followup"],
            t2_followup_dir=p["t2_followup"],
            flair_followup_dir=p["flair_followup"],
            outdir=tmp_path,
            registration_algorithm="syn",
        )

        with patch("predict_gbm.pipeline.RegisterRecurrencePipe") as pipe_mock:
            instance = pipe_mock.return_value
            processor._register_recurrence()
            pipe_mock.assert_called_once()
            assert pipe_mock.call_args.kwargs["registration_algorithm"] == "syn"
            instance.run.assert_called_once()


def test_nifti_processor_passes_registration_algorithm_to_register_pipe():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        processor = NiftiProcessor(
            patient_id="p1",
            model_id="test_model",
            t1_preop_file=tmp_path / "t1_preop.nii.gz",
            t1c_preop_file=tmp_path / "t1c_preop.nii.gz",
            t2_preop_file=tmp_path / "t2_preop.nii.gz",
            flair_preop_file=tmp_path / "flair_preop.nii.gz",
            t1_followup_file=tmp_path / "t1_followup.nii.gz",
            t1c_followup_file=tmp_path / "t1c_followup.nii.gz",
            t2_followup_file=tmp_path / "t2_followup.nii.gz",
            flair_followup_file=tmp_path / "flair_followup.nii.gz",
            outdir=tmp_path,
            registration_algorithm="syn",
        )

        with patch("predict_gbm.pipeline.RegisterRecurrencePipe") as pipe_mock:
            instance = pipe_mock.return_value
            processor._register_recurrence()
            pipe_mock.assert_called_once()
            assert pipe_mock.call_args.kwargs["registration_algorithm"] == "syn"
            instance.run.assert_called_once()
