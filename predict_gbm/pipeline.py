import os
import time
from pathlib import Path
from loguru import logger
from typing import Dict, Optional
from predict_gbm.prediction.predict import PredictTumorGrowthPipe
from predict_gbm.evaluation.evaluate import EvaluateTumorModelPipe
from predict_gbm.utils.visualization import VisualizationPipe
from predict_gbm.utils.constants import (
    PATIENT_PREOP_OUTPUT_SCHEMA,
    PATIENT_FOLLOWUP_OUTPUT_SCHEMA,
)
from predict_gbm.preprocessing.preprocess import (
    DicomPreprocessor,
    NiftiPreprocessor,
    RegisterRecurrencePipe,
)


class BaseProcessor:
    """
    Base class for Processors that perform pre-processing, prediction and evaluation.

    Parameters:
        patient_id (str): String acting as patient identifier. Included in the output directory structure.
        model_id (str): String identifying the growth model to be used.
        outdir (Path): Path to the output directory.
        cuda_device (Optional, str): The gpu device to use.
    """

    def __init__(
        self,
        patient_id: str,
        model_id: str,
        outdir: Path,
        cuda_device: str = "0",
    ) -> None:
        self.patient_id = patient_id
        self.model_id = model_id
        self.outdir = outdir
        self.cuda_device = cuda_device

        outdir_preop = PATIENT_PREOP_OUTPUT_SCHEMA.format(
            base_dir=self.outdir.absolute(), patient_id=self.patient_id
        )
        outdir_followup = PATIENT_FOLLOWUP_OUTPUT_SCHEMA.format(
            base_dir=self.outdir.absolute(), patient_id=self.patient_id
        )
        self.outdir_preop = outdir_preop
        self.outdir_followup = outdir_followup

    def run(self) -> Dict[str, float]:

        os.environ["CUDA_VISIBLE_DEVICES"] = self.cuda_device
        start_time = time.time()

        self._preprocess_preop()
        self._preprocess_followup()
        self._register_recurrence()

        predictor = PredictTumorGrowthPipe(
            preop_dir=self.outdir_preop,
            model_id=self.model_id,
            cuda_device=self.cuda_device,
        )
        predictor.run()

        evaluator = EvaluateTumorModelPipe(
            preop_dir=self.outdir_preop,
            followup_dir=self.outdir_followup,
            model_id=self.model_id,
        )
        results = evaluator.run()

        visualizer = VisualizationPipe(
            patient_id=self.patient_id,
            model_id=self.model_id,
            preop_dir=self.outdir_preop,
            followup_dir=self.outdir_followup,
        )
        visualizer.run()

        time_spent = time.time() - start_time
        logger.info(
            f"Pipeline completed in {time_spent:.2f} seconds: {results}"
            f"Pre-operative derivatives saved to {self.outdir_preop}, "
            f"follow-up derivatives saved to {self.outdir_followup}."
        )
        return results

    def _preprocess_preop(self) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def _preprocess_followup(self) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def _register_recurrence(self) -> None:  # pragma: no cover - interface
        raise NotImplementedError


class DicomProcessor(BaseProcessor):
    """
    Performs preprocessing (skull stripping, normalization, atlas registration), growth model prediction
    and evaluation from a pre-operative MRI exam and a follow-up MRI exam passed as DICOM.

    Parameters:
        patient_id (str): String acting as patient identifier. Included in the output directory structure.
        model_id (str): String identifying the growth model to be used.
        t1_preop_dir (Path): Path to the directory containing the pre-operative T1 DICOM images.
        t1c_preop_dir (Path): Path to the directory containing the pre-operative T1c DICOM images.
        t2_preop_dir (Path): Path to the directory containing the pre-operative T2 DICOM images.
        flair_preop_dir (Path): Path to the directory containing the pre-operative FLAIR DICOM images.
        t1_followup_dir (Path): Path to the directory containing the follow-up T1 DICOM images.
        t1c_followup_dir (Path): Path to the directory containing the follow-up T1c DICOM images.
        t2_followup_dir (Path): Path to the directory containing the follow-up T2 DICOM images.
        flair_followup_dir (Path): Path to the directory containing the follow-up FLAIR DICOM images.
        outdir (Path): Path to the output directory.
        dcm2niix_location (Path): Path to the dcm2niix executable.
        cuda_device (Optional, str): The gpu device to use.
    """

    def __init__(
        self,
        patient_id: str,
        model_id: str,
        t1_preop_dir: Path,
        t1c_preop_dir: Path,
        t2_preop_dir: Path,
        flair_preop_dir: Path,
        t1_followup_dir: Path,
        t1c_followup_dir: Path,
        t2_followup_dir: Path,
        flair_followup_dir: Path,
        outdir: Path,
        dcm2niix_location: Path = Path("dcm2niix"),
        cuda_device: str = "0",
    ) -> None:
        super().__init__(patient_id, model_id, outdir, cuda_device)
        self.t1_preop_dir = t1_preop_dir
        self.t1c_preop_dir = t1c_preop_dir
        self.t2_preop_dir = t2_preop_dir
        self.flair_preop_dir = flair_preop_dir
        self.t1_followup_dir = t1_followup_dir
        self.t1c_followup_dir = t1c_followup_dir
        self.t2_followup_dir = t2_followup_dir
        self.flair_followup_dir = flair_followup_dir
        self.dcm2niix_location = dcm2niix_location

    def _preprocess_preop(self) -> None:
        preprocessor = DicomPreprocessor(
            t1_dir=self.t1_preop_dir,
            t1c_dir=self.t1c_preop_dir,
            t2_dir=self.t2_preop_dir,
            flair_dir=self.flair_preop_dir,
            outdir=self.outdir_preop,
            dcm2niix_location=self.dcm2niix_location,
            pre_treatment=True,
            cuda_device=self.cuda_device,
        )
        preprocessor.run()

    def _preprocess_followup(self) -> None:
        preprocessor = DicomPreprocessor(
            t1_dir=self.t1_followup_dir,
            t1c_dir=self.t1c_followup_dir,
            t2_dir=self.t2_followup_dir,
            flair_dir=self.flair_followup_dir,
            outdir=self.outdir_followup,
            dcm2niix_location=self.dcm2niix_location,
            pre_treatment=False,
            cuda_device=self.cuda_device,
        )
        preprocessor.run()

    def _register_recurrence(self) -> None:
        registrator = RegisterRecurrencePipe(
            preop_dir=self.outdir_preop,
            followup_dir=self.outdir_followup,
        )
        registrator.run()


class NiftiProcessor(BaseProcessor):
    """
    Performs preprocessing (skull stripping, normalization, atlas registration), growth model prediction
    and evaluation from a pre-operative MRI exam and a follow-up MRI exam passed as NIfTI images.
    Intermediate results such as tumor segmentation can be provided to skip the corresponding steps.

    Parameters:
        patient_id (str): String acting as patient identifier. Included in the output directory structure.
        model_id (str): String identifying the growth model to be used.
        t1_preop_file (Path): Path to the NIfTI file containing the pre-operative T1 image.
        t1c_preop_file (Path): Path to the NIfTI file containing the pre-operative T1c image.
        t2_preop_file (Path): Path to the NIfTI file containing the pre-operative T2 image.
        flair_preop_file (Path): Path to the NIfTI file containing the pre-operative FLAIR image.
        t1_followup_file (Path): Path to the NIfTI file containing the follow-up T1 image.
        t1c_followup_file (Path): Path to the NIfTI file containing the follow-up T1c image.
        t2_followup_file (Path): Path to the NIfTI file containing the follow-up T2 image.
        flair_followup_file (Path): Path to the NIfTI file containing the follow-up FLAIR image.
        outdir (Path): Path to the output directory.
        cuda_device (Optional, str): The gpu device to use.
        tumorseg_file (Optional, Path): Path to a NIfTI containing the pre-operative tumor segmentation
            that will be used instead of the BRATS algorithms. Expects labels 1 for necrotic, 2 for edema
            and 3 for enhancing tumor.
        recurrenceseg_file (Optional, Path): Path to a NIfTI containing the follow-up recurrence segmentation
            that will be used instead of the BRATS algorithms. Expects labels 1 for necrotic, 2 for edema
            and 3 for enhancing tumor.
        is_skull_stripped (Optional, bool): If true, skips the skull stripping step.
        is_coregistered (Optional, bool): If true, skips the co-registration to atlas space step.
            Note that BRATS algorithms were trained in SRI-24 space.
    """

    def __init__(
        self,
        patient_id: str,
        model_id: str,
        t1_preop_file: Path,
        t1c_preop_file: Path,
        t2_preop_file: Path,
        flair_preop_file: Path,
        t1_followup_file: Path,
        t1c_followup_file: Path,
        t2_followup_file: Path,
        flair_followup_file: Path,
        outdir: Path,
        cuda_device: str = "0",
        tumorseg_file: Optional[Path] = None,
        recurrenceseg_file: Optional[Path] = None,
        is_skull_stripped: bool = False,
        is_coregistered: bool = False,
    ) -> None:
        super().__init__(patient_id, model_id, outdir, cuda_device)
        self.t1_preop_file = t1_preop_file
        self.t1c_preop_file = t1c_preop_file
        self.t2_preop_file = t2_preop_file
        self.flair_preop_file = flair_preop_file
        self.t1_followup_file = t1_followup_file
        self.t1c_followup_file = t1c_followup_file
        self.t2_followup_file = t2_followup_file
        self.flair_followup_file = flair_followup_file
        self.tumorseg_file = tumorseg_file
        self.recurrenceseg_file = recurrenceseg_file
        self.is_skull_stripped = is_skull_stripped
        self.is_coregistered = is_coregistered
        # TODO: This class can handle missing modalities IF the segmentations are provided.
        #      Currently, empty modalities can be handled as empty Path("") inputs.
        #      Implement this more explicitely and check if segmentations are provided properly.
        #      Might need to catch exceptions for visualization if modalities are missing.

    def _preprocess_preop(self) -> None:
        preprocessor = NiftiPreprocessor(
            t1_file=self.t1_preop_file,
            t1c_file=self.t1c_preop_file,
            t2_file=self.t2_preop_file,
            flair_file=self.flair_preop_file,
            outdir=self.outdir_preop,
            pre_treatment=True,
            cuda_device=self.cuda_device,
            is_coregistered=self.is_coregistered,
            is_skull_stripped=self.is_skull_stripped,
            tumorseg_file=self.tumorseg_file,
        )
        preprocessor.run()

    def _preprocess_followup(self) -> None:
        preprocessor = NiftiPreprocessor(
            t1_file=self.t1_followup_file,
            t1c_file=self.t1c_followup_file,
            t2_file=self.t2_followup_file,
            flair_file=self.flair_followup_file,
            outdir=self.outdir_followup,
            pre_treatment=False,
            cuda_device=self.cuda_device,
            is_coregistered=self.is_coregistered,
            is_skull_stripped=self.is_skull_stripped,
            tumorseg_file=self.recurrenceseg_file,
        )
        preprocessor.run()

    def _register_recurrence(self) -> None:
        registrator = RegisterRecurrencePipe(
            preop_dir=self.outdir_preop,
            followup_dir=self.outdir_followup,
            is_coregistered=self.is_coregistered,
        )
        registrator.run()
