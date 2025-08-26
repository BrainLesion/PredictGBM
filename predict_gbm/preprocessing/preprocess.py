import os
import ants
import time
import shutil
from pathlib import Path
from loguru import logger
from typing import Optional
from predict_gbm.base import BasePipe
from predict_gbm.preprocessing.dicom_to_nifti import dicom_to_nifti
from predict_gbm.preprocessing.tumor_segmentation import run_brats
from predict_gbm.preprocessing.norm_ss_coregistration import (
    normalize,
    norm_ss_coregister,
    register_recurrence,
)
from predict_gbm.preprocessing.tissue_segmentation import (
    generate_registration_mask,
    run_tissue_seg_registration,
)
from predict_gbm.utils.utils import make_symlink
from predict_gbm.utils.constants import (
    LONGITUDINAL_WARP_SCHEMA,
    MODALITY_CONVERTED_SCHEMA,
    MODALITY_STRIPPED_SCHEMA,
    RECURRENCE_SCHEMA,
    REGISTRATION_MASK_SCHEMA,
    REGISTRATION_TRAFO_SCHEMA,
    TUMOR_LABELS,
    TUMORSEG_SCHEMA,
)


class BasePreprocessor:
    """
    Base class for DicomPreprocessor and NiftiPreprocessor that handles shared pre-processing steps on a fixed directory
    structure.

    Parameters:
        outdir (Path): Output directory containing either nifti converted data or skull stripped niftis in the
            expected directory structure.
        pre_treatment (bool): Whether the provided DICOM are preop (True) or postop (False).
            Causes the BRATS segmentation algorithm to choose different models.
        mask_tissueseg (bool): If true, masks out the tumor region for the tissue segmentation step.
        perform_coregistration (bool): If true, performs normalization, skull stripping, co-registration
        perform_skull_stripping (bool): If true, performs skull stripping during co-registration step.
        perform_tumorseg (bool): If true, performs tumor segmentation via BRATS.
        perform_tissueseg (bool): If true, performs tissue segmentation.
        cuda_device (str): GPU device to use.
    """

    def __init__(
        self,
        outdir: Path,
        pre_treatment: bool,
        mask_tissueseg: bool = False,
        perform_coregistration: bool = True,
        perform_skull_stripping: bool = True,
        perform_tumorseg: bool = True,
        perform_tissueseg: bool = True,
        cuda_device: str = "0",
    ) -> None:
        self.outdir = outdir
        self.pre_treatment = pre_treatment
        self.mask_tissueseg = mask_tissueseg
        self.perform_coregistration = perform_coregistration
        self.perform_skull_stripping = perform_skull_stripping
        self.perform_tumorseg = perform_tumorseg
        self.perform_tissueseg = perform_tissueseg
        self.cuda_device = cuda_device

    def run(self) -> None:

        start_time = time.time()
        os.environ["CUDA_VISIBLE_DEVICES"] = self.cuda_device
        logger.info("Starting preprocessing.")

        if self.perform_coregistration:
            norm_ss_coregister(
                t1_file=MODALITY_CONVERTED_SCHEMA.format(
                    base_dir=self.outdir, modality="t1"
                ),
                t1c_file=MODALITY_CONVERTED_SCHEMA.format(
                    base_dir=self.outdir, modality="t1c"
                ),
                t2_file=MODALITY_CONVERTED_SCHEMA.format(
                    base_dir=self.outdir, modality="t2"
                ),
                flair_file=MODALITY_CONVERTED_SCHEMA.format(
                    base_dir=self.outdir, modality="flair"
                ),
                skull_strip=self.perform_skull_stripping,
                outdir=self.outdir,
            )

        if self.perform_tumorseg:
            run_brats(
                t1_file=MODALITY_STRIPPED_SCHEMA.format(
                    base_dir=self.outdir, modality="t1"
                ),
                t1c_file=MODALITY_STRIPPED_SCHEMA.format(
                    base_dir=self.outdir, modality="t1c"
                ),
                t2_file=MODALITY_STRIPPED_SCHEMA.format(
                    base_dir=self.outdir, modality="t2"
                ),
                flair_file=MODALITY_STRIPPED_SCHEMA.format(
                    base_dir=self.outdir, modality="flair"
                ),
                outdir=self.outdir,
                pre_treatment=self.pre_treatment,
                cuda_device=self.cuda_device,
            )
        elif self.perform_coregistration:
            tumorseg_file = TUMORSEG_SCHEMA.format(base_dir=self.outdir)
            transform_dir = (
                REGISTRATION_TRAFO_SCHEMA.format(base_dir=self.outdir) / "t1c"
            )
            transform_files = [str(f) for f in sorted(transform_dir.glob("*.mat"))]

            tumorseg_transformed = ants.apply_transforms(
                fixed=ants.image_read(str(tumorseg_file)),
                moving=ants.image_read(str(tumorseg_file)),
                transformlist=transform_files,
                interpolator="nearestNeighbor",
            )
            ants.image_write(tumorseg_transformed, str(tumorseg_file))

        registration_mask_file = REGISTRATION_MASK_SCHEMA.format(base_dir=self.outdir)
        generate_registration_mask(
            tumor_seg_file=TUMORSEG_SCHEMA.format(base_dir=self.outdir),
            outfile=registration_mask_file,
        )

        tissueseg_kwargs = {}
        if self.mask_tissueseg:
            tissueseg_kwargs["registration_mask_file"] = registration_mask_file

        if self.perform_tissueseg:
            run_tissue_seg_registration(
                t1_file=MODALITY_STRIPPED_SCHEMA.format(
                    base_dir=self.outdir, modality="t1c"
                ),
                outdir=self.outdir,
                **tissueseg_kwargs,
            )

        time_spent = time.time() - start_time
        logger.info(
            f"Finished preprocessing in {time_spent:.2f} seconds. Results saved to {self.outdir}."
        )


class DicomPreprocessor(BasePreprocessor):
    """
    Performs a multitude of processing steps to prepare DICOM inputs for tumor growth models.
    DICOM data is first converted to NIfTI, then preprocess_pipe is called for the rest.

    Parameters:
         t1_dir (Path): Path to the directory with the DICOM files for T1.
         t1c_dir (Path): Path to the directory with the DICOM files for T1c.
         t2_dir (Path): Path to the directory with the DICOM files for T2.
         flair_dir (Path): Path to the directory with the DICOM files for Flair.
         pre_treatment (bool): Whether the provided DICOM are preop (True) or postop (False).
             Causes the BRATS segmentation algorithm to choose different models.
         outdir (Path): Base directory for the output. Usually exam directory.
         dcm2niix_location (Path, optional): The location of the dcm2niix executable.
         cuda_device (str): GPU device to use.
    """

    def __init__(
        self,
        t1_dir: Path,
        t1c_dir: Path,
        t2_dir: Path,
        flair_dir: Path,
        pre_treatment: bool,
        outdir: Path,
        dcm2niix_location: Path = Path("dcm2niix"),
        cuda_device: str = "0",
    ) -> None:
        super().__init__(
            outdir=outdir,
            pre_treatment=pre_treatment,
            cuda_device=cuda_device,
            perform_coregistration=True,
            perform_skull_stripping=True,
            perform_tumorseg=True,
            perform_tissueseg=pre_treatment,
        )
        self.t1_dir = t1_dir
        self.t1c_dir = t1c_dir
        self.t2_dir = t2_dir
        self.flair_dir = flair_dir
        self.dcm2niix_location = dcm2niix_location

    def run(self) -> None:

        os.environ["CUDA_VISIBLE_DEVICES"] = self.cuda_device

        start_time_conversion = time.time()
        logger.info("Starting DICOM to NIfTI conversion.")

        dicom_modalities = {
            "t1": self.t1_dir,
            "t1c": self.t1c_dir,
            "t2": self.t2_dir,
            "flair": self.flair_dir,
        }
        for modality_name, dicom_dir in dicom_modalities.items():
            # remove suffixes because dcm2niix adds them automatically
            outfile_tmp = (
                MODALITY_CONVERTED_SCHEMA.format(
                    base_dir=self.outdir, modality=modality_name
                )
                .with_suffix("")
                .with_suffix("")
            )
            dicom_to_nifti(
                input_dir=dicom_dir,
                outfile=outfile_tmp,
                dcm2niix_location=self.dcm2niix_location,
            )
        time_spent_conversion = time.time() - start_time_conversion
        logger.info(f"Finished conversion step in {time_spent_conversion:.2f} seconds.")

        super().run()


class NiftiPreprocessor(BasePreprocessor):
    """
    Performs a multitude of precessing steps to prepare nifti inputs for tumor growth models. Allows passing
    available intermediate results like tumor segmentation or already skull stripped images.

    Parameters:
        t1_file (Path): Path to the NIfTI file with the t1 data.
        t1c_file (Path): Path to the NIfTI file with the t1c data.
        t2_file (Path): Path to the NIfTI file with the t2 data.
        flair_file (Path): Path to the NIfTI file with the flair data.
        pre_treatment (bool): Whether the provided DICOM are preop (True) or postop (False).
            Causes the BRATS segmentation algorithm to choose different models.
        outdir (Path): Base directory for the output. Usually exam directory.
        cuda_device (str): GPU device to use.
        is_coregistered (bool): True if the provided data has already been co-registered to SRI-24 space and skull stripped.
        is_skull_stripped (bool): True if the provided data has already been normalized, skull stripped and co-registered.
        tumorseg_file (Optional, Path): Path to the tumor segmentation.
    """

    def __init__(
        self,
        t1_file: Path,
        t1c_file: Path,
        t2_file: Path,
        flair_file: Path,
        pre_treatment: bool,
        outdir: Path,
        is_coregistered: bool,
        is_skull_stripped: bool,
        tumorseg_file: Optional[Path] = None,
        cuda_device: str = "0",
    ) -> None:
        super().__init__(
            outdir=outdir,
            pre_treatment=pre_treatment,
            cuda_device=cuda_device,
            perform_coregistration=not is_coregistered,
            perform_skull_stripping=not is_skull_stripped,
            perform_tumorseg=tumorseg_file is None,
            perform_tissueseg=pre_treatment,
        )
        self.t1_file = t1_file
        self.t1c_file = t1c_file
        self.t2_file = t2_file
        self.flair_file = flair_file
        self.is_coregistered = is_coregistered
        self.is_skull_stripped = is_skull_stripped
        self.tumorseg_file = tumorseg_file

    def run(self) -> None:

        os.environ["CUDA_VISIBLE_DEVICES"] = self.cuda_device

        modality_dict = {
            "t1": self.t1_file,
            "t1c": self.t1c_file,
            "t2": self.t2_file,
            "flair": self.flair_file,
        }

        if self.is_coregistered:
            logger.info("Running with provided skull stripped modality images.")
            for modality, path in modality_dict.items():
                if str(path) == ".":
                    continue  # ignore missing modalities
                normalize(
                    img_file=path,
                    outfile=MODALITY_STRIPPED_SCHEMA.format(
                        base_dir=self.outdir, modality=modality
                    ),
                )
        else:
            for modality, path in modality_dict.items():
                make_symlink(
                    src=path,
                    dst=MODALITY_CONVERTED_SCHEMA.format(
                        base_dir=self.outdir, modality=modality
                    ),
                )

        if self.tumorseg_file is not None:
            logger.warning(
                f"Running with provided tumor segmentation {self.tumorseg_file}. Expected labels: {TUMOR_LABELS}"
            )
            tumorseg_file_dest = TUMORSEG_SCHEMA.format(base_dir=self.outdir)
            tumorseg_file_dest.parent.mkdir(exist_ok=True, parents=True)
            shutil.copyfile(self.tumorseg_file, tumorseg_file_dest)

        super().run()


class RegisterRecurrencePipe(BasePipe):
    """Performs longitudinal registration, transforming followup t1c and recurrence segmentation to preop space."""

    def __init__(
        self,
        preop_dir: Path,
        followup_dir: Path,
        is_coregistered: bool = False,
        use_fixed_mask: bool = False,
        use_moving_mask: bool = False,
    ) -> None:
        super().__init__(preop_dir=preop_dir, followup_dir=followup_dir)
        self.is_coregistered = is_coregistered
        self.use_fixed_mask = use_fixed_mask
        self.use_moving_mask = use_moving_mask

    def run(self) -> None:  # pragma: no cover - wrapper tested via pipeline
        start_time = time.time()
        logger.info("Starting longitudinal processing.")

        t1c_pre_file = MODALITY_STRIPPED_SCHEMA.format(
            base_dir=self.preop_dir, modality="t1c"
        )
        t1c_post_file = MODALITY_STRIPPED_SCHEMA.format(
            base_dir=self.followup_dir, modality="t1c"
        )
        recurrence_seg_file = TUMORSEG_SCHEMA.format(base_dir=self.followup_dir)

        reg_kwargs = {}
        if self.use_fixed_mask:
            reg_kwargs["fixed_mask_file"] = REGISTRATION_MASK_SCHEMA.format(
                base_dir=self.preop_dir
            )
        if self.use_moving_mask:
            reg_kwargs["moving_mask_file"] = REGISTRATION_MASK_SCHEMA.format(
                base_dir=self.followup_dir
            )

        if self.is_coregistered:
            make_symlink(
                src=t1c_post_file,
                dst=LONGITUDINAL_WARP_SCHEMA.format(base_dir=self.followup_dir),
            )
            make_symlink(
                src=recurrence_seg_file,
                dst=RECURRENCE_SCHEMA.format(base_dir=self.followup_dir),
            )
        else:
            register_recurrence(
                t1c_pre_file=t1c_pre_file,
                t1c_post_file=t1c_post_file,
                recurrence_seg_file=recurrence_seg_file,
                outdir=self.followup_dir,
                **reg_kwargs,
            )

        time_spent = time.time() - start_time
        logger.info(
            f"Finished longitudinal preprocessing in {time_spent:.2f} seconds. Results saved to {self.followup_dir}."
        )
