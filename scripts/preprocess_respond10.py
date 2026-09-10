import os
import shutil
import argparse
import numpy as np
import nibabel as nib
from pathlib import Path
from loguru import logger
from scipy.ndimage import binary_fill_holes
from predict_gbm.utils.constants import (
    LONGITUDINAL_DIR,
    PathSchema,
    RECURRENCE_SCHEMA,
    SKULL_STRIP_FOLDER,
    STANDARD_PLAN_SCHEMA,
    TUMORSEG_SCHEMA,
)
from predict_gbm.utils.utils import load_segmentation
from predict_gbm.preprocessing import register_recurrence, run_brats
from predict_gbm.evaluation.evaluate import create_standard_plan

# The respond10 dataset was skull stripped outside of this repository and uses
# {modality}_bet_normalized.nii.gz instead of MODALITY_STRIPPED_SCHEMA's naming.
BET_NORMALIZED_SCHEMA = PathSchema(
    "{base_dir}/" + SKULL_STRIP_FOLDER + "/{modality}_bet_normalized.nii.gz"
)
# Longitudinal (postop/followup -> preop) results are stored in preop_space instead of
# the LONGITUDINAL_DIR used by the schemata; the registration step writes to
# LONGITUDINAL_DIR and its full content is moved to this folder afterwards.
PREOP_SPACE_FOLDER = "preop_space"

BRAIN_MASK_BET_SCHEMA = PathSchema(
    "{base_dir}/" + SKULL_STRIP_FOLDER + "/brainmask.nii.gz"
)
EXAM_PREFIXES = ("preop", "postop", "followup")
# CTV margin (mm) for the preop standard plan; matches predict_gbm.evaluation defaults.
CTV_MARGIN = 15
# Modalities (besides t1c) of the postop/followup exams that are warped into preop space.
ADDITIONAL_WARP_MODALITIES = ("flair",)


def find_exam_dirs(patient_dir: Path) -> dict:
    """
    Maps each timepoint (preop/postop/followup) to its exam directory, e.g. preop_d0,
    postop_d3, followup_d289. Raises if a timepoint has none or several exam directories.
    """
    exam_dirs = {}
    for timepoint in EXAM_PREFIXES:
        candidates = sorted(
            d for d in patient_dir.glob(f"{timepoint}_d*") if d.is_dir()
        )
        if len(candidates) != 1:
            raise FileNotFoundError(
                f"{patient_dir.name}: expected exactly one '{timepoint}_d*' directory, "
                f"found {[c.name for c in candidates]}."
            )
        exam_dirs[timepoint] = candidates[0]
    return exam_dirs


def segment_tumor(exam_dir: Path, cuda_device: str) -> None:
    """Runs BraTS tumor segmentation for an exam based on its skull stripped modalities."""
    run_brats(
        t1_file=BET_NORMALIZED_SCHEMA.format(base_dir=exam_dir, modality="t1"),
        t1c_file=BET_NORMALIZED_SCHEMA.format(base_dir=exam_dir, modality="t1c"),
        t2_file=BET_NORMALIZED_SCHEMA.format(base_dir=exam_dir, modality="t2"),
        flair_file=BET_NORMALIZED_SCHEMA.format(base_dir=exam_dir, modality="flair"),
        outdir=exam_dir,
        cuda_device=cuda_device,
    )


def create_preop_standard_plan(preop_dir: Path) -> None:
    """
    Regenerates the standard radiotherapy plan of a preop exam from its tumor segmentation,
    mirroring predict_gbm.evaluation.evaluate_tumor_model: the tumor core (necrosis and
    enhancing tumor, edema ignored) is dilated by CTV_MARGIN and restricted to the brain mask.
    """
    tumorseg_file = TUMORSEG_SCHEMA.format(base_dir=preop_dir)
    brain_mask_file = BRAIN_MASK_BET_SCHEMA.format(base_dir=preop_dir)
    affine = nib.load(str(tumorseg_file)).affine

    core_segmentation = load_segmentation(tumorseg_file)
    core_segmentation[core_segmentation == 2] = 0  # ignore edema
    core_segmentation[core_segmentation == 3] = 1

    brain_mask = load_segmentation(brain_mask_file)
    brain_mask = binary_fill_holes(brain_mask.astype(bool)).astype(np.int32)

    standard_plan = create_standard_plan(core_segmentation, CTV_MARGIN)
    standard_plan[brain_mask == 0] = 0

    outfile = STANDARD_PLAN_SCHEMA.format(base_dir=preop_dir)
    nib.save(nib.Nifti1Image(standard_plan, affine=affine), str(outfile))
    logger.info(f"{preop_dir}: saved standard plan to {outfile}.")


def register_to_preop(exam_dir: Path, preop_dir: Path) -> None:
    """
    Registers an exam (postop/followup) to preop with DIRAC and warps its tumor segmentation,
    t1c and ADDITIONAL_WARP_MODALITIES into preop space. All outputs of the registration
    step, including the DIRAC displacement fields, are moved into <exam_dir>/preop_space.
    """
    additional_modalities = {
        modality: BET_NORMALIZED_SCHEMA.format(base_dir=exam_dir, modality=modality)
        for modality in ADDITIONAL_WARP_MODALITIES
    }
    register_recurrence(
        t1c_pre_file=BET_NORMALIZED_SCHEMA.format(base_dir=preop_dir, modality="t1c"),
        t1c_post_file=BET_NORMALIZED_SCHEMA.format(base_dir=exam_dir, modality="t1c"),
        recurrence_seg_file=TUMORSEG_SCHEMA.format(base_dir=exam_dir),
        outdir=exam_dir,
        registration_algorithm="dirac",
        additional_modalities=additional_modalities,
    )

    longitudinal_dir = RECURRENCE_SCHEMA.format(base_dir=exam_dir).parent
    assert longitudinal_dir.name == LONGITUDINAL_DIR
    preop_space_dir = exam_dir / PREOP_SPACE_FOLDER
    preop_space_dir.mkdir(parents=True, exist_ok=True)
    for src in sorted(longitudinal_dir.iterdir()):
        dst = preop_space_dir / src.name
        if dst.is_dir():
            shutil.rmtree(dst)
        elif dst.exists():
            dst.unlink()
        shutil.move(str(src), str(dst))
    longitudinal_dir.rmdir()
    logger.info(f"{exam_dir}: moved longitudinal outputs to {preop_space_dir}.")


if __name__ == "__main__":
    # Example:
    # nohup python -u scripts/preprocess_respond10.py -cuda_device 0 > preprocess_respond10.out 2>&1 &
    # Add -preop to also segment the preop exams and regenerate their standard plans.
    parser = argparse.ArgumentParser()
    parser.add_argument("-cuda_device", type=str, default="0", help="GPU id to run on.")
    parser.add_argument(
        "-data_dir",
        type=str,
        default="/mnt/Drive2/lucas/predict_gbm_10_respond",
        help="Directory containing one folder per patient (respond_tum_XXX).",
    )
    parser.add_argument(
        "-preop",
        action="store_true",
        help="Also perform tumor segmentation and regenerate the standard plan for the preop exams (no registration).",
    )
    parser.add_argument(
        "-patients",
        type=str,
        nargs="+",
        default=None,
        help="Optional list of patient ids to process (e.g. respond_tum_001). Default: all.",
    )
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device

    data_dir = Path(args.data_dir)
    patient_dirs = sorted(d for d in data_dir.iterdir() if d.is_dir())

    for patient_dir in patient_dirs:
        patient_id = patient_dir.name
        if args.patients is not None and patient_id not in args.patients:
            continue

        try:
            exam_dirs = find_exam_dirs(patient_dir)
        except FileNotFoundError:
            logger.exception(f"{patient_id}: could not resolve exam directories, skipping.")
            continue

        preop_dir = exam_dirs["preop"]
        if args.preop:
            logger.info(f"{patient_id}/{preop_dir.name}: starting tumor segmentation.")
            try:
                segment_tumor(preop_dir, args.cuda_device)
                create_preop_standard_plan(preop_dir)
            except Exception:
                logger.exception(
                    f"{patient_id}/{preop_dir.name}: tumor segmentation / standard plan failed."
                )

        for timepoint in ("postop", "followup"):
            exam_dir = exam_dirs[timepoint]
            logger.info(f"{patient_id}/{exam_dir.name}: starting tumor segmentation.")
            try:
                segment_tumor(exam_dir, args.cuda_device)
            except Exception:
                logger.exception(
                    f"{patient_id}/{exam_dir.name}: tumor segmentation failed, skipping exam."
                )
                continue

            logger.info(f"{patient_id}/{exam_dir.name}: starting DIRAC registration to preop.")
            try:
                register_to_preop(exam_dir, preop_dir)
            except Exception:
                logger.exception(
                    f"{patient_id}/{exam_dir.name}: DIRAC registration failed, skipping exam."
                )
