import os
import shutil
import argparse
import numpy as np
import nibabel as nib
from pathlib import Path
from typing import Dict, List
from loguru import logger
from scipy.ndimage import binary_fill_holes
from predict_gbm.utils.constants import (
    LONGITUDINAL_DIR,
    MODEL_OUTPUT_DIR,
    PREDICTION_OUTPUT_SCHEMA,
    PathSchema,
    RECURRENCE_SCHEMA,
    SKULL_STRIP_FOLDER,
    STANDARD_PLAN_SCHEMA,
    TISSUE_PBMAP_SCHEMA,
    TUMORSEG_SCHEMA,
)
from predict_gbm.utils.utils import (
    is_binary_array,
    load_and_resample_mri_data,
    load_segmentation,
)
from predict_gbm.preprocessing import register_recurrence, run_brats
from predict_gbm.prediction import predict_tumor_growth
from predict_gbm.evaluation.evaluate import (
    create_standard_plan,
    generate_distance_fade_mask,
    topk_plan,
)

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

# Growth models run on the preop exams. pinngbm is listed last since it is by far the
# slowest model, so that the faster ones are already done if the run is interrupted.
GROWTH_MODEL_DIR = Path("/mnt/Drive4/lucas/growth_models/predict-gbm")
DEFAULT_GROWTH_MODEL_PATHS = [
    str(GROWTH_MODEL_DIR / "gliodil.tar"),
    str(GROWTH_MODEL_DIR / "sbtc.tar"),
    str(GROWTH_MODEL_DIR / "unet1.tar"),
    str(GROWTH_MODEL_DIR / "pinngbm.tar"),
]
# The model id names the prediction and plan files and defaults to the file stem of the
# docker image. The respond10 predictions of unet1.tar are named "unet", so its outputs
# keep that id and overwrite the existing ones instead of being written beside them.
MODEL_ID_OVERRIDES = {"unet1": "unet"}
# respond10 stores predictions flat in growth_models/ instead of the per model subdirectory
# of PREDICTION_OUTPUT_SCHEMA / MODEL_PLAN_SCHEMA; predictions are moved there after the run.
FLAT_PREDICTION_SCHEMA = PathSchema(
    "{base_dir}/" + MODEL_OUTPUT_DIR + "/{algo_id}_pred.nii.gz"
)
FLAT_MODEL_PLAN_SCHEMA = PathSchema(
    "{base_dir}/" + MODEL_OUTPUT_DIR + "/{algo_id}_plan.nii.gz"
)


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


def load_brain_mask(exam_dir: Path) -> np.ndarray:
    """Loads the skull stripping brain mask of an exam and fills its holes."""
    brain_mask = load_segmentation(BRAIN_MASK_BET_SCHEMA.format(base_dir=exam_dir))
    return binary_fill_holes(brain_mask.astype(bool)).astype(np.int32)


def create_preop_standard_plan(preop_dir: Path) -> None:
    """
    Regenerates the standard radiotherapy plan of a preop exam from its tumor segmentation,
    mirroring predict_gbm.evaluation.evaluate_tumor_model: the tumor core (necrosis and
    enhancing tumor, edema ignored) is dilated by CTV_MARGIN and restricted to the brain mask.
    """
    tumorseg_file = TUMORSEG_SCHEMA.format(base_dir=preop_dir)
    affine = nib.load(str(tumorseg_file)).affine

    core_segmentation = load_segmentation(tumorseg_file)
    core_segmentation[core_segmentation == 2] = 0  # ignore edema
    core_segmentation[core_segmentation == 3] = 1

    brain_mask = load_brain_mask(preop_dir)

    standard_plan = create_standard_plan(core_segmentation, CTV_MARGIN)
    standard_plan[brain_mask == 0] = 0

    outfile = STANDARD_PLAN_SCHEMA.format(base_dir=preop_dir)
    nib.save(nib.Nifti1Image(standard_plan, affine=affine), str(outfile))
    logger.info(f"{preop_dir}: saved standard plan to {outfile}.")


def resolve_growth_models(growth_models: List[str]) -> Dict[str, Path]:
    """
    Maps the model id of every given growth model docker image (*.tar) to its path, keeping
    the given order. The id is the file stem unless MODEL_ID_OVERRIDES renames it. Missing
    images are logged and skipped.
    """
    growth_model_paths = {}
    for growth_model in growth_models:
        growth_model_path = Path(growth_model)
        if not growth_model_path.is_file():
            logger.warning(f"Growth model {growth_model_path} not found, skipping model.")
            continue
        model_id = MODEL_ID_OVERRIDES.get(
            growth_model_path.stem, growth_model_path.stem
        )
        growth_model_paths[model_id] = growth_model_path
    return growth_model_paths


def create_model_plan(preop_dir: Path, model_id: str) -> None:
    """
    Creates the radiotherapy plan of a growth model prediction, mirroring the model plan of
    predict_gbm.evaluation.evaluate_tumor_model: the prediction is resampled to the grid of
    the standard plan, binary predictions are turned into a distance fade, and the highest
    scoring voxels within the brain mask are selected until the volume of the standard plan
    is reached.
    """
    pred_file = FLAT_PREDICTION_SCHEMA.format(base_dir=preop_dir, algo_id=model_id)
    standard_plan_file = STANDARD_PLAN_SCHEMA.format(base_dir=preop_dir)
    affine = nib.load(str(standard_plan_file)).affine

    standard_plan = load_segmentation(standard_plan_file)
    brain_mask = load_brain_mask(preop_dir)

    model_prediction = load_and_resample_mri_data(
        str(pred_file), resample_params=standard_plan.shape, interp_type=0
    )
    if is_binary_array(model_prediction):
        logger.info(
            f"Prediction {pred_file} is binary. Generating distance fade for radiation planning."
        )
        model_prediction = generate_distance_fade_mask(model_prediction)
    model_prediction = np.clip(model_prediction, 0.0, 1.0)

    model_plan = topk_plan(
        scores=model_prediction,
        target_voxels=int(np.sum(standard_plan)),
        mask=brain_mask,
    )

    outfile = FLAT_MODEL_PLAN_SCHEMA.format(base_dir=preop_dir, algo_id=model_id)
    nib.save(nib.Nifti1Image(model_plan, affine=affine), str(outfile))
    logger.info(f"{preop_dir}: saved {model_id} plan to {outfile}.")


def predict_growth(
    preop_dir: Path, growth_model_paths: Dict[str, Path], cuda_device: str
) -> None:
    """
    Predicts tumor cell concentration on a preop exam with every given growth model and derives
    the corresponding radiotherapy plan. predict_tumor_growth writes into the per model
    subdirectory of PREDICTION_OUTPUT_SCHEMA, from where the prediction is moved to the flat
    growth_models/{model_id}_pred.nii.gz of the respond10 layout, overwriting the previous
    prediction and plan of that model. A failing model is logged and skipped so that the
    remaining models still run.
    """
    model_inputs = {
        "tumorseg_file": TUMORSEG_SCHEMA.format(base_dir=preop_dir),
        "gm_file": TISSUE_PBMAP_SCHEMA.format(base_dir=preop_dir, tissue="gm"),
        "wm_file": TISSUE_PBMAP_SCHEMA.format(base_dir=preop_dir, tissue="wm"),
        "csf_file": TISSUE_PBMAP_SCHEMA.format(base_dir=preop_dir, tissue="csf"),
        "t1c_file": BET_NORMALIZED_SCHEMA.format(base_dir=preop_dir, modality="t1c"),
        "flair_file": BET_NORMALIZED_SCHEMA.format(base_dir=preop_dir, modality="flair"),
        "brain_mask_file": BRAIN_MASK_BET_SCHEMA.format(base_dir=preop_dir),
    }
    # TumorGrowthModel exits the process on a missing input instead of raising, which would
    # abort the whole run, so the inputs are checked before the first model is started.
    missing = [str(f) for f in model_inputs.values() if not f.exists()]
    if missing:
        logger.error(f"{preop_dir}: missing prediction inputs {missing}, skipping prediction.")
        return

    for model_id, growth_model_path in growth_model_paths.items():
        logger.info(f"{preop_dir}: starting growth prediction with {model_id}.")
        try:
            predict_tumor_growth(
                model_id=model_id,
                outdir=preop_dir,
                cuda_device=cuda_device,
                growth_model_path=growth_model_path,
                **model_inputs,
            )

            nested_pred_file = PREDICTION_OUTPUT_SCHEMA.format(
                base_dir=preop_dir.resolve(), algo_id=model_id
            )
            pred_file = FLAT_PREDICTION_SCHEMA.format(
                base_dir=preop_dir, algo_id=model_id
            )
            shutil.move(str(nested_pred_file), str(pred_file))
            shutil.rmtree(nested_pred_file.parent, ignore_errors=True)
            logger.info(f"{preop_dir}: saved {model_id} prediction to {pred_file}.")

            create_model_plan(preop_dir, model_id)
        except Exception:
            logger.exception(
                f"{preop_dir}: prediction with {model_id} failed, skipping model."
            )


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
    # Without -preop the postop and followup exams are segmented and registered to preop,
    # with -preop only the preop exams are processed (segmentation, standard plan, growth
    # model predictions and their plans). With -gliodil_plans only the gliodil plans are
    # regenerated from the existing gliodil predictions (nothing else is run).
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
        help=(
            "Only process the preop exams: tumor segmentation, standard plan and growth model "
            "prediction with the corresponding plan (no registration). The postop and followup "
            "exams are skipped."
        ),
    )
    parser.add_argument(
        "-gliodil_plans",
        action="store_true",
        help=(
            "Only regenerate the gliodil radiotherapy plans of the preop exams from the "
            "existing growth_models/gliodil_pred.nii.gz predictions, overwriting the old "
            "plans. No segmentation, prediction or registration is run."
        ),
    )
    parser.add_argument(
        "-growth_models",
        type=str,
        nargs="*",
        default=DEFAULT_GROWTH_MODEL_PATHS,
        help=(
            "Growth model docker images (*.tar) to predict with in the preop step. The file "
            "stem is used as model id unless it is renamed by MODEL_ID_OVERRIDES. Pass without "
            "arguments to skip the prediction."
        ),
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

    # Growth models only run in the preop step, so nothing is resolved for the postop/followup one.
    growth_model_paths = {}
    if args.preop:
        growth_model_paths = resolve_growth_models(args.growth_models)
        logger.info(f"Predicting with growth models {list(growth_model_paths)}.")

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
        if args.gliodil_plans:
            pred_file = FLAT_PREDICTION_SCHEMA.format(base_dir=preop_dir, algo_id="gliodil")
            if not pred_file.exists():
                logger.error(f"{patient_id}/{preop_dir.name}: {pred_file} not found, skipping.")
                continue
            try:
                create_model_plan(preop_dir, "gliodil")
            except Exception:
                logger.exception(f"{patient_id}/{preop_dir.name}: gliodil plan failed, skipping.")
            continue

        if args.preop:
            logger.info(f"{patient_id}/{preop_dir.name}: starting tumor segmentation.")
            try:
                segment_tumor(preop_dir, args.cuda_device)
                create_preop_standard_plan(preop_dir)
            except Exception:
                logger.exception(
                    f"{patient_id}/{preop_dir.name}: tumor segmentation / standard plan failed, "
                    "skipping growth prediction."
                )
                continue

            predict_growth(preop_dir, growth_model_paths, args.cuda_device)
            continue

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
