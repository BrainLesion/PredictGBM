import os
import argparse
import numpy as np
import nibabel as nib
from pathlib import Path
from typing import Dict, List
from loguru import logger
from scipy.ndimage import binary_fill_holes
from predict_gbm.utils.constants import (
    BRAIN_MASK_SCHEMA,
    MODALITY_STRIPPED_SCHEMA,
    MODEL_OUTPUT_DIR,
    MODEL_PLAN_SCHEMA,
    PREDICTION_OUTPUT_SCHEMA,
    SKULL_STRIP_FOLDER,
    STANDARD_PLAN_SCHEMA,
    TISSUE_PBMAP_SCHEMA,
    TISSUE_SEGMENTATION_FOLDER,
    TUMOR_SEGMENTATION_FOLDER,
    TUMORSEG_SCHEMA,
)
from predict_gbm.utils.utils import (
    is_binary_array,
    load_and_resample_mri_data,
    load_segmentation,
)
from predict_gbm.preprocessing import norm_ss_coregister, run_brats, run_tissue_seg
from predict_gbm.prediction import predict_tumor_growth
from predict_gbm.evaluation.evaluate import (
    create_standard_plan,
    generate_distance_fade_mask,
    topk_plan,
)

# The exam consists of one nifti per modality named <series description>_<modality>.nii.gz.
# adc is optional and treated as quantitative modality (co-registered, not normalized).
REQUIRED_MODALITIES = ("t1", "t1c", "t2", "flair")
OPTIONAL_MODALITIES = ("adc",)
# CTV margin (mm) for the standard plan; matches predict_gbm.evaluation defaults.
CTV_MARGIN = 15

# Growth models to run on the exam. The model id names the prediction and plan files and
# defaults to the file stem of the docker image, renamed by MODEL_ID_OVERRIDES.
GROWTH_MODEL_DIR = Path("/mnt/Drive4/lucas/growth_models/predict-gbm")
DEFAULT_GROWTH_MODEL_PATHS = [
    str(GROWTH_MODEL_DIR / "unet1.tar"),
    str(GROWTH_MODEL_DIR / "gliodil.tar"),
]
MODEL_ID_OVERRIDES = {"unet1": "unet"}
# Absolute tolerance for comparing output affines to the reference.
AFFINE_ATOL = 1e-3


def find_modality_files(data_dir: Path) -> Dict[str, Path]:
    """
    Maps each modality to its nifti in data_dir, identified by the '_<modality>.nii.gz'
    suffix. Raises if a required modality has none or several matches; optional modalities
    are dropped if absent.
    """
    modality_files = {}
    for modality in REQUIRED_MODALITIES + OPTIONAL_MODALITIES:
        candidates = sorted(data_dir.glob(f"*_{modality}.nii.gz"))
        if len(candidates) == 1:
            modality_files[modality] = candidates[0]
        elif len(candidates) > 1 or modality in REQUIRED_MODALITIES:
            raise FileNotFoundError(
                f"{data_dir}: expected exactly one '*_{modality}.nii.gz', "
                f"found {[c.name for c in candidates]}."
            )
    logger.info(
        f"Found modalities: { {m: f.name for m, f in modality_files.items()} }"
    )
    return modality_files


def preprocess(modality_files: Dict[str, Path], outdir: Path, override: bool) -> None:
    """Normalization, skull stripping and atlas co-registration of all modalities."""
    if MODALITY_STRIPPED_SCHEMA.format(base_dir=outdir, modality="t1c").exists() and not override:
        logger.info(f"{outdir}: skull stripping already done, skipping.")
        return
    additional_quantitative_modalities = {
        modality: modality_files[modality]
        for modality in OPTIONAL_MODALITIES
        if modality in modality_files
    } or None
    norm_ss_coregister(
        t1_file=modality_files["t1"],
        t1c_file=modality_files["t1c"],
        t2_file=modality_files["t2"],
        flair_file=modality_files["flair"],
        skull_strip=True,
        outdir=outdir,
        additional_quantitative_modalities=additional_quantitative_modalities,
    )


def segment_tumor(outdir: Path, cuda_device: str, override: bool) -> None:
    """BraTS tumor segmentation on the skull stripped modalities."""
    if TUMORSEG_SCHEMA.format(base_dir=outdir).exists() and not override:
        logger.info(f"{outdir}: tumor segmentation already done, skipping.")
        return
    run_brats(
        t1_file=MODALITY_STRIPPED_SCHEMA.format(base_dir=outdir, modality="t1"),
        t1c_file=MODALITY_STRIPPED_SCHEMA.format(base_dir=outdir, modality="t1c"),
        t2_file=MODALITY_STRIPPED_SCHEMA.format(base_dir=outdir, modality="t2"),
        flair_file=MODALITY_STRIPPED_SCHEMA.format(base_dir=outdir, modality="flair"),
        outdir=outdir,
        cuda_device=cuda_device,
    )


def segment_tissue(outdir: Path, override: bool) -> None:
    """Tissue (gm/wm/csf) probability maps based on the skull stripped t1c."""
    tissueseg_files = [
        TISSUE_PBMAP_SCHEMA.format(base_dir=outdir, tissue=tissue)
        for tissue in ("gm", "wm", "csf")
    ]
    if all(f.exists() for f in tissueseg_files) and not override:
        logger.info(f"{outdir}: tissue segmentation already done, skipping.")
        return
    run_tissue_seg(
        t1_file=MODALITY_STRIPPED_SCHEMA.format(base_dir=outdir, modality="t1c"),
        outdir=outdir,
    )


def load_brain_mask(outdir: Path) -> np.ndarray:
    """Loads the skull stripping brain mask and fills its holes."""
    brain_mask = load_segmentation(BRAIN_MASK_SCHEMA.format(base_dir=outdir))
    return binary_fill_holes(brain_mask.astype(bool)).astype(np.int32)


def create_preop_standard_plan(outdir: Path) -> None:
    """
    Standard radiotherapy plan from the tumor segmentation, mirroring
    predict_gbm.evaluation.evaluate_tumor_model: the tumor core (necrosis and enhancing
    tumor, edema ignored) is dilated by CTV_MARGIN and restricted to the brain mask.
    """
    tumorseg_file = TUMORSEG_SCHEMA.format(base_dir=outdir)
    affine = nib.load(str(tumorseg_file)).affine

    core_segmentation = load_segmentation(tumorseg_file)
    core_segmentation[core_segmentation == 2] = 0  # ignore edema
    core_segmentation[core_segmentation == 3] = 1

    standard_plan = create_standard_plan(core_segmentation, CTV_MARGIN)
    standard_plan[load_brain_mask(outdir) == 0] = 0

    outfile = STANDARD_PLAN_SCHEMA.format(base_dir=outdir)
    nib.save(nib.Nifti1Image(standard_plan, affine=affine), str(outfile))
    logger.info(f"{outdir}: saved standard plan to {outfile}.")


def resolve_growth_models(growth_models: List[str]) -> Dict[str, Path]:
    """
    Maps the model id of every given growth model docker image (*.tar) to its path, keeping
    the given order. Missing images are logged and skipped.
    """
    growth_model_paths = {}
    for growth_model in growth_models:
        growth_model_path = Path(growth_model)
        if not growth_model_path.is_file():
            logger.warning(f"Growth model {growth_model_path} not found, skipping model.")
            continue
        model_id = MODEL_ID_OVERRIDES.get(growth_model_path.stem, growth_model_path.stem)
        growth_model_paths[model_id] = growth_model_path
    return growth_model_paths


def fix_prediction_affine(outdir: Path, model_id: str) -> None:
    """
    Rewrites a growth model prediction whose affine differs from the skull stripped t1c (the
    atlas frame of all outputs) with that affine, keeping the voxel data. Some model dockers
    (e.g. gliodil) write their prediction with an identity affine although the voxel grid is
    the one of their inputs. The data is stored as float32.
    """
    pred_file = PREDICTION_OUTPUT_SCHEMA.format(base_dir=outdir, algo_id=model_id)
    ref_img = nib.load(str(MODALITY_STRIPPED_SCHEMA.format(base_dir=outdir, modality="t1c")))
    pred_img = nib.load(str(pred_file))
    if pred_img.shape != ref_img.shape:
        raise ValueError(
            f"{pred_file}: shape {pred_img.shape} differs from the skull stripped t1c "
            f"{ref_img.shape}, cannot fix the affine."
        )
    if np.allclose(pred_img.affine, ref_img.affine, atol=AFFINE_ATOL):
        return
    data = np.asanyarray(pred_img.dataobj).astype(np.float32)
    tmp_file = pred_file.with_name(pred_file.name + ".tmp.nii.gz")
    nib.save(nib.Nifti1Image(data, ref_img.affine), str(tmp_file))
    os.replace(tmp_file, pred_file)
    logger.info(
        f"{outdir}: replaced the affine of {pred_file} (translation "
        f"{pred_img.affine[:3, 3]}) with the one of the skull stripped t1c "
        f"({ref_img.affine[:3, 3]})."
    )


def create_model_plan(outdir: Path, model_id: str) -> None:
    """
    Radiotherapy plan of a growth model prediction, mirroring the model plan of
    predict_gbm.evaluation.evaluate_tumor_model: the prediction is resampled to the grid of
    the standard plan, binary predictions are turned into a distance fade, and the highest
    scoring voxels within the brain mask are selected until the volume of the standard plan
    is reached. Unlike evaluate_tumor_model the scores are not clipped to [0, 1] before the
    selection: clipping saturates every voxel above 1 (e.g. the U-Net logits) to the same
    value, and topk_plan then breaks the ties by memory order, truncating the plan along the
    first axis. Clipping does not change the ranking otherwise, so the plans of predictions
    within [0, 1] are unaffected.
    """
    pred_file = PREDICTION_OUTPUT_SCHEMA.format(base_dir=outdir, algo_id=model_id)
    standard_plan_file = STANDARD_PLAN_SCHEMA.format(base_dir=outdir)
    affine = nib.load(str(standard_plan_file)).affine

    standard_plan = load_segmentation(standard_plan_file)
    brain_mask = load_brain_mask(outdir)

    model_prediction = load_and_resample_mri_data(
        str(pred_file), resample_params=standard_plan.shape, interp_type=0
    )
    if is_binary_array(model_prediction):
        logger.info(
            f"Prediction {pred_file} is binary. Generating distance fade for radiation planning."
        )
        model_prediction = generate_distance_fade_mask(model_prediction)

    model_plan = topk_plan(
        scores=model_prediction,
        target_voxels=int(np.sum(standard_plan)),
        mask=brain_mask,
    )

    outfile = MODEL_PLAN_SCHEMA.format(base_dir=outdir, algo_id=model_id)
    nib.save(nib.Nifti1Image(model_plan, affine=affine), str(outfile))
    logger.info(f"{outdir}: saved {model_id} plan to {outfile}.")


def predict_growth(
    outdir: Path, growth_model_paths: Dict[str, Path], cuda_device: str, override: bool
) -> None:
    """
    Predicts tumor cell concentration with every given growth model and derives the
    corresponding radiotherapy plan (growth_models/<id>/<id>_pred.nii.gz and _plan.nii.gz).
    The plan is always regenerated from the prediction, the prediction only if missing or
    override is set. A failing model is logged and skipped so that the remaining models
    still run.
    """
    model_inputs = {
        "tumorseg_file": TUMORSEG_SCHEMA.format(base_dir=outdir),
        "gm_file": TISSUE_PBMAP_SCHEMA.format(base_dir=outdir, tissue="gm"),
        "wm_file": TISSUE_PBMAP_SCHEMA.format(base_dir=outdir, tissue="wm"),
        "csf_file": TISSUE_PBMAP_SCHEMA.format(base_dir=outdir, tissue="csf"),
        "t1c_file": MODALITY_STRIPPED_SCHEMA.format(base_dir=outdir, modality="t1c"),
        "flair_file": MODALITY_STRIPPED_SCHEMA.format(base_dir=outdir, modality="flair"),
        "brain_mask_file": BRAIN_MASK_SCHEMA.format(base_dir=outdir),
    }
    # TumorGrowthModel exits the process on a missing input instead of raising, which would
    # abort the whole run, so the inputs are checked before the first model is started.
    missing = [str(f) for f in model_inputs.values() if not f.exists()]
    if missing:
        logger.error(f"{outdir}: missing prediction inputs {missing}, skipping prediction.")
        return

    for model_id, growth_model_path in growth_model_paths.items():
        pred_file = PREDICTION_OUTPUT_SCHEMA.format(base_dir=outdir, algo_id=model_id)
        try:
            if pred_file.exists() and not override:
                logger.info(f"{outdir}: {model_id} prediction already exists, skipping model run.")
            else:
                logger.info(f"{outdir}: starting growth prediction with {model_id}.")
                predict_tumor_growth(
                    model_id=model_id,
                    outdir=outdir,
                    cuda_device=cuda_device,
                    growth_model_path=growth_model_path,
                    **model_inputs,
                )
            fix_prediction_affine(outdir, model_id)
            create_model_plan(outdir, model_id)
        except Exception:
            logger.exception(f"{outdir}: prediction with {model_id} failed, skipping model.")


def check_output_affines(outdir: Path) -> None:
    """
    Verifies that every 3D nifti output (skull stripped modalities, brain mask, tumor and
    tissue segmentations, standard plan, predictions and plans) has the shape and affine of
    the skull stripped t1c. Raises on the first mismatch. The 5D ANTs warp fields of the
    tissue segmentation are not outputs in the atlas frame and are skipped.
    """
    ref_file = MODALITY_STRIPPED_SCHEMA.format(base_dir=outdir, modality="t1c")
    ref_img = nib.load(str(ref_file))
    out_files = sorted(
        f
        for folder in (
            SKULL_STRIP_FOLDER,
            TUMOR_SEGMENTATION_FOLDER,
            TISSUE_SEGMENTATION_FOLDER,
            MODEL_OUTPUT_DIR,
        )
        for f in (outdir / folder).rglob("*.nii.gz")
    )
    mismatches = []
    for f in out_files:
        img = nib.load(str(f))
        if len(img.shape) != 3:
            continue
        if img.shape != ref_img.shape:
            mismatches.append(f"{f}: shape {img.shape} != {ref_img.shape}")
        elif not np.allclose(img.affine, ref_img.affine, atol=AFFINE_ATOL):
            mismatches.append(
                f"{f}: affine translation {img.affine[:3, 3]} != {ref_img.affine[:3, 3]}"
            )
    if mismatches:
        raise ValueError(
            f"{outdir}: {len(mismatches)} outputs differ from {ref_file}:\n"
            + "\n".join(mismatches)
        )
    logger.info(
        f"{outdir}: all {len(out_files)} outputs share the shape and affine of {ref_file}."
    )


if __name__ == "__main__":
    # Example:
    # nohup python -u scripts/process_aa_bene.py -cuda_device 0 > tmp_process_aa_bene.out 2>&1 &
    parser = argparse.ArgumentParser()
    parser.add_argument("-cuda_device", type=str, default="0", help="GPU id to run on.")
    parser.add_argument(
        "-data_dir",
        type=str,
        default="/mnt/Drive4/lucas/aa_bene",
        help="Directory holding the preop exam niftis, one '*_<modality>.nii.gz' per modality.",
    )
    parser.add_argument(
        "-outdir",
        type=str,
        default="/mnt/Drive4/lucas/aa_bene/processed",
        help="Directory to save processed output to (predict_gbm exam layout).",
    )
    parser.add_argument(
        "-growth_models",
        type=str,
        nargs="*",
        default=DEFAULT_GROWTH_MODEL_PATHS,
        help=(
            "Growth model docker images (*.tar) to predict with. The file stem is used as "
            "model id unless it is renamed by MODEL_ID_OVERRIDES. Pass without arguments to "
            "skip the prediction."
        ),
    )
    parser.add_argument(
        "-override",
        action="store_true",
        help="Rerun every step even if its output already exists, overwriting previous results.",
    )
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    modality_files = find_modality_files(Path(args.data_dir))
    growth_model_paths = resolve_growth_models(args.growth_models)
    logger.info(f"Predicting with growth models {list(growth_model_paths)}.")

    # i) atlas co-registration, skull stripping, normalization
    preprocess(modality_files, outdir, args.override)
    # ii) tumor segmentation and standard plan
    segment_tumor(outdir, args.cuda_device, args.override)
    create_preop_standard_plan(outdir)
    # iii) tissue segmentation
    segment_tissue(outdir, args.override)
    # Inputs of the growth models must share the atlas frame before predicting.
    check_output_affines(outdir)
    # iv) growth model predictions and plans
    predict_growth(outdir, growth_model_paths, args.cuda_device, args.override)
    check_output_affines(outdir)

    logger.info(f"Finished processing {args.data_dir}, output in {outdir}.")
