import os
import ants
import argparse
import numpy as np
import nibabel as nib
from pathlib import Path
from typing import Dict, List, Tuple
from loguru import logger
from scipy.ndimage import binary_fill_holes
from predict_gbm.utils.constants import (
    BRAIN_MASK_SCHEMA,
    MODALITY_STRIPPED_SCHEMA,
    MODEL_OUTPUT_DIR,
    MODEL_PLAN_SCHEMA,
    OUTPUT_BASE_SCHEMA,
    PREDICTION_OUTPUT_SCHEMA,
    REGISTRATION_TRAFO_SCHEMA,
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
# Atlas the exam is registered to. The unet docker crops its output to the BraTS grid
# (240x240x155) and fails on any other shape, so -unet registers to the sri24 atlas, which
# has that grid, predicts with UNET_GROWTH_MODEL_PATHS and writes to <outdir>/UNET_SUBFOLDER.
DEFAULT_ATLAS = "brats_mni152"
UNET_ATLAS = "sri24"
UNET_SUBFOLDER = "sri24"
UNET_GROWTH_MODEL_PATHS = [str(GROWTH_MODEL_DIR / "unet1.tar")]
# Absolute tolerance for comparing output affines to the reference.
AFFINE_ATOL = 1e-3

# Raw input modalities whose grid the predictions and plans are transformed back to (patient
# space), see transform_to_patient_space. Every modality of the exam is supported.
PATIENT_SPACE_MODALITIES = ("t1c",)
PATIENT_SPACE_FOLDER = "patient_space"
PATIENT_SPACE_SCHEMA = (
    OUTPUT_BASE_SCHEMA / PATIENT_SPACE_FOLDER / "{modality}" / "{name}.nii.gz"
)
# Growth model outputs transformed to patient space as (output, schema, interpolator,
# pixeltype); the file name is <model id>_<output>.nii.gz.
MODEL_OUTPUTS = (
    ("pred", PREDICTION_OUTPUT_SCHEMA, "linear", "float"),
    ("plan", MODEL_PLAN_SCHEMA, "nearestNeighbor", "unsigned char"),
)
# Segmentations transformed to patient space with -transform_segmentations as (schema,
# schema kwargs besides base_dir, interpolator, pixeltype); the file name is the one of the
# atlas space file. Label maps and masks use nearest neighbors, the tissue probability maps
# linear interpolation.
SEGMENTATION_OUTPUTS = (
    (TUMORSEG_SCHEMA, {}, "nearestNeighbor", "unsigned char"),
    (STANDARD_PLAN_SCHEMA, {}, "nearestNeighbor", "unsigned char"),
    (BRAIN_MASK_SCHEMA, {}, "nearestNeighbor", "unsigned char"),
    (TISSUE_PBMAP_SCHEMA, {"tissue": "gm"}, "linear", "float"),
    (TISSUE_PBMAP_SCHEMA, {"tissue": "wm"}, "linear", "float"),
    (TISSUE_PBMAP_SCHEMA, {"tissue": "csf"}, "linear", "float"),
)


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


def preprocess(
    modality_files: Dict[str, Path], outdir: Path, atlas: str, override: bool
) -> None:
    """Normalization, skull stripping and co-registration of all modalities to the atlas."""
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
        atlas=atlas,
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


def segment_tissue(outdir: Path, atlas: str, override: bool) -> None:
    """Tissue (gm/wm/csf) probability maps of the atlas registered to the skull stripped t1c."""
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
        atlas=atlas,
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


def transform_file_to_raw_grid(
    infile: Path,
    outfile: Path,
    raw_file: Path,
    raw_img_ants: ants.ANTsImage,
    transform_files: List[str],
    interpolator: str,
    pixeltype: str,
) -> None:
    """
    Transforms infile from the atlas frame to the grid of raw_file by applying the inverse
    of transform_files (see transform_to_patient_space) and saves it as outfile with the
    given ANTs interpolator and pixeltype. Raises if the output does not have the shape and
    affine of raw_file.
    """
    transformed = ants.apply_transforms(
        fixed=raw_img_ants,
        moving=ants.image_read(str(infile)),
        transformlist=transform_files,
        whichtoinvert=[True] * len(transform_files),
        interpolator=interpolator,
        defaultvalue=0,
    )
    outfile.parent.mkdir(parents=True, exist_ok=True)
    ants.image_write(transformed.clone(pixeltype), str(outfile))

    raw_img = nib.load(str(raw_file))
    out_img = nib.load(str(outfile))
    if out_img.shape != raw_img.shape or not np.allclose(
        out_img.affine, raw_img.affine, atol=AFFINE_ATOL
    ):
        raise ValueError(
            f"{outfile}: shape {out_img.shape} / affine translation {out_img.affine[:3, 3]} "
            f"differ from {raw_file} ({raw_img.shape} / {raw_img.affine[:3, 3]})."
        )


def transform_to_patient_space(
    outdir: Path,
    modality_files: Dict[str, Path],
    model_ids: List[str],
    transform_segmentations: bool = False,
    modalities: Tuple[str, ...] = PATIENT_SPACE_MODALITIES,
) -> None:
    """
    Transforms the growth model predictions and plans (MODEL_OUTPUTS) and, if
    transform_segmentations is set, the tumor and tissue segmentations, brain mask and
    standard plan (SEGMENTATION_OUTPUTS) from the atlas frame back to the grid of the raw
    input modalities (patient space) by inverting the affine transforms that
    norm_ss_coregister saved in skull_stripped/<modality>/: for the center modality t1c the
    atlas registration (2_M_atlas__t1c.mat), for every other modality additionally its
    co-registration to t1c (1_M_co__t1c__<modality>.mat) and its correction registration in
    atlas space (3_M_atlas_corrected__t1c__<modality>.mat). ANTs applies the transform list
    front to back to the points of the fixed (raw) image, so the sorted matrices, all inverted,
    map a raw voxel to its position in the atlas frame. Continuous images are interpolated
    linearly, label maps and masks with nearest neighbors; the outputs have the shape and
    affine of the raw modality (verified). Outputs: patient_space/<modality>/<id>_pred.nii.gz,
    <id>_plan.nii.gz and the segmentations under their atlas space file names. Missing
    inputs (e.g. of failed models) are logged and skipped.
    """
    for modality in modalities:
        if modality not in modality_files:
            logger.warning(f"{outdir}: no raw {modality} in the exam, skipping patient space.")
            continue
        raw_file = modality_files[modality]
        transform_dir = REGISTRATION_TRAFO_SCHEMA.format(base_dir=outdir) / modality
        transform_files = [str(f) for f in sorted(transform_dir.glob("*.mat"))]
        if not transform_files:
            raise FileNotFoundError(
                f"{outdir}: no registration transform of {modality} found in {transform_dir}."
            )
        logger.info(
            f"{outdir}: transforming to the raw {modality} grid with the inverse of "
            f"{[Path(f).name for f in transform_files]}."
        )
        raw_img_ants = ants.image_read(str(raw_file))

        # (infile, output name, interpolator, pixeltype)
        jobs = [
            (
                infile_schema.format(base_dir=outdir, algo_id=model_id),
                f"{model_id}_{output}",
                interpolator,
                pixeltype,
            )
            for model_id in model_ids
            for output, infile_schema, interpolator, pixeltype in MODEL_OUTPUTS
        ]
        if transform_segmentations:
            for infile_schema, schema_kwargs, interpolator, pixeltype in SEGMENTATION_OUTPUTS:
                infile = infile_schema.format(base_dir=outdir, **schema_kwargs)
                name = infile.name[: -len(".nii.gz")]
                jobs.append((infile, name, interpolator, pixeltype))
        for infile, name, interpolator, pixeltype in jobs:
            if not infile.exists():
                logger.warning(f"{outdir}: {infile} does not exist, skipping.")
                continue
            outfile = PATIENT_SPACE_SCHEMA.format(base_dir=outdir, modality=modality, name=name)
            transform_file_to_raw_grid(
                infile, outfile, raw_file, raw_img_ants, transform_files, interpolator, pixeltype
            )
            logger.info(f"{outdir}: saved {name} in {modality} space to {outfile}.")


if __name__ == "__main__":
    # Example:
    # nohup python -u scripts/process_aa_bene.py -cuda_device 0 > tmp_process_aa_bene.out 2>&1 &
    # nohup python -u scripts/process_aa_bene.py -cuda_device 0 -unet > tmp_process_aa_bene_unet.out 2>&1 &
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
        default=None,
        help=(
            "Growth model docker images (*.tar) to predict with. The file stem is used as "
            "model id unless it is renamed by MODEL_ID_OVERRIDES. Pass without arguments to "
            f"skip the prediction. Defaults to {DEFAULT_GROWTH_MODEL_PATHS}, with -unet to "
            f"{UNET_GROWTH_MODEL_PATHS}."
        ),
    )
    parser.add_argument(
        "-unet",
        action="store_true",
        help=(
            f"Run the pipeline on the {UNET_ATLAS} atlas (BraTS grid required by the unet "
            f"docker) with the unet growth model, writing to <outdir>/{UNET_SUBFOLDER}."
        ),
    )
    parser.add_argument(
        "-transform_segmentations",
        action="store_true",
        help=(
            "Also transform the tumor and tissue segmentations, brain mask and standard plan "
            f"to patient space (see {PATIENT_SPACE_FOLDER}/)."
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
    atlas = DEFAULT_ATLAS
    growth_models = args.growth_models
    if args.unet:
        outdir = outdir / UNET_SUBFOLDER
        atlas = UNET_ATLAS
        if growth_models is None:
            growth_models = UNET_GROWTH_MODEL_PATHS
    elif growth_models is None:
        growth_models = DEFAULT_GROWTH_MODEL_PATHS
    outdir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Registering to the {atlas} atlas, output in {outdir}.")
    modality_files = find_modality_files(Path(args.data_dir))
    growth_model_paths = resolve_growth_models(growth_models)
    logger.info(f"Predicting with growth models {list(growth_model_paths)}.")

    # i) atlas co-registration, skull stripping, normalization
    preprocess(modality_files, outdir, atlas, args.override)
    # ii) tumor segmentation and standard plan
    segment_tumor(outdir, args.cuda_device, args.override)
    create_preop_standard_plan(outdir)
    # iii) tissue segmentation
    segment_tissue(outdir, atlas, args.override)
    # Inputs of the growth models must share the atlas frame before predicting.
    check_output_affines(outdir)
    # iv) growth model predictions and plans
    predict_growth(outdir, growth_model_paths, args.cuda_device, args.override)
    check_output_affines(outdir)
    # v) predictions, plans (and segmentations) back to the patient space of the raw modalities
    transform_to_patient_space(
        outdir, modality_files, list(growth_model_paths), args.transform_segmentations
    )

    logger.info(f"Finished processing {args.data_dir}, output in {outdir}.")
