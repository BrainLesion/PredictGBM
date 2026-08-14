import os
import re
import json
import shutil
import tempfile
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import ants
import nibabel as nib
import numpy as np
from loguru import logger
from brainles_preprocessing.preprocessor import AtlasCentricPreprocessor
from brainles_preprocessing.registration import ANTsRegistrator
from brainles_preprocessing.brain_extraction.synthstrip import SynthStripExtractor
from brainles_preprocessing.normalization.percentile_normalizer import (
    PercentileNormalizer,
)

from predict_gbm.preprocessing import (
    norm_ss_coregister,
    register_recurrence,
    run_brats,
    run_tissue_seg,
)
from predict_gbm.preprocessing.norm_ss_coregistration import (
    SUPPORTED_ATLASES,
    initialize_center_modality,
)
from predict_gbm.preprocessing.tissue_segmentation import generate_registration_mask
from predict_gbm.prediction import predict_tumor_growth
from predict_gbm.utils.constants import (
    ATLAS_UNSTRIPPED_SCHEMA,
    BRAIN_MASK_SCHEMA,
    LONGITUDINAL_AFFINE_SCHEMA,
    LONGITUDINAL_WARP_SCHEMA,
    MODALITY_STRIPPED_SCHEMA,
    PREDICTION_OUTPUT_SCHEMA,
    RECURRENCE_SCHEMA,
    REGISTRATION_MASK_SCHEMA,
    TISSUE_LABELS,
    TISSUE_PBMAP_SCHEMA,
    TISSUE_SEG_BASE_SCHEMA,
    TUMORSEG_SCHEMA,
)

# Session directories are named ses-YYYYMMDD; the date defines the exam ordering:
# earliest exam = healthy (pre-diagnosis), 2nd earliest = pre-operative, rest = follow-up.
SESSION_DATE_PATTERN = re.compile(r"(\d{8})")

# Some exams carry a T1-weighted volume under the generic "_mr" suffix instead of "_t1"
# (e.g. the Philips s3DI_MC_HR series in sub-GR031/ses-20101004). Such a series is only
# used when no "_t1" file exists and its DICOM description looks T1-weighted, so that
# unrelated "_mr" series (e.g. a duplicated t2_tse) are not picked up by mistake.
T1_LIKE_DESCRIPTION_PATTERN = re.compile(r"t1|mprage|s3di|bravo|fspgr|tfl", re.IGNORECASE)

# Growth models run on every pre-operative and follow-up exam. The model id, which names the
# prediction output, is the file stem of the docker image (e.g. sbtc.tar -> "sbtc").
DEFAULT_GROWTH_MODEL_PATHS = [
    "/mnt/Drive4/lucas/growth_models/predict-gbm/sbtc.tar",
    "/mnt/Drive4/lucas/growth_models/predict-gbm/unet1.tar",
]

# The antsAtroposN4 tissue segmentation of the healthy exams is kept beside the atlas registration
# one instead of replacing it, so that the two can be compared. Only the directories differ: the
# file names inside them are the ones the standard schemas produce.
ATROPOS_TISSUE_SEG_FOLDER = "tissue_segmentation_atropos"
ATROPOS_LONGITUDINAL_FOLDER = "longitudinal_atropos"


def parse_session_date(session_dir: Path) -> Optional[datetime]:
    """Extracts the exam date from a session directory name such as 'ses-20230213'."""
    match = SESSION_DATE_PATTERN.search(session_dir.name)
    if match is None:
        return None
    try:
        return datetime.strptime(match.group(1), "%Y%m%d")
    except ValueError:
        return None


def collect_sessions(patient_dir: Path) -> List[Tuple[datetime, Path]]:
    """Returns the patient's session directories as (date, dir) tuples, sorted by date."""
    sessions = []
    for session_dir in sorted(p for p in patient_dir.iterdir() if p.is_dir()):
        session_date = parse_session_date(session_dir)
        if session_date is None:
            logger.warning(f"{session_dir}: no date in directory name, skipping session.")
            continue
        sessions.append((session_date, session_dir))
    return sorted(sessions, key=lambda entry: entry[0])


def find_modality_file(session_dir: Path, modality: str) -> Optional[Path]:
    """Returns the nifti of the given modality in a session directory, or None if absent."""
    matches = sorted(session_dir.glob(f"*_{modality}.nii.gz"))
    if not matches:
        return None
    if len(matches) > 1:
        logger.warning(
            f"{session_dir}: {len(matches)} candidates for modality {modality}, using {matches[0].name}."
        )
    return matches[0]


def read_series_description(nifti_file: Path) -> str:
    """Reads SeriesDescription and ProtocolName from the dcm2niix json sidecar of a nifti."""
    sidecar = nifti_file.with_name(nifti_file.name.replace(".nii.gz", ".json"))
    if not sidecar.exists():
        return ""
    try:
        with sidecar.open("r") as f:
            meta = json.load(f)
    except (OSError, json.JSONDecodeError):
        logger.warning(f"{sidecar}: could not be read, ignoring series description.")
        return ""
    return f"{meta.get('SeriesDescription', '')} {meta.get('ProtocolName', '')}"


def resolve_t1(session_dir: Path) -> Optional[Path]:
    """Returns the t1 nifti of a session, falling back to a T1-weighted '_mr' series."""
    t1_file = find_modality_file(session_dir, "t1")
    if t1_file is not None:
        return t1_file

    for mr_file in sorted(session_dir.glob("*_mr.nii.gz")):
        description = read_series_description(mr_file)
        if T1_LIKE_DESCRIPTION_PATTERN.search(description):
            logger.warning(
                f"{session_dir}: no _t1 series, using {mr_file.name} "
                f"({description.strip()}) as t1."
            )
            return mr_file
    return None


def resolve_modalities(session_dir: Path) -> Optional[Dict[str, Path]]:
    """
    Resolves t1/t1c/t2/flair paths for an exam, applying the same fallback rules as
    scripts/process_sailor.py: t1 missing -> use t1c; t2 missing -> use flair;
    flair missing -> use t2. Returns None if the exam should be excluded, i.e. if t1c
    is missing or neither t2 nor flair is available.
    """
    t1c_file = find_modality_file(session_dir, "t1c")
    if t1c_file is None:
        return None

    t1_file = resolve_t1(session_dir)
    if t1_file is None:
        logger.warning(f"{session_dir}: no t1 series, using t1c instead.")
        t1_file = t1c_file

    t2_file = find_modality_file(session_dir, "t2")
    flair_file = find_modality_file(session_dir, "flair")
    if t2_file is None and flair_file is None:
        return None
    if t2_file is None:
        logger.warning(f"{session_dir}: no t2 series, using flair instead.")
        t2_file = flair_file
    if flair_file is None:
        logger.warning(f"{session_dir}: no flair series, using t2 instead.")
        flair_file = t2_file

    return {"t1": t1_file, "t1c": t1c_file, "t2": t2_file, "flair": flair_file}


def process_healthy_exam(
    t1_file: Path, outdir: Path, atlas: str, override: bool = False
) -> None:
    """
    Processes the healthy (earliest) exam exactly as scripts/single_t1.py does: normalization,
    skull stripping and atlas co-registration with the t1 as the only, center modality, followed
    by atlas registration tissue segmentation. If override is True, all steps are rerun even if
    their outputs already exist.
    """
    outdir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Starting single-T1 preprocessing for {t1_file}.")

    t1_stripped_file = MODALITY_STRIPPED_SCHEMA.format(base_dir=outdir, modality="t1")
    if t1_stripped_file.exists() and not override:
        logger.info(f"{outdir}: skull stripping already done, skipping.")
    else:
        # Normalization, skull stripping, atlas co-registration (same steps as norm_ss_coregister,
        # but with the t1 as the only, center modality since no other modalities are available).
        percentile_normalizer = PercentileNormalizer(
            lower_percentile=0.1,
            upper_percentile=99.9,
            lower_limit=0,
            upper_limit=1,
        )
        center = initialize_center_modality(
            modality_file=t1_file,
            modality_name="t1",
            normalizer=percentile_normalizer,
            outdir=outdir,
            skull_strip=True,
        )
        atlas_schema = ATLAS_UNSTRIPPED_SCHEMA
        registrator = ANTsRegistrator(transformation_params={"defaultvalue": 0})
        preprocessor = AtlasCentricPreprocessor(
            center_modality=center,
            moving_modalities=[],
            registrator=registrator,
            brain_extractor=SynthStripExtractor(),
            atlas_image_path=atlas_schema.format(atlas=atlas),
        )
        preprocessor.run()

    # Tissue segmentation. The healthy exam has no tumor, so the atlas is registered to it without
    # a registration mask.
    if tissueseg_done(outdir) and not override:
        logger.info(f"{outdir}: tissue segmentation already done, skipping.")
    else:
        clear_tissue_seg(outdir)
        run_tissue_seg(
            t1_file=t1_stripped_file,
            outdir=outdir,
            algorithm="atlas_registration",
            atlas=atlas,
        )

    logger.info(f"Finished single-T1 preprocessing. Output saved to {outdir}.")


def clear_tissue_seg(outdir: Path) -> None:
    """
    Removes the tissue segmentation directory of an exam, so that the segmentation about to be
    run starts from an empty one.

    run_tissue_seg_atlas_registration hands the tissue segmentation directory to ants.registration
    as its outprefix, and ants.registration does not track the files it writes: it recovers them
    afterwards by globbing "<outprefix>*[0-9]*" and passes every match on as a transform. Any file
    left in that directory by an earlier run whose name contains a digit is therefore applied as a
    warp field. The antsAtroposN4 output "tissue_seg_Segmentation0N4.nii.gz" is such a file, which
    silently turns the warped probability maps into near-empty ones.
    """
    tissue_seg_dir = TISSUE_SEG_BASE_SCHEMA.format(base_dir=outdir)
    if tissue_seg_dir.exists():
        shutil.rmtree(tissue_seg_dir)


def tissueseg_done(outdir: Path) -> bool:
    """True if the gm/wm/csf probability maps already exist for an exam."""
    return all(
        TISSUE_PBMAP_SCHEMA.format(base_dir=outdir, tissue=tissue).exists()
        for tissue in ("gm", "wm", "csf")
    )


def predict_growth(
    outdir: Path,
    growth_model_paths: Dict[str, Path],
    cuda_device: str,
    override: bool = False,
) -> None:
    """
    Predicts tumor cell concentration on a preprocessed exam with every given growth model,
    using its skull-stripped images, tumor segmentation and tissue probability maps. Models are
    identified by the stem of their docker image and each prediction is written to its own
    growth_models/{model_id} directory. A failing model is logged and skipped so that the
    remaining models still run. If override is True, predictions are rerun even if they exist.
    """
    for model_id, growth_model_path in growth_model_paths.items():
        pred_file = PREDICTION_OUTPUT_SCHEMA.format(base_dir=outdir, algo_id=model_id)
        if pred_file.exists() and not override:
            logger.info(f"{outdir}: prediction with {model_id} already done, skipping.")
            continue

        try:
            predict_tumor_growth(
                tumorseg_file=TUMORSEG_SCHEMA.format(base_dir=outdir),
                gm_file=TISSUE_PBMAP_SCHEMA.format(base_dir=outdir, tissue="gm"),
                wm_file=TISSUE_PBMAP_SCHEMA.format(base_dir=outdir, tissue="wm"),
                csf_file=TISSUE_PBMAP_SCHEMA.format(base_dir=outdir, tissue="csf"),
                model_id=model_id,
                outdir=outdir,
                cuda_device=cuda_device,
                t1c_file=MODALITY_STRIPPED_SCHEMA.format(
                    base_dir=outdir, modality="t1c"
                ),
                flair_file=MODALITY_STRIPPED_SCHEMA.format(
                    base_dir=outdir, modality="flair"
                ),
                brain_mask_file=BRAIN_MASK_SCHEMA.format(base_dir=outdir),
                growth_model_path=growth_model_path,
            )
        except Exception:
            logger.exception(f"{outdir}: prediction with {model_id} failed, skipping model.")


def process_exam(
    modalities: Dict[str, Path],
    outdir: Path,
    atlas: str,
    cuda_device: str,
    growth_model_paths: Dict[str, Path],
    override: bool = False,
) -> None:
    """
    Processes a pre-operative or follow-up exam: normalization, skull stripping and atlas
    co-registration, tumor segmentation via BRATS, tissue segmentation via atlas registration
    with the tumor core masked out of the registration metric, and tumor growth prediction with
    every given growth model. If override is True, all steps are rerun even if their outputs
    already exist.
    """
    outdir.mkdir(parents=True, exist_ok=True)

    t1c_stripped = MODALITY_STRIPPED_SCHEMA.format(base_dir=outdir, modality="t1c")
    if t1c_stripped.exists() and not override:
        logger.info(f"{outdir}: skull stripping already done, skipping.")
    else:
        norm_ss_coregister(
            t1_file=modalities["t1"],
            t1c_file=modalities["t1c"],
            t2_file=modalities["t2"],
            flair_file=modalities["flair"],
            skull_strip=True,
            outdir=outdir,
            atlas=atlas,
        )

    tumorseg_file = TUMORSEG_SCHEMA.format(base_dir=outdir)
    if tumorseg_file.exists() and not override:
        logger.info(f"{outdir}: tumor segmentation already done, skipping.")
    else:
        run_brats(
            t1_file=MODALITY_STRIPPED_SCHEMA.format(base_dir=outdir, modality="t1"),
            t1c_file=t1c_stripped,
            t2_file=MODALITY_STRIPPED_SCHEMA.format(base_dir=outdir, modality="t2"),
            flair_file=MODALITY_STRIPPED_SCHEMA.format(base_dir=outdir, modality="flair"),
            outdir=outdir,
            cuda_device=cuda_device,
        )

    if tissueseg_done(outdir) and not override:
        logger.info(f"{outdir}: tissue segmentation already done, skipping.")
    else:
        registration_mask_file = REGISTRATION_MASK_SCHEMA.format(base_dir=outdir)
        generate_registration_mask(
            tumor_seg_file=tumorseg_file,
            outfile=registration_mask_file,
        )
        run_tissue_seg(
            t1_file=t1c_stripped,
            outdir=outdir,
            registration_mask_file=registration_mask_file,
            algorithm="atlas_registration",
            atlas=atlas,
        )

    predict_growth(
        outdir=outdir,
        growth_model_paths=growth_model_paths,
        cuda_device=cuda_device,
        override=override,
    )


def register_followup_to_preop(
    preop_outdir: Path,
    followup_outdir: Path,
    model_ids: List[str],
    override: bool = False,
) -> None:
    """
    Registers a follow-up exam to pre-operative space with an affine transform, warping the tumor
    segmentation, all remaining modalities (t1, t2, flair) and the growth predictions alongside
    the t1c used to drive the registration. Predictions that are missing (e.g. because the model
    failed) are skipped.
    """
    additional_modalities = {
        modality: MODALITY_STRIPPED_SCHEMA.format(
            base_dir=followup_outdir, modality=modality
        )
        for modality in ("t1", "t2", "flair")
    }
    for model_id in model_ids:
        pred_file = PREDICTION_OUTPUT_SCHEMA.format(
            base_dir=followup_outdir, algo_id=model_id
        )
        if pred_file.exists():
            additional_modalities[f"{model_id}_pred"] = pred_file
        else:
            logger.warning(
                f"{followup_outdir}: no prediction of {model_id} to warp into preop space."
            )

    recurrence_file = RECURRENCE_SCHEMA.format(base_dir=followup_outdir)
    warped_files = [recurrence_file] + [
        LONGITUDINAL_WARP_SCHEMA.format(base_dir=followup_outdir, modality=modality)
        for modality in additional_modalities
    ]
    if all(warped_file.exists() for warped_file in warped_files) and not override:
        logger.info(f"{followup_outdir}: registration to preop already done, skipping.")
        return

    register_recurrence(
        t1c_pre_file=MODALITY_STRIPPED_SCHEMA.format(
            base_dir=preop_outdir, modality="t1c"
        ),
        t1c_post_file=MODALITY_STRIPPED_SCHEMA.format(
            base_dir=followup_outdir, modality="t1c"
        ),
        recurrence_seg_file=TUMORSEG_SCHEMA.format(base_dir=followup_outdir),
        outdir=followup_outdir,
        registration_algorithm="affine",
        additional_modalities=additional_modalities,
    )


def register_healthy_to_preop(
    preop_outdir: Path, healthy_outdir: Path, override: bool = False
) -> None:
    """
    Registers the healthy exam to pre-operative space with an affine transform, warping its t1.
    This is the same step as register_followup_to_preop, except that the healthy exam has no
    tumor segmentation to co-transform. Since register_recurrence always warps a segmentation,
    an all-zero placeholder is passed and its warped copy is removed afterwards.
    """
    warped_t1_file = LONGITUDINAL_WARP_SCHEMA.format(
        base_dir=healthy_outdir, modality="t1"
    )
    if warped_t1_file.exists() and not override:
        logger.info(f"{healthy_outdir}: registration to preop already done, skipping.")
        return

    healthy_t1_file = MODALITY_STRIPPED_SCHEMA.format(
        base_dir=healthy_outdir, modality="t1"
    )
    placeholder_seg_file = healthy_outdir / "empty_seg_placeholder.nii.gz"
    healthy_t1_img = nib.load(str(healthy_t1_file))
    nib.save(
        nib.Nifti1Image(
            np.zeros(healthy_t1_img.shape, dtype=np.uint8), healthy_t1_img.affine
        ),
        str(placeholder_seg_file),
    )

    try:
        register_recurrence(
            t1c_pre_file=MODALITY_STRIPPED_SCHEMA.format(
                base_dir=preop_outdir, modality="t1c"
            ),
            t1c_post_file=healthy_t1_file,
            recurrence_seg_file=placeholder_seg_file,
            outdir=healthy_outdir,
            registration_algorithm="affine",
        )
    finally:
        placeholder_seg_file.unlink(missing_ok=True)

    # register_recurrence stores the warped moving image under the t1c name and the warped
    # placeholder as the recurrence; the healthy exam only has a t1 and no recurrence.
    LONGITUDINAL_WARP_SCHEMA.format(base_dir=healthy_outdir, modality="t1c").replace(
        warped_t1_file
    )
    RECURRENCE_SCHEMA.format(base_dir=healthy_outdir).unlink(missing_ok=True)


def warp_predictions_to_preop(
    preop_outdir: Path,
    followup_outdir: Path,
    model_ids: List[str],
    override: bool = False,
) -> None:
    """
    Warps the growth predictions of a follow-up exam into pre-operative space, reusing the affine
    transform stored by the earlier longitudinal registration. This is the deferred counterpart to
    register_followup_to_preop, which warps the predictions right away: it adds the predictions to
    an already registered exam without recomputing the registration, so the tumor segmentation and
    the mri sequences keep the exact transform they were warped with.
    """
    transform_file = LONGITUDINAL_AFFINE_SCHEMA.format(base_dir=followup_outdir)
    if not transform_file.exists():
        logger.warning(
            f"{followup_outdir}: no longitudinal transform ({transform_file}), "
            "cannot warp predictions into preop space."
        )
        return

    t1c_pre_img = ants.image_read(
        str(MODALITY_STRIPPED_SCHEMA.format(base_dir=preop_outdir, modality="t1c"))
    )
    for model_id in model_ids:
        pred_file = PREDICTION_OUTPUT_SCHEMA.format(
            base_dir=followup_outdir, algo_id=model_id
        )
        if not pred_file.exists():
            logger.warning(
                f"{followup_outdir}: no prediction of {model_id} to warp into preop space."
            )
            continue

        warped_file = LONGITUDINAL_WARP_SCHEMA.format(
            base_dir=followup_outdir, modality=f"{model_id}_pred"
        )
        if warped_file.exists() and not override:
            logger.info(
                f"{followup_outdir}: prediction of {model_id} already warped to preop, skipping."
            )
            continue

        pred_warped = ants.apply_transforms(
            fixed=t1c_pre_img,
            moving=ants.image_read(str(pred_file)),
            transformlist=[str(transform_file)],
        )
        warped_file.parent.mkdir(parents=True, exist_ok=True)
        ants.image_write(pred_warped, str(warped_file))
        logger.info(f"Warped prediction of {model_id} to {warped_file}.")


def warp_tissue_seg_to_preop(
    preop_outdir: Path, exam_outdir: Path, override: bool = False
) -> None:
    """
    Warps the gm/wm/csf probability maps of an exam into pre-operative space, reusing the affine
    transform stored by the longitudinal registration. Like warp_predictions_to_preop, this is a
    deferred step that adds maps to an already registered exam without recomputing the
    registration, so they carry the exact transform the mri sequences were warped with.
    """
    transform_file = LONGITUDINAL_AFFINE_SCHEMA.format(base_dir=exam_outdir)
    if not transform_file.exists():
        logger.warning(
            f"{exam_outdir}: no longitudinal transform ({transform_file}), "
            "cannot warp tissue segmentation into preop space."
        )
        return

    t1c_pre_img = ants.image_read(
        str(MODALITY_STRIPPED_SCHEMA.format(base_dir=preop_outdir, modality="t1c"))
    )
    for tissue in TISSUE_LABELS:
        pbmap_file = TISSUE_PBMAP_SCHEMA.format(base_dir=exam_outdir, tissue=tissue)
        if not pbmap_file.exists():
            logger.warning(
                f"{exam_outdir}: no {tissue} probability map to warp into preop space."
            )
            continue

        warped_file = LONGITUDINAL_WARP_SCHEMA.format(
            base_dir=exam_outdir, modality=f"{tissue}_pbmap"
        )
        if warped_file.exists() and not override:
            logger.info(
                f"{exam_outdir}: {tissue} probability map already warped to preop, skipping."
            )
            continue

        # Probability maps are continuous, so they are interpolated linearly like the mri
        # sequences and the predictions rather than with a label-preserving interpolator.
        pbmap_warped = ants.apply_transforms(
            fixed=t1c_pre_img,
            moving=ants.image_read(str(pbmap_file)),
            transformlist=[str(transform_file)],
            interpolator="linear",
        )
        warped_file.parent.mkdir(parents=True, exist_ok=True)
        ants.image_write(pbmap_warped, str(warped_file))
        logger.info(f"Warped {tissue} probability map to {warped_file}.")


def run_atropos_tissue_seg(exam_outdir: Path) -> None:
    """
    Runs the antsAtroposN4 tissue segmentation of an exam and stores it under
    ATROPOS_TISSUE_SEG_FOLDER instead of the standard tissue segmentation directory.

    run_tissue_seg always writes to the directory its constants schemas derive from outdir, so the
    segmentation is run on a scratch directory that mirrors the exam and its output is moved into
    place afterwards. antsAtroposN4 only reads the skull-stripped t1 and the brain mask, so linking
    the skull stripping directory is enough to mirror the exam. This leaves the atlas registration
    segmentation of the exam untouched. The step config that run_tissue_seg writes alongside the
    segmentation stays in the scratch directory, since the exam config describes the atlas
    registration segmentation that remains in place.
    """
    target_dir = exam_outdir / ATROPOS_TISSUE_SEG_FOLDER
    if target_dir.exists():
        shutil.rmtree(target_dir)

    with tempfile.TemporaryDirectory() as scratch:
        scratch_outdir = Path(scratch) / exam_outdir.name
        scratch_outdir.mkdir(parents=True)
        stripped_dir = MODALITY_STRIPPED_SCHEMA.format(
            base_dir=exam_outdir, modality="t1"
        ).parent
        (scratch_outdir / stripped_dir.name).symlink_to(stripped_dir)

        run_tissue_seg(
            t1_file=MODALITY_STRIPPED_SCHEMA.format(
                base_dir=scratch_outdir, modality="t1"
            ),
            outdir=scratch_outdir,
            algorithm="antsAtroposN4",
        )
        shutil.move(
            str(TISSUE_SEG_BASE_SCHEMA.format(base_dir=scratch_outdir)), str(target_dir)
        )

    logger.info(f"Atropos tissue segmentation saved to {target_dir}.")


def warp_atropos_tissue_seg_to_preop(
    preop_outdir: Path, exam_outdir: Path, override: bool = False
) -> None:
    """
    Warps the antsAtroposN4 probability maps of an exam into pre-operative space, reading them from
    ATROPOS_TISSUE_SEG_FOLDER and writing them to ATROPOS_LONGITUDINAL_FOLDER. Identical to
    warp_tissue_seg_to_preop apart from those two directories, and in particular it reuses the very
    same longitudinal transform, so both variants are warped exactly alike.
    """
    transform_file = LONGITUDINAL_AFFINE_SCHEMA.format(base_dir=exam_outdir)
    if not transform_file.exists():
        logger.warning(
            f"{exam_outdir}: no longitudinal transform ({transform_file}), "
            "cannot warp atropos tissue segmentation into preop space."
        )
        return

    t1c_pre_img = ants.image_read(
        str(MODALITY_STRIPPED_SCHEMA.format(base_dir=preop_outdir, modality="t1c"))
    )
    for tissue in TISSUE_LABELS:
        # Only the directories differ from the atlas registration variant, so the file names are
        # taken from the standard schemas rather than spelled out again.
        pbmap_file = (
            exam_outdir
            / ATROPOS_TISSUE_SEG_FOLDER
            / TISSUE_PBMAP_SCHEMA.format(base_dir=exam_outdir, tissue=tissue).name
        )
        if not pbmap_file.exists():
            logger.warning(
                f"{exam_outdir}: no atropos {tissue} probability map to warp into preop space."
            )
            continue

        warped_file = (
            exam_outdir
            / ATROPOS_LONGITUDINAL_FOLDER
            / LONGITUDINAL_WARP_SCHEMA.format(
                base_dir=exam_outdir, modality=f"{tissue}_pbmap"
            ).name
        )
        if warped_file.exists() and not override:
            logger.info(
                f"{exam_outdir}: atropos {tissue} probability map already warped to preop, "
                "skipping."
            )
            continue

        pbmap_warped = ants.apply_transforms(
            fixed=t1c_pre_img,
            moving=ants.image_read(str(pbmap_file)),
            transformlist=[str(transform_file)],
            interpolator="linear",
        )
        warped_file.parent.mkdir(parents=True, exist_ok=True)
        ants.image_write(pbmap_warped, str(warped_file))
        logger.info(f"Warped atropos {tissue} probability map to {warped_file}.")


def resolve_healthy_and_preop(patient_outdir: Path) -> Optional[Tuple[Path, List[Path]]]:
    """
    Returns the pre-operative exam directory of a patient together with its healthy exam
    directories, taken from the session_roles.json written during preprocessing. Returns None if
    the patient was not preprocessed or has no pre-operative exam recorded.
    """
    roles_file = patient_outdir / "session_roles.json"
    if not roles_file.exists():
        logger.warning(
            f"{patient_outdir}: no {roles_file.name}, patient not preprocessed, skipping patient."
        )
        return None
    with roles_file.open("r") as f:
        session_roles = json.load(f)

    preop_sessions = [name for name, role in session_roles.items() if role == "preop"]
    if not preop_sessions:
        logger.warning(f"{patient_outdir}: no preop exam recorded, skipping patient.")
        return None

    healthy_outdirs = [
        patient_outdir / name
        for name, role in session_roles.items()
        if role == "healthy"
    ]
    return patient_outdir / preop_sessions[0], healthy_outdirs


def recompute_and_warp_atropos_tissue_seg_patient(
    patient_outdir: Path, warp_only: bool = False
) -> None:
    """
    Runs the antsAtroposN4 tissue segmentation on the already preprocessed healthy exams of a
    single patient and warps the resulting probability maps into pre-operative space, storing both
    beside the atlas registration results rather than replacing them. Used with
    -stage warp_tissue_atropos. Both steps always overwrite their previous output, since the point
    of the stage is to replace it; pass warp_only to reuse the existing maps and only redo the
    warping.
    """
    resolved = resolve_healthy_and_preop(patient_outdir)
    if resolved is None:
        return
    preop_outdir, healthy_outdirs = resolved

    for healthy_outdir in healthy_outdirs:
        t1_stripped_file = MODALITY_STRIPPED_SCHEMA.format(
            base_dir=healthy_outdir, modality="t1"
        )
        if not t1_stripped_file.exists():
            logger.warning(
                f"{healthy_outdir}: no skull-stripped t1 ({t1_stripped_file}), "
                "healthy exam not preprocessed, skipping exam."
            )
            continue

        if not warp_only:
            try:
                run_atropos_tissue_seg(healthy_outdir)
            except Exception:
                logger.exception(
                    f"{healthy_outdir}: atropos tissue segmentation failed, skipping exam."
                )
                continue

        try:
            # The maps just changed, so their warped copies are always rewritten.
            warp_atropos_tissue_seg_to_preop(
                preop_outdir=preop_outdir,
                exam_outdir=healthy_outdir,
                override=True,
            )
        except Exception:
            logger.exception(
                f"{healthy_outdir}: warping atropos tissue segmentation to preop failed, skipping."
            )


def recompute_and_warp_tissue_seg_patient(
    patient_outdir: Path, atlas: str, warp_only: bool = False
) -> None:
    """
    Recomputes the tissue segmentation of the already preprocessed healthy exam of a single
    patient via atlas registration and warps the resulting probability maps into pre-operative space.
    Used to run this step separately from preprocessing, i.e. with -stage warp_tissue, so that the
    healthy tissue maps can be regenerated after a change to the segmentation without redoing any
    of the preprocessing. Both steps always overwrite their previous output, since the point of
    the stage is to replace it; pass warp_only to reuse the existing maps and only redo the
    warping. The exam roles are taken from the session_roles.json written during preprocessing.
    """
    resolved = resolve_healthy_and_preop(patient_outdir)
    if resolved is None:
        return
    preop_outdir, healthy_outdirs = resolved

    for healthy_outdir in healthy_outdirs:
        t1_stripped_file = MODALITY_STRIPPED_SCHEMA.format(
            base_dir=healthy_outdir, modality="t1"
        )
        if not t1_stripped_file.exists():
            logger.warning(
                f"{healthy_outdir}: no skull-stripped t1 ({t1_stripped_file}), "
                "healthy exam not preprocessed, skipping exam."
            )
            continue

        if not warp_only:
            try:
                # Same call as in process_healthy_exam: no registration mask, since the healthy
                # exam has no tumor to exclude from the registration metric.
                clear_tissue_seg(healthy_outdir)
                run_tissue_seg(
                    t1_file=t1_stripped_file,
                    outdir=healthy_outdir,
                    algorithm="atlas_registration",
                    atlas=atlas,
                )
            except Exception:
                logger.exception(
                    f"{healthy_outdir}: tissue segmentation failed, skipping exam."
                )
                continue

        try:
            # The maps just changed, so their warped copies are always rewritten.
            warp_tissue_seg_to_preop(
                preop_outdir=preop_outdir,
                exam_outdir=healthy_outdir,
                override=True,
            )
        except Exception:
            logger.exception(
                f"{healthy_outdir}: warping tissue segmentation to preop failed, skipping."
            )


def predict_patient(
    patient_outdir: Path,
    growth_model_paths: Dict[str, Path],
    cuda_device: str,
    override: bool = False,
) -> None:
    """
    Runs the growth prediction on the already preprocessed pre-operative exam of a single patient.
    Used to run the prediction as a separate pass after preprocessing, i.e. with -stage preprocess
    followed by -stage predict. The exam roles are taken from the session_roles.json written during
    preprocessing. Follow-ups are not predicted on, so nothing has to be warped into pre-operative
    space here.
    """
    resolved = resolve_healthy_and_preop(patient_outdir)
    if resolved is None:
        return
    preop_outdir, _ = resolved

    if not TUMORSEG_SCHEMA.format(base_dir=preop_outdir).exists() or not tissueseg_done(
        preop_outdir
    ):
        logger.warning(
            f"{preop_outdir}: tumor or tissue segmentation missing, skipping prediction."
        )
        return

    predict_growth(
        outdir=preop_outdir,
        growth_model_paths=growth_model_paths,
        cuda_device=cuda_device,
        override=override,
    )


def process_patient(
    patient_dir: Path,
    outdir_root: Path,
    atlas: str,
    cuda_device: str,
    growth_model_paths: Dict[str, Path],
    override: bool = False,
) -> None:
    """Processes all exams of a single patient and registers the follow-ups to preop space."""
    patient_id = patient_dir.name
    sessions = collect_sessions(patient_dir)
    if not sessions:
        logger.warning(f"{patient_id}: no dated session directories found, skipping patient.")
        return

    healthy_dir = sessions[0][1]
    session_roles = {healthy_dir.name: "healthy"}

    healthy_outdir = outdir_root / patient_id / healthy_dir.name
    healthy_processed = False
    healthy_t1_file = resolve_t1(healthy_dir)
    if healthy_t1_file is None:
        logger.warning(
            f"{patient_id}/{healthy_dir.name}: no t1 series for the healthy exam, skipping exam."
        )
    else:
        try:
            process_healthy_exam(
                t1_file=healthy_t1_file,
                outdir=healthy_outdir,
                atlas=atlas,
                override=override,
            )
            healthy_processed = True
        except Exception:
            logger.exception(
                f"{patient_id}/{healthy_dir.name}: healthy exam processing failed, skipping."
            )

    if len(sessions) < 2:
        logger.warning(
            f"{patient_id}: only one exam available, no preop exam to process, skipping patient."
        )
        return

    preop_dir = sessions[1][1]
    followup_dirs = [session_dir for _, session_dir in sessions[2:]]
    session_roles[preop_dir.name] = "preop"
    session_roles.update({session_dir.name: "followup" for session_dir in followup_dirs})
    logger.info(
        f"{patient_id}: healthy={healthy_dir.name}, preop={preop_dir.name}, "
        f"followups={[session_dir.name for session_dir in followup_dirs]}."
    )

    roles_file = outdir_root / patient_id / "session_roles.json"
    roles_file.parent.mkdir(parents=True, exist_ok=True)
    with roles_file.open("w") as f:
        json.dump(session_roles, f, indent=4)

    processed_followups = []
    for session_dir in [preop_dir] + followup_dirs:
        is_preop = session_dir is preop_dir

        modalities = resolve_modalities(session_dir)
        if modalities is None:
            logger.warning(
                f"{patient_id}/{session_dir.name}: missing t1c or unresolvable t2/flair, skipping exam."
            )
            if is_preop:
                logger.warning(
                    f"{patient_id}: preop exam unusable, skipping registration of follow-ups."
                )
                return
            continue

        exam_outdir = outdir_root / patient_id / session_dir.name
        try:
            process_exam(
                modalities=modalities,
                outdir=exam_outdir,
                atlas=atlas,
                cuda_device=cuda_device,
                growth_model_paths=growth_model_paths,
                override=override,
            )
        except Exception:
            logger.exception(f"{patient_id}/{session_dir.name}: processing failed, skipping.")
            if is_preop:
                logger.warning(
                    f"{patient_id}: preop exam failed, skipping registration of follow-ups."
                )
                return
            continue

        if not is_preop:
            processed_followups.append((session_dir.name, exam_outdir))

    preop_outdir = outdir_root / patient_id / preop_dir.name
    if healthy_processed:
        try:
            register_healthy_to_preop(preop_outdir, healthy_outdir, override)
        except Exception:
            logger.exception(
                f"{patient_id}/{healthy_dir.name}: registration to preop failed, skipping."
            )
        else:
            try:
                warp_tissue_seg_to_preop(preop_outdir, healthy_outdir, override)
            except Exception:
                logger.exception(
                    f"{patient_id}/{healthy_dir.name}: warping tissue segmentation to preop "
                    "failed, skipping."
                )

    for session_name, exam_outdir in processed_followups:
        try:
            register_followup_to_preop(
                preop_outdir=preop_outdir,
                followup_outdir=exam_outdir,
                model_ids=list(growth_model_paths),
                override=override,
            )
        except Exception:
            logger.exception(
                f"{patient_id}/{session_name}: registration to preop failed, skipping."
            )


if __name__ == "__main__":
    # Processes the GB_healthy_preop_postop_recurrence cohort. Per patient, exams are ordered by
    # the date in their session directory name: the earliest exam is the healthy one and is
    # processed as in scripts/single_t1.py, the 2nd earliest is the pre-operative exam and all
    # later exams are follow-ups, processed as in scripts/single_nifti.py. Tumor growth is then
    # predicted on the pre-operative and follow-up exams with every given growth model. The
    # healthy exam and the follow-ups are finally affinely registered to pre-operative space,
    # which warps the follow-up predictions into preop space alongside the tumor segmentation
    # and the mri sequences. The tissue segmentation of the healthy exam is warped with the same
    # transform, so its gm/wm/csf probability maps are available in pre-operative space too.
    #
    # The growth prediction and the tissue warping can also be deferred and run once all other
    # steps are done, in which case the transform stored by the earlier registration is reused
    # instead of registering again. The deferred tissue stage additionally recomputes the tissue
    # segmentation of the healthy exams first, so that a change to the segmentation can be applied
    # without redoing any of the preprocessing.
    #
    # Examples:
    # nohup python -u scripts/preprocess_marco_healthy.py -cuda_device 0 > tmp_marco_healthy.out 2>&1 &
    # nohup python -u scripts/preprocess_marco_healthy.py -cuda_device 0 -stage preprocess > tmp_marco_healthy.out 2>&1 &
    # nohup python -u scripts/preprocess_marco_healthy.py -cuda_device 0 -stage predict > tmp_marco_healthy_predict.out 2>&1 &
    # nohup python -u scripts/preprocess_marco_healthy.py -stage warp_tissue > tmp_marco_healthy_warp_tissue.out 2>&1 &
    parser = argparse.ArgumentParser()
    parser.add_argument("-cuda_device", type=str, default="0", help="GPU id to run on.")
    parser.add_argument(
        "-datadir",
        type=str,
        default="/mnt/Drive3/marco/data/GB_healthy_preop_postop_recurrence/original",
        help="Directory containing one directory per patient, each with dated session directories.",
    )
    parser.add_argument(
        "-outdir",
        type=str,
        default="/mnt/Drive3/marco/data/GB_healthy_preop_postop_recurrence/preprocessed",
        help="Directory to save processed output to.",
    )
    parser.add_argument(
        "-atlas",
        type=str,
        default="sri24",
        choices=sorted(SUPPORTED_ATLASES),
        help="Atlas to register the exams into.",
    )
    parser.add_argument(
        "-patients",
        type=str,
        nargs="*",
        default=None,
        help="Optional subset of patient directory names to process. Defaults to all.",
    )
    parser.add_argument(
        "-growth_models",
        type=str,
        nargs="*",
        default=DEFAULT_GROWTH_MODEL_PATHS,
        help="Growth model docker images (*.tar) to predict with. The file stem is used as model id.",
    )
    parser.add_argument(
        "-stage",
        type=str,
        default="all",
        choices=("all", "preprocess", "predict", "warp_tissue", "warp_tissue_atropos"),
        help=(
            "Which steps to run. 'preprocess' stops after the longitudinal registration and runs "
            "no growth model, 'predict' only runs the growth models on the already preprocessed "
            "preop exam, 'warp_tissue' recomputes the "
            "tissue segmentation of already preprocessed healthy exams and warps it into preop "
            "space, 'warp_tissue_atropos' does the same with antsAtroposN4 instead of atlas "
            f"registration and stores its output in '{ATROPOS_TISSUE_SEG_FOLDER}' and "
            f"'{ATROPOS_LONGITUDINAL_FOLDER}', beside the atlas registration one rather than "
            "replacing it, 'all' runs the full pipeline in one go."
        ),
    )
    parser.add_argument(
        "-warp_only",
        action="store_true",
        help=(
            "Only used with -stage warp_tissue and -stage warp_tissue_atropos: skip the tissue "
            "segmentation and only warp the existing probability maps of the healthy exams into "
            "preop space."
        ),
    )
    parser.add_argument(
        "-override",
        action="store_true",
        help="Rerun every step even if its output already exists, overwriting previous results.",
    )
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device

    # No growth model is loaded for the stages that do not predict; process_exam and
    # register_followup_to_preop then simply have no prediction to run and to warp.
    growth_model_paths = {}
    if args.stage in ("all", "predict"):
        for growth_model in args.growth_models:
            growth_model_path = Path(growth_model)
            if not growth_model_path.is_file():
                logger.warning(
                    f"Growth model {growth_model_path} not found, skipping model."
                )
                continue
            growth_model_paths[growth_model_path.stem] = growth_model_path
        logger.info(f"Predicting with growth models {sorted(growth_model_paths)}.")

    datadir = Path(args.datadir)
    outdir_root = Path(args.outdir)
    outdir_root.mkdir(parents=True, exist_ok=True)

    # The prediction and tissue warping stages run on preprocessed output only, so their patients
    # come from outdir rather than from the original data.
    source_dir = (
        outdir_root
        if args.stage in ("predict", "warp_tissue", "warp_tissue_atropos")
        else datadir
    )
    patient_dirs = sorted(p for p in source_dir.iterdir() if p.is_dir())
    if args.patients:
        selected = set(args.patients)
        patient_dirs = [p for p in patient_dirs if p.name in selected]
        missing = selected - {p.name for p in patient_dirs}
        if missing:
            logger.warning(f"Patients not found in {source_dir}: {sorted(missing)}.")
    logger.info(
        f"Running stage '{args.stage}' for {len(patient_dirs)} patients from {source_dir}."
    )

    for patient_dir in patient_dirs:
        try:
            if args.stage == "warp_tissue":
                recompute_and_warp_tissue_seg_patient(
                    patient_outdir=patient_dir,
                    atlas=args.atlas,
                    warp_only=args.warp_only,
                )
            elif args.stage == "warp_tissue_atropos":
                recompute_and_warp_atropos_tissue_seg_patient(
                    patient_outdir=patient_dir,
                    warp_only=args.warp_only,
                )
            elif args.stage == "predict":
                predict_patient(
                    patient_outdir=patient_dir,
                    growth_model_paths=growth_model_paths,
                    cuda_device=args.cuda_device,
                    override=args.override,
                )
            else:
                process_patient(
                    patient_dir=patient_dir,
                    outdir_root=outdir_root,
                    atlas=args.atlas,
                    cuda_device=args.cuda_device,
                    growth_model_paths=growth_model_paths,
                    override=args.override,
                )
        except Exception:
            logger.exception(f"{patient_dir.name}: processing failed, skipping patient.")

    logger.info(f"Finished processing. Results saved to {outdir_root}.")
