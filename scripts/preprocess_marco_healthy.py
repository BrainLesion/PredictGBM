import os
import re
import json
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
from predict_gbm.utils.constants import (
    ATLAS_UNSTRIPPED_SCHEMA,
    MODALITY_STRIPPED_SCHEMA,
    RECURRENCE_SCHEMA,
    REGISTRATION_MASK_SCHEMA,
    TISSUE_PBMAP_SCHEMA,
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
    by antsAtroposN4 tissue segmentation. If override is True, all steps are rerun even if their
    outputs already exist.
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

    # Tissue segmentation. antsAtroposN4 uses the atlas tissue probability maps as spatial priors
    # without registering them, so the atlas has to match the space the exam was registered into.
    if tissueseg_done(outdir) and not override:
        logger.info(f"{outdir}: tissue segmentation already done, skipping.")
    else:
        run_tissue_seg(
            t1_file=t1_stripped_file,
            outdir=outdir,
            algorithm="antsAtroposN4",
            atlas=atlas,
        )

    logger.info(f"Finished single-T1 preprocessing. Output saved to {outdir}.")


def tissueseg_done(outdir: Path) -> bool:
    """True if the gm/wm/csf probability maps already exist for an exam."""
    return all(
        TISSUE_PBMAP_SCHEMA.format(base_dir=outdir, tissue=tissue).exists()
        for tissue in ("gm", "wm", "csf")
    )


def process_exam(
    modalities: Dict[str, Path],
    outdir: Path,
    atlas: str,
    cuda_device: str,
    override: bool = False,
) -> None:
    """
    Processes a pre-operative or follow-up exam: normalization, skull stripping and atlas
    co-registration, tumor segmentation via BRATS and tissue segmentation via atlas registration
    with the tumor core masked out of the registration metric. If override is True, all steps are
    rerun even if their outputs already exist.
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


def register_followup_to_preop(
    preop_outdir: Path, followup_outdir: Path, override: bool = False
) -> None:
    """
    Registers a follow-up exam to pre-operative space, warping the tumor segmentation and all
    remaining modalities (t1, t2, flair) alongside the t1c used to drive the registration.
    """
    recurrence_file = RECURRENCE_SCHEMA.format(base_dir=followup_outdir)
    if recurrence_file.exists() and not override:
        logger.info(f"{followup_outdir}: registration to preop already done, skipping.")
        return

    additional_modalities = {
        modality: MODALITY_STRIPPED_SCHEMA.format(
            base_dir=followup_outdir, modality=modality
        )
        for modality in ("t1", "t2", "flair")
    }
    register_recurrence(
        t1c_pre_file=MODALITY_STRIPPED_SCHEMA.format(
            base_dir=preop_outdir, modality="t1c"
        ),
        t1c_post_file=MODALITY_STRIPPED_SCHEMA.format(
            base_dir=followup_outdir, modality="t1c"
        ),
        recurrence_seg_file=TUMORSEG_SCHEMA.format(base_dir=followup_outdir),
        outdir=followup_outdir,
        additional_modalities=additional_modalities,
    )


def process_patient(
    patient_dir: Path,
    outdir_root: Path,
    atlas: str,
    cuda_device: str,
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

    healthy_t1_file = resolve_t1(healthy_dir)
    if healthy_t1_file is None:
        logger.warning(
            f"{patient_id}/{healthy_dir.name}: no t1 series for the healthy exam, skipping exam."
        )
    else:
        try:
            process_healthy_exam(
                t1_file=healthy_t1_file,
                outdir=outdir_root / patient_id / healthy_dir.name,
                atlas=atlas,
                override=override,
            )
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
            process_exam(modalities, exam_outdir, atlas, cuda_device, override)
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
    for session_name, exam_outdir in processed_followups:
        try:
            register_followup_to_preop(preop_outdir, exam_outdir, override)
        except Exception:
            logger.exception(
                f"{patient_id}/{session_name}: registration to preop failed, skipping."
            )


if __name__ == "__main__":
    # Processes the GB_healthy_preop_postop_recurrence cohort. Per patient, exams are ordered by
    # the date in their session directory name: the earliest exam is the healthy one and is
    # processed as in scripts/single_t1.py, the 2nd earliest is the pre-operative exam and all
    # later exams are follow-ups, processed as in scripts/single_nifti.py and registered to
    # pre-operative space.
    #
    # Example:
    # nohup python -u scripts/preprocess_marco_healthy.py -cuda_device 0 > tmp_marco_healthy.out 2>&1 &
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
        default="/mnt/Drive3/marco/data/GB_healthy_preop_postop_recurrence/processed",
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
        "-override",
        action="store_true",
        help="Rerun every step even if its output already exists, overwriting previous results.",
    )
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device

    datadir = Path(args.datadir)
    outdir_root = Path(args.outdir)
    outdir_root.mkdir(parents=True, exist_ok=True)

    patient_dirs = sorted(p for p in datadir.iterdir() if p.is_dir())
    if args.patients:
        selected = set(args.patients)
        patient_dirs = [p for p in patient_dirs if p.name in selected]
        missing = selected - {p.name for p in patient_dirs}
        if missing:
            logger.warning(f"Patients not found in {datadir}: {sorted(missing)}.")
    logger.info(f"Processing {len(patient_dirs)} patients from {datadir}.")

    for patient_dir in patient_dirs:
        try:
            process_patient(
                patient_dir=patient_dir,
                outdir_root=outdir_root,
                atlas=args.atlas,
                cuda_device=args.cuda_device,
                override=args.override,
            )
        except Exception:
            logger.exception(f"{patient_dir.name}: processing failed, skipping patient.")

    logger.info(f"Finished processing. Results saved to {outdir_root}.")
