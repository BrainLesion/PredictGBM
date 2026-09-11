import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from loguru import logger

# The processing steps are exactly the ones of scripts/preprocess_marco_healthy.py, so they are
# imported from there rather than copied; only the cohort handling differs (see the module docs
# in __main__). scripts/ is not a package, hence the sys.path entry.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from preprocess_marco_healthy import (  # noqa: E402
    ATROPOS_LONGITUDINAL_FOLDER,
    ATROPOS_TISSUE_SEG_FOLDER,
    DEFAULT_GROWTH_MODEL_PATHS,
    T1_LIKE_DESCRIPTION_PATTERN,
    clear_tissue_seg,
    collect_sessions,
    predict_growth,
    process_exam,
    process_healthy_exam,
    read_series_description,
    register_followup_to_preop,
    register_healthy_to_preop,
    run_atropos_tissue_seg,
    tissueseg_done,
    warp_atropos_tissue_seg_to_preop,
    warp_tissue_seg_to_preop,
)
from predict_gbm.preprocessing import run_tissue_seg  # noqa: E402
from predict_gbm.preprocessing.norm_ss_coregistration import SUPPORTED_ATLASES  # noqa: E402
from predict_gbm.utils.constants import (  # noqa: E402
    MODALITY_STRIPPED_SCHEMA,
    TUMORSEG_SCHEMA,
)

# Patients whose pre-operative exam was already preprocessed elsewhere (by
# scripts/preprocess_marco_healthy.py) and is used in place: the healthy exam is registered to
# the skull-stripped t1c found there and nothing is ever written into that directory. Such a
# patient's sessions in datadir are all healthy or follow-up exams, i.e. no preop is taken from
# the date ordering. The preop directory must hold the skull-stripped t1c; for the growth
# prediction and the follow-up registration it additionally needs the tumor segmentation, the
# tissue probability maps and the predictions, all of which the original script produces.
EXTERNAL_PREOP_DIRS = {
    "sub-GR067": Path(
        "/home/home/marco/data/GB_healthy_preop_postop_recurrence/preprocessed/sub-GR067/ses-20220310"
    ),
}

# Explicit series choice for sessions that carry several files of one modality, keyed by
# (patient, session, modality). Both tumor board exams have two t1c series; the ones listed here
# are the 3D acquisitions (ax T1 FSPGR 3D KM, AI_T1 3D sag KM), the other ones are a 2D coronal
# orbit Dixon and a 2D coronal 5mm series. A missing override falls back to the first file in
# sorted order with a warning, as in the original script.
SERIES_OVERRIDES = {
    ("sub-Tumorboardexample", "ses-20250827", "t1c"): (
        "sub-Tumorboardexample_ses-20250827_sequ-10_t1c.nii.gz"
    ),
    ("sub-Tumorboardexample", "ses-20260831", "t1c"): (
        "sub-Tumorboardexample_ses-20260831_sequ-901_t1c.nii.gz"
    ),
}

# Only directories with this prefix are patients; datadir also contains the conversion logs and,
# by default, the output directory itself.
PATIENT_DIR_PREFIX = "sub-"


def find_modality_file(session_dir: Path, modality: str) -> Optional[Path]:
    """
    Returns the nifti of the given modality in a session directory, or None if absent. A series
    listed in SERIES_OVERRIDES takes precedence and must exist; otherwise the first file in sorted
    order is used, with a warning if there are several.
    """
    override = SERIES_OVERRIDES.get((session_dir.parent.name, session_dir.name, modality))
    if override is not None:
        override_file = session_dir / override
        if not override_file.exists():
            raise FileNotFoundError(
                f"{session_dir}: series override {override} for modality {modality} not found."
            )
        logger.info(f"{session_dir}: using series override {override} for modality {modality}.")
        return override_file

    matches = sorted(session_dir.glob(f"*_{modality}.nii.gz"))
    if not matches:
        return None
    if len(matches) > 1:
        logger.warning(
            f"{session_dir}: {len(matches)} candidates for modality {modality}, using {matches[0].name}."
        )
    return matches[0]


def resolve_t1(session_dir: Path) -> Optional[Path]:
    """
    Returns the t1 nifti of a session, falling back to a T1-weighted '_mr' series. Same rule as
    in the original script, repeated here so that SERIES_OVERRIDES are honored.
    """
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


def resolve_healthy_t1(session_dir: Path) -> Optional[Path]:
    """
    Returns the image to drive the single-T1 healthy pipeline with: the t1 (or a T1-weighted
    '_mr' series), falling back to the t1c if neither exists. The tumor board healthy exam only
    has a post-contrast 3D T1, which is why the fallback is needed; it mirrors the
    t1-missing-use-t1c rule of the preop pipeline. The image is still processed under the
    modality name t1, so all output file names stay the ones of a healthy exam.
    """
    t1_file = resolve_t1(session_dir)
    if t1_file is not None:
        return t1_file

    t1c_file = find_modality_file(session_dir, "t1c")
    if t1c_file is not None:
        logger.warning(
            f"{session_dir}: no t1 series for the healthy exam, using {t1c_file.name} (t1c) instead."
        )
    return t1c_file


def resolve_modalities(session_dir: Path) -> Optional[Dict[str, Path]]:
    """
    Resolves t1/t1c/t2/flair paths for a preop or follow-up exam with the same fallback rules as
    the original script: t1 missing -> use t1c; t2 missing -> use flair; flair missing -> use t2.
    Returns None if the exam should be excluded, i.e. if t1c is missing or neither t2 nor flair
    is available.
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


def resolve_healthy_and_preop(patient_outdir: Path) -> Optional[Tuple[Path, List[Path]]]:
    """
    Returns the pre-operative exam directory of a patient together with its healthy exam
    directories. The healthy exams come from the session_roles.json written during
    preprocessing; the preop is the external one for patients in EXTERNAL_PREOP_DIRS and the
    one recorded in session_roles.json otherwise. Returns None if the patient was not
    preprocessed or has no pre-operative exam.
    """
    roles_file = patient_outdir / "session_roles.json"
    if not roles_file.exists():
        logger.warning(
            f"{patient_outdir}: no {roles_file.name}, patient not preprocessed, skipping patient."
        )
        return None
    with roles_file.open("r") as f:
        session_roles = json.load(f)

    external_preop = EXTERNAL_PREOP_DIRS.get(patient_outdir.name)
    if external_preop is not None:
        preop_outdir = external_preop
    else:
        preop_sessions = [name for name, role in session_roles.items() if role == "preop"]
        if not preop_sessions:
            logger.warning(f"{patient_outdir}: no preop exam recorded, skipping patient.")
            return None
        preop_outdir = patient_outdir / preop_sessions[0]

    healthy_outdirs = [
        patient_outdir / name
        for name, role in session_roles.items()
        if role == "healthy"
    ]
    return preop_outdir, healthy_outdirs


def recompute_and_warp_atropos_tissue_seg_patient(
    patient_outdir: Path, warp_only: bool = False
) -> None:
    """
    Runs the antsAtroposN4 tissue segmentation on the already preprocessed healthy exams of a
    single patient and warps the resulting probability maps into pre-operative space, storing both
    beside the atlas registration results rather than replacing them. Used with
    -stage warp_tissue_atropos. Same as in the original script, except that the preop is resolved
    through EXTERNAL_PREOP_DIRS where configured.
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
    Recomputes the atlas registration tissue segmentation of the already preprocessed healthy
    exams of a single patient and warps the resulting probability maps into pre-operative space.
    Used with -stage warp_tissue. Same as in the original script, except that the preop is
    resolved through EXTERNAL_PREOP_DIRS where configured.
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
    Used with -stage predict. Patients with an external preop are skipped, since their preop was
    predicted on by the original script and this script never writes into that directory.
    """
    if patient_outdir.name in EXTERNAL_PREOP_DIRS:
        logger.info(
            f"{patient_outdir}: preop is external ({EXTERNAL_PREOP_DIRS[patient_outdir.name]}), "
            "not predicting on it."
        )
        return

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
    """
    Processes all exams of a single patient and registers the healthy exam and the follow-ups to
    preop space. Exams are ordered by the date in their session directory name. Without an
    external preop, the roles are healthy / preop / follow-ups as in the original script. With an
    external preop (EXTERNAL_PREOP_DIRS), the earliest exam is the healthy one and every later
    exam is a follow-up; the preop itself is used in place and never processed or written to.
    """
    patient_id = patient_dir.name
    sessions = collect_sessions(patient_dir)
    if not sessions:
        logger.warning(f"{patient_id}: no dated session directories found, skipping patient.")
        return

    external_preop = EXTERNAL_PREOP_DIRS.get(patient_id)
    healthy_dir = sessions[0][1]
    session_roles = {healthy_dir.name: "healthy"}

    healthy_outdir = outdir_root / patient_id / healthy_dir.name
    healthy_processed = False
    healthy_t1_file = resolve_healthy_t1(healthy_dir)
    if healthy_t1_file is None:
        logger.warning(
            f"{patient_id}/{healthy_dir.name}: no t1 or t1c series for the healthy exam, skipping exam."
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

    if external_preop is not None:
        preop_dir = None
        preop_outdir = external_preop
        followup_dirs = [session_dir for _, session_dir in sessions[1:]]
        preop_t1c_file = MODALITY_STRIPPED_SCHEMA.format(base_dir=preop_outdir, modality="t1c")
        if not preop_t1c_file.exists():
            logger.warning(
                f"{patient_id}: external preop has no skull-stripped t1c ({preop_t1c_file}), "
                "skipping registration to preop."
            )
            return
    else:
        if len(sessions) < 2:
            logger.warning(
                f"{patient_id}: only one exam available, no preop exam to process, skipping patient."
            )
            return
        preop_dir = sessions[1][1]
        preop_outdir = outdir_root / patient_id / preop_dir.name
        followup_dirs = [session_dir for _, session_dir in sessions[2:]]
        session_roles[preop_dir.name] = "preop"

    session_roles.update({session_dir.name: "followup" for session_dir in followup_dirs})
    logger.info(
        f"{patient_id}: healthy={healthy_dir.name}, "
        f"preop={preop_outdir if external_preop is not None else preop_dir.name}, "
        f"followups={[session_dir.name for session_dir in followup_dirs]}."
    )

    roles_file = outdir_root / patient_id / "session_roles.json"
    roles_file.parent.mkdir(parents=True, exist_ok=True)
    with roles_file.open("w") as f:
        json.dump(session_roles, f, indent=4)

    processed_followups = []
    exam_dirs = followup_dirs if preop_dir is None else [preop_dir] + followup_dirs
    for session_dir in exam_dirs:
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
    # Processes the additional patients converted to /mnt/Drive4/lucas/niftis_v2 with the same
    # steps as scripts/preprocess_marco_healthy.py (see there for the pipeline): the earliest exam
    # of a patient is the healthy one and is processed as a single-T1 exam, the pre-operative exam
    # gets the full preprocessing, tumor and tissue segmentation and the growth prediction, and
    # the healthy exam is affinely registered to preop space together with its gm/wm/csf
    # probability maps. The differences to the original script are:
    #
    # - sub-GR067: only a new, earlier healthy exam (ses-20180306) is in datadir. Its preop is the
    #   already preprocessed ses-20220310 in Marco's directory (EXTERNAL_PREOP_DIRS), which is
    #   used in place for the registration and never written to.
    # - sub-Tumorboardexample: ses-20250827 is the healthy exam, ses-20260831 the preop. The
    #   healthy exam has no native t1, so its 3D post-contrast t1c drives the single-T1 pipeline.
    #   Both sessions carry two t1c series; the 3D ones are selected via SERIES_OVERRIDES.
    #
    # Examples:
    # nohup python -u scripts/preprocess_marco_healthy_2.py -cuda_device 0 > tmp_marco_healthy_2.out 2>&1 &
    # nohup python -u scripts/preprocess_marco_healthy_2.py -cuda_device 0 -stage preprocess > tmp_marco_healthy_2.out 2>&1 &
    # nohup python -u scripts/preprocess_marco_healthy_2.py -cuda_device 0 -stage predict > tmp_marco_healthy_2_predict.out 2>&1 &
    # nohup python -u scripts/preprocess_marco_healthy_2.py -stage warp_tissue > tmp_marco_healthy_2_warp_tissue.out 2>&1 &
    parser = argparse.ArgumentParser()
    parser.add_argument("-cuda_device", type=str, default="0", help="GPU id to run on.")
    parser.add_argument(
        "-datadir",
        type=str,
        default="/mnt/Drive4/lucas/niftis_v2",
        help="Directory containing one sub-* directory per patient, each with dated session directories.",
    )
    parser.add_argument(
        "-outdir",
        type=str,
        default="/mnt/Drive4/lucas/niftis_v2/processed",
        help="Directory to save processed output to.",
    )
    parser.add_argument(
        "-atlas",
        type=str,
        default="sri24",
        choices=sorted(SUPPORTED_ATLASES),
        help="Atlas to register the exams into. Must match the one the external preops were processed with.",
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
            "preop exam (skipped for patients with an external preop), 'warp_tissue' recomputes "
            "the tissue segmentation of already preprocessed healthy exams and warps it into "
            "preop space, 'warp_tissue_atropos' does the same with antsAtroposN4 instead of atlas "
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

    source_dir = (
        outdir_root
        if args.stage in ("predict", "warp_tissue", "warp_tissue_atropos")
        else datadir
    )
    patient_dirs = sorted(
        p
        for p in source_dir.iterdir()
        if p.is_dir() and p.name.startswith(PATIENT_DIR_PREFIX)
    )
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
