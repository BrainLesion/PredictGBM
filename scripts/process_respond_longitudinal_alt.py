import os
import shutil
import tempfile
import argparse
from pathlib import Path
from typing import Dict, Optional
import ants
import numpy as np
import nibabel as nib
from loguru import logger
from brainles_preprocessing.brain_extraction.synthstrip import SynthStripExtractor
from brainles_preprocessing.modality import CenterModality, Modality
from brainles_preprocessing.normalization.percentile_normalizer import (
    PercentileNormalizer,
)
from brainles_preprocessing.preprocessor import (
    AtlasCentricPreprocessor,
    NativeSpacePreprocessor,
)
from predict_gbm.utils.constants import (
    ATLAS_UNSTRIPPED_SCHEMA,
    BRAIN_MASK_SCHEMA,
    BRAINLES_LOGFILE_SCHEMA,
    CONFIG_STEP_NORM_SS_COREGISTER,
    LONGITUDINAL_DIR,
    MODALITY_STRIPPED_SCHEMA,
    OUTPUT_BASE_SCHEMA,
    PathSchema,
    RECURRENCE_LABELS,
    RECURRENCE_SCHEMA,
    REGISTRATION_MASK_SCHEMA,
    SKULL_STRIP_ALGORITHM_SYNTHSTRIP,
    TISSUE_PBMAP_SCHEMA,
    TUMOR_SEGMENTATION_FOLDER,
    TUMORSEG_SCHEMA,
)
from predict_gbm.preprocessing import register_recurrence, run_brats, run_tissue_seg
from predict_gbm.preprocessing.norm_ss_coregistration import LinearANTsRegistrator
from predict_gbm.preprocessing.tissue_segmentation import generate_registration_mask
from predict_gbm.utils.utils import update_config
from process_respond_longitudinal import (
    DEFAULT_POSTOP_WINDOW_DAYS,
    LONGITUDINAL_TISSUES,
    MODALITIES,
    PATIENT_DIR_GLOB,
    SESSION_DIR_GLOB,
    SESSION_LABELS_FILENAME,
    TIMEPOINT_PREOP,
    assign_timepoints,
    session_date,
    write_session_labels,
)

# Alternative to process_respond_longitudinal.py that starts from the original-space
# (*space-org*) images instead of the provided skull-stripped SRI-space ones. Per exam:
# 1. atlas registration with brainles (t1c center, SRI24 unstripped atlas), without skull
#    stripping and without normalization,
# 2. postop/followup only: rigid co-registration of t1c to the preop t1c (both with skull,
#    in atlas space), applied to all modalities,
# 3. skull stripping of t1c with brainles' NativeSpacePreprocessor (no atlas registration)
#    and percentile normalization; the t1c brain mask is applied to t1/t2/flair,
# 4. tumor segmentation with BraTS25_1 (preop branch for the preop exam, post branch
#    otherwise); postop/followup: background voxels inside the preop brain mask but
#    outside the exam's own brain mask become cavity,
# 5. atlas-based tissue segmentation with the tumor core masked.
# Every postop/followup exam is then registered to the preop exam with ANTs SyN, necrosis,
# enhancing tumor and cavity (labels 1, 3, 4) excluded from the metric in the moving image.
ATLAS = "sri24"
SPACE = "org"
DEFAULT_OUTDIR_NAME = "processed_alt"
# Only these sessions are processed (tum_149: the preop and the first postop exam);
# timepoints are still assigned from all MRI sessions.
PROCESSED_SESSIONS = ("ses-20210303", "ses-20210312")

# <patient_dir>/<session>/sub-<subject>_<session>_space-org_<modality>.nii.gz
ORG_FILE_SCHEMA = PathSchema(
    "{patient_dir}/{session}/sub-{subject}_{session}_space-" + SPACE + "_{suffix}.nii.gz"
)

# Intermediate outputs: atlas-registered images with skull (brainles transforms and log
# next to them), and for postop/followup exams the same images rigidly co-registered to
# the preop exam together with the rigid transform.
ATLAS_REGISTERED_FOLDER = "atlas_registered"
ATLAS_REGISTERED_SCHEMA = (
    OUTPUT_BASE_SCHEMA / ATLAS_REGISTERED_FOLDER / "{modality}.nii.gz"
)
ATLAS_REGISTRATION_TRAFO_SCHEMA = (
    OUTPUT_BASE_SCHEMA / ATLAS_REGISTERED_FOLDER / "transformations"
)
ATLAS_REGISTRATION_LOGFILE_SCHEMA = (
    OUTPUT_BASE_SCHEMA / ATLAS_REGISTERED_FOLDER / "brainles.log"
)
PREOP_COREGISTERED_FOLDER = "preop_coregistered"
PREOP_COREGISTERED_SCHEMA = (
    OUTPUT_BASE_SCHEMA / PREOP_COREGISTERED_FOLDER / "{modality}.nii.gz"
)
PREOP_RIGID_TRAFO_SCHEMA = (
    OUTPUT_BASE_SCHEMA / PREOP_COREGISTERED_FOLDER / "rigid_to_preop.mat"
)
PREOP_COREGISTRATION_TRANSFORM = "Rigid"

# BraTS output of postop/followup exams before the cavity fill; tumor_seg.nii.gz holds the
# filled segmentation.
TUMORSEG_UNFILLED_SCHEMA = (
    OUTPUT_BASE_SCHEMA / TUMOR_SEGMENTATION_FOLDER / "tumor_seg_brats.nii.gz"
)
CAVITY_LABEL = RECURRENCE_LABELS["cavity"]
CONFIG_STEP_CAVITY_FILL = "cavity_fill"
# Moving mask of the SyN registration: 1 everywhere except in these labels of the
# exam's tumor segmentation.
SYN_EXCLUDED_LABELS = (
    RECURRENCE_LABELS["necrotic"],
    RECURRENCE_LABELS["enhancing"],
    RECURRENCE_LABELS["cavity"],
)
SYN_MOVING_MASK_SCHEMA = OUTPUT_BASE_SCHEMA / LONGITUDINAL_DIR / "syn_moving_mask.nii.gz"
LONGITUDINAL_REGISTRATION_ALGORITHM = "syn"


def subject_id(patient_dir: Path) -> Optional[str]:
    """
    Subject id as used in the file names (sub-<subject>_...), read from the first
    original-space file found in the patient directory. Returns None if there is none.
    """
    for f in sorted(patient_dir.glob(f"{SESSION_DIR_GLOB}/sub-*_space-{SPACE}_*.nii.gz")):
        return f.name.split("_")[0].removeprefix("sub-")
    return None


def resolve_modalities(
    patient_dir: Path, subject: str, session: str
) -> Optional[Dict[str, Path]]:
    """
    Resolves the original-space t1/t1c/t2/flair files of a session. t2 missing -> use
    flair. Returns None (with a warning) if t1 or flair is missing, or if the session has
    no t1c at all (PET/CT-only sessions, silently).
    """
    files = {
        modality: ORG_FILE_SCHEMA.format(
            patient_dir=patient_dir, subject=subject, session=session, suffix=modality
        )
        for modality in MODALITIES
    }
    if not files["t1c"].exists():
        return None
    missing = [m for m in ("t1", "flair") if not files[m].exists()]
    if missing:
        logger.warning(f"{patient_dir.name}/{session}: missing {missing}, skipping exam.")
        return None
    if not files["t2"].exists():
        logger.info(f"{patient_dir.name}/{session}: no t2, using flair as t2.")
        files["t2"] = files["flair"]
    return files


def register_to_atlas(modalities: Dict[str, Path], outdir: Path) -> None:
    """
    Registers t1c to the SRI24 unstripped atlas with brainles (t1/t2/flair co-registered
    to t1c first, atlas correction on, as in norm_ss_coregister) and stores all four
    modalities in atlas space with skull and unnormalized as
    <outdir>/atlas_registered/<modality>.nii.gz.
    """
    center = CenterModality(
        modality_name="t1c",
        input_path=str(modalities["t1c"]),
        raw_skull_output_path=str(
            ATLAS_REGISTERED_SCHEMA.format(base_dir=outdir, modality="t1c")
        ),
    )
    moving = [
        Modality(
            modality_name=modality,
            input_path=str(modalities[modality]),
            raw_skull_output_path=str(
                ATLAS_REGISTERED_SCHEMA.format(base_dir=outdir, modality=modality)
            ),
        )
        for modality in MODALITIES
        if modality != "t1c"
    ]
    preprocessor = AtlasCentricPreprocessor(
        center_modality=center,
        moving_modalities=moving,
        registrator=LinearANTsRegistrator(transformation_params={"defaultvalue": 0}),
        atlas_image_path=ATLAS_UNSTRIPPED_SCHEMA.format(atlas=ATLAS),
    )
    preprocessor.run(
        log_file=ATLAS_REGISTRATION_LOGFILE_SCHEMA.format(base_dir=outdir),
        save_dir_transformations=ATLAS_REGISTRATION_TRAFO_SCHEMA.format(base_dir=outdir),
    )


def coregister_to_preop(outdir: Path, preop_outdir: Path) -> None:
    """
    Rigidly registers the atlas-space t1c of an exam to the atlas-space t1c of the preop
    exam (both with skull) and resamples all four modalities with that transform into
    <outdir>/preop_coregistered/<modality>.nii.gz. The transform is stored as
    <outdir>/preop_coregistered/rigid_to_preop.mat.
    """
    fixed = ants.image_read(
        str(ATLAS_REGISTERED_SCHEMA.format(base_dir=preop_outdir, modality="t1c"))
    )
    moving = ants.image_read(
        str(ATLAS_REGISTERED_SCHEMA.format(base_dir=outdir, modality="t1c"))
    )
    trafo_file = PREOP_RIGID_TRAFO_SCHEMA.format(base_dir=outdir)
    trafo_file.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="coregister_to_preop_") as tmpdir:
        reg = ants.registration(
            fixed=fixed,
            moving=moving,
            type_of_transform=PREOP_COREGISTRATION_TRANSFORM,
            outprefix=f"{tmpdir}/rigid_",
        )
        shutil.move(reg["fwdtransforms"][0], str(trafo_file))

    for modality in MODALITIES:
        warped = ants.apply_transforms(
            fixed=fixed,
            moving=ants.image_read(
                str(ATLAS_REGISTERED_SCHEMA.format(base_dir=outdir, modality=modality))
            ),
            transformlist=[str(trafo_file)],
            interpolator="linear",
            defaultvalue=0,
        )
        ants.image_write(
            warped, str(PREOP_COREGISTERED_SCHEMA.format(base_dir=outdir, modality=modality))
        )
    logger.info(f"{outdir}: rigidly co-registered to {preop_outdir}.")


def skull_strip(modality_files: Dict[str, Path], outdir: Path) -> None:
    """
    Skull strips t1c with SynthStrip through brainles' NativeSpacePreprocessor (t1c as the
    only modality, so nothing is re-registered), applies the t1c brain mask to t1/t2/flair
    and stores all four percentile-normalized as <outdir>/skull_stripped/<modality>_
    skullstripped.nii.gz, the mask as <outdir>/skull_stripped/brain_mask.nii.gz.
    """
    percentile_normalizer = PercentileNormalizer(
        lower_percentile=0.1,
        upper_percentile=99.9,
        lower_limit=0,
        upper_limit=1,
    )
    brain_mask_file = BRAIN_MASK_SCHEMA.format(base_dir=outdir)
    center = CenterModality(
        modality_name="t1c",
        input_path=str(modality_files["t1c"]),
        normalizer=percentile_normalizer,
        normalized_bet_output_path=str(
            MODALITY_STRIPPED_SCHEMA.format(base_dir=outdir, modality="t1c")
        ),
        bet_mask_output_path=str(brain_mask_file),
    )
    brain_extractor = SynthStripExtractor()
    preprocessor = NativeSpacePreprocessor(
        center_modality=center,
        moving_modalities=[],
        registrator=LinearANTsRegistrator(transformation_params={"defaultvalue": 0}),
        brain_extractor=brain_extractor,
    )
    preprocessor.run(log_file=BRAINLES_LOGFILE_SCHEMA.format(base_dir=outdir))

    # Same masking and normalization as brainles applies to moving modalities.
    with tempfile.TemporaryDirectory(prefix="skull_strip_") as bet_dir:
        for modality in MODALITIES:
            if modality == "t1c":
                continue
            m = Modality(
                modality_name=modality,
                input_path=str(modality_files[modality]),
                normalizer=percentile_normalizer,
                normalized_bet_output_path=str(
                    MODALITY_STRIPPED_SCHEMA.format(base_dir=outdir, modality=modality)
                ),
            )
            m.apply_bet_mask(
                brain_extractor=brain_extractor, mask_path=brain_mask_file, bet_dir=bet_dir
            )
            m.save_current_image(m.normalized_bet_output_path, normalization=True)


def fill_cavity(
    tumorseg_file: Path, brain_mask_file: Path, preop_brain_mask_file: Path, outfile: Path
) -> int:
    """
    Sets the cavity label in every voxel that is background in the tumor segmentation,
    outside the exam's brain mask and inside the preop brain mask; existing labels are
    never changed. Returns the number of filled voxels.
    """
    seg_img = nib.load(str(tumorseg_file))
    mask_img = nib.load(str(brain_mask_file))
    preop_mask_img = nib.load(str(preop_brain_mask_file))
    for name, img in (("brain mask", mask_img), ("preop brain mask", preop_mask_img)):
        if img.shape != seg_img.shape or not np.allclose(img.affine, seg_img.affine, atol=1e-3):
            raise ValueError(
                f"The {name} ({img.shape}) does not share the grid of the tumor "
                f"segmentation ({seg_img.shape})."
            )
    seg = np.rint(seg_img.get_fdata()).astype(np.int32)
    brain_mask = np.rint(mask_img.get_fdata()) > 0
    preop_brain_mask = np.rint(preop_mask_img.get_fdata()) > 0

    fill = preop_brain_mask & ~brain_mask & (seg == 0)
    cavity_before = int(np.sum(seg == CAVITY_LABEL))
    seg[fill] = CAVITY_LABEL
    outfile.parent.mkdir(parents=True, exist_ok=True)
    nib.save(
        nib.Nifti1Image(seg.astype(seg_img.get_data_dtype()), seg_img.affine, seg_img.header),
        str(outfile),
    )
    n_filled = int(np.sum(fill))
    logger.info(
        f"{outfile}: filled {n_filled} cavity voxels (label {CAVITY_LABEL} voxels "
        f"{cavity_before} -> {cavity_before + n_filled})."
    )
    return n_filled


def process_exam(
    modalities: Dict[str, Path],
    outdir: Path,
    preop_outdir: Optional[Path],
    cuda_device: str,
    override: bool = False,
) -> None:
    """
    Per-exam steps: atlas registration (skull on, unnormalized), rigid co-registration to
    the preop exam (postop/followup, i.e. preop_outdir given), skull stripping and
    normalization, BraTS25_1 tumor segmentation (+ cavity fill for postop/followup) and the
    atlas-based tissue segmentation with the tumor core masked. If override is True, all
    steps are rerun even if their outputs already exist.
    """
    outdir.mkdir(parents=True, exist_ok=True)
    is_preop = preop_outdir is None

    atlas_files = [ATLAS_REGISTERED_SCHEMA.format(base_dir=outdir, modality=m) for m in MODALITIES]
    if all(f.exists() for f in atlas_files) and not override:
        logger.info(f"{outdir}: atlas registration already done, skipping.")
    else:
        register_to_atlas(modalities, outdir)

    if is_preop:
        coregistered_schema = ATLAS_REGISTERED_SCHEMA
    else:
        coregistered_schema = PREOP_COREGISTERED_SCHEMA
        coregistered_files = [
            PREOP_COREGISTERED_SCHEMA.format(base_dir=outdir, modality=m) for m in MODALITIES
        ]
        if all(f.exists() for f in coregistered_files) and not override:
            logger.info(f"{outdir}: co-registration to preop already done, skipping.")
        else:
            coregister_to_preop(outdir, preop_outdir)

    t1c_stripped = MODALITY_STRIPPED_SCHEMA.format(base_dir=outdir, modality="t1c")
    if t1c_stripped.exists() and not override:
        logger.info(f"{outdir}: skull stripping already done, skipping.")
    else:
        skull_strip(
            {m: coregistered_schema.format(base_dir=outdir, modality=m) for m in MODALITIES},
            outdir,
        )
        update_config(
            outdir,
            CONFIG_STEP_NORM_SS_COREGISTER,
            {
                "atlas": ATLAS,
                "skull_strip_algorithm": SKULL_STRIP_ALGORITHM_SYNTHSTRIP,
                "preop_coregistration": None if is_preop else PREOP_COREGISTRATION_TRANSFORM,
            },
        )

    tumorseg_file = TUMORSEG_SCHEMA.format(base_dir=outdir)
    unfilled_file = TUMORSEG_UNFILLED_SCHEMA.format(base_dir=outdir)
    if tumorseg_file.exists() and (is_preop or unfilled_file.exists()) and not override:
        logger.info(f"{outdir}: tumor segmentation already done, skipping.")
    else:
        run_brats(
            t1_file=MODALITY_STRIPPED_SCHEMA.format(base_dir=outdir, modality="t1"),
            t1c_file=t1c_stripped,
            t2_file=MODALITY_STRIPPED_SCHEMA.format(base_dir=outdir, modality="t2"),
            flair_file=MODALITY_STRIPPED_SCHEMA.format(base_dir=outdir, modality="flair"),
            outdir=outdir,
            cuda_device=cuda_device,
            is_preop=is_preop,
        )
        if not is_preop:
            shutil.copyfile(tumorseg_file, unfilled_file)
            n_filled = fill_cavity(
                tumorseg_file=unfilled_file,
                brain_mask_file=BRAIN_MASK_SCHEMA.format(base_dir=outdir),
                preop_brain_mask_file=BRAIN_MASK_SCHEMA.format(base_dir=preop_outdir),
                outfile=tumorseg_file,
            )
            update_config(
                outdir,
                CONFIG_STEP_CAVITY_FILL,
                {"preop_dir": str(preop_outdir), "filled_voxels": n_filled},
            )

    tissueseg_files = [
        TISSUE_PBMAP_SCHEMA.format(base_dir=outdir, tissue=tissue)
        for tissue in LONGITUDINAL_TISSUES
    ]
    if all(f.exists() for f in tissueseg_files) and not override:
        logger.info(f"{outdir}: tissue segmentation already done, skipping.")
    else:
        registration_mask_file = REGISTRATION_MASK_SCHEMA.format(base_dir=outdir)
        generate_registration_mask(
            tumor_seg_file=tumorseg_file, outfile=registration_mask_file
        )
        run_tissue_seg(
            t1_file=t1c_stripped,
            outdir=outdir,
            registration_mask_file=registration_mask_file,
            atlas=ATLAS,
        )


def write_syn_moving_mask(tumorseg_file: Path, outfile: Path) -> int:
    """
    Writes the moving mask of the SyN registration: 1 everywhere except in the necrosis,
    enhancing tumor and cavity (SYN_EXCLUDED_LABELS) of the tumor segmentation. Returns
    the number of excluded voxels.
    """
    seg_img = nib.load(str(tumorseg_file))
    excluded = np.isin(np.rint(seg_img.get_fdata()), SYN_EXCLUDED_LABELS)
    outfile.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image((~excluded).astype(np.uint8), seg_img.affine), str(outfile))
    return int(np.sum(excluded))


def longitudinal_modalities(exam_outdir: Path) -> Dict[str, Path]:
    """
    Additional modalities of an exam that register_recurrence co-transforms to preop space:
    the normalized t1, t2 and flair and the gm/wm/csf tissue probability maps. Outputs are
    stored as <exam_outdir>/longitudinal/<name>_warped_longitudinal.nii.gz.
    """
    modalities = {
        modality: MODALITY_STRIPPED_SCHEMA.format(base_dir=exam_outdir, modality=modality)
        for modality in ("t1", "t2", "flair")
    }
    for tissue in LONGITUDINAL_TISSUES:
        modalities[f"{tissue}_pbmap"] = TISSUE_PBMAP_SCHEMA.format(
            base_dir=exam_outdir, tissue=tissue
        )
    return modalities


if __name__ == "__main__":
    # Full alternative processing of the original-space MRI sessions of every patient;
    # steps with existing outputs are skipped unless -override is given. Outputs go to
    # <patient_dir>/processed_alt/<session>/:
    # nohup python -u scripts/process_respond_longitudinal_alt.py -cuda_device 0 > tmp_process_respond_longitudinal_alt.out 2>&1 &
    # Only one patient:
    # python -u scripts/process_respond_longitudinal_alt.py -cuda_device 0 -patients tum_149
    # Recompute the postop/followup-to-preop registrations only (per-exam outputs reused):
    # python -u scripts/process_respond_longitudinal_alt.py -cuda_device 0 -longitudinal_registration
    parser = argparse.ArgumentParser()
    parser.add_argument("-cuda_device", type=str, default="0", help="GPU id to run on.")
    parser.add_argument(
        "-data_dir",
        type=str,
        default="/mnt/Drive4/lucas/respond_longitudinal",
        help="Directory containing one directory per patient (tum_*) with ses-YYYYMMDD "
        "session directories of original-space niftis.",
    )
    parser.add_argument(
        "-outdir_name",
        type=str,
        default=DEFAULT_OUTDIR_NAME,
        help="Name of the output directory created inside each patient directory.",
    )
    parser.add_argument(
        "-patients",
        type=str,
        nargs="+",
        default=None,
        help="Patient directory names to process (e.g. tum_149). All patients if omitted.",
    )
    parser.add_argument(
        "-postop_window_days",
        type=int,
        default=DEFAULT_POSTOP_WINDOW_DAYS,
        help="MRI sessions up to this many days after the postop session (the second MRI "
        "session) are labelled postop as well; later ones are followups.",
    )
    parser.add_argument(
        "-longitudinal_registration",
        action="store_true",
        help="Only recompute the longitudinal (postop/followup to preop) registration of "
        "every processed postop/followup exam, overwriting existing results. Existing "
        "outputs of the per-exam steps are reused.",
    )
    parser.add_argument(
        "-override",
        action="store_true",
        help="Rerun every step even if its output already exists, overwriting previous results.",
    )
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device

    data_dir = Path(args.data_dir)
    patient_dirs = sorted(p for p in data_dir.glob(PATIENT_DIR_GLOB) if p.is_dir())
    if args.patients is not None:
        patient_dirs = [p for p in patient_dirs if p.name in args.patients]
    logger.info(f"Found {len(patient_dirs)} patients in {data_dir}.")

    for patient_dir in patient_dirs:
        patient_id = patient_dir.name
        subject = subject_id(patient_dir)
        if subject is None:
            logger.warning(f"{patient_id}: no original-space files found, skipping patient.")
            continue
        outdir_root = patient_dir / args.outdir_name

        # MRI sessions: those with an original-space t1c (PET/CT-only sessions have none).
        sessions = [
            d.name
            for d in sorted(patient_dir.glob(SESSION_DIR_GLOB))
            if d.is_dir()
            and ORG_FILE_SCHEMA.format(
                patient_dir=patient_dir, subject=subject, session=d.name, suffix="t1c"
            ).exists()
        ]
        if len(sessions) < 2:
            logger.warning(
                f"{patient_id}: only {len(sessions)} MRI session(s), need preop and postop, "
                "skipping patient."
            )
            continue
        timepoints = assign_timepoints(sessions, args.postop_window_days)
        logger.info(
            f"{patient_id} (sub-{subject}): "
            + ", ".join(f"{s} {timepoints[s]}" for s in sorted(timepoints, key=session_date))
        )

        # The preop exam is processed first: the others are co-registered to it and their
        # cavity fill needs its brain mask. It is the first session by date.
        exam_outdirs: Dict[str, Path] = {}  # session -> outdir of processed exams
        preop_outdir: Optional[Path] = None
        for session in sorted(timepoints, key=session_date):
            if session not in PROCESSED_SESSIONS:
                continue
            is_preop = timepoints[session] == TIMEPOINT_PREOP
            if not is_preop and preop_outdir is None:
                logger.warning(f"{patient_id}/{session}: no processed preop exam, skipping exam.")
                continue
            modalities = resolve_modalities(patient_dir, subject, session)
            if modalities is None:
                continue

            exam_outdir = outdir_root / session
            if args.longitudinal_registration:
                if TUMORSEG_SCHEMA.format(base_dir=exam_outdir).exists():
                    exam_outdirs[session] = exam_outdir
                    if is_preop:
                        preop_outdir = exam_outdir
                else:
                    logger.warning(f"{patient_id}/{session}: not processed yet, skipping exam.")
                continue

            try:
                process_exam(
                    modalities=modalities,
                    outdir=exam_outdir,
                    preop_outdir=None if is_preop else preop_outdir,
                    cuda_device=args.cuda_device,
                    override=args.override,
                )
                exam_outdirs[session] = exam_outdir
                if is_preop:
                    preop_outdir = exam_outdir
            except Exception:
                logger.exception(f"{patient_id}/{session}: processing failed, skipping.")

        outdir_root.mkdir(parents=True, exist_ok=True)
        write_session_labels(
            outfile=outdir_root / SESSION_LABELS_FILENAME,
            subject=subject,
            timepoints=timepoints,
            processed={s: s in exam_outdirs for s in timepoints},
        )

        if preop_outdir is None:
            logger.warning(
                f"{patient_id}: no processed preop exam, skipping longitudinal registration."
            )
            continue

        # Every postop and followup exam is registered to the preop exam.
        for session, exam_outdir in exam_outdirs.items():
            if timepoints[session] == TIMEPOINT_PREOP:
                continue
            recurrence_file = RECURRENCE_SCHEMA.format(base_dir=exam_outdir)
            if args.longitudinal_registration:
                logger.info(f"{patient_id}/{session}: recomputing longitudinal registration.")
            elif recurrence_file.exists() and not args.override:
                logger.info(
                    f"{patient_id}/{session}: longitudinal registration already done, skipping."
                )
                continue
            try:
                tumorseg_file = TUMORSEG_SCHEMA.format(base_dir=exam_outdir)
                moving_mask_file = SYN_MOVING_MASK_SCHEMA.format(base_dir=exam_outdir)
                n_excluded = write_syn_moving_mask(tumorseg_file, moving_mask_file)
                logger.info(
                    f"{patient_id}/{session}: {n_excluded} voxels of labels "
                    f"{list(SYN_EXCLUDED_LABELS)} excluded from the SyN metric."
                )
                register_recurrence(
                    t1c_pre_file=MODALITY_STRIPPED_SCHEMA.format(
                        base_dir=preop_outdir, modality="t1c"
                    ),
                    t1c_post_file=MODALITY_STRIPPED_SCHEMA.format(
                        base_dir=exam_outdir, modality="t1c"
                    ),
                    recurrence_seg_file=tumorseg_file,
                    outdir=exam_outdir,
                    moving_mask_file=moving_mask_file,
                    registration_algorithm=LONGITUDINAL_REGISTRATION_ALGORITHM,
                    additional_modalities=longitudinal_modalities(exam_outdir),
                )
            except Exception:
                logger.exception(
                    f"{patient_id}/{session}: longitudinal registration failed, skipping."
                )
