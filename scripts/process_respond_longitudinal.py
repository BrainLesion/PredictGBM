import os
import csv
import shutil
import tempfile
import argparse
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import ants
import numpy as np
import nibabel as nib
from loguru import logger
from scipy.ndimage import distance_transform_edt
from dipy.core.gradients import gradient_table
from dipy.io.gradients import read_bvals_bvecs
from dipy.reconst.dti import TensorModel
from dipy.segment.mask import median_otsu
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
    LONGITUDINAL_AFFINE_SCHEMA,
    LONGITUDINAL_WARP_SCHEMA,
    MODALITY_STRIPPED_SCHEMA,
    OUTPUT_BASE_SCHEMA,
    PathSchema,
    RECURRENCE_SCHEMA,
    REGISTRATION_MASK_SCHEMA,
    SKULL_STRIP_ALGORITHM_SYNTHSTRIP,
    TISSUE_PBMAP_SCHEMA,
    TUMORSEG_SCHEMA,
)
from predict_gbm.preprocessing import register_recurrence, run_brats, run_tissue_seg
from predict_gbm.preprocessing.norm_ss_coregistration import LinearANTsRegistrator
from predict_gbm.preprocessing.tissue_segmentation import generate_registration_mask
from predict_gbm.utils.utils import update_config

# Processing starts from the original-space (*space-org*) images. Per exam:
# 1. atlas registration with brainles (t1c center, SRI24 unstripped atlas), without skull
#    stripping and without normalization,
# 2. postop/followup only: rigid co-registration of t1c to the preop t1c (both with skull,
#    in atlas space), applied to all modalities,
# 3. skull stripping of t1c with brainles' NativeSpacePreprocessor (no atlas registration)
#    and percentile normalization; the t1c brain mask is applied to t1/t2/flair,
# 4. tumor segmentation with BraTS25_1 (preop branch for the preop exam, post branch
#    otherwise),
# 5. atlas-based tissue segmentation with the tumor core masked.
# Every postop/followup exam is then registered to the preop exam with four-modality DIRAC.
# Exams with a DTI series additionally get a mean diffusivity map (dipy tensor fit), which
# goes through steps 1-3 as an additional moving modality but is never normalized
# (quantitative, mm^2/s) and is co-transformed by DIRAC.
ATLAS = "sri24"
SPACE = "org"
# Patient directories under the data dir (e.g. tum_149) and session directories within.
PATIENT_DIR_GLOB = "tum_*"
SESSION_DIR_GLOB = "ses-*"
# Name of the output directory created inside each patient directory.
DEFAULT_OUTDIR_NAME = "processed"
SESSION_LABELS_FILENAME = "session_labels.tsv"

# <patient_dir>/<session>/sub-<subject>_<session>_space-org_<suffix>.nii.gz
ORG_FILE_SCHEMA = PathSchema(
    "{patient_dir}/{session}/sub-{subject}_{session}_space-" + SPACE + "_{suffix}.nii.gz"
)
MODALITIES = ("t1", "t1c", "t2", "flair")

# DTI: <session>/sub-<subject>_<session>_space-org_dti.{nii.gz,bval,bvec}. The mean
# diffusivity is fitted in the DTI space (<outdir>/dti/md.nii.gz) and then treated as an
# additional moving modality without normalization.
DTI_SUFFIX = "dti"
MD_MODALITY = "md"
QUANTITATIVE_MODALITIES = (MD_MODALITY,)
DTI_FOLDER = "dti"
MD_NATIVE_SCHEMA = OUTPUT_BASE_SCHEMA / DTI_FOLDER / "md.nii.gz"
DTI_FIT_METHOD = "WLS"
# The tensor is only fitted inside a median_otsu brain mask of the b0 volumes (MD = 0
# outside), as in the legacy gliomap MD computation; b-values below this are b0.
DTI_B0_THRESHOLD = 50
DTI_MASK_MEDIAN_RADIUS = 3
DTI_MASK_NUMPASS = 2
CONFIG_STEP_COMPUTE_MD = "compute_md"

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

# Timepoints: the first MRI session is preop, the second postop, and later sessions
# within POSTOP_WINDOW_DAYS of the postop session are postop as well (e.g. a second scan
# the day after surgery). All remaining sessions are followups. PET/CT sessions (no
# original-space t1c) are processed separately, see the PET constants below.
TIMEPOINT_PREOP = "preop"
TIMEPOINT_POSTOP = "postop"
TIMEPOINT_FOLLOWUP = "followup"
DEFAULT_POSTOP_WINDOW_DAYS = 7

# Tissue probability maps co-transformed to preop space alongside t1c and flair.
LONGITUDINAL_TISSUES = ("gm", "wm", "csf")
# Modalities of both exams that drive the four-modality DIRAC longitudinal registration.
DIRAC_MODALITIES = ("t1", "t1c", "t2", "flair")

# PET sessions: sessions with at least one PET (*_pt / *_pet) and one CT (*_ct) image, in
# scanner space. The CT is rigidly registered (Mattes) to the atlas-registered preop t1c
# with skull, with the dilated preop brain mask as fixed mask (keeps the CT couch out of
# the metric), the PET
# is warped with that transform, masked with the preop brain mask, and the mean background
# (brain mask without the tumor mask) is subtracted.
PET_FILE_GLOBS = ("*_pt.nii.gz", "*_pet.nii.gz")
CT_FILE_GLOB = "*_ct.nii.gz"
# Sessions with several PET or CT images: (patient dir, session) -> (PET, CT) file name
# infixes. Any other session with more than one candidate is skipped with a warning.
PET_CT_SERIES = {("tum_149", "ses-20210309"): ("_sequ-5_", "_sequ-4_")}
PET_BRAIN_MASK_DILATION_MM = 10.0
# CT intensities are clipped to this HU window for the registration only (the warped CT
# output keeps raw HU) and shifted to be non-negative for the center-of-mass initialization.
CT_REGISTRATION_HU_WINDOW = (-200.0, 1000.0)
PET_REGISTRATION_TRANSFORM = "Rigid"
PET_REGISTRATION_METRIC = "mattes"
PET_MODALITY = "pet"
CT_MODALITY = "ct"
CONFIG_STEP_REGISTER_PET = "register_pet"


def session_date(session: str) -> date:
    """Parses the date of a session directory name ses-YYYYMMDD."""
    return datetime.strptime(session.removeprefix("ses-"), "%Y%m%d").date()


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


def resolve_dti(
    patient_dir: Path, subject: str, session: str
) -> Optional[Tuple[Path, Path, Path]]:
    """
    The original-space DTI series of a session as (nifti, bval, bvec), or None if there is
    no DTI (silently) or its bval/bvec file is missing (with a warning).
    """
    dti_file = ORG_FILE_SCHEMA.format(
        patient_dir=patient_dir, subject=subject, session=session, suffix=DTI_SUFFIX
    )
    if not dti_file.exists():
        return None
    bval_file = dti_file.with_name(dti_file.name.replace(".nii.gz", ".bval"))
    bvec_file = dti_file.with_name(dti_file.name.replace(".nii.gz", ".bvec"))
    missing = [f.name for f in (bval_file, bvec_file) if not f.exists()]
    if missing:
        logger.warning(f"{patient_dir.name}/{session}: DTI without {missing}, skipping MD.")
        return None
    return dti_file, bval_file, bvec_file


def compute_md(dti_file: Path, bval_file: Path, bvec_file: Path, outfile: Path) -> None:
    """
    Fits the diffusion tensor (dipy TensorModel, DTI_FIT_METHOD) inside a median_otsu brain
    mask of the b0 volumes and writes the mean diffusivity (trace/3, mm^2/s, 0 outside the
    mask) on the DTI grid to outfile. NaN/inf and negative values are set to 0.
    """
    dti_img = nib.load(str(dti_file))
    bvals, bvecs = read_bvals_bvecs(str(bval_file), str(bvec_file))
    masked_data, mask = median_otsu(
        dti_img.get_fdata(dtype=np.float32),
        vol_idx=np.where(bvals < DTI_B0_THRESHOLD)[0].tolist(),
        median_radius=DTI_MASK_MEDIAN_RADIUS,
        numpass=DTI_MASK_NUMPASS,
    )
    gtab = gradient_table(bvals, bvecs=bvecs)
    fit = TensorModel(gtab, fit_method=DTI_FIT_METHOD).fit(masked_data, mask=mask)
    md = np.nan_to_num(fit.md, nan=0.0, posinf=0.0, neginf=0.0)
    md[md < 0] = 0.0

    md_img = nib.Nifti1Image(md.astype(np.float32), dti_img.affine)
    md_img.set_qform(dti_img.affine, code=int(dti_img.header["qform_code"]))
    md_img.set_sform(dti_img.affine, code=int(dti_img.header["sform_code"]))
    md_img.header.set_xyzt_units(xyz="mm")
    outfile.parent.mkdir(parents=True, exist_ok=True)
    nib.save(md_img, str(outfile))
    logger.info(f"{outfile}: mean diffusivity fitted from {dti_file.name} ({DTI_FIT_METHOD}).")


def assign_timepoints(sessions: List[str], postop_window_days: int) -> Dict[str, str]:
    """
    Maps the (date-sorted) MRI sessions of a patient to preop/postop/followup: the first
    session is preop, the second postop, further sessions within postop_window_days of the
    postop session are postop too, and all later ones are followups.
    """
    sessions = sorted(sessions, key=session_date)
    timepoints = {}
    postop_date = session_date(sessions[1]) if len(sessions) > 1 else None
    for i, session in enumerate(sessions):
        if i == 0:
            timepoints[session] = TIMEPOINT_PREOP
        elif (session_date(session) - postop_date).days <= postop_window_days:
            timepoints[session] = TIMEPOINT_POSTOP
        else:
            timepoints[session] = TIMEPOINT_FOLLOWUP
    return timepoints


def register_to_atlas(modalities: Dict[str, Path], outdir: Path) -> None:
    """
    Registers t1c to the SRI24 unstripped atlas with brainles (all other modalities, e.g.
    t1/t2/flair/md, co-registered to t1c first, atlas correction on, as in
    norm_ss_coregister) and stores all modalities in atlas space with skull and
    unnormalized as <outdir>/atlas_registered/<modality>.nii.gz.
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
        for modality in modalities
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


def coregister_to_preop(
    outdir: Path, preop_outdir: Path, modality_names: Sequence[str]
) -> None:
    """
    Rigidly registers the atlas-space t1c of an exam to the atlas-space t1c of the preop
    exam (both with skull) and resamples the given modalities with that transform into
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

    for modality in modality_names:
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
    only modality, so nothing is re-registered), applies the t1c brain mask to the other
    modalities and stores them as <outdir>/skull_stripped/<modality>_skullstripped.nii.gz,
    percentile-normalized except the QUANTITATIVE_MODALITIES (e.g. md), which keep their
    values. The mask is stored as <outdir>/skull_stripped/brain_mask.nii.gz.
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

    # Same masking and normalization as brainles applies to moving modalities; quantitative
    # modalities are masked only.
    with tempfile.TemporaryDirectory(prefix="skull_strip_") as bet_dir:
        for modality, modality_file in modality_files.items():
            if modality == "t1c":
                continue
            stripped_file = MODALITY_STRIPPED_SCHEMA.format(base_dir=outdir, modality=modality)
            quantitative = modality in QUANTITATIVE_MODALITIES
            if quantitative:
                m = Modality(
                    modality_name=modality,
                    input_path=str(modality_file),
                    raw_bet_output_path=str(stripped_file),
                )
            else:
                m = Modality(
                    modality_name=modality,
                    input_path=str(modality_file),
                    normalizer=percentile_normalizer,
                    normalized_bet_output_path=str(stripped_file),
                )
            m.apply_bet_mask(
                brain_extractor=brain_extractor, mask_path=brain_mask_file, bet_dir=bet_dir
            )
            m.save_current_image(stripped_file, normalization=not quantitative)


def process_exam(
    modalities: Dict[str, Path],
    dti_files: Optional[Tuple[Path, Path, Path]],
    outdir: Path,
    preop_outdir: Optional[Path],
    cuda_device: str,
    override: bool = False,
) -> None:
    """
    Per-exam steps: mean diffusivity fit (if dti_files are given), atlas registration
    (skull on, unnormalized), rigid co-registration to the preop exam (postop/followup,
    i.e. preop_outdir given), skull stripping and normalization (md masked only), BraTS25_1
    tumor segmentation (t1/t1c/t2/flair) and the atlas-based tissue segmentation with the
    tumor core masked. If override is True, all steps are rerun even if their outputs
    already exist.
    """
    outdir.mkdir(parents=True, exist_ok=True)
    is_preop = preop_outdir is None

    modalities = dict(modalities)
    if dti_files is not None:
        md_native_file = MD_NATIVE_SCHEMA.format(base_dir=outdir)
        if md_native_file.exists() and not override:
            logger.info(f"{outdir}: mean diffusivity already fitted, skipping.")
        else:
            compute_md(*dti_files, outfile=md_native_file)
            update_config(
                outdir,
                CONFIG_STEP_COMPUTE_MD,
                {
                    "dti_file": str(dti_files[0]),
                    "fit_method": DTI_FIT_METHOD,
                    "fit_mask": "median_otsu",
                    "fit_mask_median_radius": DTI_MASK_MEDIAN_RADIUS,
                    "fit_mask_numpass": DTI_MASK_NUMPASS,
                    "negative_to_zero": True,
                },
            )
        modalities[MD_MODALITY] = md_native_file

    atlas_files = [ATLAS_REGISTERED_SCHEMA.format(base_dir=outdir, modality=m) for m in modalities]
    if all(f.exists() for f in atlas_files) and not override:
        logger.info(f"{outdir}: atlas registration already done, skipping.")
    else:
        register_to_atlas(modalities, outdir)

    if is_preop:
        coregistered_schema = ATLAS_REGISTERED_SCHEMA
    else:
        coregistered_schema = PREOP_COREGISTERED_SCHEMA
        coregistered_files = [
            PREOP_COREGISTERED_SCHEMA.format(base_dir=outdir, modality=m) for m in modalities
        ]
        if all(f.exists() for f in coregistered_files) and not override:
            logger.info(f"{outdir}: co-registration to preop already done, skipping.")
        else:
            coregister_to_preop(outdir, preop_outdir, list(modalities))

    t1c_stripped = MODALITY_STRIPPED_SCHEMA.format(base_dir=outdir, modality="t1c")
    stripped_files = [
        MODALITY_STRIPPED_SCHEMA.format(base_dir=outdir, modality=m) for m in modalities
    ]
    if all(f.exists() for f in stripped_files) and not override:
        logger.info(f"{outdir}: skull stripping already done, skipping.")
    else:
        skull_strip(
            {m: coregistered_schema.format(base_dir=outdir, modality=m) for m in modalities},
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
            is_preop=is_preop,
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


def dirac_modality_files(preop_outdir: Path, exam_outdir: Path) -> Dict[str, Path]:
    """
    Skull-stripped t1, t1c, t2 and flair of the preop and the followup exam as the
    {modality}_pre_file / {modality}_post_file arguments of register_recurrence, which the
    four-modality DIRAC model needs.
    """
    files: Dict[str, Path] = {}
    for modality in DIRAC_MODALITIES:
        files[f"{modality}_pre_file"] = MODALITY_STRIPPED_SCHEMA.format(
            base_dir=preop_outdir, modality=modality
        )
        files[f"{modality}_post_file"] = MODALITY_STRIPPED_SCHEMA.format(
            base_dir=exam_outdir, modality=modality
        )
    return files


def longitudinal_modalities(exam_outdir: Path) -> Dict[str, Path]:
    """
    Additional modalities of an exam that register_recurrence co-transforms to preop space:
    the normalized flair, the mean diffusivity (if the exam has one) and the gm/wm/csf
    tissue probability maps. Outputs are stored as
    <exam_outdir>/longitudinal/<name>_warped_longitudinal.nii.gz.
    """
    modalities = {
        "flair": MODALITY_STRIPPED_SCHEMA.format(base_dir=exam_outdir, modality="flair")
    }
    md_file = MODALITY_STRIPPED_SCHEMA.format(base_dir=exam_outdir, modality=MD_MODALITY)
    if md_file.exists():
        modalities[MD_MODALITY] = md_file
    for tissue in LONGITUDINAL_TISSUES:
        modalities[f"{tissue}_pbmap"] = TISSUE_PBMAP_SCHEMA.format(
            base_dir=exam_outdir, tissue=tissue
        )
    return modalities


def resolve_pet_ct(patient_dir: Path, session: str) -> Optional[Tuple[Path, Path]]:
    """
    The (PET, CT) image pair of a session, or None if the session has no PET or no CT
    (silently) or if several candidates exist without a PET_CT_SERIES entry, or the entry
    does not match exactly one PET and one CT (with a warning).
    """
    session_dir = patient_dir / session
    pet_files = sorted(f for pattern in PET_FILE_GLOBS for f in session_dir.glob(pattern))
    ct_files = sorted(session_dir.glob(CT_FILE_GLOB))
    if not pet_files or not ct_files:
        return None
    if len(pet_files) > 1 or len(ct_files) > 1:
        series = PET_CT_SERIES.get((patient_dir.name, session))
        if series is None:
            logger.warning(
                f"{patient_dir.name}/{session}: {len(pet_files)} PET and {len(ct_files)} CT "
                "images but no PET_CT_SERIES entry, skipping PET session."
            )
            return None
        pet_files = [f for f in pet_files if series[0] in f.name]
        ct_files = [f for f in ct_files if series[1] in f.name]
        if len(pet_files) != 1 or len(ct_files) != 1:
            logger.warning(
                f"{patient_dir.name}/{session}: PET_CT_SERIES entry {series} matches "
                f"{len(pet_files)} PET and {len(ct_files)} CT images, skipping PET session."
            )
            return None
    return pet_files[0], ct_files[0]


def pet_tumor_mask_file(
    pet_session: str, timepoints: Dict[str, str], exam_outdirs: Dict[str, Path]
) -> Optional[Path]:
    """
    Tumor mask in preop space that is excluded from the PET background region. PET sessions
    dated before the postop MRI use the preop tumor segmentation, later ones the
    recurrence_preop of the date-nearest processed postop/followup exam (ties go to the
    earlier exam). Returns None if there is none.
    """
    pet_date = session_date(pet_session)
    postop_dates = [session_date(s) for s, t in timepoints.items() if t == TIMEPOINT_POSTOP]
    if not postop_dates or pet_date < min(postop_dates):
        preop_sessions = [s for s in exam_outdirs if timepoints[s] == TIMEPOINT_PREOP]
        return TUMORSEG_SCHEMA.format(base_dir=exam_outdirs[preop_sessions[0]])
    candidates = [
        s
        for s, exam_outdir in exam_outdirs.items()
        if timepoints[s] != TIMEPOINT_PREOP
        and RECURRENCE_SCHEMA.format(base_dir=exam_outdir).exists()
    ]
    if not candidates:
        return None
    nearest = min(
        candidates,
        key=lambda s: (abs((session_date(s) - pet_date).days), session_date(s)),
    )
    return RECURRENCE_SCHEMA.format(base_dir=exam_outdirs[nearest])


def register_pet(
    pet_file: Path,
    ct_file: Path,
    preop_outdir: Path,
    tumor_mask_file: Path,
    outdir: Path,
) -> None:
    """
    Rigidly registers the CT to the atlas-registered preop t1c with skull (Mattes, dilated
    preop brain mask as fixed mask, CT clipped to CT_REGISTRATION_HU_WINDOW), warps CT and
    PET to preop space with the resulting transform, masks the PET with the preop brain
    mask and subtracts the mean PET of the brain mask without the tumor mask (all nonzero
    labels), clipping at 0. Writes <outdir>/longitudinal/{ct,pet}_warped_longitudinal.nii.gz
    and longitudinal_trafo.mat.
    """
    t1c_pre_img = ants.image_read(
        str(ATLAS_REGISTERED_SCHEMA.format(base_dir=preop_outdir, modality="t1c"))
    )
    brain_mask_img = ants.image_read(str(BRAIN_MASK_SCHEMA.format(base_dir=preop_outdir)))
    brain_mask = brain_mask_img.numpy() > 0
    dilated_mask = (
        distance_transform_edt(~brain_mask, sampling=brain_mask_img.spacing)
        <= PET_BRAIN_MASK_DILATION_MM
    )
    fixed_mask_img = brain_mask_img.new_image_like(dilated_mask.astype(np.float32))

    ct_img = ants.image_read(str(ct_file))
    hu_min, hu_max = CT_REGISTRATION_HU_WINDOW
    ct_reg_img = ct_img.new_image_like(np.clip(ct_img.numpy(), hu_min, hu_max) - hu_min)

    affine_file = LONGITUDINAL_AFFINE_SCHEMA.format(base_dir=outdir)
    affine_file.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="register_pet_") as tmpdir:
        reg = ants.registration(
            fixed=t1c_pre_img,
            moving=ct_reg_img,
            type_of_transform=PET_REGISTRATION_TRANSFORM,
            aff_metric=PET_REGISTRATION_METRIC,
            mask=fixed_mask_img,
            outprefix=f"{tmpdir}/ants_",
        )
        shutil.move(reg["fwdtransforms"][0], str(affine_file))
    transformlist = [str(affine_file)]

    ct_warped = ants.apply_transforms(
        fixed=t1c_pre_img,
        moving=ct_img,
        transformlist=transformlist,
        defaultvalue=float(ct_img.min()),
    )
    ants.image_write(
        ct_warped, str(LONGITUDINAL_WARP_SCHEMA.format(base_dir=outdir, modality=CT_MODALITY))
    )

    pet_warped = ants.apply_transforms(
        fixed=t1c_pre_img, moving=ants.image_read(str(pet_file)), transformlist=transformlist
    )
    pet = pet_warped.numpy() * brain_mask
    tumor_mask = np.rint(ants.image_read(str(tumor_mask_file)).numpy()) > 0
    background = float(pet[brain_mask & ~tumor_mask].mean())
    pet = np.clip(pet - background, 0, None)
    ants.image_write(
        pet_warped.new_image_like(pet.astype(np.float32)),
        str(LONGITUDINAL_WARP_SCHEMA.format(base_dir=outdir, modality=PET_MODALITY)),
    )
    logger.info(
        f"{outdir}: registered {ct_file.name}, warped {pet_file.name}, background "
        f"{background:.2f} (tumor mask {tumor_mask_file})."
    )

    update_config(
        outdir,
        CONFIG_STEP_REGISTER_PET,
        {
            "pet_file": str(pet_file),
            "ct_file": str(ct_file),
            "fixed_file": str(
                ATLAS_REGISTERED_SCHEMA.format(base_dir=preop_outdir, modality="t1c")
            ),
            "registration_transform": PET_REGISTRATION_TRANSFORM,
            "registration_metric": PET_REGISTRATION_METRIC,
            "brain_mask_dilation_mm": PET_BRAIN_MASK_DILATION_MM,
            "ct_registration_hu_window": list(CT_REGISTRATION_HU_WINDOW),
            "tumor_mask_file": str(tumor_mask_file),
            "background_mean": background,
            "clipped_at_zero": True,
        },
    )


def write_session_labels(
    outfile: Path, subject: str, timepoints: Dict[str, str], processed: Dict[str, bool]
) -> None:
    """
    Writes a tsv with columns subject, session, timepoint and processed (whether the
    exam's per-exam steps completed) for the MRI sessions of a patient.
    """
    with open(outfile, "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t", lineterminator="\n")
        writer.writerow(["subject", "session", "timepoint", "processed"])
        for session in sorted(timepoints, key=session_date):
            writer.writerow(
                [subject, session, timepoints[session], int(processed.get(session, False))]
            )
    logger.info(f"Wrote {outfile}.")


if __name__ == "__main__":
    # Full processing (per-exam steps, longitudinal registration) of the original-space MRI
    # sessions and of the PET/CT sessions (registration to the preop exam) of every patient;
    # steps with existing outputs are skipped unless -override is given. Outputs go to
    # <patient_dir>/processed/<session>/:
    # nohup python -u scripts/process_respond_longitudinal.py -cuda_device 0 > tmp_process_respond_longitudinal.out 2>&1 &
    # Only one patient:
    # python -u scripts/process_respond_longitudinal.py -cuda_device 0 -patients tum_149
    # Recompute the postop/followup-to-preop registrations only (per-exam outputs reused):
    # python -u scripts/process_respond_longitudinal.py -cuda_device 0 -longitudinal_registration
    # PET sessions only (needs the processed MRI exams):
    # python -u scripts/process_respond_longitudinal.py -pet_only
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
        "-pet_only",
        action="store_true",
        help="Only process the PET sessions (CT/PET registration to the preop exam), reusing "
        "the processed MRI exams; with -override existing PET outputs are recomputed.",
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

        # The preop exam is processed first: the others are co-registered to it. It is the
        # first session by date.
        exam_outdirs: Dict[str, Path] = {}  # session -> outdir of processed exams
        preop_outdir: Optional[Path] = None
        for session in sorted(timepoints, key=session_date):
            is_preop = timepoints[session] == TIMEPOINT_PREOP
            if not is_preop and preop_outdir is None:
                logger.warning(f"{patient_id}/{session}: no processed preop exam, skipping exam.")
                continue
            modalities = resolve_modalities(patient_dir, subject, session)
            if modalities is None:
                continue

            exam_outdir = outdir_root / session
            if args.longitudinal_registration or args.pet_only:
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
                    dti_files=resolve_dti(patient_dir, subject, session),
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

        if not args.pet_only:
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
            if args.pet_only or timepoints[session] == TIMEPOINT_PREOP:
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
                # DIRAC (four-modality model): t1, t1c, t2 and flair of both exams.
                register_recurrence(
                    recurrence_seg_file=TUMORSEG_SCHEMA.format(base_dir=exam_outdir),
                    outdir=exam_outdir,
                    additional_modalities=longitudinal_modalities(exam_outdir),
                    **dirac_modality_files(preop_outdir, exam_outdir),
                )
            except Exception:
                logger.exception(
                    f"{patient_id}/{session}: longitudinal registration failed, skipping."
                )

        # PET sessions: CT rigidly registered to the atlas-registered preop t1c with skull,
        # PET warped with it.
        for session_dir in sorted(patient_dir.glob(SESSION_DIR_GLOB)):
            session = session_dir.name
            pet_ct = resolve_pet_ct(patient_dir, session) if session_dir.is_dir() else None
            if pet_ct is None:
                continue
            if session in timepoints:
                # Outputs would collide with the MRI exam's longitudinal registration.
                logger.warning(
                    f"{patient_id}/{session}: PET/CT session is also an MRI session, "
                    "skipping PET session."
                )
                continue
            pet_outdir = outdir_root / session
            pet_outfile = LONGITUDINAL_WARP_SCHEMA.format(
                base_dir=pet_outdir, modality=PET_MODALITY
            )
            if pet_outfile.exists() and not args.override:
                logger.info(f"{patient_id}/{session}: PET already processed, skipping.")
                continue
            tumor_mask_file = pet_tumor_mask_file(session, timepoints, exam_outdirs)
            if tumor_mask_file is None:
                logger.warning(
                    f"{patient_id}/{session}: no tumor mask for the PET background, "
                    "skipping PET session."
                )
                continue
            try:
                register_pet(
                    pet_file=pet_ct[0],
                    ct_file=pet_ct[1],
                    preop_outdir=preop_outdir,
                    tumor_mask_file=tumor_mask_file,
                    outdir=pet_outdir,
                )
            except Exception:
                logger.exception(f"{patient_id}/{session}: PET processing failed, skipping.")
