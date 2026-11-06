import os
import csv
import json
import shutil
import tempfile
import argparse
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple
import ants
import numpy as np
import nibabel as nib
import openpyxl
from loguru import logger
from scipy.ndimage import binary_opening, distance_transform_edt
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
    RECURRENCE_CORRECTED_SCHEMA,
    RECURRENCE_SCHEMA,
    REGISTRATION_MASK_SCHEMA,
    SKULL_STRIP_ALGORITHM_SYNTHSTRIP,
    TISSUE_PBMAP_SCHEMA,
    TUMORSEG_CORRECTED_SCHEMA,
    TUMORSEG_SCHEMA,
)
from predict_gbm.preprocessing import (
    longitudinal_transform_file,
    register_recurrence,
    run_brats,
    run_tissue_seg,
)
from ants.internal import get_pointer_string
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
# 4. tumor segmentation with BraTS25_1 (preop branch for preop-labelled exams, post branch
#    otherwise),
# 5. corrected tumor segmentation: morphological opening of the enhancing tumor, removed
#    voxels become edema (tumor_seg_corrected.nii.gz next to tumor_seg.nii.gz),
# 6. atlas-based tissue segmentation with the tumor core masked (original segmentation).
# Every exam except the reference preop exam (the first preop session, everything is
# co-registered to it) is then registered to it with four-modality DIRAC, which also warps
# both segmentations (longitudinal/recurrence_preop{,_corrected}.nii.gz).
# Exams with a DTI series additionally get a mean diffusivity map (dipy tensor fit). It is
# kept out of brainles (its center-of-mass initialization and atlas correction misplace the
# skull-free MD by up to ~100 mm): MD is rigidly registered to the original-space t1c from
# the header alignment, moved to atlas space with the t1c atlas transform, then follows
# step 2. Before masking (step 3, never normalized: quantitative, mm^2/s) its EPI
# susceptibility distortion is corrected with an A-P-only deformable registration of the b0
# to the t2. It is co-transformed by DIRAC.
# -compute_trace (separate step, not part of the regular run): the trace-weighted image of
# every processed exam with a DTI, the geometric mean of the diffusion-weighted volumes,
# brought to the same spaces as MD with the transforms stored for MD (process_trace).
# -register_adc (separate step as well): the scanner's ADC map of every processed exam
# without a DTI, rigidly registered to the t1c like MD and brought to the same spaces
# (process_adc), so that exams without a trace have a diffusion image there too.
ATLAS = "sri24"
SPACE = "org"
# Patient directories under the data dir (default glob, e.g. sub-GR008; -patient_glob 'tum_*'
# for the original tum_149 directory) and session directories within.
DEFAULT_PATIENT_DIR_GLOB = "sub-GR*"
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
# diffusivity and the mean b0 are written in the DTI space (<outdir>/dti/{md,b0}.nii.gz) and
# then registered and resampled separately from the other modalities, without normalization.
DTI_SUFFIX = "dti"
MD_MODALITY = "md"
B0_MODALITY = "b0"
QUANTITATIVE_MODALITIES = (MD_MODALITY,)
DTI_FOLDER = "dti"
MD_NATIVE_SCHEMA = OUTPUT_BASE_SCHEMA / DTI_FOLDER / "md.nii.gz"
B0_NATIVE_SCHEMA = OUTPUT_BASE_SCHEMA / DTI_FOLDER / "b0.nii.gz"
DTI_FIT_METHOD = "WLS"
# The tensor is only fitted inside a median_otsu brain mask of the b0 volumes (MD = 0
# outside), as in the legacy gliomap MD computation; b-values below this are b0.
DTI_B0_THRESHOLD = 50
DTI_MASK_MEDIAN_RADIUS = 3
DTI_MASK_NUMPASS = 2
# MD is clipped to [0, MD_MAX] (mm^2/s, just above free water at body temperature): single
# failed voxel fits reach MD ~200 and collapse the Mattes histogram of the registration.
MD_MAX = 4e-3
CONFIG_STEP_COMPUTE_MD = "compute_md"
# MD -> original-space t1c (with skull): rigid, Mattes, initialized with the header
# alignment (same session) instead of ANTs' center of mass, which the neck in the t1c
# field of view pulls far off. Applied together with the brainles t1c atlas transform.
MD_REGISTRATION_TRANSFORM = "Rigid"
MD_REGISTRATION_METRIC = "mattes"
MD_REGISTRATION_INIT = "identity"
CONFIG_STEP_REGISTER_MD = "register_md"
# EPI susceptibility distortion of the DTI (phase encoding A-P, no reverse phase-encoded b0
# for topup): the brain-masked b0 is registered to the skull-stripped t2 (similar contrast)
# with a smooth B-spline SyN field restricted to the physical A-P axis (LPS y; the DTI phase
# encoding axis is tilted by the slice angulation, ~9 deg for tum_149, so this is an
# approximation). No fixed mask: both images are already brain-masked, and with a fixed mask
# ANTs finds no valid metric points at shrink factor 4, so that level was silently skipped.
# The field is applied to MD in the exam's co-registered space. Exams without a t2 (flair
# stand-in, inverted CSF contrast against the b0) are not corrected.
DISTORTION_FIXED_MODALITY = "t2"
DISTORTION_TRANSFORM = "BSplineSyN[0.1,40,0,3]"  # B-spline knot spacing 40 mm
DISTORTION_METRIC_RADIUS = 4  # CC radius (voxels)
DISTORTION_RESTRICT_DEFORMATION = "0x1x0"
DISTORTION_CONVERGENCE = "[100x70x30,1e-6,10]"
DISTORTION_SHRINK_FACTORS = "4x2x1"
DISTORTION_SMOOTHING_SIGMAS = "2x1x0vox"
DISTORTION_WARP_SCHEMA = OUTPUT_BASE_SCHEMA / DTI_FOLDER / "distortion_warp.nii.gz"
CONFIG_STEP_CORRECT_MD_DISTORTION = "correct_md_distortion"
# Trace (-compute_trace only): the geometric mean of the diffusion-weighted volumes (b >=
# DTI_B0_THRESHOLD) of the DTI series, i.e. the isotropic / trace-weighted DWI, on the DTI
# grid (<outdir>/dti/trace.nii.gz, raw signal, unmasked). It takes the MD path with the
# transforms stored for MD: MD -> t1c rigid plus the t1c atlas transform
# (atlas_registered/trace.nii.gz), the rigid transform to the preop exam
# (preop_coregistered/trace.nii.gz), the DTI distortion field if one was estimated for
# the MD, then the t1c brain mask with percentile normalization like the structural
# modalities (skull_stripped/trace_skullstripped.nii.gz) and the stored longitudinal
# transform (longitudinal/trace_warped_longitudinal.nii.gz).
TRACE_MODALITY = "trace"
TRACE_NATIVE_SCHEMA = OUTPUT_BASE_SCHEMA / DTI_FOLDER / "trace.nii.gz"
CONFIG_STEP_COMPUTE_TRACE = "compute_trace"
# ADC (-register_adc only): the scanner's ADC map (<session>/..._space-org_adc.nii.gz, from
# a clinical DWI on its own grid, no sidecar) of every processed exam without a DTI. It is
# rigidly registered to the original-space t1c with the MD settings (for the registration
# only, values above its ADC_CLIP_PERCENTILE are clipped: stray extreme voxels collapse
# the Mattes histogram) and takes the MD path: the t1c atlas transform
# (atlas_registered/adc.nii.gz), the rigid transform to the preop exam
# (preop_coregistered/adc.nii.gz), the t1c brain mask with percentile normalization
# (skull_stripped/adc_skullstripped.nii.gz) and the stored longitudinal transform
# (longitudinal/adc_warped_longitudinal.nii.gz). No distortion correction (no b0).
ADC_SUFFIX = "adc"
ADC_MODALITY = "adc"
ADC_CLIP_PERCENTILE = 99.9
CONFIG_STEP_REGISTER_ADC = "register_adc"

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
# brainles' t1c -> atlas transform (its file naming) and the separate MD -> t1c transform.
ATLAS_T1C_TRAFO_SCHEMA = ATLAS_REGISTRATION_TRAFO_SCHEMA / "t1c" / "2_M_atlas__t1c.mat"
MD_TO_T1C_TRAFO_SCHEMA = ATLAS_REGISTRATION_TRAFO_SCHEMA / MD_MODALITY / "md_to_t1c.mat"
ADC_TO_T1C_TRAFO_SCHEMA = ATLAS_REGISTRATION_TRAFO_SCHEMA / ADC_MODALITY / "adc_to_t1c.mat"
PREOP_COREGISTERED_FOLDER = "preop_coregistered"
PREOP_COREGISTERED_SCHEMA = (
    OUTPUT_BASE_SCHEMA / PREOP_COREGISTERED_FOLDER / "{modality}.nii.gz"
)
PREOP_RIGID_TRAFO_SCHEMA = (
    OUTPUT_BASE_SCHEMA / PREOP_COREGISTERED_FOLDER / "rigid_to_preop.mat"
)
PREOP_COREGISTRATION_TRANSFORM = "Rigid"
# Normalization of the skull-stripped non-quantitative modalities (stateless, shared).
PERCENTILE_NORMALIZER = PercentileNormalizer(
    lower_percentile=0.1,
    upper_percentile=99.9,
    lower_limit=0,
    upper_limit=1,
)

# Timepoints come from the surgery dates sheet (DATES_FILENAME in the data directory, one
# row per subject id): MRI sessions before the Resection date are preop, sessions on it
# intraop, sessions after it up to the postop date postop, and later ones followups. The
# Baseline session must be the first MRI session and the postop date an MRI session,
# otherwise (and for subjects missing from the sheet) the rule is used instead: the first
# MRI session is preop, the second postop, and later sessions within POSTOP_WINDOW_DAYS of
# the postop session are postop as well (e.g. a second scan the day after surgery), all
# remaining ones followups. The first preop session is the reference exam everything is
# registered to; further preop sessions only differ from postop ones in the BraTS branch.
# PET/CT sessions (no original-space t1c) are processed separately, see the PET constants
# below.
TIMEPOINT_PREOP = "preop"
TIMEPOINT_INTRAOP = "intraop"
TIMEPOINT_POSTOP = "postop"
TIMEPOINT_FOLLOWUP = "followup"
DEFAULT_POSTOP_WINDOW_DAYS = 7
DATES_FILENAME = "dates.xlsx"
DATES_ID_COLUMN = "ID"
DATES_BASELINE_COLUMN = "Baseline"
DATES_RESECTION_COLUMN = "Resection"
DATES_POSTOP_COLUMN = "postop"
LABEL_SOURCE_SHEET = "sheet"
LABEL_SOURCE_RULE = "rule"

# Corrected tumor segmentation: binary opening (erosion followed by dilation, scipy's default
# 6-connected 3D structuring element) of the enhancing tumor; removed voxels become edema.
ENHANCING_LABEL = 3
EDEMA_LABEL = 2
ENHANCING_OPENING_ITERATIONS = 2
CONFIG_STEP_CORRECT_TUMOR_SEG = "correct_tumor_seg"

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
# macOS AppleDouble files (._<name>) next to the images match the globs and are ignored.
APPLEDOUBLE_PREFIX = "._"
# Sessions often hold several PET reconstructions and CT series (plus derived fusion MPRs).
# The series is picked from the json sidecar: ImageType ORIGINAL and a SeriesDescription
# matching (whitespace removed, case-insensitive) the attenuation-corrected iterative PET
# ("PET it+TOF AC" / "PET it + TOF AC") and the 3 mm low-dose skull CT, as chosen for
# tum_149 ses-20210309 (sequ-5 / sequ-4). Sessions without exactly one match each are
# skipped with a warning.
PET_SERIES_DESCRIPTION = "PET it+TOF AC"
CT_SERIES_DESCRIPTION = "Schädel LD 3mm SAF"
PET_BRAIN_MASK_DILATION_MM = 10.0
# CT intensities are clipped to this HU window for the registration only (the warped CT
# output keeps raw HU) and shifted to be non-negative for the center-of-mass initialization.
CT_REGISTRATION_HU_WINDOW = (-200.0, 1000.0)
PET_REGISTRATION_TRANSFORM = "Rigid"
PET_REGISTRATION_METRIC = "mattes"
PET_REGISTRATION_SEED = 1
# The CT -> t1c Mattes registration has wrong local minima ~10-20 mm from the right one
# (6/10 failures from starts 15 mm off the center of mass, for both tum_149 PET sessions).
# QC: fraction of the preop brain mask (deeper than PET_QC_BRAIN_DEPTH_MM) where the warped
# CT is bone (> PET_QC_BONE_HU); good registrations have ~0.2%, wrong ones 5-13%. Above
# PET_QC_MAX_BONE_FRACTION the registration is retried from the center of mass shifted by
# +-PET_RETRY_OFFSET_MM along each axis and the start with the least bone in the brain is
# kept (a warning is logged if it still exceeds the threshold).
PET_QC_BONE_HU = 300.0
PET_QC_BRAIN_DEPTH_MM = 3.0
PET_QC_MAX_BONE_FRACTION = 0.01
PET_RETRY_OFFSET_MM = 15.0
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


def resolve_adc(patient_dir: Path, subject: str, session: str) -> Optional[Path]:
    """The original-space ADC map of a session, or None if there is none."""
    adc_file = ORG_FILE_SCHEMA.format(
        patient_dir=patient_dir, subject=subject, session=session, suffix=ADC_SUFFIX
    )
    return adc_file if adc_file.exists() else None


def compute_md(
    dti_file: Path, bval_file: Path, bvec_file: Path, outfile: Path, b0_outfile: Path
) -> None:
    """
    Fits the diffusion tensor (dipy TensorModel, DTI_FIT_METHOD) inside a median_otsu brain
    mask of the b0 volumes and writes the mean diffusivity (trace/3, mm^2/s, 0 outside the
    mask) on the DTI grid to outfile. NaN/inf and negative values are set to 0, values
    above MD_MAX are clipped to MD_MAX. The mean of the (unmasked) b0 volumes is written
    to b0_outfile.
    """
    dti_img = nib.load(str(dti_file))
    bvals, bvecs = read_bvals_bvecs(str(bval_file), str(bvec_file))
    data = dti_img.get_fdata(dtype=np.float32)
    b0_idx = np.where(bvals < DTI_B0_THRESHOLD)[0]
    masked_data, mask = median_otsu(
        data,
        vol_idx=b0_idx.tolist(),
        median_radius=DTI_MASK_MEDIAN_RADIUS,
        numpass=DTI_MASK_NUMPASS,
    )
    gtab = gradient_table(bvals, bvecs=bvecs)
    fit = TensorModel(gtab, fit_method=DTI_FIT_METHOD).fit(masked_data, mask=mask)
    md = np.nan_to_num(fit.md, nan=0.0, posinf=0.0, neginf=0.0)
    md[md < 0] = 0.0
    n_clipped = int((md > MD_MAX).sum())
    md[md > MD_MAX] = MD_MAX

    outfile.parent.mkdir(parents=True, exist_ok=True)
    for volume, volume_file in ((md, outfile), (data[..., b0_idx].mean(axis=-1), b0_outfile)):
        img = nib.Nifti1Image(volume.astype(np.float32), dti_img.affine)
        img.set_qform(dti_img.affine, code=int(dti_img.header["qform_code"]))
        img.set_sform(dti_img.affine, code=int(dti_img.header["sform_code"]))
        img.header.set_xyzt_units(xyz="mm")
        nib.save(img, str(volume_file))
    logger.info(
        f"{outfile}: mean diffusivity fitted from {dti_file.name} ({DTI_FIT_METHOD}), "
        f"{n_clipped} voxels clipped to {MD_MAX}."
    )


def compute_trace(dti_file: Path, bval_file: Path, bvec_file: Path, outfile: Path) -> None:
    """
    Writes the trace-weighted image of the DTI series on its grid to outfile: the geometric
    mean of the diffusion-weighted volumes (b >= DTI_B0_THRESHOLD), raw signal, unmasked,
    0 where any of the volumes is <= 0.
    """
    dti_img = nib.load(str(dti_file))
    bvals, _ = read_bvals_bvecs(str(bval_file), str(bvec_file))
    dwi_idx = np.where(bvals >= DTI_B0_THRESHOLD)[0]
    dwi = dti_img.get_fdata(dtype=np.float32)[..., dwi_idx]
    valid = np.all(dwi > 0, axis=-1)
    trace = np.zeros(dwi.shape[:-1], dtype=np.float32)
    trace[valid] = np.exp(np.log(dwi[valid]).mean(axis=-1))

    outfile.parent.mkdir(parents=True, exist_ok=True)
    img = nib.Nifti1Image(trace, dti_img.affine)
    img.set_qform(dti_img.affine, code=int(dti_img.header["qform_code"]))
    img.set_sform(dti_img.affine, code=int(dti_img.header["sform_code"]))
    img.header.set_xyzt_units(xyz="mm")
    nib.save(img, str(outfile))
    logger.info(
        f"{outfile}: trace computed as the geometric mean of {len(dwi_idx)} diffusion-weighted "
        f"volumes (b = {sorted(set(np.round(bvals[dwi_idx]).astype(int).tolist()))}) of "
        f"{dti_file.name}."
    )


def register_md_to_atlas(md_file: Path, b0_file: Path, t1c_file: Path, outdir: Path) -> None:
    """
    Rigidly registers the mean diffusivity map to the original-space t1c with skull
    (MD_REGISTRATION_TRANSFORM/_METRIC from the header alignment) and resamples it and the
    b0 (same DTI grid) with that transform followed by the brainles t1c atlas transform
    (register_to_atlas must have run) to <outdir>/atlas_registered/{md,b0}.nii.gz. The
    MD -> t1c transform is stored as <outdir>/atlas_registered/transformations/md/md_to_t1c.mat.
    """
    md_img = ants.image_read(str(md_file))
    trafo_file = MD_TO_T1C_TRAFO_SCHEMA.format(base_dir=outdir)
    trafo_file.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="register_md_") as tmpdir:
        reg = ants.registration(
            fixed=ants.image_read(str(t1c_file)),
            moving=md_img,
            type_of_transform=MD_REGISTRATION_TRANSFORM,
            aff_metric=MD_REGISTRATION_METRIC,
            initial_transform=MD_REGISTRATION_INIT,
            outprefix=f"{tmpdir}/rigid_",
        )
        shutil.move(reg["fwdtransforms"][0], str(trafo_file))

    t1c_atlas_img = ants.image_read(
        str(ATLAS_REGISTERED_SCHEMA.format(base_dir=outdir, modality="t1c"))
    )
    for modality, img in ((MD_MODALITY, md_img), (B0_MODALITY, ants.image_read(str(b0_file)))):
        warped = ants.apply_transforms(
            fixed=t1c_atlas_img,
            moving=img,
            transformlist=[str(ATLAS_T1C_TRAFO_SCHEMA.format(base_dir=outdir)), str(trafo_file)],
            interpolator="linear",
            defaultvalue=0,
        )
        ants.image_write(
            warped, str(ATLAS_REGISTERED_SCHEMA.format(base_dir=outdir, modality=modality))
        )
    logger.info(f"{outdir}: mean diffusivity registered to t1c and moved to atlas space.")


def correct_md_distortion(outdir: Path, coregistered_schema: PathSchema) -> None:
    """
    Estimates the EPI susceptibility distortion of the exam's DTI in its co-registered space
    (atlas space for the preop exam, preop space otherwise): the b0
    (<coregistered_schema>/b0.nii.gz) masked with the t1c brain mask is registered to the
    skull-stripped DISTORTION_FIXED_MODALITY with DISTORTION_TRANSFORM restricted to the A-P
    axis. The displacement field is stored as <outdir>/dti/distortion_warp.nii.gz.
    """
    fixed_img = ants.image_read(
        str(MODALITY_STRIPPED_SCHEMA.format(base_dir=outdir, modality=DISTORTION_FIXED_MODALITY))
    ).clone("float")
    brain_mask_img = ants.image_read(str(BRAIN_MASK_SCHEMA.format(base_dir=outdir))).clone(
        "float"
    )
    moving_img = (
        ants.image_read(
            str(coregistered_schema.format(base_dir=outdir, modality=B0_MODALITY))
        ).clone("float")
        * brain_mask_img
    )
    warp_file = DISTORTION_WARP_SCHEMA.format(base_dir=outdir)
    with tempfile.TemporaryDirectory(prefix="correct_md_distortion_") as tmpdir:
        # Raw antsRegistration arguments: ants.registration only forwards the deformation
        # restriction for its antsRegistrationSyN[...] presets.
        args = [
            "-d", "3",
            "-v", "0",
            "--float", "1",
            "-o", f"[{tmpdir}/distortion_]",
            "-n", "Linear",
            "-m",
            f"CC[{get_pointer_string(fixed_img)},{get_pointer_string(moving_img)},1,"
            f"{DISTORTION_METRIC_RADIUS}]",
            "-t", DISTORTION_TRANSFORM,
            "-c", DISTORTION_CONVERGENCE,
            "-f", DISTORTION_SHRINK_FACTORS,
            "-s", DISTORTION_SMOOTHING_SIGMAS,
            "-g", DISTORTION_RESTRICT_DEFORMATION,
        ]  # fmt: skip
        ants.registration(fixed=args, moving=None)
        shutil.move(f"{tmpdir}/distortion_0Warp.nii.gz", str(warp_file))

    displacement = np.abs(ants.image_read(str(warp_file)).numpy()[..., 1])[
        brain_mask_img.numpy() > 0
    ]
    logger.info(
        f"{outdir}: DTI distortion estimated, A-P displacement in the brain mean "
        f"{displacement.mean():.1f} mm, p99 {np.percentile(displacement, 99):.1f} mm, "
        f"max {displacement.max():.1f} mm."
    )


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


class SurgeryDates(NamedTuple):
    """Dates of one subject from the surgery dates sheet (DATES_FILENAME)."""

    baseline: date
    resection: date
    postop: date


def read_surgery_dates(dates_file: Path) -> Dict[str, SurgeryDates]:
    """
    Reads the surgery dates sheet (first worksheet, header row with DATES_ID_COLUMN,
    DATES_BASELINE_COLUMN, DATES_RESECTION_COLUMN and DATES_POSTOP_COLUMN) and returns
    subject id -> SurgeryDates. Rows with any of these dates missing are left out with a
    warning.
    """
    worksheet = openpyxl.load_workbook(dates_file, read_only=True, data_only=True).active
    rows = worksheet.iter_rows(values_only=True)
    header = list(next(rows))
    columns = {
        name: header.index(name)
        for name in (
            DATES_ID_COLUMN,
            DATES_BASELINE_COLUMN,
            DATES_RESECTION_COLUMN,
            DATES_POSTOP_COLUMN,
        )
    }
    surgery_dates = {}
    for row in rows:
        subject = row[columns[DATES_ID_COLUMN]]
        if subject is None:
            continue
        values = [
            row[columns[name]]
            for name in (DATES_BASELINE_COLUMN, DATES_RESECTION_COLUMN, DATES_POSTOP_COLUMN)
        ]
        if any(not isinstance(v, datetime) for v in values):
            logger.warning(f"{dates_file}: {subject} has missing/non-date dates {values}, ignored.")
            continue
        surgery_dates[str(subject).strip()] = SurgeryDates(*(v.date() for v in values))
    return surgery_dates


def assign_timepoints_from_dates(
    sessions: List[str], surgery_dates: SurgeryDates
) -> Optional[Dict[str, str]]:
    """
    Maps the MRI sessions of a patient to preop (before the resection date), intraop (on
    it), postop (after it, up to the postop date) and followup (later). Returns None if the
    dates do not fit the sessions: the baseline session must be the first MRI session, the
    postop date an MRI session and baseline < resection <= postop.
    """
    sessions = sorted(sessions, key=session_date)
    dates = [session_date(s) for s in sessions]
    if dates[0] != surgery_dates.baseline:
        logger.warning(
            f"Baseline {surgery_dates.baseline} is not the first MRI session ({sessions[0]})."
        )
        return None
    if surgery_dates.postop not in dates:
        logger.warning(f"postop date {surgery_dates.postop} is not an MRI session.")
        return None
    if not surgery_dates.baseline < surgery_dates.resection <= surgery_dates.postop:
        logger.warning(f"Dates are not ordered baseline < resection <= postop: {surgery_dates}.")
        return None
    timepoints = {}
    for session, session_day in zip(sessions, dates):
        if session_day < surgery_dates.resection:
            timepoints[session] = TIMEPOINT_PREOP
        elif session_day == surgery_dates.resection:
            timepoints[session] = TIMEPOINT_INTRAOP
        elif session_day <= surgery_dates.postop:
            timepoints[session] = TIMEPOINT_POSTOP
        else:
            timepoints[session] = TIMEPOINT_FOLLOWUP
    return timepoints


def reference_session(timepoints: Dict[str, str]) -> str:
    """The first preop session: the exam all others are co-registered and registered to."""
    return min((s for s, t in timepoints.items() if t == TIMEPOINT_PREOP), key=session_date)


def register_to_atlas(modalities: Dict[str, Path], outdir: Path) -> None:
    """
    Registers t1c to the SRI24 unstripped atlas with brainles (all other modalities, i.e.
    t1/t2/flair, co-registered to t1c first, atlas correction on, as in
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
        resample_to_preop(outdir, preop_outdir, modality)
    logger.info(f"{outdir}: rigidly co-registered to {preop_outdir}.")


def resample_to_preop(outdir: Path, preop_outdir: Path, modality: str) -> None:
    """
    Resamples <outdir>/atlas_registered/<modality>.nii.gz with the stored rigid transform
    to the preop exam (coregister_to_preop) into <outdir>/preop_coregistered/.
    """
    warped = ants.apply_transforms(
        fixed=ants.image_read(
            str(ATLAS_REGISTERED_SCHEMA.format(base_dir=preop_outdir, modality="t1c"))
        ),
        moving=ants.image_read(
            str(ATLAS_REGISTERED_SCHEMA.format(base_dir=outdir, modality=modality))
        ),
        transformlist=[str(PREOP_RIGID_TRAFO_SCHEMA.format(base_dir=outdir))],
        interpolator="linear",
        defaultvalue=0,
    )
    ants.image_write(
        warped, str(PREOP_COREGISTERED_SCHEMA.format(base_dir=outdir, modality=modality))
    )


def skull_strip(modality_files: Dict[str, Path], outdir: Path) -> None:
    """
    Skull strips t1c with SynthStrip through brainles' NativeSpacePreprocessor (t1c as the
    only modality, so nothing is re-registered), applies the t1c brain mask to the other
    modalities and stores them as <outdir>/skull_stripped/<modality>_skullstripped.nii.gz,
    percentile-normalized except the QUANTITATIVE_MODALITIES (e.g. md), which keep their
    values. The mask is stored as <outdir>/skull_stripped/brain_mask.nii.gz.
    """
    brain_mask_file = BRAIN_MASK_SCHEMA.format(base_dir=outdir)
    center = CenterModality(
        modality_name="t1c",
        input_path=str(modality_files["t1c"]),
        normalizer=PERCENTILE_NORMALIZER,
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

    for modality, modality_file in modality_files.items():
        if modality != "t1c":
            apply_brain_mask(modality, modality_file, outdir)


def apply_brain_mask(modality: str, modality_file: Path, outdir: Path) -> None:
    """
    Applies the t1c brain mask of skull_strip (<outdir>/skull_stripped/brain_mask.nii.gz)
    to modality_file and stores it as <outdir>/skull_stripped/<modality>_skullstripped.nii.gz,
    percentile-normalized like brainles' moving modalities except the
    QUANTITATIVE_MODALITIES, which are masked only.
    """
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
            normalizer=PERCENTILE_NORMALIZER,
            normalized_bet_output_path=str(stripped_file),
        )
    with tempfile.TemporaryDirectory(prefix="skull_strip_") as bet_dir:
        m.apply_bet_mask(
            brain_extractor=SynthStripExtractor(),
            mask_path=BRAIN_MASK_SCHEMA.format(base_dir=outdir),
            bet_dir=bet_dir,
        )
        m.save_current_image(stripped_file, normalization=not quantitative)


def correct_tumor_seg(outdir: Path, override: bool = False) -> None:
    """
    Writes the corrected tumor segmentation of an exam: its enhancing tumor is opened
    (binary_opening, ENHANCING_OPENING_ITERATIONS) and the removed voxels become edema; all
    other labels are kept. Existing outputs are kept unless override is True.
    """
    corrected_file = TUMORSEG_CORRECTED_SCHEMA.format(base_dir=outdir)
    if corrected_file.exists() and not override:
        logger.info(f"{outdir}: corrected tumor segmentation already done, skipping.")
        return
    seg_nifti = nib.load(str(TUMORSEG_SCHEMA.format(base_dir=outdir)))
    seg = np.rint(seg_nifti.get_fdata()).astype(np.int32)
    enhancing = seg == ENHANCING_LABEL
    removed = enhancing & ~binary_opening(enhancing, iterations=ENHANCING_OPENING_ITERATIONS)
    seg[removed] = EDEMA_LABEL
    nib.save(
        nib.Nifti1Image(seg.astype(seg_nifti.get_data_dtype()), seg_nifti.affine, seg_nifti.header),
        str(corrected_file),
    )
    update_config(
        outdir,
        CONFIG_STEP_CORRECT_TUMOR_SEG,
        {
            "opened_label": ENHANCING_LABEL,
            "replacement_label": EDEMA_LABEL,
            "opening_iterations": ENHANCING_OPENING_ITERATIONS,
        },
    )
    logger.info(
        f"{outdir}: corrected tumor segmentation written, {int(removed.sum())} of "
        f"{int(enhancing.sum())} enhancing voxels relabelled as edema."
    )


def process_exam(
    modalities: Dict[str, Path],
    dti_files: Optional[Tuple[Path, Path, Path]],
    outdir: Path,
    preop_outdir: Optional[Path],
    preop_timepoint: bool,
    cuda_device: str,
    override: bool = False,
) -> None:
    """
    Per-exam steps: atlas registration (skull on, unnormalized), rigid co-registration to
    the reference preop exam (all other exams, i.e. preop_outdir given), skull stripping
    and normalization, BraTS25_1 tumor segmentation (t1/t1c/t2/flair; preop branch if
    preop_timepoint, post branch otherwise), its corrected version (correct_tumor_seg), the
    atlas-based tissue segmentation with the tumor core masked and, if dti_files are given,
    the mean diffusivity (process_md). If
    override is True, all steps are rerun even if their outputs already exist.
    """
    outdir.mkdir(parents=True, exist_ok=True)
    is_preop = preop_outdir is None  # reference preop exam

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
            is_preop=preop_timepoint,
        )

    correct_tumor_seg(outdir, override=override)

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

    if dti_files is not None:
        process_md(
            dti_files,
            modalities["t1c"],
            outdir,
            preop_outdir,
            correct_distortion=modalities["t2"] != modalities["flair"],
            override=override,
        )


def process_md(
    dti_files: Tuple[Path, Path, Path],
    t1c_file: Path,
    outdir: Path,
    preop_outdir: Optional[Path],
    correct_distortion: bool,
    override: bool = False,
) -> None:
    """
    Mean diffusivity of an exam whose other per-exam steps are done: tensor fit, MD -> t1c
    registration and resampling of MD and b0 to atlas space (register_md_to_atlas),
    resampling to preop space with the exam's rigid transform (postop/followup), EPI
    distortion correction (correct_md_distortion, if correct_distortion) and masking with
    the t1c brain mask (unnormalized). Each output is skipped on its own if it exists
    (unless override), so only missing MD outputs are produced.
    """
    md_native_file = MD_NATIVE_SCHEMA.format(base_dir=outdir)
    b0_native_file = B0_NATIVE_SCHEMA.format(base_dir=outdir)
    if md_native_file.exists() and b0_native_file.exists() and not override:
        logger.info(f"{outdir}: mean diffusivity already fitted, skipping.")
    else:
        compute_md(*dti_files, outfile=md_native_file, b0_outfile=b0_native_file)
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
                "clip_max": MD_MAX,
            },
        )

    dti_modalities = (MD_MODALITY, B0_MODALITY)
    atlas_files = [
        ATLAS_REGISTERED_SCHEMA.format(base_dir=outdir, modality=m) for m in dti_modalities
    ]
    if all(f.exists() for f in atlas_files) and not override:
        logger.info(f"{outdir}: mean diffusivity atlas registration already done, skipping.")
    else:
        register_md_to_atlas(md_native_file, b0_native_file, t1c_file, outdir)
        update_config(
            outdir,
            CONFIG_STEP_REGISTER_MD,
            {
                "fixed_file": str(t1c_file),
                "registration_transform": MD_REGISTRATION_TRANSFORM,
                "registration_metric": MD_REGISTRATION_METRIC,
                "registration_init": MD_REGISTRATION_INIT,
                "transform_file": str(MD_TO_T1C_TRAFO_SCHEMA.format(base_dir=outdir)),
            },
        )

    if preop_outdir is None:
        coregistered_schema = ATLAS_REGISTERED_SCHEMA
    else:
        coregistered_schema = PREOP_COREGISTERED_SCHEMA
        for modality in dti_modalities:
            coregistered_file = PREOP_COREGISTERED_SCHEMA.format(base_dir=outdir, modality=modality)
            if coregistered_file.exists() and not override:
                logger.info(f"{outdir}: {modality} preop co-registration done, skipping.")
            else:
                resample_to_preop(outdir, preop_outdir, modality)
    md_coregistered_file = coregistered_schema.format(base_dir=outdir, modality=MD_MODALITY)

    warp_file = DISTORTION_WARP_SCHEMA.format(base_dir=outdir)
    if not correct_distortion:
        logger.warning(
            f"{outdir}: no {DISTORTION_FIXED_MODALITY}, mean diffusivity is not corrected "
            "for EPI distortion."
        )
    elif warp_file.exists() and not override:
        logger.info(f"{outdir}: DTI distortion correction already done, skipping.")
    else:
        correct_md_distortion(outdir, coregistered_schema)
        update_config(
            outdir,
            CONFIG_STEP_CORRECT_MD_DISTORTION,
            {
                "fixed_modality": DISTORTION_FIXED_MODALITY,
                "moving_modality": B0_MODALITY,
                "transform": DISTORTION_TRANSFORM,
                "metric": f"CC[radius {DISTORTION_METRIC_RADIUS}]",
                "restrict_deformation": DISTORTION_RESTRICT_DEFORMATION,
                "convergence": DISTORTION_CONVERGENCE,
                "shrink_factors": DISTORTION_SHRINK_FACTORS,
                "smoothing_sigmas": DISTORTION_SMOOTHING_SIGMAS,
                "warp_file": str(warp_file),
            },
        )

    md_stripped_file = MODALITY_STRIPPED_SCHEMA.format(base_dir=outdir, modality=MD_MODALITY)
    if md_stripped_file.exists() and not override:
        logger.info(f"{outdir}: mean diffusivity skull stripping already done, skipping.")
    elif not correct_distortion:
        apply_brain_mask(MD_MODALITY, md_coregistered_file, outdir)
    else:
        apply_distortion_and_brain_mask(MD_MODALITY, md_coregistered_file, warp_file, outdir)


def apply_distortion_and_brain_mask(
    modality: str, coregistered_file: Path, warp_file: Path, outdir: Path
) -> None:
    """
    Applies the DTI distortion field (correct_md_distortion) to a co-registered DTI-derived
    image and then the t1c brain mask (apply_brain_mask, normalized unless the modality is
    quantitative) into <outdir>/skull_stripped/<modality>_skullstripped.nii.gz.
    """
    with tempfile.TemporaryDirectory(prefix=f"{modality}_distortion_corrected_") as tmpdir:
        corrected_file = Path(tmpdir) / f"{modality}.nii.gz"
        img = ants.image_read(str(coregistered_file))
        ants.image_write(
            ants.apply_transforms(
                fixed=img,
                moving=img,
                transformlist=[str(warp_file)],
                interpolator="linear",
                defaultvalue=0,
            ),
            str(corrected_file),
        )
        apply_brain_mask(modality, corrected_file, outdir)


def process_trace(
    dti_files: Tuple[Path, Path, Path],
    outdir: Path,
    preop_outdir: Optional[Path],
    override: bool = False,
) -> None:
    """
    Trace-weighted image of a processed exam with a DTI whose MD outputs exist
    (-compute_trace): compute_trace on the DTI grid, resampling to atlas space with the
    stored MD -> t1c and t1c atlas transforms, to preop space with the exam's rigid
    transform (postop/followup), the DTI distortion field if it was estimated for the MD
    (otherwise a warning), the t1c brain mask with percentile normalization and, if the
    exam's longitudinal transform exists, the warp to the preop space of the longitudinal
    registration. Each output is skipped on its own if it exists (unless override).
    """
    md_trafo_file = MD_TO_T1C_TRAFO_SCHEMA.format(base_dir=outdir)
    atlas_trafo_file = ATLAS_T1C_TRAFO_SCHEMA.format(base_dir=outdir)
    missing = [f for f in (md_trafo_file, atlas_trafo_file) if not f.exists()]
    if missing:
        raise FileNotFoundError(f"MD transforms missing (run the MD step first): {missing}")

    trace_native_file = TRACE_NATIVE_SCHEMA.format(base_dir=outdir)
    if trace_native_file.exists() and not override:
        logger.info(f"{outdir}: trace already computed, skipping.")
    else:
        compute_trace(*dti_files, outfile=trace_native_file)

    atlas_file = ATLAS_REGISTERED_SCHEMA.format(base_dir=outdir, modality=TRACE_MODALITY)
    if atlas_file.exists() and not override:
        logger.info(f"{outdir}: trace atlas registration already done, skipping.")
    else:
        warped = ants.apply_transforms(
            fixed=ants.image_read(
                str(ATLAS_REGISTERED_SCHEMA.format(base_dir=outdir, modality="t1c"))
            ),
            moving=ants.image_read(str(trace_native_file)),
            transformlist=[str(atlas_trafo_file), str(md_trafo_file)],
            interpolator="linear",
            defaultvalue=0,
        )
        ants.image_write(warped, str(atlas_file))
        logger.info(f"{outdir}: trace moved to atlas space with the MD transforms.")

    if preop_outdir is None:
        coregistered_schema = ATLAS_REGISTERED_SCHEMA
    else:
        coregistered_schema = PREOP_COREGISTERED_SCHEMA
        coregistered_file = PREOP_COREGISTERED_SCHEMA.format(base_dir=outdir, modality=TRACE_MODALITY)
        if coregistered_file.exists() and not override:
            logger.info(f"{outdir}: trace preop co-registration done, skipping.")
        else:
            resample_to_preop(outdir, preop_outdir, TRACE_MODALITY)
    trace_coregistered_file = coregistered_schema.format(base_dir=outdir, modality=TRACE_MODALITY)

    warp_file = DISTORTION_WARP_SCHEMA.format(base_dir=outdir)
    correct_distortion = warp_file.exists()
    stripped_file = MODALITY_STRIPPED_SCHEMA.format(base_dir=outdir, modality=TRACE_MODALITY)
    if stripped_file.exists() and not override:
        logger.info(f"{outdir}: trace skull stripping already done, skipping.")
    elif not correct_distortion:
        logger.warning(
            f"{outdir}: no DTI distortion field (none estimated for the MD), trace is not "
            "corrected for EPI distortion."
        )
        apply_brain_mask(TRACE_MODALITY, trace_coregistered_file, outdir)
    else:
        apply_distortion_and_brain_mask(TRACE_MODALITY, trace_coregistered_file, warp_file, outdir)

    longitudinal_trafo_file = None
    if preop_outdir is not None:
        longitudinal_trafo_file = longitudinal_transform_file(outdir)
        warped_file = LONGITUDINAL_WARP_SCHEMA.format(base_dir=outdir, modality=TRACE_MODALITY)
        if not longitudinal_trafo_file.exists():
            logger.warning(f"{outdir}: no longitudinal transform, trace not warped to preop space.")
            longitudinal_trafo_file = None
        elif warped_file.exists() and not override:
            logger.info(f"{outdir}: trace longitudinal warp already done, skipping.")
        else:
            warp_longitudinal(preop_outdir, outdir, TRACE_MODALITY)

    update_config(
        outdir,
        CONFIG_STEP_COMPUTE_TRACE,
        {
            "dti_file": str(dti_files[0]),
            "trace": "geometric mean of the diffusion-weighted volumes",
            "b0_threshold": DTI_B0_THRESHOLD,
            "md_to_t1c_transform_file": str(md_trafo_file),
            "atlas_transform_file": str(atlas_trafo_file),
            "preop_rigid_transform_file": (
                None if preop_outdir is None else str(PREOP_RIGID_TRAFO_SCHEMA.format(base_dir=outdir))
            ),
            "distortion_warp_file": str(warp_file) if correct_distortion else None,
            "normalization": "percentile",
            "longitudinal_transform_file": (
                None if longitudinal_trafo_file is None else str(longitudinal_trafo_file)
            ),
        },
    )


def register_adc_to_atlas(adc_file: Path, t1c_file: Path, outdir: Path) -> None:
    """
    Rigidly registers the ADC map (clipped to its ADC_CLIP_PERCENTILE for the registration)
    to the original-space t1c with skull with the MD settings
    (MD_REGISTRATION_TRANSFORM/_METRIC/_INIT), stores the transform as
    <outdir>/atlas_registered/transformations/adc/adc_to_t1c.mat and resamples the raw ADC
    with it followed by the brainles t1c atlas transform to
    <outdir>/atlas_registered/adc.nii.gz.
    """
    adc_img = ants.image_read(str(adc_file))
    values = adc_img.numpy()
    positive = values[values > 0]
    clip = float(np.percentile(positive, ADC_CLIP_PERCENTILE)) if positive.size else 0.0
    registration_img = adc_img.new_image_like(np.minimum(values, clip).astype(np.float32))
    trafo_file = ADC_TO_T1C_TRAFO_SCHEMA.format(base_dir=outdir)
    trafo_file.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="register_adc_") as tmpdir:
        reg = ants.registration(
            fixed=ants.image_read(str(t1c_file)),
            moving=registration_img,
            type_of_transform=MD_REGISTRATION_TRANSFORM,
            aff_metric=MD_REGISTRATION_METRIC,
            initial_transform=MD_REGISTRATION_INIT,
            outprefix=f"{tmpdir}/rigid_",
        )
        shutil.move(reg["fwdtransforms"][0], str(trafo_file))

    warped = ants.apply_transforms(
        fixed=ants.image_read(str(ATLAS_REGISTERED_SCHEMA.format(base_dir=outdir, modality="t1c"))),
        moving=adc_img,
        transformlist=[str(ATLAS_T1C_TRAFO_SCHEMA.format(base_dir=outdir)), str(trafo_file)],
        interpolator="linear",
        defaultvalue=0,
    )
    ants.image_write(warped, str(ATLAS_REGISTERED_SCHEMA.format(base_dir=outdir, modality=ADC_MODALITY)))
    logger.info(
        f"{outdir}: ADC registered to t1c (clipped at {clip:.0f} for the registration) and "
        "moved to atlas space."
    )


def process_adc(
    adc_file: Path,
    t1c_file: Path,
    outdir: Path,
    preop_outdir: Optional[Path],
    override: bool = False,
) -> None:
    """
    ADC map of a processed exam without a DTI (-register_adc): ADC -> t1c registration and
    resampling to atlas space (register_adc_to_atlas), to preop space with the exam's rigid
    transform (postop/followup), the t1c brain mask with percentile normalization and, if
    the exam's longitudinal transform exists, the warp to the preop space of the
    longitudinal registration. Each output is skipped on its own if it exists (unless
    override).
    """
    atlas_trafo_file = ATLAS_T1C_TRAFO_SCHEMA.format(base_dir=outdir)
    if not atlas_trafo_file.exists():
        raise FileNotFoundError(f"t1c atlas transform missing: {atlas_trafo_file}")

    atlas_file = ATLAS_REGISTERED_SCHEMA.format(base_dir=outdir, modality=ADC_MODALITY)
    if atlas_file.exists() and not override:
        logger.info(f"{outdir}: ADC atlas registration already done, skipping.")
    else:
        register_adc_to_atlas(adc_file, t1c_file, outdir)

    if preop_outdir is None:
        coregistered_schema = ATLAS_REGISTERED_SCHEMA
    else:
        coregistered_schema = PREOP_COREGISTERED_SCHEMA
        coregistered_file = PREOP_COREGISTERED_SCHEMA.format(base_dir=outdir, modality=ADC_MODALITY)
        if coregistered_file.exists() and not override:
            logger.info(f"{outdir}: ADC preop co-registration done, skipping.")
        else:
            resample_to_preop(outdir, preop_outdir, ADC_MODALITY)

    stripped_file = MODALITY_STRIPPED_SCHEMA.format(base_dir=outdir, modality=ADC_MODALITY)
    if stripped_file.exists() and not override:
        logger.info(f"{outdir}: ADC skull stripping already done, skipping.")
    else:
        apply_brain_mask(
            ADC_MODALITY, coregistered_schema.format(base_dir=outdir, modality=ADC_MODALITY), outdir
        )

    longitudinal_trafo_file = None
    if preop_outdir is not None:
        longitudinal_trafo_file = longitudinal_transform_file(outdir)
        warped_file = LONGITUDINAL_WARP_SCHEMA.format(base_dir=outdir, modality=ADC_MODALITY)
        if not longitudinal_trafo_file.exists():
            logger.warning(f"{outdir}: no longitudinal transform, ADC not warped to preop space.")
            longitudinal_trafo_file = None
        elif warped_file.exists() and not override:
            logger.info(f"{outdir}: ADC longitudinal warp already done, skipping.")
        else:
            warp_longitudinal(preop_outdir, outdir, ADC_MODALITY)

    update_config(
        outdir,
        CONFIG_STEP_REGISTER_ADC,
        {
            "adc_file": str(adc_file),
            "fixed_file": str(t1c_file),
            "registration_transform": MD_REGISTRATION_TRANSFORM,
            "registration_metric": MD_REGISTRATION_METRIC,
            "registration_init": MD_REGISTRATION_INIT,
            "registration_clip_percentile": ADC_CLIP_PERCENTILE,
            "transform_file": str(ADC_TO_T1C_TRAFO_SCHEMA.format(base_dir=outdir)),
            "atlas_transform_file": str(atlas_trafo_file),
            "preop_rigid_transform_file": (
                None if preop_outdir is None else str(PREOP_RIGID_TRAFO_SCHEMA.format(base_dir=outdir))
            ),
            "distortion_corrected": False,
            "normalization": "percentile",
            "longitudinal_transform_file": (
                None if longitudinal_trafo_file is None else str(longitudinal_trafo_file)
            ),
        },
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


def warp_longitudinal(preop_outdir: Path, exam_outdir: Path, modality: str) -> None:
    """
    Warps a skull-stripped modality of an exam (e.g. the mean diffusivity or the trace) to
    preop space with its stored longitudinal transform (as register_recurrence does for its
    additional modalities), for outputs that postdate the longitudinal registration.
    """
    warped = ants.apply_transforms(
        fixed=ants.image_read(
            str(MODALITY_STRIPPED_SCHEMA.format(base_dir=preop_outdir, modality="t1c"))
        ),
        moving=ants.image_read(
            str(MODALITY_STRIPPED_SCHEMA.format(base_dir=exam_outdir, modality=modality))
        ),
        transformlist=[str(longitudinal_transform_file(exam_outdir))],
    )
    ants.image_write(
        warped, str(LONGITUDINAL_WARP_SCHEMA.format(base_dir=exam_outdir, modality=modality))
    )
    logger.info(f"{exam_outdir}: {modality} warped to preop space.")


def warp_corrected_recurrence(preop_outdir: Path, exam_outdir: Path) -> None:
    """
    Warps the corrected tumor segmentation of an exam to preop space with its stored
    longitudinal transform, as register_recurrence does for the original one (nearest
    neighbor). The corrected segmentation is written first if the exam does not have one yet.
    """
    correct_tumor_seg(exam_outdir)
    fixed = ants.image_read(
        str(MODALITY_STRIPPED_SCHEMA.format(base_dir=preop_outdir, modality="t1c"))
    )
    warped = ants.apply_transforms(
        fixed=fixed.clone("unsigned int"),
        moving=ants.image_read(str(TUMORSEG_CORRECTED_SCHEMA.format(base_dir=exam_outdir))),
        transformlist=[str(longitudinal_transform_file(exam_outdir))],
        interpolator="nearestNeighbor",
    )
    ants.image_write(warped, str(RECURRENCE_CORRECTED_SCHEMA.format(base_dir=exam_outdir)))
    logger.info(f"{exam_outdir}: corrected tumor segmentation warped to preop space.")


def normalize_series_description(description: str) -> str:
    """SeriesDescription without whitespace, case-folded, for matching."""
    return "".join(description.split()).casefold()


def is_series(image_file: Path, series_description: str) -> bool:
    """
    Whether the json sidecar of image_file has ImageType ORIGINAL and a SeriesDescription
    equal to series_description (normalize_series_description). False without sidecar.
    """
    json_file = image_file.with_name(image_file.name.replace(".nii.gz", ".json"))
    if not json_file.exists():
        return False
    with open(json_file) as f:
        sidecar = json.load(f)
    image_type = sidecar.get("ImageType") or [""]
    return image_type[0] == "ORIGINAL" and normalize_series_description(
        sidecar.get("SeriesDescription", "")
    ) == normalize_series_description(series_description)


def resolve_pet_ct(patient_dir: Path, session: str) -> Optional[Tuple[Path, Path]]:
    """
    The (PET, CT) image pair of a session, selected by PET_SERIES_DESCRIPTION and
    CT_SERIES_DESCRIPTION (is_series), ignoring AppleDouble files. Returns None if the session
    has no PET or no CT image (silently) or not exactly one matching series of each (with a
    warning).
    """
    session_dir = patient_dir / session
    pet_files = sorted(
        f
        for pattern in PET_FILE_GLOBS
        for f in session_dir.glob(pattern)
        if not f.name.startswith(APPLEDOUBLE_PREFIX)
    )
    ct_files = sorted(
        f for f in session_dir.glob(CT_FILE_GLOB) if not f.name.startswith(APPLEDOUBLE_PREFIX)
    )
    if not pet_files or not ct_files:
        return None
    pet_matches = [f for f in pet_files if is_series(f, PET_SERIES_DESCRIPTION)]
    ct_matches = [f for f in ct_files if is_series(f, CT_SERIES_DESCRIPTION)]
    if len(pet_matches) != 1 or len(ct_matches) != 1:
        logger.warning(
            f"{patient_dir.name}/{session}: {len(pet_matches)}/{len(pet_files)} PET images "
            f"match '{PET_SERIES_DESCRIPTION}' and {len(ct_matches)}/{len(ct_files)} CT images "
            f"match '{CT_SERIES_DESCRIPTION}', need one each, skipping PET session."
        )
        return None
    return pet_matches[0], ct_matches[0]


def pet_tumor_mask_file(
    pet_session: str,
    timepoints: Dict[str, str],
    exam_outdirs: Dict[str, Path],
    reference: str,
    resection_date: Optional[date],
) -> Optional[Path]:
    """
    Tumor mask in preop space that is excluded from the PET background region. PET sessions
    dated before the resection (resection_date from the dates sheet, otherwise the first
    non-preop MRI session) use the tumor segmentation of the reference preop exam, later
    ones the recurrence_preop of the date-nearest processed non-preop exam (ties go to the
    earlier exam). Returns None if there is none.
    """
    pet_date = session_date(pet_session)
    if resection_date is None:
        resection_dates = [
            session_date(s) for s, t in timepoints.items() if t != TIMEPOINT_PREOP
        ]
        resection_date = min(resection_dates) if resection_dates else None
    if resection_date is None or pet_date < resection_date:
        return TUMORSEG_SCHEMA.format(base_dir=exam_outdirs[reference])
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
    labels), clipping at 0. The registration is checked with the bone-in-brain QC and
    retried from shifted starts if it fails (see PET_QC_* / PET_RETRY_OFFSET_MM). Writes
    <outdir>/longitudinal/{ct,pet}_warped_longitudinal.nii.gz and longitudinal_trafo.mat.
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
    deep_brain_mask = (
        distance_transform_edt(brain_mask, sampling=brain_mask_img.spacing)
        > PET_QC_BRAIN_DEPTH_MM
    )

    ct_img = ants.image_read(str(ct_file))
    hu_min, hu_max = CT_REGISTRATION_HU_WINDOW
    ct_reg_img = ct_img.new_image_like(np.clip(ct_img.numpy(), hu_min, hu_max) - hu_min)

    def register_ct(initial_transform: Optional[str], tmpdir: str, tag: str) -> Tuple[str, float]:
        """Rigid CT registration from initial_transform (None: center of mass); returns the
        transform file and the fraction of the deep brain mask that is bone in the warped CT."""
        reg = ants.registration(
            fixed=t1c_pre_img,
            moving=ct_reg_img,
            type_of_transform=PET_REGISTRATION_TRANSFORM,
            aff_metric=PET_REGISTRATION_METRIC,
            mask=fixed_mask_img,
            initial_transform=initial_transform,
            random_seed=PET_REGISTRATION_SEED,
            outprefix=f"{tmpdir}/{tag}_",
        )
        warped = ants.apply_transforms(
            fixed=t1c_pre_img,
            moving=ct_img,
            transformlist=reg["fwdtransforms"],
            defaultvalue=float(ct_img.min()),
        )
        bone_fraction = float((warped.numpy()[deep_brain_mask] > PET_QC_BONE_HU).mean())
        return reg["fwdtransforms"][0], bone_fraction

    affine_file = LONGITUDINAL_AFFINE_SCHEMA.format(base_dir=outdir)
    affine_file.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="register_pet_") as tmpdir:
        best_file, best_fraction = register_ct(None, tmpdir, "com")
        n_starts = 1
        if best_fraction > PET_QC_MAX_BONE_FRACTION:
            logger.info(
                f"{outdir}: CT registration QC failed ({100 * best_fraction:.1f}% bone in the "
                f"brain), retrying from {PET_RETRY_OFFSET_MM:.0f} mm shifted starts."
            )
            # Center-of-mass translation (fixed -> moving points), as ANTs initializes it.
            com_shift = np.array(ants.get_center_of_mass(ct_reg_img)) - np.array(
                ants.get_center_of_mass(t1c_pre_img)
            )
            for axis in range(3):
                for sign in (-1, 1):
                    offset = np.zeros(3)
                    offset[axis] = sign * PET_RETRY_OFFSET_MM
                    init_file = f"{tmpdir}/init_{axis}_{sign}.mat"
                    ants.write_transform(
                        ants.create_ants_transform(
                            transform_type="AffineTransform",
                            dimension=3,
                            translation=tuple(com_shift + offset),
                        ),
                        init_file,
                    )
                    trafo_file, fraction = register_ct(init_file, tmpdir, f"retry_{axis}_{sign}")
                    n_starts += 1
                    if fraction < best_fraction:
                        best_file, best_fraction = trafo_file, fraction
        shutil.move(best_file, str(affine_file))
    if best_fraction > PET_QC_MAX_BONE_FRACTION:
        logger.warning(
            f"{outdir}: CT registration still fails QC after {n_starts} starts "
            f"({100 * best_fraction:.1f}% bone in the brain), check the PET registration."
        )
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
        f"{outdir}: registered {ct_file.name} ({n_starts} start(s), "
        f"{100 * best_fraction:.2f}% bone in the brain), warped {pet_file.name}, background "
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
            "registration_seed": PET_REGISTRATION_SEED,
            "registration_starts": n_starts,
            "qc_bone_in_brain_fraction": best_fraction,
            "qc_bone_hu": PET_QC_BONE_HU,
            "qc_brain_depth_mm": PET_QC_BRAIN_DEPTH_MM,
            "qc_max_bone_fraction": PET_QC_MAX_BONE_FRACTION,
            "retry_offset_mm": PET_RETRY_OFFSET_MM,
            "brain_mask_dilation_mm": PET_BRAIN_MASK_DILATION_MM,
            "ct_registration_hu_window": list(CT_REGISTRATION_HU_WINDOW),
            "tumor_mask_file": str(tumor_mask_file),
            "background_mean": background,
            "clipped_at_zero": True,
        },
    )


def write_session_labels(
    outfile: Path,
    subject: str,
    timepoints: Dict[str, str],
    processed: Dict[str, bool],
    reference: str,
    label_source: str,
) -> None:
    """
    Writes a tsv with columns subject, session, timepoint, processed (whether the exam's
    per-exam steps completed), reference (1 for the exam all others are registered to)
    and label_source (LABEL_SOURCE_SHEET or LABEL_SOURCE_RULE) for the MRI sessions of a
    patient.
    """
    with open(outfile, "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t", lineterminator="\n")
        writer.writerow(
            ["subject", "session", "timepoint", "processed", "reference", "label_source"]
        )
        for session in sorted(timepoints, key=session_date):
            writer.writerow(
                [
                    subject,
                    session,
                    timepoints[session],
                    int(processed.get(session, False)),
                    int(session == reference),
                    label_source,
                ]
            )
    logger.info(f"Wrote {outfile}.")


if __name__ == "__main__":
    # Full processing (per-exam steps, longitudinal registration) of the original-space MRI
    # sessions and of the PET/CT sessions (registration to the preop exam) of every patient;
    # steps with existing outputs are skipped unless -override is given. Outputs go to
    # <patient_dir>/processed/<session>/:
    # nohup python -u scripts/process_respond_longitudinal.py -cuda_device 0 > tmp_process_respond_longitudinal.out 2>&1 &
    # Only one patient:
    # python -u scripts/process_respond_longitudinal.py -cuda_device 0 -patients sub-GR008
    # The original tum_149 directory:
    # python -u scripts/process_respond_longitudinal.py -cuda_device 0 -data_dir /mnt/Drive4/lucas/respond_longitudinal -patient_glob 'tum_*'
    # Recompute the postop/followup-to-preop registrations only (per-exam outputs reused):
    # python -u scripts/process_respond_longitudinal.py -cuda_device 0 -longitudinal_registration
    # PET sessions only (needs the processed MRI exams):
    # python -u scripts/process_respond_longitudinal.py -pet_only
    # Trace of every processed exam with a DTI only (needs the MD outputs; CPU only):
    # python -u scripts/process_respond_longitudinal.py -compute_trace
    # ADC of every processed exam without a DTI only (registered to the t1c; CPU only):
    # python -u scripts/process_respond_longitudinal.py -register_adc
    parser = argparse.ArgumentParser()
    parser.add_argument("-cuda_device", type=str, default="0", help="GPU id to run on.")
    parser.add_argument(
        "-data_dir",
        type=str,
        default="/mnt/Drive4/lucas/respond_longitudinal/lucas_long_followup",
        help="Directory containing one directory per patient (matching -patient_glob) with "
        "ses-YYYYMMDD session directories of original-space niftis.",
    )
    parser.add_argument(
        "-patient_glob",
        type=str,
        default=DEFAULT_PATIENT_DIR_GLOB,
        help="Glob of the patient directory names in -data_dir (e.g. 'tum_*').",
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
        help="Patient directory names to process (e.g. sub-GR008). All patients if omitted.",
    )
    parser.add_argument(
        "-dates_file",
        type=str,
        default=None,
        help=f"Surgery dates sheet the timepoints are taken from. Defaults to "
        f"<data_dir>/{DATES_FILENAME} if it exists; subjects missing from it (or without a "
        "sheet) are labelled by the first/second-session rule.",
    )
    parser.add_argument(
        "-postop_window_days",
        type=int,
        default=DEFAULT_POSTOP_WINDOW_DAYS,
        help="Rule labelling only (subjects not in the dates sheet): MRI sessions up to this "
        "many days after the postop session (the second MRI session) are labelled postop as "
        "well; later ones are followups.",
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
        "-compute_trace",
        action="store_true",
        help="Only compute the trace-weighted image (geometric mean of the diffusion-weighted "
        "DTI volumes) of every processed exam with a DTI, in the spaces of the MD outputs "
        "with the stored MD transforms; with -override existing trace outputs are recomputed.",
    )
    parser.add_argument(
        "-register_adc",
        action="store_true",
        help="Only register the ADC map of every processed exam without a DTI to the t1c and "
        "bring it to the spaces of the MD/trace outputs; with -override existing ADC outputs "
        "are recomputed. Can be combined with -compute_trace.",
    )
    parser.add_argument(
        "-override",
        action="store_true",
        help="Rerun every step even if its output already exists, overwriting previous results.",
    )
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device

    data_dir = Path(args.data_dir)
    patient_dirs = sorted(p for p in data_dir.glob(args.patient_glob) if p.is_dir())
    if args.patients is not None:
        patient_dirs = [p for p in patient_dirs if p.name in args.patients]
    logger.info(f"Found {len(patient_dirs)} patients in {data_dir}.")

    dates_file = Path(args.dates_file) if args.dates_file else data_dir / DATES_FILENAME
    if dates_file.exists():
        all_surgery_dates = read_surgery_dates(dates_file)
        logger.info(f"Read surgery dates of {len(all_surgery_dates)} subjects from {dates_file}.")
    elif args.dates_file:
        raise FileNotFoundError(f"Dates file {dates_file} not found.")
    else:
        all_surgery_dates = {}
        logger.info(f"No {dates_file}, labelling all subjects by the session rule.")

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
        surgery_dates = all_surgery_dates.get(subject)
        timepoints = None
        if surgery_dates is not None:
            timepoints = assign_timepoints_from_dates(sessions, surgery_dates)
            if timepoints is None:
                logger.warning(
                    f"{patient_id}: dates sheet does not fit the MRI sessions, falling back "
                    "to the session rule."
                )
        if timepoints is None:
            surgery_dates = None
            timepoints = assign_timepoints(sessions, args.postop_window_days)
        label_source = LABEL_SOURCE_RULE if surgery_dates is None else LABEL_SOURCE_SHEET
        reference = reference_session(timepoints)
        logger.info(
            f"{patient_id} (sub-{subject}, {label_source}, reference {reference}): "
            + ", ".join(f"{s} {timepoints[s]}" for s in sorted(timepoints, key=session_date))
        )

        # The reference (first preop) exam is processed first: the others are co-registered
        # to it. It is the first session by date.
        exam_outdirs: Dict[str, Path] = {}  # session -> outdir of processed exams
        preop_outdir: Optional[Path] = None
        for session in sorted(timepoints, key=session_date):
            is_preop = session == reference
            if not is_preop and preop_outdir is None:
                logger.warning(f"{patient_id}/{session}: no processed preop exam, skipping exam.")
                continue
            modalities = resolve_modalities(patient_dir, subject, session)
            if modalities is None:
                continue

            exam_outdir = outdir_root / session
            if args.longitudinal_registration or args.pet_only or args.compute_trace or args.register_adc:
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
                    preop_timepoint=timepoints[session] == TIMEPOINT_PREOP,
                    cuda_device=args.cuda_device,
                    override=args.override,
                )
                exam_outdirs[session] = exam_outdir
                if is_preop:
                    preop_outdir = exam_outdir
            except Exception:
                logger.exception(f"{patient_id}/{session}: processing failed, skipping.")

        if not (args.pet_only or args.compute_trace or args.register_adc):
            outdir_root.mkdir(parents=True, exist_ok=True)
            write_session_labels(
                outfile=outdir_root / SESSION_LABELS_FILENAME,
                subject=subject,
                timepoints=timepoints,
                processed={s: s in exam_outdirs for s in timepoints},
                reference=reference,
                label_source=label_source,
            )

        if preop_outdir is None:
            logger.warning(
                f"{patient_id}: no processed preop exam, skipping longitudinal registration."
            )
            continue

        if args.compute_trace or args.register_adc:
            # Trace of every processed exam with a DTI (with the transforms stored for its MD)
            # and / or the ADC of every processed exam without one.
            for session, exam_outdir in exam_outdirs.items():
                dti_files = resolve_dti(patient_dir, subject, session)
                exam_preop_outdir = None if session == reference else preop_outdir
                if args.compute_trace and dti_files is not None:
                    try:
                        process_trace(
                            dti_files, exam_outdir, preop_outdir=exam_preop_outdir, override=args.override
                        )
                    except Exception:
                        logger.exception(f"{patient_id}/{session}: trace failed, skipping.")
                if args.register_adc and dti_files is None:
                    adc_file = resolve_adc(patient_dir, subject, session)
                    if adc_file is None:
                        logger.warning(f"{patient_id}/{session}: neither DTI nor ADC, nothing to register.")
                        continue
                    try:
                        process_adc(
                            adc_file,
                            ORG_FILE_SCHEMA.format(
                                patient_dir=patient_dir, subject=subject, session=session, suffix="t1c"
                            ),
                            exam_outdir,
                            preop_outdir=exam_preop_outdir,
                            override=args.override,
                        )
                    except Exception:
                        logger.exception(f"{patient_id}/{session}: ADC registration failed, skipping.")
            continue

        # Every exam except the reference one is registered to the reference preop exam.
        for session, exam_outdir in exam_outdirs.items():
            if args.pet_only or session == reference:
                continue
            recurrence_file = RECURRENCE_SCHEMA.format(base_dir=exam_outdir)
            if args.longitudinal_registration:
                logger.info(f"{patient_id}/{session}: recomputing longitudinal registration.")
            elif recurrence_file.exists() and not args.override:
                logger.info(
                    f"{patient_id}/{session}: longitudinal registration already done, skipping."
                )
                # The MD may have been (re)computed after the registration.
                md_warped_file = LONGITUDINAL_WARP_SCHEMA.format(
                    base_dir=exam_outdir, modality=MD_MODALITY
                )
                has_md = MD_MODALITY in longitudinal_modalities(exam_outdir)
                if has_md and not md_warped_file.exists():
                    try:
                        warp_longitudinal(preop_outdir, exam_outdir, MD_MODALITY)
                    except Exception:
                        logger.exception(f"{patient_id}/{session}: MD warp failed, skipping.")
                # The corrected segmentation may postdate the registration as well.
                if not RECURRENCE_CORRECTED_SCHEMA.format(base_dir=exam_outdir).exists():
                    try:
                        warp_corrected_recurrence(preop_outdir, exam_outdir)
                    except Exception:
                        logger.exception(
                            f"{patient_id}/{session}: corrected segmentation warp failed, skipping."
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
                warp_corrected_recurrence(preop_outdir, exam_outdir)
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
            tumor_mask_file = pet_tumor_mask_file(
                session,
                timepoints,
                exam_outdirs,
                reference=reference,
                resection_date=None if surgery_dates is None else surgery_dates.resection,
            )
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
