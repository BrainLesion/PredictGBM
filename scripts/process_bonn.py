import os
import re
import argparse
import ants
import nibabel as nib
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from loguru import logger
from predict_gbm.utils.constants import (
    BRAIN_MASK_SCHEMA,
    MODALITY_STRIPPED_SCHEMA,
    OUTPUT_BASE_SCHEMA,
    PathSchema,
    REGISTRATION_MASK_SCHEMA,
    REGISTRATION_TRAFO_SCHEMA,
    TISSUE_PBMAP_SCHEMA,
    TUMORSEG_SCHEMA,
)
from predict_gbm.preprocessing import norm_ss_coregister, run_brats, run_tissue_seg
from predict_gbm.preprocessing.tissue_segmentation import generate_registration_mask
from predict_gbm.utils.utils import update_config

# Atlas the modalities are registered to (and the tissue segmentation is based on).
ATLAS = "sri24"
NIFTI_SUFFIX = ".nii.gz"
# Case directories are named "nifti", "nifti 2", ..., "nifti 10" (with spaces); the
# outputs use bonn_1 ... bonn_10 (bonn_1 == "nifti", matching the existing test dataset).
CASE_DIR_PATTERN = re.compile(r"^nifti(?: (\d+))?$")
CASE_ID_SCHEMA = "bonn_{number}"

# Per-case files of the four BraTS modalities. t1c is the center modality of the
# co-registration; everything else in the case directory is registered to it.
CASE_MODALITY_FILES: Dict[str, Dict[str, str]] = {
    "nifti": {
        "t1c": "T1_Navigator_KM",
        "t1": "T1_Navigator_nativ",
        "t2": "T2W_iso_Planung",
        "flair": "3D_FLAIR_cs6",
    },
    "nifti 2": {
        "t1c": "T1_Navigator_KM",
        "t1": "T1_Navigator_nativ",
        "t2": "T2W_iso_Planung",
        "flair": "3D_FLAIR_cs6",
    },
    "nifti 3": {
        "t1c": "Navigator_KM",
        "t1": "Navigator_nativ",
        "t2": "T2W_TSE_iso_Voxel",
        "flair": "3D_Brain_VIEW_FLAIR",
    },
    "nifti 9": {
        "t1c": "Navigator_KM",
        "t1": "Navigator_nativ",
        "t2": "T2W_TSE_iso_Voxel",
        "flair": "3D_Brain_VIEW_FLAIR",
    },
}
for _case in ("nifti 4", "nifti 5", "nifti 6", "nifti 7", "nifti 8", "nifti 10"):
    CASE_MODALITY_FILES[_case] = {
        "t1c": "REC_SAG_0,5mm_KM",
        "t1": "REC_SAG_0,5mm",
        "t2": "T2W_iso_Planung",
        "flair": "3D_FLAIR_cs6",
    }

# Physical/quantitative series (file name patterns, case-insensitive) that are
# co-registered without intensity normalization: ADC and diffusion-weighted images,
# perfusion (dynamic susceptibility contrast, EPI and PRESTO), DTI and the SWI phase.
QUANTITATIVE_PATTERNS = ("adc", "diff", "perfusion", "presto", "dti", "phase")

# The perfusion, DTI and diffusion series were converted as 3D stacks of all volumes along
# z, with the slice spacing in the header set to the slab extent divided by the number of
# images. Two stacking orders occur (verified on the data in sagittal view): "slice" major
# (all volumes of slice 0, then slice 1, ...; perfusion, PRESTO, DTI, S_DIFFssh) and
# "volume" major (all slices of volume 0, then volume 1, ...; DIFF_SENSE2.2). Each entry
# maps a file name pattern to the number of volumes (25 slices x 40 dynamics, 30 x 40,
# 70 x 34 and 24 x 3 b-values), the stacking order, and how the 3D reference image that
# drives the registration of the whole series is derived ("mean" over the volumes or the
# "first" volume, i.e. b0 for diffusion data).
STACKED_SERIES: Tuple[Tuple[str, int, str, str], ...] = (
    ("perfusion", 40, "slice", "mean"),
    ("presto", 40, "slice", "mean"),
    ("dti", 34, "slice", "first"),
    ("diffssh", 3, "slice", "first"),
    ("diff_sense", 3, "volume", "first"),
)
STACKED_REFERENCE_SUFFIX = "_ref"
# Unstacked 4D series (corrected header) and their 3D references, used as registration
# input; the registered 4D series are stored next to the other modalities.
UNSTACKED_SCHEMA = OUTPUT_BASE_SCHEMA / "unstacked" / "{modality}.nii.gz"
CONFIG_STEP_STACKED_SERIES = "unstack_4d_series"
# Normalization is skipped for all 4D series (physical modalities). The 4D output is
# resampled with linear interpolation and masked with the brain mask like the 3D ones.
FOURD_INTERPOLATOR = "linear"


def case_id(case_dir: Path) -> Optional[str]:
    """bonn_<n> for a case directory named "nifti" (n=1) or "nifti <n>"; None otherwise."""
    match = CASE_DIR_PATTERN.match(case_dir.name)
    if match is None:
        return None
    return CASE_ID_SCHEMA.format(number=match.group(1) or "1")


def sanitize_modality_name(stem: str) -> str:
    """Turns a nifti filename stem into a lowercase, filesystem-safe modality name."""
    return re.sub(r"[^0-9a-zA-Z]+", "_", stem).strip("_").lower()


def is_quantitative(name: str) -> bool:
    """Whether a file name denotes a physical series that must not be normalized."""
    return any(p in name.lower() for p in QUANTITATIVE_PATTERNS)


def stacked_layout(name: str) -> Optional[Tuple[int, str, str]]:
    """(volumes, stacking order, reference kind) if the file is a stacked 4D series, else None."""
    sanitized = sanitize_modality_name(name)
    for pattern, n_volumes, order, reference in STACKED_SERIES:
        if pattern in sanitized:
            return n_volumes, order, reference
    return None


def discover_files(case_dir: Path) -> List[Path]:
    """
    All nifti files of a case directory, excluding the macOS "._" sidecar files. Only
    .nii.gz files are considered (some cases hold duplicate .nii copies).
    """
    return sorted(
        f for f in case_dir.glob(f"*{NIFTI_SUFFIX}") if not f.name.startswith("._")
    )


def unstack_4d_series(
    stacked_file: Path,
    n_volumes: int,
    order: str,
    reference: str,
    outfile: Path,
    reference_file: Path,
) -> Dict[str, object]:
    """
    Converts a 3D stack of n_volumes volumes (order "slice": index = slice * n_volumes +
    volume; order "volume": index = volume * n_slices + slice) into a 4D nifti with
    corrected slice spacing, and derives the 3D reference image (mean over the volumes or
    the first volume) used to register the series. The z direction and origin (first
    slice) of the stack are kept; the slice spacing is recomputed from the slab extent
    encoded in the stacked header: extent = spacing * (n_images - 1) = thickness *
    (n_slices - 1). Returns the layout parameters for the config file.
    """
    img = nib.load(str(stacked_file))
    data = np.asanyarray(img.dataobj)
    if data.ndim != 3:
        raise ValueError(f"{stacked_file.name}: expected a 3D stack, got {data.ndim}D.")
    nx, ny, n_images = data.shape
    if n_images % n_volumes != 0:
        raise ValueError(
            f"{stacked_file.name}: {n_images} images are not divisible by {n_volumes} "
            "volumes per slice."
        )
    n_slices = n_images // n_volumes
    if n_slices < 2:
        raise ValueError(f"{stacked_file.name}: only {n_slices} slice(s) after unstacking.")

    if order == "slice":
        data_4d = data.reshape(nx, ny, n_slices, n_volumes)
    elif order == "volume":
        data_4d = data.reshape(nx, ny, n_volumes, n_slices).transpose(0, 1, 3, 2)
    else:
        raise ValueError(f"Unknown stacking order '{order}'.")
    data_4d = np.ascontiguousarray(data_4d, dtype=np.float32)

    affine = img.affine.copy()
    stacked_spacing = float(np.linalg.norm(affine[:3, 2]))
    thickness = stacked_spacing * (n_images - 1) / (n_slices - 1)
    affine[:3, 2] = affine[:3, 2] / stacked_spacing * thickness

    outfile.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(data_4d, affine), str(outfile))

    if reference == "mean":
        reference_data = data_4d.mean(axis=3)
    elif reference == "first":
        reference_data = data_4d[..., 0]
    else:
        raise ValueError(f"Unknown reference kind '{reference}'.")
    nib.save(nib.Nifti1Image(reference_data, affine), str(reference_file))

    logger.info(
        f"Unstacked {stacked_file.name}: {n_images} images -> {n_slices} slices x "
        f"{n_volumes} volumes ({order}-major), slice thickness {thickness:.2f} mm, "
        f"reference '{reference}'. "
        f"Saved to {outfile}."
    )
    return {
        "source": str(stacked_file),
        "n_slices": n_slices,
        "n_volumes": n_volumes,
        "order": order,
        "slice_thickness_mm": round(thickness, 4),
        "reference": reference,
    }


def modality_transform_files(outdir: Path, modality: str) -> List[str]:
    """
    The ANTs transforms brainles applied to a moving modality (coregistration to the center,
    center to atlas, atlas correction), stored as <step>_M_*.mat in the modality's transform
    directory, in the order ants.apply_transforms expects (last applied first).
    """
    transform_dir = REGISTRATION_TRAFO_SCHEMA.format(base_dir=outdir) / modality
    files = sorted(transform_dir.glob("*.mat"), key=lambda f: int(f.name.split("_")[0]))
    if not files:
        raise FileNotFoundError(f"No transforms found for modality {modality} in {transform_dir}.")
    return [str(f) for f in reversed(files)]


def transform_4d_series(
    series_file: Path, reference_modality: str, outdir: Path, outfile: Path
) -> None:
    """
    Resamples a 4D series into atlas space with the transforms brainles computed for its 3D
    reference modality (see modality_transform_files), onto the grid of the skull-stripped
    t1c, and masks every volume with the brain mask. Written as float32 4D nifti.
    """
    t1c_file = MODALITY_STRIPPED_SCHEMA.format(base_dir=outdir, modality="t1c")
    transforms = modality_transform_files(outdir, reference_modality)
    logger.info(
        f"Transforming 4D series {series_file.name} with the transforms of "
        f"{reference_modality}: {[Path(t).name for t in transforms]}."
    )
    warped = ants.apply_transforms(
        fixed=ants.image_read(str(t1c_file)),
        moving=ants.image_read(str(series_file)),
        transformlist=transforms,
        interpolator=FOURD_INTERPOLATOR,
        imagetype=3,
        defaultvalue=0,
    )
    warped_file = outfile.with_name(outfile.name.replace(NIFTI_SUFFIX, f"_unmasked{NIFTI_SUFFIX}"))
    ants.image_write(warped, str(warped_file))

    warped_img = nib.load(str(warped_file))
    mask = nib.load(str(BRAIN_MASK_SCHEMA.format(base_dir=outdir))).get_fdata() > 0
    data = np.asanyarray(warped_img.dataobj).astype(np.float32) * mask[..., None]
    nib.save(nib.Nifti1Image(data, warped_img.affine, warped_img.header), str(outfile))
    os.remove(warped_file)
    logger.info(f"Saved skull-stripped 4D series to {outfile} (shape {data.shape}).")


def collect_modalities(
    case_dir: Path, outdir: Path, override: bool = False
) -> Tuple[Dict[str, Path], Dict[str, Path], Dict[str, Path], Dict[str, Tuple[Path, str]]]:
    """
    Sorts the nifti files of a case into the four BraTS modalities, the additional
    modalities to normalize, the additional quantitative modalities (not normalized) and
    the stacked 4D series. The latter are unstacked into <outdir>/unstacked (skipped if the
    outputs exist and override is False); their 3D references join the quantitative
    modalities under the name <modality>_ref, and the 4D files are returned as
    {modality: (4D file, reference modality name)} for transform_4d_series. Files with
    more than three dimensions (the RGB nrCBV maps, which carry no geometry) are skipped.
    """
    role_files = CASE_MODALITY_FILES[case_dir.name]
    role_by_stem = {stem: role for role, stem in role_files.items()}
    modalities: Dict[str, Path] = {}
    additional: Dict[str, Path] = {}
    quantitative: Dict[str, Path] = {}
    stacked: Dict[str, Tuple[Path, str]] = {}
    layouts: Dict[str, object] = {}

    for nifti_file in discover_files(case_dir):
        stem = nifti_file.name[: -len(NIFTI_SUFFIX)]
        if stem in role_by_stem:
            modalities[role_by_stem[stem]] = nifti_file
            continue
        name = sanitize_modality_name(stem)

        ndim = len(nib.load(str(nifti_file)).shape)
        if ndim != 3:
            logger.warning(f"Skipping {nifti_file.name}: {ndim}D data without geometry.")
            continue

        layout = stacked_layout(stem)
        if layout is not None:
            n_volumes, order, reference = layout
            unstacked_file = UNSTACKED_SCHEMA.format(base_dir=outdir, modality=name)
            reference_name = f"{name}{STACKED_REFERENCE_SUFFIX}"
            reference_file = UNSTACKED_SCHEMA.format(base_dir=outdir, modality=reference_name)
            if unstacked_file.exists() and reference_file.exists() and not override:
                logger.info(f"{nifti_file.name}: already unstacked, skipping.")
            else:
                layouts[name] = unstack_4d_series(
                    stacked_file=nifti_file,
                    n_volumes=n_volumes,
                    order=order,
                    reference=reference,
                    outfile=unstacked_file,
                    reference_file=reference_file,
                )
            quantitative[reference_name] = reference_file
            stacked[name] = (unstacked_file, reference_name)
            continue

        if is_quantitative(stem):
            quantitative[name] = nifti_file
        else:
            additional[name] = nifti_file

    missing = [role for role in ("t1", "t1c", "t2", "flair") if role not in modalities]
    if missing:
        raise FileNotFoundError(f"{case_dir.name}: modality files not found for {missing}.")
    if layouts:
        update_config(outdir, CONFIG_STEP_STACKED_SERIES, layouts)
    return modalities, additional, quantitative, stacked


def process_case(
    case_dir: Path, outdir: Path, cuda_device: str, override: bool = False
) -> None:
    """
    Runs normalization (non-physical modalities only), skull stripping and co-registration
    of every nifti of the case to the t1c center modality and the atlas, transforms the
    unstacked 4D series with the transforms of their 3D references, and runs the pre-op
    tumor segmentation (BraTS25_1) and the atlas-based tissue segmentation with the tumor
    core masked from the registration metric. Steps whose outputs exist are skipped unless
    override is True.
    """
    outdir.mkdir(parents=True, exist_ok=True)
    modalities, additional, quantitative, stacked = collect_modalities(
        case_dir, outdir, override=override
    )
    logger.info(
        f"{case_dir.name}: center t1c {modalities['t1c'].name}; t1 {modalities['t1'].name}, "
        f"t2 {modalities['t2'].name}, flair {modalities['flair'].name}; "
        f"normalized additional modalities {sorted(additional)}; quantitative "
        f"{sorted(quantitative)}; 4D series {sorted(stacked)}."
    )

    t1c_stripped = MODALITY_STRIPPED_SCHEMA.format(base_dir=outdir, modality="t1c")
    if t1c_stripped.exists() and not override:
        logger.info(f"{outdir}: skull stripping/co-registration already done, skipping.")
    else:
        norm_ss_coregister(
            t1_file=modalities["t1"],
            t1c_file=modalities["t1c"],
            t2_file=modalities["t2"],
            flair_file=modalities["flair"],
            outdir=outdir,
            skull_strip=True,
            additional_modalities=additional,
            additional_quantitative_modalities=quantitative,
            atlas=ATLAS,
        )

    for name, (series_file, reference_name) in stacked.items():
        series_outfile = MODALITY_STRIPPED_SCHEMA.format(base_dir=outdir, modality=name)
        if series_outfile.exists() and not override:
            logger.info(f"{outdir}: 4D series {name} already transformed, skipping.")
            continue
        transform_4d_series(
            series_file=series_file,
            reference_modality=reference_name,
            outdir=outdir,
            outfile=series_outfile,
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
            is_preop=True,
        )

    tissueseg_files = [
        TISSUE_PBMAP_SCHEMA.format(base_dir=outdir, tissue=tissue)
        for tissue in ("gm", "wm", "csf")
    ]
    if all(f.exists() for f in tissueseg_files) and not override:
        logger.info(f"{outdir}: tissue segmentation already done, skipping.")
    else:
        registration_mask_file = REGISTRATION_MASK_SCHEMA.format(base_dir=outdir)
        generate_registration_mask(tumor_seg_file=tumorseg_file, outfile=registration_mask_file)
        run_tissue_seg(
            t1_file=t1c_stripped,
            outdir=outdir,
            registration_mask_file=registration_mask_file,
            atlas=ATLAS,
        )


if __name__ == "__main__":
    # Full processing of every case (normalization, skull stripping, co-registration of all
    # niftis to the t1c and SRI24, 4D series, tumor + tissue segmentation); steps with
    # existing outputs are skipped unless -override is given:
    # nohup python -u scripts/process_bonn.py -cuda_device 0 > tmp_process_bonn.out 2>&1 &
    # Only one case:
    # python -u scripts/process_bonn.py -cuda_device 0 -cases bonn_1
    parser = argparse.ArgumentParser()
    parser.add_argument("-cuda_device", type=str, default="0", help="GPU id to run on.")
    parser.add_argument(
        "-data_dir",
        type=str,
        default="/mnt/Drive4/lucas/bonn/PreOp_nifti",
        help="Directory containing one directory of raw niftis per case (nifti, nifti 2, ...).",
    )
    parser.add_argument(
        "-outdir",
        type=str,
        default="/mnt/Drive4/lucas/bonn/processed",
        help="Directory to save processed output to (<outdir>/bonn_<n>/).",
    )
    parser.add_argument(
        "-cases",
        type=str,
        nargs="+",
        default=None,
        help="Case ids to process (e.g. bonn_1 bonn_2). Processes all cases if omitted.",
    )
    parser.add_argument(
        "-override",
        action="store_true",
        help="Rerun every step even if its output already exists, overwriting previous results.",
    )
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device

    data_dir = Path(args.data_dir)
    outdir_root = Path(args.outdir)
    outdir_root.mkdir(parents=True, exist_ok=True)

    cases = []
    for case_dir in sorted(d for d in data_dir.iterdir() if d.is_dir()):
        cid = case_id(case_dir)
        if cid is None:
            logger.warning(f"Ignoring directory {case_dir.name}: not a case directory.")
            continue
        if case_dir.name not in CASE_MODALITY_FILES:
            logger.warning(f"{cid} ({case_dir.name}): no modality mapping defined, skipping.")
            continue
        if args.cases is not None and cid not in args.cases:
            continue
        cases.append((cid, case_dir))
    cases.sort(key=lambda c: int(c[0].split("_")[1]))
    logger.info(f"Found {len(cases)} cases in {data_dir}: {[c for c, _ in cases]}.")

    for cid, case_dir in cases:
        logger.info(f"Processing {cid} ({case_dir}).")
        try:
            process_case(
                case_dir=case_dir,
                outdir=outdir_root / cid,
                cuda_device=args.cuda_device,
                override=args.override,
            )
        except Exception:
            logger.exception(f"{cid}: processing failed, skipping.")

    logger.info("Finished processing Bonn dataset.")
