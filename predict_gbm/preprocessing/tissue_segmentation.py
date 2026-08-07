import ants
import time
import shlex
import shutil
import subprocess
import numpy as np
import nibabel as nib
from pathlib import Path
from typing import Literal
from loguru import logger
from predict_gbm.utils.constants import (
    ATLAS_STRIPPED_SCHEMA,
    ATLAS_TISSUE_PBMAP_SCHEMA,
    BRAIN_MASK_SCHEMA,
    CONFIG_STEP_TISSUE_SEG,
    TISSUE_LABELS,
    TISSUE_PBMAP_SCHEMA,
    TISSUE_SEG_BASE_SCHEMA,
)
from predict_gbm.utils.utils import update_config

SUPPORTED_ATLASES = ("brats_mni152", "mni152", "sri24")


def generate_healthy_brain_mask(
    brain_mask_file: Path, tumor_seg_file: Path, outfile: Path
) -> None:
    """
    Generate a healthy brain mask by subtracting the tumor segmentation from the brain mask.

    Parameters:
        brain_mask_file (Path): Path to the brain mask NIfTI file.
        tumor_seg_file (Path): Path to the tumor segmentation NIfTI file.
        outfile (Path): Output file path where the healthy brain mask will be saved.

    Returns:
        None
    """
    logger.info("Generating healthy brain mask.")
    # Load niftis
    brain_nifti = nib.load(str(brain_mask_file))
    brain_data = np.rint(brain_nifti.get_fdata()).astype(np.int32)

    tumor_data = np.rint(nib.load(str(tumor_seg_file)).get_fdata()).astype(np.int32)
    tumor_mask = (tumor_data > 0).astype(np.int32)

    # Generate the healthy brain mask.
    healthy_data = np.where(tumor_mask > 0, 0, brain_data).astype(np.int32)

    # Generate output nifti and save it
    outfile.parent.mkdir(parents=True, exist_ok=True)
    healthy_mask_nifti = nib.Nifti1Image(
        healthy_data,
        affine=brain_nifti.affine,
    )
    nib.save(healthy_mask_nifti, str(outfile))

    logger.info(f"Healthy brain mask generated succesfully and saved to {outfile}.")


def generate_registration_mask(tumor_seg_file: Path, outfile: Path) -> None:
    """
    Generate the inverse of the tumor mask to be used with registration during tissue segmentation.

    Parameters:
        tumor_seg_file (Path): Path to the tumor segmentation NIfTI file.
        outfile (Path): Output file path where the mask will be saved.

    Returns:
        None
    """
    logger.info("Generating registration mask.")

    # Load data
    tumor_nifti = nib.load(str(tumor_seg_file))

    # Generate mask
    tumor_seg = np.rint(tumor_nifti.get_fdata()).astype(np.int32)
    tumor_seg[tumor_seg == 2] = 0  # discard edema, only use core as mask
    no_tumor_mask = (tumor_seg < 0.5).astype(np.int32)

    # Save
    outfile.parent.mkdir(parents=True, exist_ok=True)
    no_tumor_mask_nifti = nib.Nifti1Image(
        no_tumor_mask,
        affine=tumor_nifti.affine,
    )
    nib.save(no_tumor_mask_nifti, str(outfile))

    logger.info(f"Registration mask generated successfully and save to {outfile}.")


def derive_tissue_labelmap_from_pbmaps(
    csf_pbmap_path: Path, gm_pbmap_path: Path, wm_pbmap_path: Path
) -> nib.Nifti1Image:
    """
    Derives a discrete tissue labelmap from the csf/gm/wm probability maps by taking the
    argmax across the maps at each voxel.

    Parameters:
        csf_pbmap_path (Path): Path to the csf probability map nifti.
        gm_pbmap_path (Path): Path to the gm probability map nifti.
        wm_pbmap_path (Path): Path to the wm probability map nifti.
            All maps must share the same affine and shape.

    Returns:
        nib.Nifti1Image: The derived discrete tissue labelmap.
    """
    pbmap_paths = {"csf": csf_pbmap_path, "gm": gm_pbmap_path, "wm": wm_pbmap_path}
    reference_nifti = nib.load(str(csf_pbmap_path))
    pbmap_stack = np.stack(
        [nib.load(str(pbmap_paths[tissue])).get_fdata() for tissue in pbmap_paths],
        axis=0,
    )

    background_mask = np.all(np.isclose(pbmap_stack, 0), axis=0)
    labels = np.array([TISSUE_LABELS[tissue] for tissue in pbmap_paths])
    labelmap = labels[np.argmax(pbmap_stack, axis=0)]
    labelmap[background_mask] = 0

    return nib.Nifti1Image(labelmap.astype(np.int32), affine=reference_nifti.affine)


def derive_tissue_labelmap(outdir: Path) -> nib.Nifti1Image:
    """
    Derives the discrete tissue labelmap for an exam directory from its previously
    generated csf/gm/wm probability maps.

    Parameters:
        outdir (Path): Path to exam directory containing tissue probability maps under standard layout.

    Returns:
        nib.Nifti1Image: The derived discrete tissue labelmap.
    """
    return derive_tissue_labelmap_from_pbmaps(
        csf_pbmap_path=TISSUE_PBMAP_SCHEMA.format(base_dir=outdir, tissue="csf"),
        gm_pbmap_path=TISSUE_PBMAP_SCHEMA.format(base_dir=outdir, tissue="gm"),
        wm_pbmap_path=TISSUE_PBMAP_SCHEMA.format(base_dir=outdir, tissue="wm"),
    )


def run_tissue_seg(
    t1_file: Path,
    outdir: Path,
    registration_mask_file: Path = None,
    algorithm: Literal["atlas_registration", "antsAtroposN4"] = "atlas_registration",
    atlas: str = "brats_mni152",
) -> None:
    """
    Performs tissue segmentation for gm, wm, csf using the given algorithm.

    Parameters:
        t1_file (Path): Path to the t1 nifti.
        outdir (Path): Path to output directory. Usually exam directory.
        registration_mask_file (Path): Path to a mask for registration metric. Voxel with value 0 are ignored.
            Only used when algorithm is "atlas_registration".
        algorithm (str): Tissue segmentation algorithm to use. Supports "atlas_registration", "antsAtroposN4"
        atlas (str): Atlas whose skull-stripped T1 template and tissue probability maps are used.
            One of "brats_mni152" (default), "mni152", or "sri24".

    Returns:
        None
    """
    if atlas not in SUPPORTED_ATLASES:
        raise ValueError(
            f"Unsupported atlas '{atlas}'. Expected one of: {sorted(SUPPORTED_ATLASES)}."
        )

    if algorithm == "atlas_registration":
        run_tissue_seg_atlas_registration(
            t1_file=t1_file,
            outdir=outdir,
            registration_mask_file=registration_mask_file,
            atlas=atlas,
        )
    elif algorithm == "antsAtroposN4":
        run_tissue_seg_atropos_n4(t1_file=t1_file, outdir=outdir, atlas=atlas)
    else:
        raise ValueError(
            f"Unknown algorithm {algorithm!r}. Expected 'atlas_registration' or 'antsAtroposN4'."
        )

    update_config(
        outdir,
        CONFIG_STEP_TISSUE_SEG,
        {
            "algorithm": algorithm,
            "atlas": atlas,
            "registration_mask_used": registration_mask_file is not None,
        },
    )


def run_tissue_seg_atlas_registration(
    t1_file: Path,
    outdir: Path,
    registration_mask_file: Path = None,
    atlas: str = "brats_mni152",
) -> None:
    """
    Performs tissue segmentation for gm, wm, csf by registering an atlas to the input t1 file and transforming atlas tissue
    probability maps using the obtained transformation. Produces one probability map per tissue.

    Parameters:
        t1_file (Path): Path to the t1 nifti.
        outdir (Path): Path to output directory. Usually exam directory.
        registration_mask_file (Path): Path to a mask for registration metric. Voxel with value 0 are ignored.
        atlas (str): Atlas whose skull-stripped T1 template and tissue probability maps are used.
            One of "brats_mni152" (default), "mni152", or "sri24".

    Returns:
        None
    """
    start_time = time.time()
    logger.info("Starting tissue segmentation.")

    # Prepare directories
    atlas_pbmap_dirs = {
        tissue: ATLAS_TISSUE_PBMAP_SCHEMA.format(atlas=atlas, tissue=tissue)
        for tissue in ["csf", "gm", "wm"]
    }
    outprefix = TISSUE_SEG_BASE_SCHEMA.format(base_dir=outdir)
    outprefix.mkdir(parents=True, exist_ok=True)

    # Read images
    t1_patient = ants.image_read(str(t1_file))
    t1_atlas = ants.image_read(str(ATLAS_STRIPPED_SCHEMA.format(atlas=atlas)))

    reg_kwargs = {}
    if registration_mask_file is not None:
        logger.info(
            f"Using provided mask for registration {str(registration_mask_file)}."
        )
        registration_mask = ants.image_read(str(registration_mask_file))
        reg_kwargs = {"mask": registration_mask}

    # Register atlas to patient
    reg = ants.registration(
        fixed=t1_patient,
        moving=t1_atlas,
        type_of_transform="antsRegistrationSyN[s,2]",
        outprefix=str(outprefix) + "/",
        **reg_kwargs,
    )
    transforms_path = reg["fwdtransforms"]

    logger.info("Generating pbmaps...")

    # Transform atlas tissue probability maps
    for tissue, pbmap_dir in atlas_pbmap_dirs.items():
        pbmap = ants.image_read(str(pbmap_dir))
        warped_pbmap = ants.apply_transforms(
            fixed=t1_patient,
            moving=pbmap,
            transformlist=transforms_path,
            interpolator="linear",
        )
        ants.image_write(
            warped_pbmap,
            str(TISSUE_PBMAP_SCHEMA.format(base_dir=outdir, tissue=tissue)),
        )

    time_spent = time.time() - start_time
    logger.info(
        f"Finished tissue segmentation in {time_spent:.2f} seconds. Results saved to {outdir}."
    )


def run_tissue_seg_atropos_n4(
    t1_file: Path, outdir: Path, atlas: str = "brats_mni152"
) -> None:
    """
    Performs tissue segmentation for gm, wm, csf by running the antsAtroposN4.sh binary with the
    atlas tissue probability maps as spatial priors and the brain mask as constraint. Produces one
    probability map per tissue.

    Parameters:
        t1_file (Path): Path to the t1 nifti.
        outdir (Path): Path to output directory. Usually exam directory.
        atlas (str): Atlas whose tissue probability maps are used as spatial priors. One of
            "brats_mni152" (default), "mni152", or "sri24".

    Returns:
        None
    """
    start_time = time.time()
    logger.info("Starting tissue segmentation via antsAtroposN4.")

    outprefix = TISSUE_SEG_BASE_SCHEMA.format(base_dir=outdir)
    outprefix.mkdir(parents=True, exist_ok=True)

    brain_mask_file = BRAIN_MASK_SCHEMA.format(base_dir=outdir)

    # antsAtroposN4.sh expects priors as a %d-indexed filename pattern, with the index matching
    # the label of the corresponding class in the output segmentation. Stage the atlas tissue
    # probability maps under those names, in TISSUE_LABELS order (csf=1, gm=2, wm=3).
    priors_dir = outprefix / "priors"
    priors_dir.mkdir(parents=True, exist_ok=True)
    for tissue, label in TISSUE_LABELS.items():
        shutil.copyfile(
            str(ATLAS_TISSUE_PBMAP_SCHEMA.format(atlas=atlas, tissue=tissue)),
            str(priors_dir / f"prior{int(label)}.nii.gz"),
        )

    seg_prefix = outprefix / "tissue_seg_"
    cmd = [
        "antsAtroposN4.sh",
        "-d", "3",
        "-a", str(t1_file),
        "-x", str(brain_mask_file),
        "-c", "3",
        "-p", str(priors_dir / "prior%d.nii.gz"),
        "-w", "0.25",
        "-y", "2",
        "-y", "3",
        "-o", str(seg_prefix),
    ]
    logger.info(f"Running: {shlex.join(cmd)}")
    proc = subprocess.run(cmd, check=False)
    if proc.returncode != 0:
        raise RuntimeError(
            f"antsAtroposN4.sh exited {proc.returncode} for {t1_file}; see logs in {outprefix}"
        )

    for tissue, label in TISSUE_LABELS.items():
        shutil.move(
            f"{seg_prefix}SegmentationPosteriors{int(label)}.nii.gz",
            str(TISSUE_PBMAP_SCHEMA.format(base_dir=outdir, tissue=tissue)),
        )

    time_spent = time.time() - start_time
    logger.info(
        f"Finished tissue segmentation in {time_spent:.2f} seconds. Results saved to {outdir}."
    )
