import ants
import time
import numpy as np
import nibabel as nib
from pathlib import Path
from loguru import logger
from predict_gbm.utils.constants import (
    ATLAS_T1_DIR,
    ATLAS_TISSUES_DIR,
    ATLAS_TISSUE_PBMAPS_DIR,
    TISSUE_LABELS,
    TISSUE_PBMAP_SCHEMA,
    TISSUE_SCHEMA,
    TISSUE_SEG_SCHEMA,
    TISSUE_SEG_BASE_SCHEMA,
)


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
    healthy_mask_nifti = nib.Nifti1Image(healthy_data, affine=np.eye(4))
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
    no_tumor_mask_nifti = nib.Nifti1Image(no_tumor_mask, affine=np.eye(4))
    nib.save(no_tumor_mask_nifti, str(outfile))

    logger.info(f"Registration mask generated successfully and save to {outfile}.")


def run_tissue_seg_registration(
    t1_file: Path, outdir: Path, registration_mask_file: Path = None
) -> None:
    """
    Performs tissue segmentation for gm, wm, csf by registering an atlas to the input t1 file and transforming atlas tissue maps using
    the obtained transformation.

    Parameters:
        t1_file (Path): Path to the t1 nifti.
        outdir (Path): Path to output directory. Usually exam directory.
        registration_mask_file (Path): Path to a mask for registration metric. Voxel with value 0 are ignored.

    Returns:
        None
    """
    start_time = time.time()
    logger.info("Starting tissue segmentation.")

    # Prepare directories
    atlas_pbmap_dirs = {
        tissue: ATLAS_TISSUE_PBMAPS_DIR.format(tissue=tissue)
        for tissue in ["csf", "gm", "wm"]
    }
    outprefix = TISSUE_SEG_BASE_SCHEMA.format(base_dir=outdir)
    outprefix.mkdir(parents=True, exist_ok=True)

    # Read images
    t1_patient = ants.image_read(str(t1_file))
    t1_atlas = ants.image_read(str(ATLAS_T1_DIR))

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

    # Transform atlas tissues
    tissues_atlas = ants.image_read(str(ATLAS_TISSUES_DIR))
    tissues_warped = ants.apply_transforms(
        fixed=t1_patient.clone("unsigned int"),
        moving=tissues_atlas,
        transformlist=transforms_path,
        interpolator="nearestNeighbor",
    )
    ants.image_write(tissues_warped, str(TISSUE_SEG_SCHEMA.format(base_dir=outdir)))

    logger.debug(
        f"Registration step done, saving output to {TISSUE_SEG_SCHEMA.format(base_dir=outdir)}"
    )
    logger.info("Generating pbmaps...")

    # Transform atlas tissue segmentations
    tissues_warped_nifti = nib.load(str(TISSUE_SEG_SCHEMA.format(base_dir=outdir)))
    for tissue, label in TISSUE_LABELS.items():
        eq = np.rint(tissues_warped_nifti.get_fdata()).astype(np.int32)
        tissue_mask = (np.isclose(eq, label)).astype(np.int32)
        tissue_mask_nifti = nib.Nifti1Image(tissue_mask, affine=np.eye(4))
        nib.save(
            tissue_mask_nifti, str(TISSUE_SCHEMA.format(base_dir=outdir, tissue=tissue))
        )

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
