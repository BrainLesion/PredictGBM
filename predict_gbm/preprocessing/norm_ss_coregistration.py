import ants
import time
import shutil
from pathlib import Path
from loguru import logger
from typing import List
from brainles_preprocessing.normalization import Normalizer
from brainles_preprocessing.preprocessor import AtlasCentricPreprocessor
from brainles_preprocessing.registration import ANTsRegistrator
from brainles_preprocessing.modality import Modality, CenterModality
from brainles_preprocessing.normalization.percentile_normalizer import (
    PercentileNormalizer,
)
from predict_gbm.utils.constants import (
    BRAIN_MASK_SCHEMA,
    BRAINLES_LOGFILE_SCHEMA,
    LONGITUDINAL_TRAFO_SCHEMA,
    LONGITUDINAL_WARP_SCHEMA,
    MODALITY_STRIPPED_SCHEMA,
    RECURRENCE_SCHEMA,
    REGISTRATION_TRAFO_SCHEMA,
)


def normalize(img_file: Path, outfile: Path) -> None:
    """Performs the normalization step from norm_ss_coregister using a percentile normalizer."""
    logger.info(f"Running plain normalization for {img_file}.")
    percentile_normalizer = PercentileNormalizer(
        lower_percentile=0.1,
        upper_percentile=99.9,
        lower_limit=0,
        upper_limit=1,
    )

    mod = Modality(
        modality_name="tmp",
        input_path=str(img_file),
        normalizer=percentile_normalizer,
        normalized_bet_output_path="test",
    )

    mod.save_current_image(str(outfile), normalization=True)
    print("Normalization done.")


def initialize_center_modality(
    modality_file: Path,
    modality_name: str,
    normalizer: Normalizer,
    outdir: Path,
    skull_strip: bool = True,
) -> CenterModality:
    """
    Initializes and returns a CenterModality object configured for a specific imaging modality.

    Parameters:
        modality_file (Path): The path to the center/fixed modality nifti used for registration..
        modality_name (str): A descriptive name for the modality (e.g., 't1c').
        normalizer (Normalizer): An instance of the Normalizer class that defines the normalization parameters.
        outdir (Path): The directory where the output files (normalized image and mask) will be saved. Usually the exam dir.
        skull_strip (bool): If true, performs skull stripping via HDBet

    Returns:
        CenterModality: An instance of CenterModality configured with the input file, normalization settings
    """
    modality_outfile = MODALITY_STRIPPED_SCHEMA.format(
        base_dir=outdir, modality=modality_name
    )
    mask_outfile = BRAIN_MASK_SCHEMA.format(base_dir=outdir)

    if skull_strip:
        center = CenterModality(
            modality_name=modality_name,
            input_path=str(modality_file),
            normalizer=normalizer,
            normalized_bet_output_path=str(modality_outfile),
            bet_mask_output_path=str(mask_outfile),
        )
    else:
        logger.info(f"normalized_skull_output_path: {str(modality_outfile)}")
        center = CenterModality(
            modality_name=modality_name,
            input_path=str(modality_file),
            normalizer=normalizer,
            normalized_skull_output_path=str(modality_outfile),
        )

    return center


def initialize_moving_modalities(
    modality_files: List[Path],
    modality_names: List[Path],
    normalizer: Normalizer,
    outdir: Path,
    skull_strip: bool = True,
) -> Modality:
    """
    Initializes and returns a list of Modality objects with moving modalities for registration.

    Parameters:
        modality_file (List[Path]): List of paths to the moving modality nifti used for registration.
        modality_name (List[Path]): List of descriptive names for the modalities.
        normalizer (Normalizer): An instance of the Normalizer class that defines the normalization parameters.
        outdir (Path): The directory where the output files will be saved. Usually the exam dir.
        skull_strip (bool): If true, performs skull stripping via HDBet

    Returns:
        List[Modality]: A list of Modality instances configured for the moving modalities.
    """
    moving_modalities = []
    for mod_file, mod_name in zip(modality_files, modality_names):

        stripped_modality_outfile = MODALITY_STRIPPED_SCHEMA.format(
            base_dir=outdir, modality=mod_name
        )

        if skull_strip:
            m = Modality(
                input_path=mod_file,
                modality_name=mod_name,
                normalizer=normalizer,
                normalized_bet_output_path=stripped_modality_outfile,
            )
        else:
            logger.info(f"normalized_skull_output_path: {stripped_modality_outfile}")
            m = Modality(
                input_path=mod_file,
                modality_name=mod_name,
                normalizer=normalizer,
                normalized_skull_output_path=stripped_modality_outfile,
            )

        moving_modalities.append(m)
    return moving_modalities


def norm_ss_coregister(
    t1_file: Path,
    t1c_file: Path,
    t2_file: Path,
    flair_file: Path,
    outdir: Path,
    skull_strip: bool = True,
) -> None:
    """
    Performs normalization, skull stripping and co-registration based on the brainles preprocessing module.

    Parameters:
        t1_file (Path): Path to the t1 nifti.
        t1c_file (Path): Path to the t1c nifti.
        t2_file (Path): Path to the t2 nifti.
        flair_file (Path): Path to the flair nifti.
        outdir (Path): Base directory where the output will be saved. Usually exam dir.
        skull_strip (bool): If true, performs skull stripping via HDBet

    Returns:
        None
    """
    start_time = time.time()
    logger.info(
        "Starting normalization, skull strippping, co-registration step. Starting brainles preprocessing."
    )
    logger.info(f"skull_strip: {skull_strip}")

    Path(outdir).mkdir(parents=True, exist_ok=True)
    percentile_normalizer = PercentileNormalizer(
        lower_percentile=0.1,
        upper_percentile=99.9,
        lower_limit=0,
        upper_limit=1,
    )

    center = initialize_center_modality(
        modality_file=str(t1c_file),
        modality_name="t1c",
        normalizer=percentile_normalizer,
        outdir=str(outdir),
        skull_strip=skull_strip,
    )
    moving = initialize_moving_modalities(
        modality_files=[str(t1_file), str(t2_file), str(flair_file)],
        modality_names=["t1", "t2", "flair"],
        normalizer=percentile_normalizer,
        outdir=str(outdir),
        skull_strip=skull_strip,
    )

    # brainles-preprocessing also uses sri24, to set atlas here use atlas_image_path
    registrator = ANTsRegistrator(transformation_params={"defaultvalue": 0})
    preprocessor = AtlasCentricPreprocessor(
        center_modality=center,
        moving_modalities=moving,
        registrator=registrator,
    )

    preprocessor.run(
        log_file=BRAINLES_LOGFILE_SCHEMA.format(base_dir=outdir),
        save_dir_transformations=REGISTRATION_TRAFO_SCHEMA.format(base_dir=outdir),
    )
    time_spent = time.time() - start_time
    logger.info(
        f"Finished normalization, skull stripping, co-registration step in {time_spent:.2f} seconds. Output saved to {outdir}."
    )


def register_recurrence(
    t1c_pre_file: Path,
    t1c_post_file: Path,
    recurrence_seg_file: Path,
    outdir: Path,
    fixed_mask_file: Path = None,
    moving_mask_file: Path = None,
) -> None:
    """
    Register a postop image with recurrence to preop.

    Parameters:
        t1c_pre_file (Path): Path to the pre-operative T1c image.
        t1c_post_file (Path): Path to the post-operative T1c image.
        recurrence_seg_file (Path): Path to the tumor segmentation.
        outdir (Path): Directory where the output files will be saved.
        fixed_mask_file (Path): Path to a mask in fixed space used for registration.
        moving_mask_file (Path): Path to a mask in moving space used for registration.

    Returns:
        None
    """
    start_time = time.time()
    logger.info("Starting longitudinal co-registration.")

    t1c_pre_img = ants.image_read(str(t1c_pre_file))
    t1c_post_img = ants.image_read(str(t1c_post_file))

    # Masks
    reg_kwargs = {}
    if fixed_mask_file is not None:
        reg_kwargs["mask"] = ants.image_read(str(fixed_mask_file))
        logger.info(f"Running with provided mask (fixed space) {str(fixed_mask_file)}.")
    if moving_mask_file is not None:
        reg_kwargs["moving_mask"] = ants.image_read(str(moving_mask_file))
        logger.info(
            f"Running with provided mask (moving space) {str(moving_mask_file)}."
        )

    # SyN Registration
    reg = ants.registration(
        fixed=t1c_pre_img,
        moving=t1c_post_img,
        type_of_transform="antsRegistrationSyN[s,2]",
    )

    recurrence_outdir = RECURRENCE_SCHEMA.format(base_dir=outdir)
    recurrence_outdir.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(
        src=reg["fwdtransforms"][0],
        dst=str(LONGITUDINAL_TRAFO_SCHEMA.format(base_dir=outdir)),
    )
    ants.image_write(
        reg["warpedmovout"], str(LONGITUDINAL_WARP_SCHEMA.format(base_dir=outdir))
    )

    recurrence_seg = ants.image_read(str(recurrence_seg_file))
    recurrence_warped = ants.apply_transforms(
        fixed=t1c_pre_img.clone("unsigned int"),
        moving=recurrence_seg,
        transformlist=reg["fwdtransforms"],
        interpolator="nearestNeighbor",
    )
    ants.image_write(recurrence_warped, str(recurrence_outdir))
    time_spent = time.time() - start_time
    logger.info(
        f"Finished longitudinal co-registration in {time_spent:.2f} seconds. Output saved to {str(recurrence_outdir)}."
    )
