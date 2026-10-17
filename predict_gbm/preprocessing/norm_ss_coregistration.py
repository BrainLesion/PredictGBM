import ants
import shutil
import tempfile
import time
import numpy as np
from pathlib import Path
from loguru import logger
from typing import Dict, List, Optional, Tuple
from brainles_preprocessing.normalization import Normalizer
from brainles_preprocessing.preprocessor import AtlasCentricPreprocessor
from brainles_preprocessing.registration import ANTsRegistrator
from brainles_preprocessing.modality import Modality, CenterModality
from brainles_preprocessing.brain_extraction.synthstrip import SynthStripExtractor
from brainles_preprocessing.normalization.percentile_normalizer import (
    PercentileNormalizer,
)
from predict_gbm.preprocessing.dirac import (
    MODALITIES_FOUR,
    MODALITIES_SINGLE,
    run_dirac_registration,
)
from predict_gbm.utils.constants import (
    ATLAS_STRIPPED_SCHEMA,
    ATLAS_UNSTRIPPED_SCHEMA,
    BRAIN_MASK_SCHEMA,
    BRAINLES_LOGFILE_SCHEMA,
    CONFIG_STEP_NORM_SS_COREGISTER,
    CONFIG_STEP_REGISTER_RECURRENCE,
    LONGITUDINAL_AFFINE_MAX_ABS_LOG_DET,
    LONGITUDINAL_AFFINE_SCHEMA,
    LONGITUDINAL_AFFINE_MAX_ROTATION_DEG,
    LONGITUDINAL_AFFINE_MAX_TRANSLATION_MM,
    LONGITUDINAL_AFFINE_SINGULAR_VALUE_RANGE,
    LONGITUDINAL_DISP_SCHEMA,
    LONGITUDINAL_WARP_SCHEMA,
    MODALITY_STRIPPED_SCHEMA,
    RECURRENCE_SCHEMA,
    REGISTRATION_TRAFO_SCHEMA,
    SKULL_STRIP_ALGORITHM_SYNTHSTRIP,
)
from predict_gbm.utils.utils import update_config, validate_additional_modality_names

SUPPORTED_ATLASES = ("brats_mni152", "mni152", "sri24")

# Longitudinal registration algorithms backed by ANTs, mapping the public algorithm name to
# the ants.registration type_of_transform. The linear ones store their forward transform as
# the ANTs .mat file (LONGITUDINAL_AFFINE_SCHEMA), syn and dirac as one dense displacement
# field (LONGITUDINAL_DISP_SCHEMA); longitudinal_transform_file resolves the stored one.
ANTS_REGISTRATION_TRANSFORMS = {
    "syn": "antsRegistrationSyN[s,2]",
    "affine": "Affine",
    "rigid": "Rigid",
}
LINEAR_REGISTRATION_ALGORITHMS = ("affine", "rigid")
SUPPORTED_REGISTRATION_ALGORITHMS = ("dirac",) + tuple(ANTS_REGISTRATION_TRANSFORMS)
# ANTs transform fitted on t1c before DIRAC, without masks.
DIRAC_AFFINE_TRANSFORM = "Affine"


class LinearANTsRegistrator(ANTsRegistrator):
    """
    ANTsRegistrator that resamples with linear interpolation by default.

    Temporary solution: brainles_preprocessing (up to 0.6.13) defaults ANTsRegistrator.transform
    to "nearestNeighbor", which is only appropriate for label maps, and the interpolator cannot
    be set via transformation_params (it is a separate argument of transform). Remove this class
    once brainles_preprocessing exposes a proper way to select the interpolator for the MRI
    modalities.
    """

    def transform(self, *args, interpolator: str = "linear", **kwargs) -> None:
        super().transform(*args, interpolator=interpolator, **kwargs)


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
        skull_strip (bool): If true, performs skull stripping via SynthStrip

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
    outdir: Path,
    normalizer: Optional[Normalizer] = None,
    skull_strip: bool = True,
    normalize: bool = True,
) -> Modality:
    """
    Initializes and returns a list of Modality objects with moving modalities for registration.

    Parameters:
        modality_file (List[Path]): List of paths to the moving modality nifti used for registration.
        modality_name (List[Path]): List of descriptive names for the modalities.
        outdir (Path): The directory where the output files will be saved. Usually the exam dir.
        normalizer (Optional[Normalizer]): An instance of the Normalizer class that defines the
            normalization parameters. Required when normalize=True, ignored otherwise.
        skull_strip (bool): If true, performs skull stripping via SynthStrip
        normalize (bool): If true, applies the normalizer and writes the normalized output.

    Returns:
        List[Modality]: A list of Modality instances configured for the moving modalities.
    """
    moving_modalities = []
    for mod_file, mod_name in zip(modality_files, modality_names):

        stripped_modality_outfile = MODALITY_STRIPPED_SCHEMA.format(
            base_dir=outdir, modality=mod_name
        )

        if skull_strip and normalize:
            m = Modality(
                input_path=mod_file,
                modality_name=mod_name,
                normalizer=normalizer,
                normalized_bet_output_path=stripped_modality_outfile,
            )
        elif skull_strip and not normalize:
            m = Modality(
                input_path=mod_file,
                modality_name=mod_name,
                raw_bet_output_path=stripped_modality_outfile,
            )
        elif not skull_strip and normalize:
            logger.info(f"normalized_skull_output_path: {stripped_modality_outfile}")
            m = Modality(
                input_path=mod_file,
                modality_name=mod_name,
                normalizer=normalizer,
                normalized_skull_output_path=stripped_modality_outfile,
            )
        else:
            logger.info(f"raw_skull_output_path: {stripped_modality_outfile}")
            m = Modality(
                input_path=mod_file,
                modality_name=mod_name,
                raw_skull_output_path=stripped_modality_outfile,
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
    additional_modalities: Optional[Dict[str, Path]] = None,
    additional_quantitative_modalities: Optional[Dict[str, Path]] = None,
    atlas: str = "sri24",
) -> None:
    """
    Performs normalization, skull stripping and co-registration based on the brainles preprocessing module.

    Parameters:
        t1_file (Path): Path to the t1 nifti.
        t1c_file (Path): Path to the t1c nifti.
        t2_file (Path): Path to the t2 nifti.
        flair_file (Path): Path to the flair nifti.
        outdir (Path): Base directory where the output will be saved. Usually exam dir.
        skull_strip (bool): If true, performs skull stripping via SynthStrip
        additional_modalities (Optional[Dict[str, Path]]): Mapping of additional modality
            name to nifti path, processed alongside t1/t2/flair with intensity normalization.
        additional_quantitative_modalities (Optional[Dict[str, Path]]): Same as
            additional_modalities, but processed without intensity normalization (e.g. for
            quantitative maps like ADC). Names must not collide with each other, with
            additional_modalities, or with the reserved modality names (t1, t1c, t2, flair).
        atlas (str): Atlas to register the patient into. One of "sri24" (default, the
            240x240x155 grid the DIRAC longitudinal registration was built for),
            "brats_mni152", or "mni152". The tissue segmentation must use the same atlas.

    Returns:
        None
    """
    start_time = time.time()
    logger.info(
        "Starting normalization, skull strippping, co-registration step. Starting brainles preprocessing."
    )
    logger.info(f"skull_strip: {skull_strip}")

    additional_modalities = additional_modalities or {}
    additional_quantitative_modalities = additional_quantitative_modalities or {}
    validate_additional_modality_names(
        additional_modalities, additional_quantitative_modalities
    )

    if atlas not in SUPPORTED_ATLASES:
        raise ValueError(
            f"Unsupported atlas '{atlas}'. Expected one of: {sorted(SUPPORTED_ATLASES)}."
        )

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
        modality_files=[str(t1_file), str(t2_file), str(flair_file)]
        + [str(p) for p in additional_modalities.values()],
        modality_names=["t1", "t2", "flair"] + list(additional_modalities.keys()),
        normalizer=percentile_normalizer,
        outdir=str(outdir),
        skull_strip=skull_strip,
    )
    if additional_quantitative_modalities:
        moving += initialize_moving_modalities(
            modality_files=[
                str(p) for p in additional_quantitative_modalities.values()
            ],
            modality_names=list(additional_quantitative_modalities.keys()),
            outdir=str(outdir),
            skull_strip=skull_strip,
            normalize=False,
        )

    atlas_schema = ATLAS_UNSTRIPPED_SCHEMA if skull_strip else ATLAS_STRIPPED_SCHEMA

    registrator = LinearANTsRegistrator(transformation_params={"defaultvalue": 0})
    preprocessor = AtlasCentricPreprocessor(
        center_modality=center,
        moving_modalities=moving,
        registrator=registrator,
        brain_extractor=SynthStripExtractor(),
        atlas_image_path=atlas_schema.format(atlas=atlas),
    )

    preprocessor.run(
        log_file=BRAINLES_LOGFILE_SCHEMA.format(base_dir=outdir),
        save_dir_transformations=REGISTRATION_TRAFO_SCHEMA.format(base_dir=outdir),
    )
    config_params = {"atlas": atlas}
    if skull_strip:
        config_params["skull_strip_algorithm"] = SKULL_STRIP_ALGORITHM_SYNTHSTRIP
    update_config(outdir, CONFIG_STEP_NORM_SS_COREGISTER, config_params)

    time_spent = time.time() - start_time
    logger.info(
        f"Finished normalization, skull stripping, co-registration step in {time_spent:.2f} seconds. Output saved to {outdir}."
    )


def longitudinal_transform_file(outdir: Path) -> Path:
    """
    Path of the follow-up-to-preop transform stored by register_recurrence in outdir: the
    affine .mat if present (affine/rigid), otherwise the displacement field (dirac/syn).
    Either can be passed to ants.apply_transforms as is.
    """
    affine_file = LONGITUDINAL_AFFINE_SCHEMA.format(base_dir=outdir)
    if affine_file.exists():
        return affine_file
    return LONGITUDINAL_DISP_SCHEMA.format(base_dir=outdir)


def read_ants_affine_pre_to_post_mm(transform_file: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Read the affine written by ants.registration(fixed=pre-op, moving=follow-up) as a 4x4
    matrix together with its rotation centre.

    Parameters:
        transform_file (Path): The ANTs affine transform file (0GenericAffine.mat).

    Returns:
        Tuple[np.ndarray, np.ndarray]: affine_pre_to_post_mm (4x4, physical LPS millimetres)
        and the rotation centre c (3,).
    """
    transform = ants.read_transform(str(transform_file))
    parameters = np.asarray(transform.parameters, dtype=np.float64)
    centre = np.asarray(transform.fixed_parameters, dtype=np.float64)
    if parameters.size != 12 or centre.size != 3:
        raise ValueError(
            f"Expected a 3-D affine transform with 12 parameters and 3 fixed parameters in "
            f"{transform_file}, got {parameters.size} and {centre.size}."
        )
    matrix = parameters[:9].reshape(3, 3)
    translation = parameters[9:]
    affine_pre_to_post_mm = np.eye(4)
    affine_pre_to_post_mm[:3, :3] = matrix
    affine_pre_to_post_mm[:3, 3] = translation + centre - matrix @ centre
    return affine_pre_to_post_mm, centre


def check_affine_plausibility(affine_pre_to_post_mm: np.ndarray, centre: np.ndarray) -> None:
    """
    Log a warning for every implausible property of the affine pre-registration.

    Parameters:
        affine_pre_to_post_mm (np.ndarray): 4x4 matrix from read_ants_affine_pre_to_post_mm.
        centre (np.ndarray): The rotation centre c of the ITK transform (3,).

    Returns:
        None
    """
    matrix = affine_pre_to_post_mm[:3, :3]
    det = float(np.linalg.det(matrix))
    if det <= 0:
        logger.warning(
            f"Affine pre-registration: determinant {det:.4f} of the linear part is not "
            "positive (reflection or degenerate fit)."
        )
    elif abs(np.log(det)) > LONGITUDINAL_AFFINE_MAX_ABS_LOG_DET:
        logger.warning(
            f"Affine pre-registration: |log det A| = {abs(np.log(det)):.3f} exceeds "
            f"{LONGITUDINAL_AFFINE_MAX_ABS_LOG_DET} (det A = {det:.4f})."
        )

    u, singular_values, vt = np.linalg.svd(matrix)
    lower, upper = LONGITUDINAL_AFFINE_SINGULAR_VALUE_RANGE
    if singular_values.min() < lower or singular_values.max() > upper:
        logger.warning(
            f"Affine pre-registration: singular values {np.round(singular_values, 4).tolist()} "
            f"outside [{lower}, {upper}]."
        )

    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vt
    angle_deg = float(np.degrees(np.arccos(np.clip((np.trace(rotation) - 1) / 2, -1.0, 1.0))))
    if angle_deg > LONGITUDINAL_AFFINE_MAX_ROTATION_DEG:
        logger.warning(
            f"Affine pre-registration: rotation angle {angle_deg:.2f} deg exceeds "
            f"{LONGITUDINAL_AFFINE_MAX_ROTATION_DEG} deg."
        )

    translation = affine_pre_to_post_mm[:3, 3] + (matrix - np.eye(3)) @ centre
    translation_mm = float(np.linalg.norm(translation))
    if translation_mm > LONGITUDINAL_AFFINE_MAX_TRANSLATION_MM:
        logger.warning(
            f"Affine pre-registration: translation {translation_mm:.2f} mm "
            f"({np.round(translation, 2).tolist()}) exceeds "
            f"{LONGITUDINAL_AFFINE_MAX_TRANSLATION_MM} mm."
        )


def register_recurrence(
    t1c_pre_file: Path,
    t1c_post_file: Path,
    recurrence_seg_file: Path,
    outdir: Path,
    fixed_mask_file: Path = None,
    moving_mask_file: Path = None,
    registration_algorithm: str = "dirac",
    additional_modalities: Optional[Dict[str, Path]] = None,
    t1_pre_file: Optional[Path] = None,
    t2_pre_file: Optional[Path] = None,
    flair_pre_file: Optional[Path] = None,
    t1_post_file: Optional[Path] = None,
    t2_post_file: Optional[Path] = None,
    flair_post_file: Optional[Path] = None,
    dirac_single_modality: bool = False,
) -> None:
    """
    Register a postop image with recurrence to preop.

    The follow-up-to-pre-op transform is stored as the ANTs affine (longitudinal_trafo.mat)
    for "affine" and "rigid" and as one dense displacement field (longitudinal_trafo.nii.gz)
    for "dirac" and "syn"; see longitudinal_transform_file. t1c, the recurrence segmentation
    and the additional modalities are warped with it.

    Parameters:
        t1c_pre_file (Path): Path to the pre-operative T1c image.
        t1c_post_file (Path): Path to the post-operative T1c image.
        recurrence_seg_file (Path): Path to the tumor segmentation.
        outdir (Path): Directory where the output files will be saved.
        fixed_mask_file (Path): Path to a mask in fixed space used for registration. Only used
            by the ANTs algorithms; ignored (with a warning) by "dirac".
        moving_mask_file (Path): Path to a mask in moving space used for registration. Only
            used by the ANTs algorithms; ignored (with a warning) by "dirac".
        registration_algorithm (str): Longitudinal registration approach. Supported
            values are "dirac", "syn" (ANTs SyN), "affine" (ANTs affine) and
            "rigid" (ANTs rigid).
        additional_modalities (Optional[Dict[str, Path]]): Mapping of modality name to its
            post-operative (followup-space) file. Each is warped into preop space alongside
            t1c_post_file. Must not contain the key "t1c".
        t1_pre_file, t2_pre_file, flair_pre_file (Optional[Path]): Pre-operative T1, T2 and
            FLAIR images. Required by "dirac" unless dirac_single_modality is set.
        t1_post_file, t2_post_file, flair_post_file (Optional[Path]): The same for the
            post-operative exam.
        dirac_single_modality (bool): Run DIRAC with the t1c-only checkpoint instead of the
            four-modality one, so that t1, t2 and flair are not needed.

    Returns:
        None
    """
    start_time = time.time()
    logger.info("Starting longitudinal co-registration.")

    additional_modalities = additional_modalities or {}
    if "t1c" in additional_modalities:
        raise ValueError(
            "additional_modalities must not contain 't1c'; use t1c_pre_file/t1c_post_file instead."
        )

    algorithm = registration_algorithm.lower()
    if algorithm not in SUPPORTED_REGISTRATION_ALGORITHMS:
        raise ValueError(
            f"Unsupported registration_algorithm '{registration_algorithm}'. "
            f"Expected one of: {sorted(SUPPORTED_REGISTRATION_ALGORITHMS)}."
        )

    config_params = {
        "registration_algorithm": algorithm,
        "use_fixed_mask": fixed_mask_file is not None,
        "use_moving_mask": moving_mask_file is not None,
    }

    if algorithm == "dirac":
        pre_files = {
            "t1": t1_pre_file,
            "t1c": t1c_pre_file,
            "flair": flair_pre_file,
            "t2": t2_pre_file,
        }
        post_files = {
            "t1": t1_post_file,
            "t1c": t1c_post_file,
            "flair": flair_post_file,
            "t2": t2_post_file,
        }
        dirac_modalities = MODALITIES_SINGLE if dirac_single_modality else MODALITIES_FOUR
        missing = [
            f"{modality}_{exam}_file"
            for exam, files in (("pre", pre_files), ("post", post_files))
            for modality in dirac_modalities
            if files[modality] is None or not Path(files[modality]).exists()
        ]
        if missing and dirac_single_modality:
            raise ValueError(
                f"DIRAC needs t1c of both exams; missing or not found: {missing}."
            )
        if missing:
            raise ValueError(
                "DIRAC with the four-modality model needs t1, t1c, t2 and flair of both exams; "
                f"missing or not found: {missing}. Pass dirac_single_modality=True to run "
                "the t1c-only model instead."
            )
        if fixed_mask_file is not None:
            logger.warning(
                f"fixed_mask_file ({fixed_mask_file}) is currently ignored for DIRAC-based recurrence registration."
            )
        if moving_mask_file is not None:
            logger.warning(
                f"moving_mask_file ({moving_mask_file}) is currently ignored for DIRAC-based recurrence registration."
            )

    recurrence_outfile = RECURRENCE_SCHEMA.format(base_dir=outdir)
    recurrence_outfile.parent.mkdir(parents=True, exist_ok=True)
    disp_file = LONGITUDINAL_DISP_SCHEMA.format(base_dir=outdir)
    affine_file = LONGITUDINAL_AFFINE_SCHEMA.format(base_dir=outdir)
    # A rerun with another algorithm must not leave the other format behind.
    disp_file.unlink(missing_ok=True)
    affine_file.unlink(missing_ok=True)
    trafo_file = affine_file if algorithm in LINEAR_REGISTRATION_ALGORITHMS else disp_file

    t1c_pre_img = ants.image_read(str(t1c_pre_file))
    t1c_post_img = ants.image_read(str(t1c_post_file))

    with tempfile.TemporaryDirectory(prefix="register_recurrence_") as tmpdir:
        if algorithm in ANTS_REGISTRATION_TRANSFORMS:
            type_of_transform = ANTS_REGISTRATION_TRANSFORMS[algorithm]
            logger.info(
                f"Running ANTs registration with type_of_transform {type_of_transform}."
            )

            reg_kwargs = {}
            if fixed_mask_file is not None:
                reg_kwargs["mask"] = ants.image_read(str(fixed_mask_file))
                logger.info(
                    f"Running with provided mask (fixed space) {str(fixed_mask_file)}."
                )
            if moving_mask_file is not None:
                reg_kwargs["moving_mask"] = ants.image_read(str(moving_mask_file))
                logger.info(
                    f"Running with provided mask (moving space) {str(moving_mask_file)}."
                )

            reg = ants.registration(
                fixed=t1c_pre_img,
                moving=t1c_post_img,
                type_of_transform=type_of_transform,
                outprefix=f"{tmpdir}/ants_",
                **reg_kwargs,
            )
            if algorithm in LINEAR_REGISTRATION_ALGORITHMS:
                # A single .mat, kept as is (a few hundred bytes).
                shutil.move(reg["fwdtransforms"][0], str(trafo_file))
            else:
                # One dense field holding all forward transforms (affine + warp for syn).
                # With compose set, apply_transforms writes the composed displacement field
                # instead of a warped image and returns the path of that file.
                composed_file = ants.apply_transforms(
                    fixed=t1c_pre_img,
                    moving=t1c_post_img,
                    transformlist=reg["fwdtransforms"],
                    compose=f"{tmpdir}/composed_",
                )
                shutil.move(str(composed_file), str(trafo_file))
        else:
            logger.info(
                f"Running ANTs {DIRAC_AFFINE_TRANSFORM} pre-registration on t1c for DIRAC."
            )
            reg = ants.registration(
                fixed=t1c_pre_img,
                moving=t1c_post_img,
                type_of_transform=DIRAC_AFFINE_TRANSFORM,
                outprefix=f"{tmpdir}/affine_",
            )
            affine_pre_to_post_mm, centre = read_ants_affine_pre_to_post_mm(
                reg["fwdtransforms"][0]
            )
            check_affine_plausibility(affine_pre_to_post_mm, centre)

            disp_pre_to_post_mm = run_dirac_registration(
                pre_files=pre_files,
                post_files=post_files,
                affine_pre_to_post_mm=affine_pre_to_post_mm,
                single_modality=dirac_single_modality,
            )
            disp_img = ants.from_numpy(
                disp_pre_to_post_mm,
                origin=t1c_pre_img.origin,
                spacing=t1c_pre_img.spacing,
                direction=t1c_pre_img.direction,
                has_components=True,
            )
            ants.image_write(disp_img, str(trafo_file))
            config_params["dirac_model"] = (
                "t1ce" if dirac_single_modality else "four_modalities"
            )
            config_params["dirac_affine_preregistration"] = DIRAC_AFFINE_TRANSFORM

    # Warp t1c, the recurrence segmentation and the additional modalities with the stored transform.
    transformlist = [str(trafo_file)]
    t1c_warped = ants.apply_transforms(
        fixed=t1c_pre_img, moving=t1c_post_img, transformlist=transformlist
    )
    ants.image_write(
        t1c_warped,
        str(LONGITUDINAL_WARP_SCHEMA.format(base_dir=outdir, modality="t1c")),
    )

    recurrence_seg = ants.image_read(str(recurrence_seg_file))
    recurrence_warped = ants.apply_transforms(
        fixed=t1c_pre_img.clone("unsigned int"),
        moving=recurrence_seg,
        transformlist=transformlist,
        interpolator="nearestNeighbor",
    )
    ants.image_write(recurrence_warped, str(recurrence_outfile))

    for modality_name, modality_post_file in additional_modalities.items():
        modality_warped = ants.apply_transforms(
            fixed=t1c_pre_img,
            moving=ants.image_read(str(modality_post_file)),
            transformlist=transformlist,
        )
        ants.image_write(
            modality_warped,
            str(LONGITUDINAL_WARP_SCHEMA.format(base_dir=outdir, modality=modality_name)),
        )

    update_config(outdir, CONFIG_STEP_REGISTER_RECURRENCE, config_params)

    time_spent = time.time() - start_time
    logger.info(
        f"Finished longitudinal co-registration in {time_spent:.2f} seconds. Output saved to {str(recurrence_outfile)}."
    )
