import os
import ants
import json
import tempfile
import numpy as np
import nibabel as nib
from pathlib import Path
from loguru import logger
from pypdf import PdfWriter
from contextlib import contextmanager
from scipy.ndimage import center_of_mass
from typing import Any, Dict, Iterable, List, Tuple, Optional, Union
from brats.utils.data_handling import remove_tmp_folder
from predict_gbm.utils.constants import CONFIG_SCHEMA, RESERVED_MODALITY_NAMES


def compute_center_of_mass(
    seg_data: np.ndarray,
    mri_data: np.ndarray,
    classes: List[int] = [1, 2, 3],
) -> Tuple[int, int, int]:

    mask = np.isin(seg_data, classes)

    # Check if the mask contains any non-zero values (i.e., non-empty segmentation)
    if not np.any(mask):
        logger.warning("Segmentation is empty, returning middle slices of the MRI.")
        # Return the middle slices of the MRI volume as default
        return (mri_data.shape[0] // 2, mri_data.shape[1] // 2, mri_data.shape[2] // 2)

    # Compute center of mass if the segmentation is non-empty
    com = center_of_mass(mask)
    return tuple(map(int, com))


def load_mri_data(filepath: Union[Path, str]) -> np.ndarray:
    img = nib.load(str(filepath))
    data = img.get_fdata()
    return data


def load_and_resample_mri_data(
    filepath: Union[str, Path],
    resample_params: Tuple[int, int, int],
    interp_type: Optional[int] = 0,
) -> np.ndarray:

    img = ants.image_read(str(filepath))
    img = ants.resample_image(
        image=img,
        resample_params=resample_params,
        use_voxels=True,
        interp_type=interp_type,
    )
    # There used to be an ants function for this
    tmp_file = tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False)
    tmp_file.close()
    try:
        ants.image_write(img, tmp_file.name)
        data = nib.load(tmp_file.name).get_fdata()
    finally:
        os.remove(tmp_file.name)
    return data


def load_segmentation(filepath: Union[Path, str]) -> np.ndarray:
    return np.rint(load_mri_data(str(filepath))).astype(np.int32)


def merge_pdfs(pdf_list: List[Union[str, Path]], output_pdf: Union[str, Path]) -> None:
    """Merge multiple PDFs into a single PDF using pypdf>=4.0"""
    pdf_writer = PdfWriter()

    for pdf in pdf_list:
        pdf_writer.append(str(pdf))

    with open(output_pdf, "wb") as f:
        pdf_writer.write(f)

    logger.info(f"Combined PDF saved as {str(output_pdf)}")


def validate_additional_modality_names(
    additional_modality_names: Iterable[str],
    additional_quantitative_modality_names: Iterable[str],
) -> None:
    """
    Ensures additional modality names don't collide with the reserved modality
    names (t1, t1c, t2, flair) or with each other.

    Parameters:
        additional_modality_names (Iterable[str]): Names processed with intensity normalization.
        additional_quantitative_modality_names (Iterable[str]): Names processed without
            intensity normalization.

    Raises:
        ValueError: If any name collides with a reserved modality name, or if the two
            iterables share a name.
    """
    additional_modality_names = set(additional_modality_names)
    additional_quantitative_modality_names = set(additional_quantitative_modality_names)

    collisions = (
        additional_modality_names | additional_quantitative_modality_names
    ) & RESERVED_MODALITY_NAMES
    if collisions:
        raise ValueError(
            f"additional modality names {sorted(collisions)} collide with reserved "
            f"modality names {sorted(RESERVED_MODALITY_NAMES)}."
        )

    overlap = additional_modality_names & additional_quantitative_modality_names
    if overlap:
        raise ValueError(
            f"additional_modality_names and additional_quantitative_modality_names "
            f"share names {sorted(overlap)}."
        )


def is_binary_array(arr: np.ndarray) -> bool:
    allowed_values = {0, 1, 0.0, 1.0, False, True}
    return np.all(np.isin(arr, list(allowed_values)))


@contextmanager
def temporary_tmpdir(base_dir: Union[str, Path]) -> Path:
    """Create and clean up a temporary directory used as TMPDIR.

    All files written to the system temporary directory during the context
    lifetime will be redirected to this folder. In addition to setting the
    ``TMPDIR`` environment variable, this also updates ``tempfile.tempdir`` so
    libraries that cache the temporary directory respect the new location.
    The ``base_dir`` folder will be created if it does not already exist.
    """
    base_dir = Path(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    tmpdir = Path(tempfile.mkdtemp(dir=str(base_dir), prefix="tmp_"))
    logger.info(f"Created temporary directory at {tmpdir}")
    old_tmpdir = os.environ.get("TMPDIR")
    old_tempfile_dir = tempfile.tempdir
    os.environ["TMPDIR"] = str(tmpdir)
    tempfile.tempdir = str(tmpdir)
    try:
        yield tmpdir
    finally:
        if old_tmpdir is not None:
            os.environ["TMPDIR"] = old_tmpdir
        else:
            os.environ.pop("TMPDIR", None)
        tempfile.tempdir = old_tempfile_dir
        remove_tmp_folder(tmpdir)
        logger.info(f"Removed temporary directory at {tmpdir}")


def update_config(outdir: Union[str, Path], step_name: str, params: Dict[str, Any]) -> None:
    """
    Create or update the predict_gbm_config.json file in outdir, recording params under
    step_name. Creates the file (and outdir) if they don't exist yet, and overwrites any
    existing entry for step_name, leaving entries for other steps untouched.

    Parameters:
        outdir (Union[str, Path]): Directory containing (or to contain) the config file.
        step_name (str): Name of the processing step, used as the top-level key. Should be
            one of the CONFIG_STEP_* constants in predict_gbm.utils.constants.
        params (Dict[str, Any]): JSON-serializable parameters to record for this step.

    Returns:
        None
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    config_file = CONFIG_SCHEMA.format(base_dir=outdir)

    if config_file.exists():
        with open(config_file, "r") as f:
            config = json.load(f)
    else:
        config = {}

    config[step_name] = params

    tmp_file = config_file.with_name(config_file.name + ".tmp")
    with open(tmp_file, "w") as f:
        json.dump(config, f, indent=2)
    tmp_file.replace(config_file)
