import time
import numpy as np
import nibabel as nib
from pathlib import Path
from loguru import logger
from brats import AdultGliomaPreAndPostTreatmentSegmenter
from brats.constants import AdultGliomaPreAndPostTreatmentAlgorithms
from predict_gbm.utils.constants import (
    CONFIG_STEP_TUMOR_SEG,
    TUMORSEG_EDEMA_SCHEMA,
    TUMORSEG_SCHEMA,
    TUMORSEG_CORE_SCHEMA,
)
from predict_gbm.utils.utils import update_config


def split_segmentation(
    tumor_seg_file: Path,
    outdir: Path,
    necrotic_label: int = 1,
    edema_label: int = 2,
    enhancing_label: int = 3,
) -> None:
    """
    Split a composite tumor segmentation into separate binary segmentation files
    for enhancing/non-enhancing tumor and peritumoral edema.

    Parameters:
        tumor_seg_file (Path): Path to the input tumor segmentation NIfTI file.
        outdir (Path): Path to the output directory. Usually exam directory.
        necrotic_label (int): Label for necrotic / non-enhancing tissue in the segmentation.
        edema_label (int): Label for edema in the segmentation.
        enhancing_label (int): Label for enhancing tumor in the segmentation.

    Returns:
        None
    """
    logger.debug("Splitting tumor segmentation into core and edema.")
    tumor_seg = nib.load(str(tumor_seg_file))
    seg_data = np.rint(tumor_seg.get_fdata()).astype(np.int32)

    # Create a binary mask for non-enhancing and enhancing tumor (labels 1 and 3).
    enhancing_non_enhancing = nib.Nifti1Image(
        np.rint((seg_data == necrotic_label) | (seg_data == enhancing_label)).astype(
            np.int32
        ),
        affine=tumor_seg.affine,
    )

    # Create a binary mask for edema (label 2).
    edema = nib.Nifti1Image(
        np.rint(seg_data == edema_label).astype(np.int32),
        affine=tumor_seg.affine,
    )

    nib.save(enhancing_non_enhancing, str(TUMORSEG_CORE_SCHEMA.format(base_dir=outdir)))
    nib.save(edema, str(TUMORSEG_EDEMA_SCHEMA.format(base_dir=outdir)))
    logger.debug(
        f"Finished splitting segmentation. Output saved to {TUMORSEG_CORE_SCHEMA.format(base_dir=outdir).parent}."
    )


def run_brats(
    t1_file: Path,
    t1c_file: Path,
    t2_file: Path,
    flair_file: Path,
    outdir: Path,
    cuda_device: str = "0",
) -> None:
    """
    Segments tumor based on common MRI modalities using brainles BRATS module.

    Parameters:
        t1_file (Path): Path to the T1 file.
        t1c_file (Path): Path to the T1c file.
        t2_file (Path): Path to the T2 file.
        flair_file (Path): Path to the flair file.
        outdir (Path): Directory to save the output to. Usually exam directory.
        cuda_device (str): The GPU device to run on.

    Returns:
        None
    """
    start_time = time.time()
    logger.info("Starting tumor segmentation via BRATS.")
    algorithm = AdultGliomaPreAndPostTreatmentAlgorithms.BraTS25_1
    segmenter = AdultGliomaPreAndPostTreatmentSegmenter(
        algorithm=algorithm,
        cuda_devices=cuda_device,
    )

    seg_outfile = str(TUMORSEG_SCHEMA.format(base_dir=outdir))
    segmenter.infer_single(
        t1n=str(t1_file),
        t1c=str(t1c_file),
        t2w=str(t2_file),
        t2f=str(flair_file),
        output_file=seg_outfile,
    )

    split_segmentation(seg_outfile, outdir)

    update_config(outdir, CONFIG_STEP_TUMOR_SEG, {"algorithm": algorithm.value})

    time_spent = time.time() - start_time
    logger.info(
        f"Finished tumor segmentation in {time_spent:.2f} seconds. Saved output to {seg_outfile}."
    )
