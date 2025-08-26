import json
import numpy as np
import nibabel as nib
from pathlib import Path
from loguru import logger
from typing import Any, Dict, Tuple
from scipy.ndimage import distance_transform_edt
from predict_gbm.base import BasePipe
from predict_gbm.evaluation.metrics import recurrence_coverage
from predict_gbm.utils.utils import (
    load_mri_data,
    load_and_resample_mri_data,
    load_segmentation,
    is_binary_array,
)
from predict_gbm.utils.constants import (
    BRAIN_MASK_SCHEMA,
    METRICS_SCHEMA,
    MODALITY_STRIPPED_SCHEMA,
    MODEL_PLAN_SCHEMA,
    PREDICTION_OUTPUT_SCHEMA,
    RECURRENCE_SCHEMA,
    STANDARD_PLAN_SCHEMA,
    TISSUE_SEG_SCHEMA,
    TISSUE_LABELS,
    TUMORSEG_SCHEMA,
)


def create_standard_plan(core_segmentation: np.ndarray, ctv_margin: int) -> np.ndarray:
    """
    Creates a target volume mask by dilating the tumor core segmentation with ctv_margin

    Parameters:
        core_segmentation (np.ndarray): A NumPy array representing the core segmentation mask, where non-zero
            values indicate the region of interest.
        ctv_margin (int): The margin to dilate the core segmentation.

    Returns:
        np.ndarray: A binary NumPy array of the same shape as core_segmentation, where
            pixels within the ctv_margin from the core segmentation are True, and
            all other pixels are False.
    """
    if ctv_margin <= 0:
        raise ValueError("ctv_margin must be a positive int.")
    distance_transform = distance_transform_edt(~(core_segmentation > 0))
    dilated_core = distance_transform <= ctv_margin
    return dilated_core.astype(np.int32)


def find_threshold(
    volume: np.ndarray,
    target_volume: float,
    tolerance: float = 0.01,
    initial_threshold: float = 0.2,
    maxIter: int = 10000,
):
    """
    Compute the threshold that produces a specified volume after masking the input array using the threshold.

    Parameters:
        volume (np.ndarray): A NumPy array representing the input volume to be thresholded (tumor cell concentration).
        target_volume (float): The desired volume size.
        tolerance (float, optional): The acceptable relative difference between the target volume and the thresholded volume. Default is 0.01 (1%).
        initial_threshold (float, optional): The starting threshold value for the segmentation. Should be between 0 and 1.
        max_iter (int, optional): The maximum number of iterations to perform. If the threshold is not found within this number of iterations,
            the function will terminate.

    Returns:
        float: The threshold value that segments the volume to match the target volume within the specified tolerance.
            Returns 1.01 if the threshold exceeds the valid range (0, 1), indicating an "above the model" condition.
            Returns 0 if the maximum number of iterations is reached without finding a suitable threshold.
    """

    if np.sum(volume > 0) < target_volume:
        print("Volume is too small")
        return 0

    # Define the initial threshold, step, and previous direction
    threshold = initial_threshold
    step = 0.1
    previous_direction = None

    # Calculate the current volume
    current_volume = np.sum(volume > threshold)

    # Iterate until the current volume is within the tolerance of the target volume
    while abs(current_volume - target_volume) / target_volume > tolerance:
        # Determine the current direction
        if current_volume > target_volume:
            direction = "increase"
        else:
            direction = "decrease"

        # Adjust the threshold
        if direction == "increase":
            threshold += step
        else:
            threshold -= step

        # Check if the threshold is out of bounds
        if threshold < 0 or threshold > 1:
            return 1.01  # above the model

        # Update the current volume
        current_volume = np.sum(volume > threshold)

        # Reduce the step size if the direction has alternated
        if previous_direction and previous_direction != direction:
            step *= 0.5

        # Update the previous direction
        previous_direction = direction

        maxIter -= 1
        if maxIter < 0:
            print("Max Iter reached, no threshold found")
            return 0

    return threshold


def generate_distance_fade_mask(binary_model_prediction: np.ndarray) -> np.ndarray:
    """Generates a fade out from a binary segmentation. Output is 1 in the segmentation and then falls off to 0."""
    if not is_binary_array(binary_model_prediction):
        raise ValueError(
            "Model prediction is not binary: {np.unique(binary_model_prediction)}"
        )

    data = np.rint(binary_model_prediction).astype(np.int32)

    # Compute distance transform on background
    distance = distance_transform_edt(data == 0)

    # Normalize distances to [0, 1] and invert: closer to mask = higher value
    max_dist = np.max(distance) if distance.max() > 0 else 1.0
    fade = 1 - (distance / max_dist)

    fade[data == 1] = 1  # Ensure mask stays at 1
    return fade.astype(np.float32)


def generate_distance_fade_mask_no_plateau(
    binary_model_prediction: np.ndarray, visible_tumor_threshold: float = 0.6
) -> np.ndarray:
    """Generates a fade out from 1.0 in the center of a binary segmentation to 0.0. The lower limit within the segmentation can be set."""
    if not is_binary_array(binary_model_prediction):
        raise ValueError(
            "Model prediction is not binary: {np.unique(binary_model_prediction)}"
        )

    data = np.rint(binary_model_prediction).astype(np.int32)

    distance_outer = distance_transform_edt(data == 0)
    distance_inner = distance_transform_edt(data != 0)

    max_outer = np.max(distance_outer)
    max_inner = np.max(distance_inner)

    distance_outer = distance_outer / max_outer
    distance_inner = distance_inner / max_inner

    fade = (
        1 - distance_outer
    ) * visible_tumor_threshold  # fade from threshold to 0 outside
    fade[data == 1] = visible_tumor_threshold + distance_inner[data == 1] * (
        1 - visible_tumor_threshold
    )  # fade from 1 to threshold inside
    return fade.astype(np.float32)


def evaluate_tumor_model(
    tumorseg_file: Path,
    recurrence_file: Path,
    pred_file: Path,
    t1c_file: Path = None,
    brain_mask_file: Path = None,
    tissue_segmentation_file: Path = None,
    ctv_margin: int = 15,
    csf_mask: bool = False,
) -> Tuple[Dict[str, Any], nib.Nifti1Image, nib.Nifti1Image]:
    """
    Evaluate a tumor model by computing recurrence coverage for standard and
    model-based radiotherapy plans.

    Parameters:
        t1c_file (Path): Path to the t1c NIfTI.
        tumorseg_file (Path): Path to the tumor segmentation NIfTI (preop)
        recurrence_file (Path): Path to the recurrence segmentation NIfTI (follow-up).
        pred_file (Path): Path to model prediction NIfTI for the tumor cell concentration.
        brain_mask_file (Path, optional): Path to brain mask NIfTI. If unavailable, a mask is extracted from t1c.
        tissue_segmentation_file (Path, optional): Path to tissue segmentation NIfTI containing csf segmentation with label 2.
        ctv_margin (int, optional): Margin used to expand the clinical target volume for the standard plan in mm. Defaults to 15.
        csf_mask (bool, optional): If true, does not consider predictions/recurrences in CSF in any way by masking it out.

    Returns:
        Dict[str, Any]: Dictionary with computed metrics
    """
    results = {}

    if t1c_file is None and brain_mask_file is None:
        raise ValueError(
            "t1c_file and brain_mask_file None. At least one has to be a valid path."
        )

    # Brain masking
    affine = nib.load(str(t1c_file)).affine
    if brain_mask_file is None:
        t1c = load_mri_data(str(t1c_file))
        background = np.min(t1c)
        brain_mask = np.rint(t1c > background)
    else:
        brain_mask = load_segmentation(brain_mask_file)

    # CSF masking
    if csf_mask:
        if tissue_segmentation_file is None:
            raise ValueError(
                "Please provide a tissue_segmentation_file when using csf_masking."
            )
        tissue_segmentation = load_segmentation(tissue_segmentation_file)
        brain_mask[tissue_segmentation == TISSUE_LABELS["csf"]] = 0

    # Load tumor/recurrence ROIs, explicit code for fast adjustments
    core_segmentation = load_segmentation(tumorseg_file)
    core_segmentation[core_segmentation == 2] = 0  # ignore edma
    core_segmentation[core_segmentation == 3] = 1

    recurrence_segmentation = load_segmentation(recurrence_file)
    recurrence_segmentation[recurrence_segmentation == 1] = 0  # ignore necrosis
    recurrence_segmentation[recurrence_segmentation == 2] = 0  # ignore edema
    recurrence_segmentation[recurrence_segmentation == 3] = 1
    recurrence_segmentation[recurrence_segmentation == 4] = 0  # ignore resection cavity

    recurrence_segmentation_all = load_segmentation(recurrence_file)
    recurrence_segmentation_all[recurrence_segmentation_all == 4] = 0
    recurrence_segmentation_all[recurrence_segmentation_all == 1] = 1
    recurrence_segmentation_all[recurrence_segmentation_all == 2] = 1
    recurrence_segmentation_all[recurrence_segmentation_all == 3] = 1

    # Load model prediction and prepare
    model_prediction = load_and_resample_mri_data(
        str(pred_file), resample_params=core_segmentation.shape, interp_type=0
    )
    if is_binary_array(model_prediction):
        logger.info(
            f"Prediction {str(pred_file)} is binary. Generating distance fade for radiation planning."
        )
        model_prediction = generate_distance_fade_mask_no_plateau(model_prediction)

    # Create standard plan
    standard_plan = create_standard_plan(core_segmentation, ctv_margin)
    standard_plan[brain_mask == 0] = 0
    standard_plan_volume = np.sum(standard_plan)
    standard_plan_nii = nib.Nifti1Image(standard_plan, affine=affine)

    # Create model based plan
    tumor_threshold = find_threshold(
        model_prediction, standard_plan_volume, initial_threshold=0.2
    )
    model_plan = (model_prediction > tumor_threshold).astype(np.int32)
    model_plan[brain_mask == 0] = 0
    model_plan_nii = nib.Nifti1Image(model_plan, affine=affine)

    # Compute coverage
    results["recurrence_coverage_standard"] = recurrence_coverage(
        recurrence_segmentation, standard_plan
    )
    results["recurrence_coverage_standard_all"] = recurrence_coverage(
        recurrence_segmentation_all, standard_plan
    )
    results["recurrence_coverage_model"] = recurrence_coverage(
        recurrence_segmentation, model_plan
    )
    results["recurrence_coverage_model_all"] = recurrence_coverage(
        recurrence_segmentation_all, model_plan
    )

    return results, standard_plan_nii, model_plan_nii


class EvaluateTumorModelPipe(BasePipe):
    """
    Evaluates tumor model with the fixed directory structure of PredictGBM to obtain radiation plans and recurrence coverage.

    Parameters:
        preop_dir (Path): Directory to the preoperative exam that has been preprocessed. Should contain the folder with the output.
        followup_dir (Path): Directory to the postoperative exam that has been preprocessed. Should contain the folder with the output.
        pred_file (Path): File path containing model prediction MRI data.
        model_id (str): Identifier for the model. Used for the name of the output file.
        ctv_margin (int, optional): Margin used to expand the clinical target volume for the standard plan in mm. Defaults to 15.
        csf_mask (bool, optional): If true, does not consider predictions/recurrences in CSF in any way by masking it out.
    """

    def __init__(
        self,
        preop_dir: Path,
        followup_dir: Path,
        model_id: str,
        ctv_margin: int = 15,
        csf_mask: bool = False,
    ) -> None:
        super().__init__(preop_dir=preop_dir, followup_dir=followup_dir)
        self.model_id = model_id
        self.ctv_margin = ctv_margin
        self.csf_mask = csf_mask

    def run(self) -> Dict[str, Any]:  # pragma: no cover - wrapper tested via pipeline
        brain_mask_file = BRAIN_MASK_SCHEMA.format(base_dir=str(self.preop_dir))
        t1c_file = MODALITY_STRIPPED_SCHEMA.format(
            base_dir=str(self.preop_dir), modality="t1c"
        )
        tissue_segmentation_file = TISSUE_SEG_SCHEMA.format(
            base_dir=str(self.preop_dir)
        )
        tumorseg_file = TUMORSEG_SCHEMA.format(base_dir=str(self.preop_dir))
        recurrence_file = RECURRENCE_SCHEMA.format(base_dir=str(self.followup_dir))
        pred_file = PREDICTION_OUTPUT_SCHEMA.format(
            base_dir=str(self.preop_dir), algo_id=self.model_id
        )

        if not brain_mask_file.exists():
            brain_mask_file = None

        results, standard_plan_nii, model_plan_nii = evaluate_tumor_model(
            t1c_file=t1c_file,
            tumorseg_file=tumorseg_file,
            recurrence_file=recurrence_file,
            pred_file=pred_file,
            brain_mask_file=brain_mask_file,
            tissue_segmentation_file=tissue_segmentation_file,
            ctv_margin=self.ctv_margin,
            csf_mask=self.csf_mask,
        )

        # Save plans
        outfile_standard = STANDARD_PLAN_SCHEMA.format(base_dir=str(self.preop_dir))
        outfile_model = MODEL_PLAN_SCHEMA.format(
            base_dir=str(self.preop_dir), algo_id=self.model_id
        )

        outfile_standard.parent.mkdir(parents=True, exist_ok=True)
        outfile_model.parent.mkdir(parents=True, exist_ok=True)

        nib.save(standard_plan_nii, outfile_standard)
        nib.save(model_plan_nii, outfile_model)

        # Save results
        save_file = METRICS_SCHEMA.format(
            base_dir=self.followup_dir, algo_id=self.model_id
        )
        save_file.parent.mkdir(exist_ok=True, parents=True)
        with open(save_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        logger.info(
            f"Finished evaluation of {self.preop_dir}. Saved results to {save_file}."
        )
        return results
