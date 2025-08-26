import numpy as np
from typing import List, Tuple
from sklearn.metrics import roc_curve, auc
from predict_gbm.utils.utils import is_binary_array


def recurrence_coverage(
    recurrence_segmentation: np.ndarray, target_volume: np.ndarray
) -> float:
    """
    Calculate the coverage of tumor recurrence by the treatment plan volume.

    Parameters:
        recurrence_segmentation (np.ndarray): A (boolean) NumPy array indicating the presence of tumor recurrence.
        target_volume (np.ndarray): A (boolean) NumPy array indicating the area covered by the treatment plan.

    Returns:
        float: The coverage ratio of the treatment plan over the tumor recurrence (0-1.0).
            Returns 1.0 if there is no tumor recurrence.
    """
    if not is_binary_array(recurrence_segmentation):
        raise ValueError(
            "recurrence_segmentation values have to be in (True, False, 0, 1, 0.0, 1.0)."
        )
    if not is_binary_array(target_volume):
        raise ValueError(
            "target_volume values have to be in (True, False, 0, 1, 0.0, 1.0)."
        )
    if recurrence_segmentation.shape != target_volume.shape:
        raise ValueError(
            "Dimension mismatch between recurrence_segmentation and target_volume."
        )

    # If there is no recurrence, return 1
    if np.sum(recurrence_segmentation) <= 0.00001:
        return 1

    # Calculate the intersection between the recurrence and the plan
    intersection = np.logical_and(recurrence_segmentation, target_volume)

    # Calculate the coverage as the ratio of the intersection to the recurrence
    coverage = np.sum(intersection) / np.sum(recurrence_segmentation)
    return coverage


def missed_voxels(
    recurrence_segmentation: np.ndarray, target_volume: np.ndarray
) -> int:
    """
    Count recurrence voxels not covered by a treatment plan.

    Parameters:
        recurrence_segmentation (np.ndarray): Boolean array marking the observed recurrence region.
        target_volume (np.ndarray): Boolean array representing the treatment plan volume.

    Returns:
        int: Number of recurrence voxels outside of the treatment plan volume.
    """
    if not is_binary_array(recurrence_segmentation):
        raise ValueError(
            "recurrence_segmentation values have to be in (True, False, 0, 1, 0.0, 1.0)."
        )
    if not is_binary_array(target_volume):
        raise ValueError(
            "target_volume values have to be in (True, False, 0, 1, 0.0, 1.0)."
        )
    if recurrence_segmentation.shape != target_volume.shape:
        raise ValueError(
            "Dimension mismatch between recurrence_segmentation and target_volume."
        )

    missed = np.logical_and(recurrence_segmentation, np.logical_not(target_volume))
    return int(np.sum(missed))


def roc_auc(
    pred: np.ndarray,
    seg: np.ndarray,
    mask: np.ndarray,
    labels_of_interest: List[int] = [1, 3],
    drop_intermediate: bool = True,
    threshold_range: Tuple = (0.20, 0.70),
) -> float:
    """
    Computes roc auc for a prediction array and a segmentation. ROI are specified via labels_of_interest.
    """
    if mask is None:
        mask = np.ones_like(seg, dtype=bool)
    mask_bool = mask.astype(bool)

    # Flatten only voxels under mask
    scores = pred[mask_bool].ravel()
    labels = seg[mask_bool].ravel()

    # Get recurrence core according to labels of interest
    y_true = np.isin(labels, labels_of_interest).astype(int)

    # Guard against degenerate cases
    pos = y_true.sum()
    neg = y_true.size - pos
    if pos == 0 or neg == 0:
        return 0.0

    # ROC and AUC
    fpr, tpr, thresholds = roc_curve(
        y_true, scores, drop_intermediate=drop_intermediate
    )
    auc_value = auc(fpr, tpr)

    return auc_value
