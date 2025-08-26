import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from pathlib import Path
from typing import Dict, List, Tuple
from predict_gbm.base import BasePipe
from predict_gbm.utils.utils import (
    compute_center_of_mass,
    load_mri_data,
    load_and_resample_mri_data,
)
from predict_gbm.utils.constants import (
    LONGITUDINAL_WARP_SCHEMA,
    MODALITY_STRIPPED_SCHEMA,
    PREDICTION_OUTPUT_SCHEMA,
    RECURRENCE_SCHEMA,
    TISSUE_SEG_SCHEMA,
    TUMORSEG_SCHEMA,
    TUMOR_VISUALIZATION_SCHEMA,
    RECURRENCE_VISUALIZATION_SCHEMA,
)


def get_slices(
    center: Tuple[int, int, int],
    num_slices: int,
    step_size: int,
    patient_dim: Tuple[int, int, int],
):
    axial_slices = [
        center[2] + ind * step_size - 2 * step_size for ind in range(0, num_slices)
    ]
    axial_slices = [
        min(max(0, ax_slice), patient_dim[2] - 1) for ax_slice in axial_slices
    ]
    coronal_slices = [
        center[1] + ind * step_size - 2 * step_size for ind in range(0, num_slices)
    ]
    coronal_slices = [
        min(max(0, cor_slice), patient_dim[1] - 1) for cor_slice in coronal_slices
    ]
    return axial_slices, coronal_slices


def get_cmap_norm_patches_tumorseg(classes_of_interest: List[int]):
    # Tumor segmentation legend (1: non enhancing, 2: edema, 3: enhancing)
    colors = [
        (0, 0, 0, 0),
        (1, 127 / 255, 0, 1),
        (30 / 255, 144 / 255, 1, 1),
        (138 / 255, 43 / 255, 226 / 255, 1),
    ]
    color_labels = ["Non-enhancing Tumor", "Peritumoral Edema", "Enhancing Tumor"]
    cmap = mcolors.ListedColormap(colors)
    bounds = [0, 0.5, 1.5, 2.5, 3.5]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    patches = [
        mpatches.Patch(color=c, label=l) for (c, l) in zip(colors[1:], color_labels)
    ]
    return cmap, norm, patches


def get_cmap_norm_patches_tumorseg_5(classes_of_interest: List[int]):
    # Tumor segmentation legend (1: non enhancing, 2: edema, 3: enhancing)
    colors = [
        (0, 0, 0, 0),
        (1, 127 / 255, 0, 1),
        (30 / 255, 144 / 255, 1, 1),
        (138 / 255, 43 / 255, 226 / 255, 1),
        (34 / 255.0, 139 / 255.0, 34 / 255.0, 1),
        (210 / 255.0, 43 / 255.0, 43 / 255.0, 1),
    ]
    color_labels = [
        "Necrosis",
        "Peritumoral Edema",
        "Enhancing Tumor",
        "Standard Plan",
        "Model Plan",
    ]
    cmap = mcolors.ListedColormap(colors)
    bounds = [0, 0.5, 1.5, 2.5, 3.5]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    patches = [
        mpatches.Patch(color=c, label=l) for (c, l) in zip(colors[1:], color_labels)
    ]
    return cmap, norm, patches


def get_segmentation_projection(
    segmentation: np.ndarray, label: int, axis: int
) -> np.ndarray:
    seg_data = segmentation.copy()
    seg_data[seg_data != label] = 0
    projection = np.rint(np.sum(seg_data, axis=axis) > 0)
    return projection


def grid_plot(
    image_tensor: np.ndarray,
    imshow_args: List[Dict],
    header: str,
    col_titles: List[str],
    row_titles: List[str],
    outfile: str,
    legend_handles: List[mpatches.Patch] = None,
) -> None:
    """
    A generic function to create a grid plot with multiple layers / overlays.

    Args:
        image_tensor: A numpy array with dimension 3 (n_layers, n_cols, n_rows) where each point is a 2D-image or None.
        imshow_args: A list of dictionaries containing arguments for imshow calls for each image layer (e.g. {"cmap": "gray"}).
        header: String to be displayd at the top of the image.
        col_titles: List of strings used as column titles.
        row_titles: List of strings used as row titles.
        outfile: File that the pdf is saved to.
        legend_handles: List of matplotlib.patches.Patch to be displayed in a legend.
    """

    if image_tensor.ndim != 3:
        raise ValueError(
            "Dimension mismatch. image_tensor dimension should be 3: (n_layers, n_cols, n_rows)"
        )

    if len(imshow_args) != image_tensor.shape[0]:
        raise ValueError(
            f"Dimension mismatch. imshow_args should be the same length as image_tensor.shape[0] = {image_tensor.shape[0]}."
        )

    if len(row_titles) != image_tensor.shape[1]:
        raise ValueError(
            f"Dimension mismatch. row_titles should be the same length as image_tensor.shape[1] = {image_tensor.shape[1]}."
        )

    n_row = image_tensor.shape[1]
    n_col = image_tensor.shape[2]

    # Create figure and fill axes
    fig, axs = plt.subplots(n_row, n_col, figsize=(5 * n_col, 4 * n_row))
    for image_layer, imshow_args in zip(image_tensor, imshow_args):
        for row in range(n_row):
            for col in range(n_col):
                if image_layer[row, col] is not None:
                    axs[row, col].imshow(np.rot90(image_layer[row, col]), **imshow_args)
                    axs[row, col].axis("off")

    if len(axs.flatten()) == len(col_titles):
        for ct, ax in zip(col_titles, axs.flatten()):
            ax.set_title(ct, fontsize=16, pad=20)
    else:
        for ind, col_title in enumerate(col_titles):
            axs[0, ind].set_title(col_title, fontsize=16, fontweight="bold", pad=20)

    # Row titles
    for ind, row_title in enumerate(row_titles):
        axs[ind, 0].axis("on")
        axs[ind, 0].tick_params(
            left=False, bottom=False, labelleft=False, labelbottom=False
        )
        axs[ind, 0].set_ylabel(row_title, fontweight="bold", labelpad=20, fontsize=16)

    # Header
    fig.subplots_adjust(top=0.85)
    fig.suptitle(
        header,
        horizontalalignment="left",
        fontsize=20,
        fontweight="bold",
        color="black",
        y=0.92,
        x=0.0665,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.7),
    )

    # Color legends
    if legend_handles is not None:
        fig.legend(
            handles=legend_handles,
            loc="upper right",
            bbox_to_anchor=(0.96, 0.890),
            ncol=3,
        )

    plt.tight_layout(rect=[0, 0.05, 1.0, 0.9])
    plt.tight_layout(rect=[0, 0.0, 1.0, 0.9])
    Path(outfile).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outfile, format="pdf")
    print(f"Plot saved as {outfile}")
    plt.close(fig)


def plot_model_multislice(
    patient_id: str,
    model_id: str,
    exam_dir: Path,
    outfile: str,
    classes_of_interest: List[int] = [1, 2, 3],
) -> None:

    c_threshold = 0.01  # tumor cell concentration threshold
    n_layers = 3  # one layer for each imshow config

    # Load data
    t1c_data = load_mri_data(
        MODALITY_STRIPPED_SCHEMA.format(base_dir=exam_dir, modality="t1c")
    )
    tumorseg_data = load_mri_data(TUMORSEG_SCHEMA.format(base_dir=exam_dir))
    tissueseg_data = load_mri_data(TISSUE_SEG_SCHEMA.format(base_dir=exam_dir))
    model_data = load_and_resample_mri_data(
        PREDICTION_OUTPUT_SCHEMA.format(base_dir=exam_dir, algo_id=model_id.lower()),
        resample_params=t1c_data.shape,
        interp_type=1,
    )

    # Compute tumor center of mass
    center = compute_center_of_mass(tumorseg_data, t1c_data, classes_of_interest)

    # Create axial/coronal slices
    step_size = 10
    num_slices = 5
    patient_dim = t1c_data.shape
    axial_slices, coronal_slices = get_slices(
        center, num_slices, step_size, patient_dim
    )

    # Tumor segmentation args
    cmap, norm, patches = get_cmap_norm_patches_tumorseg(classes_of_interest)

    # Titles
    col_titles = ["T1C", "TUMORSEG", f"{model_id.upper()}", "TISSUESEG"]
    row_titles = axial_slices + coronal_slices
    header = (
        f"Patient: {patient_id}\n"
        f"Model: {model_id}\n"
        f"Tumor cell concentration threshold: {c_threshold}\n"
    )

    # Build image tensor
    image_tensor = np.empty((n_layers, num_slices * 2, 4), dtype=object)

    # Layer 1: T1c, T1c, T1c, Tissueseg
    layer_1_args = {"cmap": "gray", "interpolation": "none"}
    for ind, ax_slice, cor_slice in zip(
        range(num_slices), axial_slices, coronal_slices
    ):

        image_tensor[0, ind, 0] = t1c_data[:, :, ax_slice]
        image_tensor[0, ind, 1] = t1c_data[:, :, ax_slice]
        image_tensor[0, ind, 2] = t1c_data[:, :, ax_slice]
        image_tensor[0, ind, 3] = tissueseg_data[:, :, ax_slice]

        image_tensor[0, ind + num_slices, 0] = t1c_data[:, cor_slice, :]
        image_tensor[0, ind + num_slices, 1] = t1c_data[:, cor_slice, :]
        image_tensor[0, ind + num_slices, 2] = t1c_data[:, cor_slice, :]
        image_tensor[0, ind + num_slices, 3] = tissueseg_data[:, cor_slice, :]

    # Layer 2: None, Tumorseg, None, None
    layer_2_args = {"cmap": cmap, "norm": norm, "alpha": 0.9, "interpolation": "none"}
    for ind, ax_slice, cor_slice in zip(
        range(num_slices), axial_slices, coronal_slices
    ):

        image_tensor[1, ind, 1] = tumorseg_data[:, :, ax_slice]
        image_tensor[1, ind + num_slices, 1] = tumorseg_data[:, cor_slice, :]

    # Layer 3: None, None, Model, None
    layer_3_args = {
        "cmap": "inferno",
        "alpha": 0.90,
        "vmin": 0.0,
        "vmax": 1.0,
        "interpolation": "none",
    }
    for ind, ax_slice, cor_slice in zip(
        range(num_slices), axial_slices, coronal_slices
    ):

        image_tensor[2, ind, 2] = model_data[:, :, ax_slice]
        image_tensor[2, ind + num_slices, 2] = model_data[:, cor_slice, :]

    # Imshow arguments
    imshow_args = [layer_1_args, layer_2_args, layer_3_args]

    grid_plot(
        image_tensor=image_tensor,
        imshow_args=imshow_args,
        header=header,
        col_titles=col_titles,
        row_titles=row_titles,
        outfile=outfile,
        legend_handles=patches,
    )


def plot_recurrence_multislice(
    patient_id: str,
    preop_dir: Path,
    followup_dir: Path,
    outfile: str,
    classes_of_interest: List[int] = [1, 2, 3],
) -> None:

    n_layers = 2  # one layer for each imshow config

    t1c_pre_dir = MODALITY_STRIPPED_SCHEMA.format(base_dir=preop_dir, modality="t1c")
    t1c_post_dir = LONGITUDINAL_WARP_SCHEMA.format(base_dir=followup_dir)
    tumor_seg_dir = TUMORSEG_SCHEMA.format(base_dir=preop_dir)
    recurrence_seg_dir = RECURRENCE_SCHEMA.format(base_dir=followup_dir)

    # Load images
    t1c_data_pre = load_mri_data(t1c_pre_dir)
    seg_data_pre = load_mri_data(tumor_seg_dir)
    t1c_data_post = load_mri_data(t1c_post_dir)
    seg_data_post = load_mri_data(recurrence_seg_dir)
    seg_data_post[seg_data_post == 4] = 0  # ignore ressection cavity label

    # Create axial/coronal slices
    center = compute_center_of_mass(seg_data_pre, t1c_data_pre, classes_of_interest)
    step_size = 10
    num_slices = 5
    patient_dim = t1c_data_pre.shape
    axial_slices, coronal_slices = get_slices(
        center, num_slices, step_size, patient_dim
    )

    # Tumor segmentation legend (1: non enhancing, 2: edema, 3: enhancing)
    cmap, norm, patches = get_cmap_norm_patches_tumorseg(classes_of_interest)

    # Titles
    col_titles = [
        "T1C (preop)",
        "T1C (preop)+Tumor",
        "T1C (follow up)",
        "T1C (follow up) + Recurrence",
    ]
    row_titles = axial_slices + coronal_slices
    header = (
        f"Patient: {patient_id}\n"
        f"CoM slice (axial/coronal): {center[2]}/{center[1]}\n"
    )

    # Build image tensor
    image_tensor = np.empty((n_layers, num_slices * 2, 4), dtype=object)

    # Layer 1: T1c (pre), T1c (pre), T1c (post, T1c (post)
    layer_1_args = {"cmap": "gray", "interpolation": "none"}
    for ind, ax_slice, cor_slice in zip(
        range(num_slices), axial_slices, coronal_slices
    ):

        image_tensor[0, ind, 0] = t1c_data_pre[:, :, ax_slice]
        image_tensor[0, ind, 1] = t1c_data_pre[:, :, ax_slice]
        image_tensor[0, ind, 2] = t1c_data_post[:, :, ax_slice]
        image_tensor[0, ind, 3] = t1c_data_post[:, :, ax_slice]

        image_tensor[0, ind + num_slices, 0] = t1c_data_pre[:, cor_slice, :]
        image_tensor[0, ind + num_slices, 1] = t1c_data_pre[:, cor_slice, :]
        image_tensor[0, ind + num_slices, 2] = t1c_data_post[:, cor_slice, :]
        image_tensor[0, ind + num_slices, 3] = t1c_data_post[:, cor_slice, :]

    # Layer 2: None, Tumorseg (pre), None, Tumorseg (post)
    layer_2_args = {"cmap": cmap, "norm": norm, "alpha": 0.9, "interpolation": "none"}
    for ind, ax_slice, cor_slice in zip(
        range(num_slices), axial_slices, coronal_slices
    ):

        image_tensor[1, ind, 1] = seg_data_pre[:, :, ax_slice]
        image_tensor[1, ind, 3] = seg_data_post[:, :, ax_slice]
        image_tensor[1, ind + num_slices, 1] = seg_data_pre[:, cor_slice, :]
        image_tensor[1, ind + num_slices, 3] = seg_data_post[:, cor_slice, :]

    # Imshow arguments
    imshow_args = [layer_1_args, layer_2_args]

    grid_plot(
        image_tensor=image_tensor,
        imshow_args=imshow_args,
        header=header,
        col_titles=col_titles,
        row_titles=row_titles,
        outfile=outfile,
        legend_handles=patches,
    )


def visualization_pipe(
    patient_id: str,
    model_id: str,
    preop_dir: Path,
    followup_dir: Path,
) -> None:
    """
    Generates common visualizations from pipeline outputs.

    Parameters:
        patient_id (str): String acting as patient identifier. Included in the output directory structure.
        model_id (str): String identifying the growth model to be used.
        preop_dir (Path): Path to the output directory containing the pre-operative pipeline outputs.
        followup_dir (Path): Path to the output directory containing the follow-up pipeline outputs.
    """
    outfile_tumor_vis = TUMOR_VISUALIZATION_SCHEMA.format(base_dir=preop_dir)
    outfile_recurrence_vis = RECURRENCE_VISUALIZATION_SCHEMA.format(
        base_dir=followup_dir
    )

    plot_model_multislice(
        patient_id=patient_id,
        model_id=model_id,
        exam_dir=preop_dir,
        outfile=str(outfile_tumor_vis),
    )

    plot_recurrence_multislice(
        patient_id=patient_id,
        preop_dir=preop_dir,
        followup_dir=followup_dir,
        outfile=str(outfile_recurrence_vis),
    )


class VisualizationPipe(BasePipe):
    """
    Generates common visualizations from pipeline outputs.

    Parameters:
        patient_id (str): String acting as patient identifier. Included in the output directory structure.
        model_id (str): String identifying the growth model to be used.
        preop_dir (Path): Path to the output directory containing the pre-operative pipeline outputs.
        followup_dir (Path): Path to the output directory containing the follow-up pipeline outputs.
    """

    def __init__(
        self,
        patient_id: str,
        model_id: str,
        preop_dir: Path,
        followup_dir: Path,
    ) -> None:
        super().__init__(preop_dir=preop_dir, followup_dir=followup_dir)
        self.patient_id = patient_id
        self.model_id = model_id

    def run(self) -> None:  # pragma: no cover - wrapper for visual output
        outfile_tumor_vis = TUMOR_VISUALIZATION_SCHEMA.format(base_dir=self.preop_dir)
        outfile_recurrence_vis = RECURRENCE_VISUALIZATION_SCHEMA.format(
            base_dir=self.followup_dir
        )

        plot_model_multislice(
            patient_id=self.patient_id,
            model_id=self.model_id,
            exam_dir=self.preop_dir,
            outfile=str(outfile_tumor_vis),
        )

        plot_recurrence_multislice(
            patient_id=self.patient_id,
            preop_dir=self.preop_dir,
            followup_dir=self.followup_dir,
            outfile=str(outfile_recurrence_vis),
        )
