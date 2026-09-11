import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from loguru import logger
from matplotlib.backends.backend_pdf import PdfPages

from predict_gbm.utils.constants import (
    MODEL_OUTPUT_DIR,
    SKULL_STRIP_FOLDER,
    TUMOR_SEGMENTATION_FOLDER,
)
from predict_gbm.utils.utils import compute_center_of_mass

# Exams (in row order) and the folder that holds their outputs in preop space; see
# scripts/preprocess_respond10.py for the respond10 layout.
EXAMS = ("preop", "postop", "followup")
PREOP_SPACE_FOLDER = "preop_space"
# The volumes follow the (x, y, z) = (sagittal, coronal, axial) convention, so each view
# is defined by the axis it slices along.
VIEWS = ("Axial", "Sagittal", "Coronal")
# Tumor core labels of the preop segmentation through whose center of mass all slices go.
CORE_CLASSES = [1, 3]
# Tumor cell concentrations below this value are not drawn, so the T1c stays visible.
PREDICTION_THRESHOLD = 0.01

# Segmentation legend: 1 necrosis/non-enhancing, 2 edema, 3 enhancing, 4 resection cavity
# (label 4 only occurs in the longitudinal recurrence segmentations).
SEG_COLORS = [
    (0, 0, 0, 0),
    (1, 127 / 255, 0, 1),
    (30 / 255, 144 / 255, 1, 1),
    (138 / 255, 43 / 255, 226 / 255, 1),
    (34 / 255, 139 / 255, 34 / 255, 1),
]
SEG_LABELS = ["Necrosis / non-enhancing", "Edema", "Enhancing tumor", "Resection cavity"]
SEG_CMAP = mcolors.ListedColormap(SEG_COLORS)
SEG_NORM = mcolors.BoundaryNorm([0, 0.5, 1.5, 2.5, 3.5, 4.5], SEG_CMAP.N)
SEG_PATCHES = [mpatches.Patch(color=c, label=l) for c, l in zip(SEG_COLORS[1:], SEG_LABELS)]
PREDICTION_CMAP = "inferno"


def find_exam_dirs(patient_dir: Path) -> Dict[str, Optional[Path]]:
    """
    Maps each exam (preop/postop/followup) to its directory (e.g. preop_d0, postop_d3), or
    None if there is no unique candidate.
    """
    exam_dirs = {}
    for exam in EXAMS:
        candidates = sorted(d for d in patient_dir.glob(f"{exam}_d*") if d.is_dir())
        if len(candidates) != 1:
            logger.warning(
                f"{patient_dir.name}: expected one '{exam}_d*' directory, found "
                f"{[c.name for c in candidates]}."
            )
            exam_dirs[exam] = None
        else:
            exam_dirs[exam] = candidates[0]
    return exam_dirs


def resolve_patient_files(
    patient_dir: Path, exam_dirs: Dict[str, Optional[Path]], models: List[str]
) -> Dict[str, Optional[Path]]:
    """
    Resolves the images of a patient. The preop exam provides the reference T1c, the tumor
    segmentation and the growth model predictions; each postop/followup exam provides its
    raw (skull stripped) T1c and, from the longitudinal registration, the T1c and tumor
    segmentation warped to preop space. Missing files are mapped to None and logged.
    """
    files = {}
    preop_dir = exam_dirs["preop"]
    if preop_dir is not None:
        files["preop_t1c"] = preop_dir / SKULL_STRIP_FOLDER / "t1c_bet_normalized.nii.gz"
        files["preop_seg"] = preop_dir / TUMOR_SEGMENTATION_FOLDER / "tumor_seg.nii.gz"
        for model in models:
            files[f"pred_{model}"] = preop_dir / MODEL_OUTPUT_DIR / f"{model}_pred.nii.gz"
    for exam in ("postop", "followup"):
        exam_dir = exam_dirs[exam]
        if exam_dir is None:
            continue
        files[f"{exam}_t1c_raw"] = exam_dir / SKULL_STRIP_FOLDER / "t1c_bet_normalized.nii.gz"
        files[f"{exam}_t1c_warped"] = (
            exam_dir / PREOP_SPACE_FOLDER / "t1c_warped_longitudinal.nii.gz"
        )
        files[f"{exam}_seg_warped"] = exam_dir / PREOP_SPACE_FOLDER / "recurrence_preop.nii.gz"

    for key, f in files.items():
        if not f.is_file():
            logger.warning(f"{patient_dir.name}: {key} not found ({f}), panel left empty.")
            files[key] = None
    return files


def load_volume(nifti_file: Path) -> Tuple[np.ndarray, Tuple[float, float, float]]:
    """Loads the voxel data and the voxel spacing of a nifti file."""
    img = nib.load(str(nifti_file))
    return np.asanyarray(img.dataobj).astype(np.float32), tuple(
        float(z) for z in img.header.get_zooms()[:3]
    )


def get_slice(data: np.ndarray, view: str, center: Tuple[int, int, int]) -> np.ndarray:
    """Extracts the 2D slice of a view through the center voxel, rotated for display."""
    x, y, z = (min(max(0, c), s - 1) for c, s in zip(center, data.shape))
    if view == "Axial":
        return np.rot90(data[:, :, z])
    if view == "Sagittal":
        return np.rot90(data[x, :, :])
    if view == "Coronal":
        return np.rot90(data[:, y, :])
    raise ValueError(f"Unknown view {view}.")


def get_aspect(zooms: Tuple[float, float, float], view: str) -> float:
    """
    Aspect ratio (height of a pixel over its width) of a view, so anisotropic voxels are
    displayed with the correct proportions. After the rotation in get_slice the display axes
    are (row, col) = (y, x) for axial, (z, y) for sagittal and (z, x) for coronal.
    """
    zoom_x, zoom_y, zoom_z = zooms
    if view == "Axial":
        return zoom_y / zoom_x
    if view == "Sagittal":
        return zoom_z / zoom_y
    if view == "Coronal":
        return zoom_z / zoom_x
    raise ValueError(f"Unknown view {view}.")


def get_intensity_window(data: np.ndarray, percentile: float = 99.5) -> float:
    """Upper display intensity of an MRI volume, clipped at a percentile of the foreground."""
    foreground = data[data > 0]
    if foreground.size == 0:
        return 1.0
    return float(np.percentile(foreground, percentile)) or float(data.max()) or 1.0


def show_panel(
    ax: plt.Axes,
    view: str,
    center: Tuple[int, int, int],
    background: Optional[Tuple[np.ndarray, Tuple[float, float, float]]],
    seg: Optional[np.ndarray] = None,
    prediction: Optional[np.ndarray] = None,
    prediction_vmax: float = 1.0,
) -> None:
    """
    Draws one panel: the background T1c in gray, optionally overlaid with a label map
    (segmentation colors) or a growth model prediction (inferno, transparent below
    PREDICTION_THRESHOLD). A missing background is marked as such.
    """
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    if background is None:
        ax.text(0.5, 0.5, "missing", ha="center", va="center", transform=ax.transAxes)
        return

    data, zooms = background
    aspect = get_aspect(zooms, view)
    ax.imshow(
        get_slice(data, view, center),
        cmap="gray",
        vmin=0,
        vmax=get_intensity_window(data),
        aspect=aspect,
        interpolation="nearest",
    )
    if seg is not None:
        ax.imshow(
            get_slice(seg, view, center),
            cmap=SEG_CMAP,
            norm=SEG_NORM,
            alpha=0.6,
            aspect=aspect,
            interpolation="nearest",
        )
    if prediction is not None:
        pred_slice = get_slice(prediction, view, center)
        ax.imshow(
            np.ma.masked_less_equal(pred_slice, PREDICTION_THRESHOLD),
            cmap=PREDICTION_CMAP,
            vmin=0,
            vmax=prediction_vmax,
            alpha=0.75,
            aspect=aspect,
            interpolation="nearest",
        )


def plot_patient(
    patient_id: str,
    exam_dirs: Dict[str, Optional[Path]],
    files: Dict[str, Optional[Path]],
    models: List[str],
) -> plt.Figure:
    """
    Plots one page of a patient with 9 rows (axial, sagittal and coronal view of the preop,
    postop and followup exam) sliced through the center of mass of the preop tumor core.
    Columns: preop T1c, raw T1c of the exam, T1c of the exam warped to preop space, tumor
    segmentation of the exam in preop space (on the warped T1c) and the growth model
    predictions of the preop exam (on the preop T1c). For the preop rows the exam columns
    show the preop exam itself.
    """
    volumes = {key: load_volume(f) if f is not None else None for key, f in files.items()}

    preop_t1c = volumes.get("preop_t1c")
    preop_seg = volumes.get("preop_seg")
    if preop_t1c is not None and preop_seg is not None:
        center = compute_center_of_mass(preop_seg[0], preop_t1c[0], classes=CORE_CLASSES)
    elif preop_t1c is not None:
        logger.warning(f"{patient_id}: no preop tumor segmentation, slicing volume center.")
        center = tuple(s // 2 for s in preop_t1c[0].shape)
    else:
        center = (120, 120, 77)  # SRI atlas center, only used to mark the missing panels

    # Prediction display range: [0, 1] for cell concentrations, the data range for models
    # that output something else (e.g. logits), which is then noted in the column title.
    pred_titles = []
    pred_vmax = {}
    for model in models:
        pred = volumes.get(f"pred_{model}")
        if pred is None:
            pred_titles.append(f"{model} prediction\n(preop, missing)")
            pred_vmax[model] = 1.0
            continue
        pred_min, pred_max = float(pred[0].min()), float(pred[0].max())
        if pred_min < 0.0 or pred_max > 1.0:
            pred_titles.append(
                f"{model} prediction\n(preop, raw range [{pred_min:.1f}, {pred_max:.1f}])"
            )
            pred_vmax[model] = max(pred_max, PREDICTION_THRESHOLD)
        else:
            pred_titles.append(f"{model} prediction\n(preop, threshold {PREDICTION_THRESHOLD})")
            pred_vmax[model] = 1.0

    col_titles = [
        "T1c preop",
        "T1c exam\n(raw)",
        "T1c exam\n(registered to preop)",
        "Tumor seg exam\n(registered to preop)",
    ] + pred_titles
    n_rows = len(EXAMS) * len(VIEWS)
    n_cols = len(col_titles)

    # Rows are shorter than wide since the sagittal/coronal slices are 155 voxels high.
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(3.4 * n_cols, 2.7 * n_rows))
    for exam_ind, exam in enumerate(EXAMS):
        if exam == "preop":
            exam_t1c_raw, exam_t1c_warped, exam_seg = preop_t1c, preop_t1c, preop_seg
        else:
            exam_t1c_raw = volumes.get(f"{exam}_t1c_raw")
            exam_t1c_warped = volumes.get(f"{exam}_t1c_warped")
            exam_seg = volumes.get(f"{exam}_seg_warped")
        exam_dir = exam_dirs[exam]
        exam_name = exam_dir.name if exam_dir is not None else f"{exam} (missing)"

        for view_ind, view in enumerate(VIEWS):
            row = exam_ind * len(VIEWS) + view_ind
            show_panel(axs[row, 0], view, center, preop_t1c)
            show_panel(axs[row, 1], view, center, exam_t1c_raw)
            show_panel(axs[row, 2], view, center, exam_t1c_warped)
            show_panel(
                axs[row, 3],
                view,
                center,
                exam_t1c_warped,
                seg=exam_seg[0] if exam_seg is not None else None,
            )
            for model_ind, model in enumerate(models):
                pred = volumes.get(f"pred_{model}")
                show_panel(
                    axs[row, 4 + model_ind],
                    view,
                    center,
                    preop_t1c,
                    prediction=pred[0] if pred is not None else None,
                    prediction_vmax=pred_vmax[model],
                )
            axs[row, 0].set_ylabel(
                f"{exam_name}\n{view}", fontsize=13, fontweight="bold", labelpad=12
            )

    for col, title in enumerate(col_titles):
        axs[0, col].set_title(title, fontsize=12, fontweight="bold", pad=12)

    fig.suptitle(
        f"{patient_id}   |   CoM of preop tumor core (labels {CORE_CLASSES}) at voxel "
        f"(x, y, z) = {tuple(center)}",
        fontsize=18,
        fontweight="bold",
        y=0.995,
    )
    fig.legend(handles=SEG_PATCHES, loc="lower center", ncol=len(SEG_PATCHES), fontsize=11)
    fig.tight_layout(rect=[0, 0.015, 1, 0.98], h_pad=0.6)
    return fig


if __name__ == "__main__":
    # Plots one pdf page per patient of the respond10 dataset: 9 rows (axial, sagittal,
    # coronal view of the preop, postop and followup exam) through the center of mass of the
    # preop tumor core, with the preop T1c, the raw and the longitudinally registered T1c of
    # the exam, the registered tumor segmentation of the exam and the growth model
    # predictions of the preop exam as columns. Each page is saved as <outdir>/<patient>.pdf
    # and all pages are merged into <outdir>/respond10_visualization.pdf.
    #
    # Example:
    # python scripts/visualize_respond_10.py -patients respond_tum_001 respond_tum_002
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-data_dir",
        type=str,
        default="/mnt/Drive2/lucas/predict_gbm_10_respond",
        help="Directory containing one folder per patient (respond_tum_XXX).",
    )
    parser.add_argument(
        "-outdir",
        type=str,
        default=None,
        help="Directory for the per patient pdfs and the merged pdf. Default: <data_dir>/visualization.",
    )
    parser.add_argument(
        "-models",
        type=str,
        nargs="*",
        default=["gliodil", "unet"],
        help="Growth models whose preop predictions (growth_models/<model>_pred.nii.gz) are shown.",
    )
    parser.add_argument(
        "-patients",
        type=str,
        nargs="+",
        default=None,
        help="Optional list of patient ids to plot (e.g. respond_tum_001). Default: all.",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    outdir = Path(args.outdir) if args.outdir else data_dir / "visualization"
    outdir.mkdir(parents=True, exist_ok=True)
    merged_file = outdir / "respond10_visualization.pdf"

    patient_dirs = sorted(d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith("respond_tum_"))
    if args.patients is not None:
        selected = set(args.patients)
        patient_dirs = [p for p in patient_dirs if p.name in selected]
        missing = selected - {p.name for p in patient_dirs}
        if missing:
            logger.warning(f"Patients not found in {data_dir}: {sorted(missing)}.")
    logger.info(f"Plotting {len(patient_dirs)} patients from {data_dir} to {outdir}.")

    plotted = 0
    with PdfPages(str(merged_file)) as merged_pdf:
        for patient_dir in patient_dirs:
            patient_id = patient_dir.name
            try:
                exam_dirs = find_exam_dirs(patient_dir)
                files = resolve_patient_files(patient_dir, exam_dirs, args.models)
                fig = plot_patient(patient_id, exam_dirs, files, args.models)
                patient_file = outdir / f"{patient_id}.pdf"
                fig.savefig(str(patient_file), format="pdf")
                merged_pdf.savefig(fig)
                plt.close(fig)
                plotted += 1
                logger.info(f"{patient_id}: saved {patient_file}.")
            except Exception:
                logger.exception(f"{patient_id}: plotting failed, skipping patient.")

    logger.info(f"Finished plotting {plotted} patients. Merged pdf saved to {merged_file}.")
