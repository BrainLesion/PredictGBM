import argparse
import importlib.util
import os
import shutil
import subprocess
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
    LONGITUDINAL_DIR,
    MODEL_OUTPUT_DIR,
    RECURRENCE_SCHEMA,
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
# Tumor cell concentrations at or below this value are not drawn (0.0: every positive voxel).
PREDICTION_THRESHOLD = 0.0
# Patients excluded by -ten, which combines the first ten remaining patients (by id).
TEN_EXCLUDED_PATIENTS = ("respond_tum_081",)
TEN_NUM_PATIENTS = 10
# -old_dirac reruns the longitudinal registration with the instance optimization of the main
# branch, i.e. before the fix of commit 437b5e0 on the sailor branch, as a visual sanity
# check of the fix. The old module is extracted from OLD_DIRAC_COMMIT with git into
# <data_dir>/OLD_DIRAC_FOLDER (kept there as provenance) and its optimize_warp_field is
# swapped into register_recurrence; the DIRAC network inference is identical on both
# commits. Outputs go to <data_dir>/OLD_DIRAC_FOLDER/<patient>/<exam>/PREOP_SPACE_FOLDER,
# the existing patient folders are only read.
OLD_DIRAC_COMMIT = "d5a67de5b718e8c35c8e66821d888d0c121167af"
OLD_DIRAC_FILE = "predict_gbm/preprocessing/dirac.py"
OLD_DIRAC_FOLDER = "old_dirac"
OLD_DIRAC_EXAMS = ("postop", "followup")
UNNORMALIZED_FOLDER = "unnormalized"
REPO_ROOT = Path(__file__).resolve().parents[1]

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
# What a growth model predicts, used in its column title (models not listed: "prediction").
MODEL_OUTPUT_NAMES = {"gliodil": "tumor conc.", "unet": "risk map"}
# Model name in the column title (models not listed: their id).
MODEL_DISPLAY_NAMES = {"gliodil": "GliODIL", "unet": "U-Net"}
# Models whose prediction is only nonzero on their solution box (gliodil); zeros are
# replaced by the smallest positive value before plotting so the overlay covers the image.
ZERO_FILL_MODELS = ("gliodil",)


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
    Columns: preop T1c, T1c of the exam in SRI space (rigid), T1c of the exam warped to
    preop space (deformable), tumor segmentation of the exam in preop space (on the warped
    T1c) and the growth model predictions of the preop exam (on the preop T1c, only in
    the preop rows). For the preop rows the exam columns show the preop exam itself.
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
    # that output something else (e.g. logits).
    pred_titles = []
    pred_vmax = {}
    for model in models:
        title = f"{MODEL_DISPLAY_NAMES.get(model, model)} {MODEL_OUTPUT_NAMES.get(model, 'prediction')}"
        pred = volumes.get(f"pred_{model}")
        if pred is None:
            pred_titles.append(f"{title}\n(preop, missing)")
            pred_vmax[model] = 1.0
            continue
        pred_titles.append(f"{title}\n(preop)")
        if model in ZERO_FILL_MODELS and np.any(pred[0] > 0):
            pred[0][pred[0] <= 0] = pred[0][pred[0] > 0].min()
        pred_min, pred_max = float(pred[0].min()), float(pred[0].max())
        if pred_min < 0.0 or pred_max > 1.0:
            pred_vmax[model] = max(pred_max, PREDICTION_THRESHOLD)
        else:
            pred_vmax[model] = 1.0

    # Column titles are repeated above the first row of every exam, naming that exam. The
    # preop rows show the preop exam itself, which is not deformably registered, and the
    # prediction columns are only titled (and drawn) for the preop exam.
    def col_titles(exam: str) -> List[str]:
        space = "SRI space (rigid)" if exam == "preop" else "Pre-op space (deformable)"
        return [
            "T1c preop",
            f"T1c {exam}\nSRI space (rigid)",
            f"T1c {exam}\n{space}",
            f"Tumor seg {exam}\n{space}",
        ] + (pred_titles if exam == "preop" else [])

    n_cols = len(col_titles("preop"))
    first_pred_col = n_fixed_cols = n_cols - len(pred_titles)
    n_rows = len(EXAMS) * len(VIEWS)

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
            show_panel(axs[row, n_fixed_cols - 2], view, center, exam_t1c_warped)
            show_panel(
                axs[row, n_fixed_cols - 1],
                view,
                center,
                exam_t1c_warped,
                seg=exam_seg[0] if exam_seg is not None else None,
            )
            # The predictions belong to the preop exam; their panels stay empty otherwise.
            for model_ind, model in enumerate(models):
                pred_ax = axs[row, first_pred_col + model_ind]
                if exam != "preop":
                    pred_ax.set_axis_off()
                    continue
                pred = volumes.get(f"pred_{model}")
                show_panel(
                    pred_ax,
                    view,
                    center,
                    preop_t1c,
                    prediction=pred[0] if pred is not None else None,
                    prediction_vmax=pred_vmax[model],
                )
            axs[row, 0].set_ylabel(
                f"{exam_name}\n{view}", fontsize=13, fontweight="bold", labelpad=12
            )
        for col, title in enumerate(col_titles(exam)):
            axs[exam_ind * len(VIEWS), col].set_title(
                title, fontsize=12, fontweight="bold", pad=12
            )

    fig.suptitle(
        f"{patient_id}   |   CoM of preop tumor core",
        fontsize=18,
        fontweight="bold",
        y=0.995,
    )
    fig.legend(handles=SEG_PATCHES, loc="lower center", ncol=len(SEG_PATCHES), fontsize=11)
    fig.tight_layout(rect=[0, 0.015, 1, 0.98], h_pad=1.5)
    return fig


def load_old_dirac_module(old_dirac_dir: Path):
    """
    Extracts OLD_DIRAC_FILE at OLD_DIRAC_COMMIT from the git history into old_dirac_dir and
    imports it as a standalone module (the file has no imports from predict_gbm).
    """
    old_dirac_dir.mkdir(parents=True, exist_ok=True)
    module_file = old_dirac_dir / f"dirac_{OLD_DIRAC_COMMIT[:7]}.py"
    source = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "show", f"{OLD_DIRAC_COMMIT}:{OLD_DIRAC_FILE}"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    module_file.write_text(source)
    spec = importlib.util.spec_from_file_location("dirac_old", module_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    logger.info(f"Loaded DIRAC of commit {OLD_DIRAC_COMMIT[:7]} from {module_file}.")
    return module


def register_old_dirac(
    patient_id: str, exam_dirs: Dict[str, Optional[Path]], old_dirac_dir: Path
) -> None:
    """
    Registers the postop and followup exam of a patient to preop with register_recurrence
    (whose optimize_warp_field has to be replaced by the old one beforehand, see
    load_old_dirac_module) and warps the T1c and tumor segmentation into preop space, as in
    scripts/preprocess_respond10.py but without additional modalities. Inputs are read from
    the exam directories, outputs are written to old_dirac_dir/<patient>/<exam>/preop_space.
    An exam whose warped T1c already exists there is skipped; a failing registration is
    logged and leaves its panels empty.
    """
    from predict_gbm.preprocessing import register_recurrence

    preop_dir = exam_dirs["preop"]
    if preop_dir is None:
        logger.error(f"{patient_id}: no preop exam, skipping registration.")
        return
    t1c_pre_file = preop_dir / SKULL_STRIP_FOLDER / "t1c_bet_normalized.nii.gz"

    for exam in OLD_DIRAC_EXAMS:
        exam_dir = exam_dirs[exam]
        if exam_dir is None:
            continue
        outdir = old_dirac_dir / patient_id / exam_dir.name
        preop_space_dir = outdir / PREOP_SPACE_FOLDER
        if (preop_space_dir / "t1c_warped_longitudinal.nii.gz").is_file():
            logger.info(f"{patient_id}/{exam_dir.name}: old DIRAC output exists, skipping.")
            continue

        logger.info(f"{patient_id}/{exam_dir.name}: starting old DIRAC registration to preop.")
        try:
            register_recurrence(
                t1c_pre_file=t1c_pre_file,
                t1c_post_file=exam_dir / SKULL_STRIP_FOLDER / "t1c_bet_normalized.nii.gz",
                recurrence_seg_file=exam_dir / TUMOR_SEGMENTATION_FOLDER / "tumor_seg.nii.gz",
                outdir=outdir,
                registration_algorithm="dirac",
            )
        except Exception:
            logger.exception(f"{patient_id}/{exam_dir.name}: old DIRAC registration failed.")
            continue

        longitudinal_dir = RECURRENCE_SCHEMA.format(base_dir=outdir).parent
        assert longitudinal_dir.name == LONGITUDINAL_DIR
        preop_space_dir.mkdir(parents=True, exist_ok=True)
        for src in sorted(longitudinal_dir.iterdir()):
            dst = preop_space_dir / src.name
            if dst.is_dir():
                shutil.rmtree(dst)
            elif dst.exists():
                dst.unlink()
            shutil.move(str(src), str(dst))
        longitudinal_dir.rmdir()
        logger.info(f"{patient_id}/{exam_dir.name}: saved old DIRAC outputs to {preop_space_dir}.")


def resolve_old_dirac_files(
    patient_dir: Path, exam_dirs: Dict[str, Optional[Path]], old_dirac_dir: Path
) -> Dict[str, Optional[Path]]:
    """
    Resolves the images of the old vs. new DIRAC comparison: the preop T1c and tumor
    segmentation, and per postop/followup exam the T1c warped to preop space by the old
    DIRAC (old_dirac_dir) and by the new one (exam directory) as well as the unnormalized T1c
    in SRI space. Missing files are mapped to None and logged.
    """
    files = {}
    preop_dir = exam_dirs["preop"]
    if preop_dir is not None:
        files["preop_t1c"] = preop_dir / SKULL_STRIP_FOLDER / "t1c_bet_normalized.nii.gz"
        files["preop_seg"] = preop_dir / TUMOR_SEGMENTATION_FOLDER / "tumor_seg.nii.gz"
    for exam in OLD_DIRAC_EXAMS:
        exam_dir = exam_dirs[exam]
        if exam_dir is None:
            continue
        files[f"{exam}_t1c_old"] = (
            old_dirac_dir
            / patient_dir.name
            / exam_dir.name
            / PREOP_SPACE_FOLDER
            / "t1c_warped_longitudinal.nii.gz"
        )
        files[f"{exam}_t1c_new"] = (
            exam_dir / PREOP_SPACE_FOLDER / "t1c_warped_longitudinal.nii.gz"
        )
        files[f"{exam}_t1c_unnormalized"] = exam_dir / UNNORMALIZED_FOLDER / "t1c.nii.gz"

    for key, f in files.items():
        if not f.is_file():
            logger.warning(f"{patient_dir.name}: {key} not found ({f}), panel left empty.")
            files[key] = None
    return files


def plot_old_dirac_patient(
    patient_id: str,
    exam_dirs: Dict[str, Optional[Path]],
    files: Dict[str, Optional[Path]],
) -> plt.Figure:
    """
    Plots one page of a patient with 6 rows (axial, sagittal and coronal view of the postop
    and followup exam) sliced through the center of mass of the preop tumor core. Columns:
    stripped preop T1c, T1c of the exam warped to preop space by the old and by the new
    DIRAC, and the unnormalized (not skull stripped) T1c of the exam in SRI space.
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

    # Column titles are repeated above the first row of every exam, naming that exam.
    def col_titles(exam: str) -> List[str]:
        return [
            "T1c preop\n(stripped)",
            f"T1c {exam}\nPre-op space (old DIRAC)",
            f"T1c {exam}\nPre-op space (new DIRAC)",
            f"T1c {exam}\nSRI space (unnormalized)",
        ]

    n_rows = len(OLD_DIRAC_EXAMS) * len(VIEWS)
    n_cols = len(col_titles(OLD_DIRAC_EXAMS[0]))

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(3.4 * n_cols, 2.7 * n_rows))
    for exam_ind, exam in enumerate(OLD_DIRAC_EXAMS):
        exam_dir = exam_dirs[exam]
        exam_name = exam_dir.name if exam_dir is not None else f"{exam} (missing)"
        columns = [
            preop_t1c,
            volumes.get(f"{exam}_t1c_old"),
            volumes.get(f"{exam}_t1c_new"),
            volumes.get(f"{exam}_t1c_unnormalized"),
        ]
        for view_ind, view in enumerate(VIEWS):
            row = exam_ind * len(VIEWS) + view_ind
            for col, volume in enumerate(columns):
                show_panel(axs[row, col], view, center, volume)
            axs[row, 0].set_ylabel(
                f"{exam_name}\n{view}", fontsize=13, fontweight="bold", labelpad=12
            )
        for col, title in enumerate(col_titles(exam)):
            axs[exam_ind * len(VIEWS), col].set_title(
                title, fontsize=12, fontweight="bold", pad=12
            )

    fig.suptitle(
        f"{patient_id}   |   old (commit {OLD_DIRAC_COMMIT[:7]}) vs. new DIRAC\n"
        "CoM of preop tumor core",
        fontsize=16,
        fontweight="bold",
        y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.965], h_pad=1.5)
    return fig


if __name__ == "__main__":
    # Plots one pdf page per patient of the respond10 dataset: 9 rows (axial, sagittal,
    # coronal view of the preop, postop and followup exam) through the center of mass of the
    # preop tumor core, with the preop T1c, the raw and the longitudinally registered T1c of
    # the exam, the registered tumor segmentation of the exam and the growth model
    # predictions of the preop exam as columns. Each page is saved as <outdir>/<patient>.pdf
    # and all pages are merged into <outdir>/respond10_visualization.pdf. With -ten only the
    # first ten patients by id (excluding TEN_EXCLUDED_PATIENTS) are plotted, into
    # <outdir>/<patient>_ten.pdf and <outdir>/respond10_visualization_ten.pdf.
    # With -old_dirac the postop and followup exams are instead registered to preop with the
    # old DIRAC instance optimization (see OLD_DIRAC_COMMIT) into <data_dir>/old_dirac, and
    # one merged pdf <outdir>/respond10_visualization_old_dirac.pdf with 6 rows per patient
    # (axial, sagittal, coronal view of the postop and followup exam) compares the old and
    # new registration next to the stripped preop T1c and the unnormalized T1c of the exam.
    #
    # Examples:
    # python scripts/visualize_respond_10.py -patients respond_tum_001 respond_tum_002
    # python scripts/visualize_respond_10.py -ten
    # nohup python -u scripts/visualize_respond_10.py -old_dirac -cuda_device 0 > old_dirac.out 2>&1 &
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
    parser.add_argument(
        "-ten",
        action="store_true",
        help=(
            f"Only plot the first {TEN_NUM_PATIENTS} patients ordered by id, excluding "
            f"{list(TEN_EXCLUDED_PATIENTS)}. Output files get a '_ten' suffix. Combined with "
            "-patients the selection is applied to the given patients."
        ),
    )
    parser.add_argument(
        "-old_dirac",
        action="store_true",
        help=(
            "Register the postop and followup exams to preop with the old DIRAC instance "
            f"optimization (commit {OLD_DIRAC_COMMIT[:7]}) into <data_dir>/{OLD_DIRAC_FOLDER} "
            "(exams with existing output are skipped) and plot the old vs. new registration "
            f"into <outdir>/respond10_visualization_old_dirac.pdf, where <outdir> defaults to "
            f"<data_dir>/{OLD_DIRAC_FOLDER}/visualization. Nothing else is plotted."
        ),
    )
    parser.add_argument(
        "-cuda_device", type=str, default="0", help="GPU id for the -old_dirac registration."
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    old_dirac_dir = data_dir / OLD_DIRAC_FOLDER
    if args.outdir:
        outdir = Path(args.outdir)
    elif args.old_dirac:
        outdir = old_dirac_dir / "visualization"
    else:
        outdir = data_dir / "visualization"
    outdir.mkdir(parents=True, exist_ok=True)
    suffix = "_old_dirac" if args.old_dirac else "_ten" if args.ten else ""
    merged_file = outdir / f"respond10_visualization{suffix}.pdf"

    patient_dirs = sorted(d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith("respond_tum_"))
    if args.patients is not None:
        selected = set(args.patients)
        patient_dirs = [p for p in patient_dirs if p.name in selected]
        missing = selected - {p.name for p in patient_dirs}
        if missing:
            logger.warning(f"Patients not found in {data_dir}: {sorted(missing)}.")
    if args.ten:
        patient_dirs = [
            p for p in patient_dirs if p.name not in TEN_EXCLUDED_PATIENTS
        ][:TEN_NUM_PATIENTS]
    logger.info(f"Plotting {len(patient_dirs)} patients from {data_dir} to {outdir}.")

    if args.old_dirac:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device
        from predict_gbm.preprocessing import norm_ss_coregistration

        # register_recurrence looks optimize_warp_field up in its module, so replacing the
        # attribute makes it run the old instance optimization (inference is unchanged).
        old_dirac = load_old_dirac_module(old_dirac_dir)
        norm_ss_coregistration.optimize_warp_field = old_dirac.optimize_warp_field

        plotted = 0
        with PdfPages(str(merged_file)) as merged_pdf:
            for patient_dir in patient_dirs:
                patient_id = patient_dir.name
                try:
                    exam_dirs = find_exam_dirs(patient_dir)
                    register_old_dirac(patient_id, exam_dirs, old_dirac_dir)
                    files = resolve_old_dirac_files(patient_dir, exam_dirs, old_dirac_dir)
                    fig = plot_old_dirac_patient(patient_id, exam_dirs, files)
                    merged_pdf.savefig(fig)
                    plt.close(fig)
                    plotted += 1
                    logger.info(f"{patient_id}: plotted old vs. new DIRAC.")
                except Exception:
                    logger.exception(f"{patient_id}: old DIRAC step failed, skipping patient.")
        logger.info(f"Finished plotting {plotted} patients. Merged pdf saved to {merged_file}.")
        raise SystemExit(0)

    plotted = 0
    with PdfPages(str(merged_file)) as merged_pdf:
        for patient_dir in patient_dirs:
            patient_id = patient_dir.name
            try:
                exam_dirs = find_exam_dirs(patient_dir)
                files = resolve_patient_files(patient_dir, exam_dirs, args.models)
                fig = plot_patient(patient_id, exam_dirs, files, args.models)
                patient_file = outdir / f"{patient_id}{suffix}.pdf"
                fig.savefig(str(patient_file), format="pdf")
                merged_pdf.savefig(fig)
                plt.close(fig)
                plotted += 1
                logger.info(f"{patient_id}: saved {patient_file}.")
            except Exception:
                logger.exception(f"{patient_id}: plotting failed, skipping patient.")

    logger.info(f"Finished plotting {plotted} patients. Merged pdf saved to {merged_file}.")
