"""Post-op tumor segmentations of SAILOR: BraTS25_1 (pipeline) vs. BraTS24_1 side by side.

Writes one page per post-op session with four rows (top to bottom): the pre-op T1c, the
post-op T1c, the post-op T1c overlaid with the pipeline segmentation (BraTS25_1,
tumor_seg.nii.gz) and overlaid with the BraTS 2024 post-treatment segmentation (BraTS24_1,
tumor_seg_24_1.nii.gz, see the -brats24_postop flag of scripts/process_sailor.py). All
images are the skull-stripped T1c in atlas space, so the pre-op exam is shown at the same
voxel slices. The columns are the axial, sagittal and coronal slices through the center of
mass of the resection cavity, enhancing tumor and necrosis labels of the BraTS24_1
segmentation. Style, labels and colors follow scripts/visualize_respond_10.py.

The pre-op and post-op sessions of a patient are taken from the session labels table (a
patient can have several post-op sessions, each gets a page). Sessions with another label
that have a BraTS24_1 segmentation (e.g. a follow-up re-segmented by hand because it looks
like a re-resection, such as sub-21 ses-11) get a page too, in session order. Patients
without a post-op session or without both segmentations are skipped with a warning; a
missing pre-op T1c leaves its row marked as missing. All pages are combined into one PDF.

Usage:
    python scripts/visualize_sailor_24_25.py
    python scripts/visualize_sailor_24_25.py -patients sub-15 sub-18 -outfile /tmp/seg_24_25.pdf
"""

import argparse
import csv
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
    MODALITY_STRIPPED_SCHEMA,
    OUTPUT_BASE_SCHEMA,
    TUMOR_SEGMENTATION_FOLDER,
    TUMORSEG_SCHEMA,
)
from predict_gbm.utils.utils import compute_center_of_mass

# Output of the BraTS24_1 post-op segmentation of scripts/process_sailor.py (-brats24_postop).
BRATS24_TUMORSEG_SCHEMA = (
    OUTPUT_BASE_SCHEMA / TUMOR_SEGMENTATION_FOLDER / "tumor_seg_24_1.nii.gz"
)
# Rows in page order: (row title, segmentation schema).
SEGMENTATIONS = (("BraTS25_1", TUMORSEG_SCHEMA), ("BraTS24_1", BRATS24_TUMORSEG_SCHEMA))
# The volumes follow the (x, y, z) = (sagittal, coronal, axial) convention, so each view
# is defined by the axis it slices along.
VIEWS = ("Axial", "Sagittal", "Coronal")
# Labels of the BraTS24_1 segmentation through whose center of mass all slices go:
# 1 necrosis / non-enhancing, 3 enhancing tumor, 4 resection cavity.
CENTER_CLASSES = [1, 3, 4]
# Session labels (column "label" of the session labels table) of the pre-op and post-op exams.
PREOP_LABEL = "preop"
POSTOP_LABEL = "postop"
# Titles of the T1c rows without overlay, in page order above the segmentation rows.
T1C_ROWS = ("T1c pre-op", "T1c post-op")

# Segmentation legend: 1 necrosis/non-enhancing, 2 edema, 3 enhancing, 4 resection cavity
# (as in scripts/visualize_respond_10.py).
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
# Short label names of the voxel counts in the row titles, in label order 1-4.
SEG_SHORT_LABELS = ["NEC", "ED", "ET", "RC"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="BraTS25_1 vs. BraTS24_1 post-op segmentations of SAILOR as one PDF."
    )
    parser.add_argument(
        "-data_dir",
        type=Path,
        default=Path("/mnt/Drive4/lucas/SAILOR/processed"),
        help="Processed SAILOR directory (<data_dir>/<patient>/<session>).",
    )
    parser.add_argument(
        "-session_labels_tsv",
        type=Path,
        default=Path("/mnt/Drive4/lucas/SAILOR/session_labels.tsv"),
        help="Session labels table, defines the post-op session(s) of each patient.",
    )
    parser.add_argument(
        "-outfile",
        type=Path,
        default=Path("/mnt/Drive4/lucas/SAILOR/postop_seg_24_25.pdf"),
        help="Combined output PDF, one page per post-op session.",
    )
    parser.add_argument(
        "-patients",
        type=str,
        nargs="+",
        default=None,
        help="Patients to visualize (default: all patient directories in data_dir).",
    )
    return parser.parse_args()


def read_session_labels(
    session_labels_file: Path,
) -> Dict[str, Dict[str, List[Tuple[str, str]]]]:
    """Reads the session labels as {patient: {label: [(session, day offset), in table order]}}."""
    sessions: Dict[str, Dict[str, List[Tuple[str, str]]]] = {}
    with open(session_labels_file, newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            labels = sessions.setdefault(row["subject"], {})
            labels.setdefault(row["label"], []).append((row["session"], row["day_offset"]))
    return sessions


def t1c_file(exam_dir: Path) -> Path:
    return MODALITY_STRIPPED_SCHEMA.format(base_dir=exam_dir, modality="t1c")


def find_sessions(
    patient_dir: Path, labels: Dict[str, List[Tuple[str, str]]]
) -> List[Tuple[str, str, str]]:
    """
    Sessions of a patient to plot as (session, label, day offset): the post-op sessions of
    the labels table plus any other labelled session that has a BraTS24_1 segmentation, in
    session order.
    """
    sessions = []
    for label, label_sessions in labels.items():
        for session, day_offset in label_sessions:
            has_brats24 = BRATS24_TUMORSEG_SCHEMA.format(base_dir=patient_dir / session).is_file()
            if label == POSTOP_LABEL or has_brats24:
                sessions.append((session, label, day_offset))
    return sorted(sessions)


def resolve_files(
    patient_dir: Path, session: str, labels: Dict[str, List[Tuple[str, str]]]
) -> Optional[Dict[str, Optional[Path]]]:
    """
    T1c and both segmentations of a session, None (and a warning) if one is missing, plus
    the pre-op T1c of the patient (first pre-op session of the labels), which is None if
    there is no pre-op session or file.
    """
    exam_dir = patient_dir / session
    files: Dict[str, Optional[Path]] = {"t1c": t1c_file(exam_dir)}
    for name, schema in SEGMENTATIONS:
        files[name] = schema.format(base_dir=exam_dir)
    missing = [str(f) for f in files.values() if not f.is_file()]
    if missing:
        logger.warning(f"{exam_dir}: missing {missing}, skipping.")
        return None

    preop_sessions = labels.get(PREOP_LABEL, [])
    preop_t1c = t1c_file(patient_dir / preop_sessions[0][0]) if preop_sessions else None
    if preop_t1c is None or not preop_t1c.is_file():
        logger.warning(f"{exam_dir}: no pre-op T1c ({preop_t1c}), its row is left empty.")
        preop_t1c = None
    files["preop_t1c"] = preop_t1c
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
) -> None:
    """
    Draws one panel: the T1c in gray, optionally overlaid with a label map in the
    segmentation colors. A missing background is marked as such.
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


def label_counts(seg: np.ndarray) -> str:
    """Voxel counts of the labels 1-4 as one line, e.g. 'NEC 0 | ED 66657 | ET 2440 | RC 6318'."""
    counts = [int(np.sum(seg == label)) for label in range(1, len(SEG_SHORT_LABELS) + 1)]
    return " | ".join(f"{name} {count}" for name, count in zip(SEG_SHORT_LABELS, counts))


def plot_session(
    patient_id: str,
    session: str,
    label: str,
    day_offset: str,
    files: Dict[str, Optional[Path]],
) -> plt.Figure:
    """
    One page of a post-op session: rows are the pre-op and post-op T1c (T1C_ROWS) and the
    post-op T1c with the segmentations (SEGMENTATIONS order), columns the views through the
    center of mass of CENTER_CLASSES of the BraTS24_1 segmentation.
    """
    t1c = load_volume(files["t1c"])
    preop_t1c = load_volume(files["preop_t1c"]) if files["preop_t1c"] is not None else None
    if preop_t1c is not None and preop_t1c[0].shape != t1c[0].shape:
        raise ValueError(
            f"pre-op T1c shape {preop_t1c[0].shape} differs from post-op shape {t1c[0].shape}."
        )
    segs = {name: load_volume(files[name])[0] for name, _ in SEGMENTATIONS}
    for name, seg in segs.items():
        if seg.shape != t1c[0].shape:
            raise ValueError(f"{name} shape {seg.shape} differs from T1c shape {t1c[0].shape}.")
    center = compute_center_of_mass(segs["BraTS24_1"], t1c[0], classes=CENTER_CLASSES)

    n_rows, n_cols = len(T1C_ROWS) + len(SEGMENTATIONS), len(VIEWS)
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(3.4 * n_cols, 2.9 * n_rows))
    for row, (title, background) in enumerate(zip(T1C_ROWS, (preop_t1c, t1c))):
        for col, view in enumerate(VIEWS):
            show_panel(axs[row, col], view, center, background)
        axs[row, 0].set_ylabel(title, fontsize=13, fontweight="bold", labelpad=12)
    for seg_ind, (name, _) in enumerate(SEGMENTATIONS):
        row = len(T1C_ROWS) + seg_ind
        for col, view in enumerate(VIEWS):
            show_panel(axs[row, col], view, center, t1c, segs[name])
        axs[row, 0].set_ylabel(name, fontsize=13, fontweight="bold", labelpad=12)
        # Voxel counts of the labels under the first panel of the row (ticks are removed).
        axs[row, 0].set_xlabel(label_counts(segs[name]), fontsize=8, labelpad=4)
    for col, view in enumerate(VIEWS):
        axs[0, col].set_title(view, fontsize=12, fontweight="bold", pad=12)

    fig.suptitle(
        f"{patient_id} {session}   |   {label} day {day_offset}   |   "
        f"CoM of BraTS24_1 cavity + enhancing + necrosis",
        fontsize=14,
        fontweight="bold",
        y=0.995,
    )
    fig.legend(handles=SEG_PATCHES, loc="lower center", ncol=len(SEG_PATCHES), fontsize=10)
    fig.tight_layout(rect=[0, 0.03, 1, 0.97], h_pad=1.5)
    return fig


def main() -> None:
    args = parse_args()
    session_labels = read_session_labels(args.session_labels_tsv)
    patient_ids = args.patients or sorted(d.name for d in args.data_dir.iterdir() if d.is_dir())
    logger.info(f"Visualizing {len(patient_ids)} patients of {args.data_dir}.")

    args.outfile.parent.mkdir(parents=True, exist_ok=True)
    plotted = 0
    with PdfPages(str(args.outfile)) as merged_pdf:
        for patient_id in patient_ids:
            labels = session_labels.get(patient_id, {})
            sessions = find_sessions(args.data_dir / patient_id, labels)
            if not sessions:
                logger.warning(f"{patient_id}: no {POSTOP_LABEL} session in the labels, skipping.")
                continue
            for session, label, day_offset in sessions:
                files = resolve_files(args.data_dir / patient_id, session, labels)
                if files is None:
                    continue
                try:
                    fig = plot_session(patient_id, session, label, day_offset, files)
                    merged_pdf.savefig(fig)
                    plt.close(fig)
                    plotted += 1
                    logger.info(f"{patient_id} {session}: plotted.")
                except Exception:
                    logger.exception(f"{patient_id} {session}: plotting failed, skipping.")
    logger.info(f"Saved {plotted} pages to {args.outfile}.")


if __name__ == "__main__":
    main()
