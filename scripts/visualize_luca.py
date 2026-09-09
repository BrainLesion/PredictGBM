import argparse
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from loguru import logger
from matplotlib.backends.backend_pdf import PdfPages

from predict_gbm.utils.utils import compute_center_of_mass

# Row order of the figure. The volumes follow the (x, y, z) = (sagittal, coronal, axial)
# convention, so each view is defined by the axis it slices along.
VIEWS = ("Axial", "Sagittal", "Coronal")

COLUMN_TITLES = (
    "Dose (preop space)",
    "Preop t1c (SRI)",
    "Postop t1c (preop space)",
    "Postop t1c (native)",
)

DOSE_CMAP = "turbo"


def find_single(directory: Path, pattern: str, exclude: str = None) -> Optional[Path]:
    """
    Returns the single file of a directory matching the pattern, or None if there is no or
    more than one match. Hidden files are skipped, since macOS AppleDouble copies ("._*") sit
    next to the images.
    """
    matches = sorted(
        f
        for f in directory.glob(pattern)
        if not f.name.startswith(".") and (exclude is None or exclude not in f.name)
    )
    if len(matches) != 1:
        return None
    return matches[0]


def resolve_patient_files(patient_dir: Path, processed_dir: Path) -> Optional[Dict[str, Path]]:
    """
    Resolves the four images of a patient: the dose map warped to pre-op space and the post-op
    t1c warped to pre-op space (both in the processed folder), the skull-stripped pre-op t1c in
    SRI space and the post-op t1c in native space (both in the input folder). Returns None if
    any of them is missing or ambiguous.
    """
    candidates = {
        "dose_preop": (processed_dir, "*_space-preop_rtdose.nii.gz", None),
        "preop_t1c": (patient_dir, "*_space-sri_t1c.nii.gz", None),
        "postop_t1c_preop": (processed_dir, "*_space-preop_t1c.nii.gz", None),
        "postop_t1c_native": (patient_dir, "*_t1c.nii.gz", "space-sri"),
    }

    files = {}
    for key, (directory, pattern, exclude) in candidates.items():
        match = find_single(directory, pattern, exclude) if directory.is_dir() else None
        if match is None:
            logger.warning(
                f"{patient_dir.name}: no unique {key} file ({directory}/{pattern}), "
                f"skipping patient."
            )
            return None
        files[key] = match
    return files


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


def load_volume(nifti_file: Path) -> Tuple[np.ndarray, Tuple[float, float, float]]:
    """Loads the voxel data and the voxel spacing of a nifti file."""
    img = nib.load(str(nifti_file))
    return np.asanyarray(img.dataobj, dtype=np.float32), tuple(
        float(z) for z in img.header.get_zooms()[:3]
    )


def get_intensity_window(data: np.ndarray, percentile: float = 99.5) -> float:
    """
    Upper display intensity of an MRI volume. Some of the scans have extreme outlier
    intensities (up to 1e6), which would render the brain almost black on a min-max scale, so
    the window is clipped at a percentile of the non-background voxels.
    """
    foreground = data[data > 0]
    if foreground.size == 0:
        return 1.0
    return float(np.percentile(foreground, percentile)) or float(data.max()) or 1.0


def plot_patient(patient_id: str, files: Dict[str, Path], pdf: PdfPages) -> None:
    """
    Plots one page with the three views (rows) of the four images (columns) of a patient. The
    first three columns share the SRI grid and are sliced through the center of mass of the
    dose map, so the irradiated region is in view; the native post-op t1c is on its own grid
    and is sliced through the volume center.
    """
    volumes = {key: load_volume(f) for key, f in files.items()}

    dose_data = volumes["dose_preop"][0]
    # compute_center_of_mass expects a segmentation, so the irradiated volume is passed as a
    # single-class mask. It falls back to the volume center if the dose map is empty.
    dose_center = compute_center_of_mass(
        seg_data=(dose_data > 0).astype(np.uint8), mri_data=dose_data, classes=[1]
    )
    native_center = tuple(s // 2 for s in volumes["postop_t1c_native"][0].shape)

    columns = [
        ("dose_preop", dose_center),
        ("preop_t1c", dose_center),
        ("postop_t1c_preop", dose_center),
        ("postop_t1c_native", native_center),
    ]

    fig, axs = plt.subplots(len(VIEWS), len(columns), figsize=(4 * len(columns), 4 * len(VIEWS)))
    for col, (key, center) in enumerate(columns):
        data, zooms = volumes[key]
        is_dose = key == "dose_preop"
        vmax = float(data.max()) if is_dose else get_intensity_window(data)
        for row, view in enumerate(VIEWS):
            ax = axs[row, col]
            image = ax.imshow(
                get_slice(data, view, center),
                cmap=DOSE_CMAP if is_dose else "gray",
                vmin=0,
                vmax=max(vmax, 1e-6),
                aspect=get_aspect(zooms, view),
                interpolation="nearest",
            )
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if is_dose:
                colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
                colorbar.ax.tick_params(labelsize=8)
            if row == 0:
                ax.set_title(COLUMN_TITLES[col], fontsize=13, fontweight="bold", pad=12)
            if col == 0:
                ax.set_ylabel(view, fontsize=13, fontweight="bold", labelpad=12)

    sessions = ", ".join(
        f"{COLUMN_TITLES[i]}: {files[key].name}" for i, (key, _) in enumerate(columns)
    )
    fig.suptitle(patient_id, fontsize=18, fontweight="bold", y=0.99)
    fig.text(0.01, 0.005, sessions, fontsize=6, color="dimgray", ha="left")
    fig.tight_layout(rect=[0, 0.02, 1, 0.97])
    pdf.savefig(fig)
    plt.close(fig)


if __name__ == "__main__":
    # Plots one page per patient of /mnt/Drive4/lucas/register_copy into a single pdf in the
    # processed folder. Rows are the axial, sagittal and coronal view; columns are the dose map
    # in pre-op space, the pre-op t1c in SRI space, the post-op t1c registered to pre-op space
    # and the post-op t1c in native space. Patients whose images are missing are skipped.
    #
    # Example:
    # python scripts/visualize_luca.py
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-datadir",
        type=str,
        default="/mnt/Drive4/lucas/register_copy",
        help="Directory containing one folder per patient with the input images.",
    )
    parser.add_argument(
        "-processed_dir",
        type=str,
        default="/mnt/Drive4/lucas/register_copy/processed",
        help="Directory containing one folder per patient with the processed images.",
    )
    parser.add_argument(
        "-outfile",
        type=str,
        default=None,
        help="Pdf to save the plots to. Defaults to <processed_dir>/visualizations.pdf.",
    )
    parser.add_argument(
        "-patients",
        type=str,
        nargs="*",
        default=None,
        help="Optional subset of patient directory names to plot. Defaults to all.",
    )
    args = parser.parse_args()

    datadir = Path(args.datadir)
    processed_root = Path(args.processed_dir)
    outfile = Path(args.outfile) if args.outfile else processed_root / "visualizations.pdf"

    patient_dirs = sorted(
        p for p in datadir.iterdir() if p.is_dir() and p.name.startswith("sub-")
    )
    if args.patients:
        selected = set(args.patients)
        patient_dirs = [p for p in patient_dirs if p.name in selected]
        missing = selected - {p.name for p in patient_dirs}
        if missing:
            logger.warning(f"Patients not found in {datadir}: {sorted(missing)}.")
    logger.info(f"Plotting {len(patient_dirs)} patients from {datadir}.")

    outfile.parent.mkdir(parents=True, exist_ok=True)
    plotted = 0
    with PdfPages(str(outfile)) as pdf:
        for patient_dir in patient_dirs:
            try:
                files = resolve_patient_files(
                    patient_dir=patient_dir, processed_dir=processed_root / patient_dir.name
                )
                if files is None:
                    continue
                plot_patient(patient_id=patient_dir.name, files=files, pdf=pdf)
                plotted += 1
                logger.info(f"{patient_dir.name}: plotted.")
            except Exception:
                logger.exception(f"{patient_dir.name}: plotting failed, skipping patient.")

    logger.info(f"Finished plotting {plotted} patients. Output saved to {outfile}.")
