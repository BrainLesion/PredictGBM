"""Visualize the spatial relation of two binary masks derived from NIfTI label maps.

Writes a single-page PDF (portrait A4) with
  rows 1-3: axial, sagittal and coronal slice of the T1c through the center of mass of M1,
            overlaid with seg1, with seg2 and with the difference of the masks (M1 only,
            M2 only),
  row 4:    thickness-weighted three-class projections (sagittal, coronal, axial),
  row 5:    Mollweide map of the radial extent difference r(M2) - r(M1) about the centroid
            of M1, and a text box with volumes, Dice, spacing and gap fractions.

Example:
    python scripts/visualize_cavity.py \
        --t1c t1c_skullstripped.nii.gz \
        --seg1 tumor_seg.nii.gz --labels1 1 3 \
        --seg2 recurrence_preop.nii.gz --labels2 4 \
        --id sub-01 --outfile /tmp/sub-01_compare.pdf
"""

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.patheffects as patheffects
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from scipy.ndimage import map_coordinates

# Class colors of the projections: M1 only, M2 only, overlap.
COLOR_M1 = "#d62728"
COLOR_M2 = "#1f77b4"
COLOR_OVERLAP = "#2ca02c"
# Directions of the radial map in which neither mask is hit.
COLOR_NO_HIT = "#b0b0b0"
COLOR_TEXT = "#222222"
COLOR_MUTED = "#666666"
COLOR_WARNING = "#c00000"
COLOR_FRAME = "#cccccc"
# Segmentation overlay of the slices as in scripts/visualize_respond_10.py: 1 necrosis /
# non-enhancing, 2 edema, 3 enhancing, 4 resection cavity (label 0 is transparent).
SEG_COLORS = [
    (0, 0, 0, 0),
    (1, 127 / 255, 0, 1),
    (30 / 255, 144 / 255, 1, 1),
    (138 / 255, 43 / 255, 226 / 255, 1),
    (34 / 255, 139 / 255, 34 / 255, 1),
]
SEG_LABELS = ["Necrosis / non-enhancing", "Edema", "Enhancing tumor", "Resection cavity"]
# Difference overlay of the slices: nothing where the masks agree, M1 only, M2 only.
DIFFERENCE_COLORS = [(0, 0, 0, 0), COLOR_M1, COLOR_M2]
OVERLAY_ALPHA = 0.6
# Upper display intensity of the T1c as a percentile of its foreground.
T1C_WINDOW_PERCENTILE = 99.5
# Slice rows as (title, voxel axis that is sliced), in page order.
SLICE_VIEWS = (("Axial", 2), ("Sagittal", 0), ("Coronal", 1))
# Anatomical letters at the (negative, positive) end of each RAS voxel axis.
AXIS_LETTERS = (("L", "R"), ("P", "A"), ("I", "S"))
# Projection panels as (title, voxel axis that is projected along), in page order.
VIEWS = (("Sagittal (along R-L)", 0), ("Coronal (along A-P)", 1), ("Axial (along S-I)", 2))
CROP_MARGIN_VOX = 5
ALPHA_PERCENTILE = 95.0
SCALE_BAR_MM = 10.0
# Radial map: margin added to the largest mask distance, floor and percentile of the limits.
R_MARGIN_MM = 5.0
DELTA_MIN_LIMIT_MM = 5.0
DELTA_LIMIT_PERCENTILE = 99.0
# Outline of the directions in which a ray enters a mask more than once.
UNRELIABLE_LINEWIDTH = 0.6
# Largest off-diagonal direction cosine that still counts as axis-aligned voxels.
OBLIQUE_TOLERANCE = 1e-3
A4_PORTRAIT_INCH = (8.27, 11.69)
# Resolution of the rasterized parts of the page (slices, projections, radial map).
FIGURE_DPI = 200
RADIAL_TITLE = "Radial extent difference (view from M1 centroid outward)"


@dataclass(frozen=True)
class Case:
    """Volumes of a case on a common RAS voxel grid; labels: label lists of M1, M2 as text."""

    case_id: str
    t1c: np.ndarray
    seg1: np.ndarray
    seg2: np.ndarray
    m1: np.ndarray
    m2: np.ndarray
    spacing: np.ndarray
    labels: List[str]


@dataclass(frozen=True)
class Stats:
    """Volumes in mL and Dice of the two masks."""

    vol_m1: float
    vol_m2: float
    vol_overlap: float
    vol_m1_only: float
    vol_m2_only: float
    dice: float


@dataclass(frozen=True)
class RadialResult:
    """
    Radial extent difference r(M2) - r(M1) in mm, masked where neither mask is hit, and the
    directions in which a ray enters either mask more than once (extent is unreliable there).
    """

    delta: np.ma.MaskedArray
    unreliable: np.ndarray
    gap_m1: float
    gap_m2: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize the spatial relation of two binary masks as a one-page PDF."
    )
    parser.add_argument(
        "--t1c", type=Path, required=True, help="NIfTI T1c image, same space as seg1"
    )
    parser.add_argument("--seg1", type=Path, required=True, help="NIfTI label map (int)")
    parser.add_argument(
        "--seg2", type=Path, required=True, help="NIfTI label map (int), same space as seg1"
    )
    parser.add_argument(
        "--labels1", type=int, nargs="+", required=True, help="Labels of seg1 that form M1"
    )
    parser.add_argument(
        "--labels2", type=int, nargs="+", required=True, help="Labels of seg2 that form M2"
    )
    parser.add_argument("--id", type=str, required=True, help="Case id, used as page title")
    parser.add_argument("--outfile", type=Path, required=True, help="Output PDF")
    return parser.parse_args()


def warn(message: str) -> None:
    print(f"WARNING: {message}", file=sys.stderr)


def load_volume(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load a 3D volume in RAS voxel order, returns (data, affine, spacing in mm)."""
    img = nib.as_closest_canonical(nib.load(path))
    data = np.asanyarray(img.dataobj)
    if data.ndim != 3:
        sys.exit(f"ERROR: {path} is not a 3D volume (shape {data.shape}).")
    spacing = np.asarray(img.header.get_zooms()[:3], dtype=float)
    return data, np.asarray(img.affine, dtype=float), spacing


def load_label_map(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load a label map in RAS voxel order, returns (labels, affine, spacing in mm)."""
    data, affine, spacing = load_volume(path)
    labels = np.rint(data)
    if not np.array_equal(labels, data):
        warn(f"{path} has non-integer values, they are rounded to the nearest label.")
    return labels.astype(np.int32), affine, spacing


def check_same_space(
    name: str, shape: Tuple[int, ...], affine: np.ndarray, seg1: np.ndarray, affine1: np.ndarray
) -> None:
    """Exit if a volume is not on the voxel grid of seg1 (nothing is resampled)."""
    if shape != seg1.shape:
        sys.exit(f"ERROR: seg1 and {name} differ in shape ({seg1.shape} vs. {shape}).")
    if not np.allclose(affine1, affine):
        sys.exit(
            f"ERROR: seg1 and {name} differ in their affine after reorientation to RAS, "
            f"they are not in the same space.\nseg1:\n{affine1}\n{name}:\n{affine}"
        )


def load_case(args: argparse.Namespace) -> Case:
    """Load the T1c and the two label maps of a case and binarize the masks."""
    seg1, affine1, spacing = load_label_map(args.seg1)
    seg2, affine2, _ = load_label_map(args.seg2)
    t1c, affine_t1c, _ = load_volume(args.t1c)
    check_same_space("seg2", seg2.shape, affine2, seg1, affine1)
    check_same_space("t1c", t1c.shape, affine_t1c, seg1, affine1)
    warn_if_oblique(affine1, spacing)
    return Case(
        case_id=args.id,
        t1c=t1c.astype(np.float32),
        seg1=seg1,
        seg2=seg2,
        m1=np.isin(seg1, args.labels1),
        m2=np.isin(seg2, args.labels2),
        spacing=spacing,
        labels=[label_list(args.labels1), label_list(args.labels2)],
    )


def warn_if_oblique(affine: np.ndarray, spacing: np.ndarray) -> None:
    """The radial method takes the voxel axes as the R, A and S directions."""
    cosines = affine[:3, :3] / spacing[None, :]
    off_diagonal = np.abs(cosines - np.diag(np.diag(cosines))).max()
    if off_diagonal > OBLIQUE_TOLERANCE:
        warn(
            f"the affine has off-diagonal rotation terms (max {off_diagonal:.4f}), the radial "
            "map assumes axis-aligned voxels and its directions are only approximate."
        )


def compute_stats(m1: np.ndarray, m2: np.ndarray, spacing: np.ndarray) -> Stats:
    ml_per_voxel = float(np.prod(spacing)) / 1000.0
    n1, n2 = int(m1.sum()), int(m2.sum())
    n12 = int((m1 & m2).sum())
    dice = 2.0 * n12 / (n1 + n2) if n1 + n2 > 0 else float("nan")
    return Stats(
        vol_m1=n1 * ml_per_voxel,
        vol_m2=n2 * ml_per_voxel,
        vol_overlap=n12 * ml_per_voxel,
        vol_m1_only=(n1 - n12) * ml_per_voxel,
        vol_m2_only=(n2 - n12) * ml_per_voxel,
        dice=dice,
    )


def crop_slices(mask: np.ndarray, margin: int) -> Tuple[slice, slice, slice]:
    """Bounding box of a non-empty mask plus a margin, clipped to the volume."""
    idx = np.argwhere(mask)
    lo = np.maximum(idx.min(axis=0) - margin, 0)
    hi = np.minimum(idx.max(axis=0) + margin + 1, mask.shape)
    return slice(lo[0], hi[0]), slice(lo[1], hi[1]), slice(lo[2], hi[2])


def project_rgb(m1: np.ndarray, m2: np.ndarray, axis: int) -> np.ndarray:
    """Thickness-weighted three-class projection along a voxel axis, RGB on white.

    Per ray the class colors are averaged with the voxel counts of M1 only, overlap and
    M2 only; the opacity is the total count over its 95th percentile over nonzero rays.
    """
    n1 = (m1 & ~m2).sum(axis=axis).astype(float)
    n12 = (m1 & m2).sum(axis=axis).astype(float)
    n2 = (m2 & ~m1).sum(axis=axis).astype(float)
    n = n1 + n12 + n2
    rgb = np.ones(n.shape + (3,))
    if not (n > 0).any():
        return rgb
    colors = [np.array(mcolors.to_rgb(c)) for c in (COLOR_M1, COLOR_OVERLAP, COLOR_M2)]
    mix = sum(count[..., None] * color for count, color in zip((n1, n12, n2), colors))
    mix = mix / np.maximum(n, 1.0)[..., None]
    alpha = np.clip(n / np.percentile(n[n > 0], ALPHA_PERCENTILE), 0.0, 1.0)[..., None]
    return alpha * mix + (1.0 - alpha) * rgb


def radial_extent(
    mask: np.ndarray,
    center: np.ndarray,
    spacing: np.ndarray,
    n_theta: int = 90,
    n_phi: int = 180,
    r_max_mm: float = 80.0,
    dr_mm: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Returns (r_outer[n_theta,n_phi] in mm, ray-hit boolean array, gap_fraction).
    theta = polar angle from +S (z), phi = azimuth from +R (x) toward +A (y)."""
    theta = np.linspace(0.0, np.pi, n_theta)
    phi = np.linspace(-np.pi, np.pi, n_phi)
    r = np.arange(0.0, r_max_mm, dr_mm)
    T, P, R = np.meshgrid(theta, phi, r, indexing="ij")
    u = np.stack([np.sin(T) * np.cos(P), np.sin(T) * np.sin(P), np.cos(T)])
    coords = center[:, None, None, None] + R * u / spacing[:, None, None, None]
    hit = map_coordinates(mask.astype(np.uint8), coords.reshape(3, -1), order=0, cval=0)
    hit = hit.reshape(T.shape).astype(bool)
    last = hit.shape[-1] - 1 - np.argmax(hit[..., ::-1], axis=-1)
    r_outer = np.where(hit.any(-1), r[last], 0.0)
    entries = hit[..., 0].astype(int) + (np.diff(hit.astype(np.int8), axis=-1) == 1).sum(-1)
    gap_fraction = float((entries > 1).mean())
    return r_outer, hit, gap_fraction


def multiple_entries(hit: np.ndarray) -> np.ndarray:
    """Rays of radial_extent that enter the mask more than once (as in its gap fraction)."""
    entries = hit[..., 0].astype(int) + (np.diff(hit.astype(np.int8), axis=-1) == 1).sum(-1)
    return entries > 1


def compute_radial(m1: np.ndarray, m2: np.ndarray, spacing: np.ndarray) -> RadialResult:
    """Radial extent difference of two non-empty masks about the centroid of M1.

    A mask that is not hit along a ray has radius 0 there; rays that hit neither mask
    are masked.
    """
    center = np.argwhere(m1).mean(axis=0)
    offsets_mm = (np.argwhere(m1 | m2) - center) * spacing
    r_max_mm = float(np.linalg.norm(offsets_mm, axis=1).max()) + R_MARGIN_MM
    r1, hit1, gap_m1 = radial_extent(m1, center, spacing, r_max_mm=r_max_mm)
    r2, hit2, gap_m2 = radial_extent(m2, center, spacing, r_max_mm=r_max_mm)
    no_hit = ~(hit1.any(axis=-1) | hit2.any(axis=-1))
    unreliable = multiple_entries(hit1) | multiple_entries(hit2)
    return RadialResult(np.ma.masked_array(r2 - r1, mask=no_hit), unreliable, gap_m1, gap_m2)


def mollweide_centers(n_theta: int, n_phi: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cell centers of the (theta, phi) grid of radial_extent on the Mollweide map.

    Latitude = 90 deg - theta, longitude = 90 deg - phi wrapped to [-180, 180] deg (A at
    0, R at +90, L at -90, P at +-180), i.e. the view from the center outward. Returns
    (longitudes in ascending order, latitudes, order of the phi columns by longitude); the
    last phi column duplicates the first and is left out.
    """
    lat = np.pi / 2.0 - np.linspace(0.0, np.pi, n_theta)
    phi = np.linspace(-np.pi, np.pi, n_phi)[:-1]
    lon = (np.pi / 2.0 - phi + np.pi) % (2.0 * np.pi) - np.pi
    order = np.argsort(lon)
    return lon[order], lat, order


def cell_edges(centers: np.ndarray, first: float, last: float) -> np.ndarray:
    """Edges halfway between neighboring cell centers, with the given outer edges."""
    return np.concatenate([[first], (centers[:-1] + centers[1:]) / 2.0, [last]])


def label_list(labels: List[int]) -> str:
    return ", ".join(str(label) for label in labels)


def style_frame(ax: Axes, title: str, pad: float = 6.0) -> None:
    """Panel without ticks, with a light frame."""
    ax.set_title(title, fontsize=10, color=COLOR_TEXT, pad=pad)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor(COLOR_FRAME)


def draw_skipped(ax: Axes, title: str, reason: str) -> None:
    """Placeholder for a panel that can not be drawn."""
    style_frame(ax, title)
    ax.text(
        0.5,
        0.5,
        f"skipped: {reason}",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=9,
        color=COLOR_MUTED,
    )


def slice_center(m1: np.ndarray, m2: np.ndarray) -> Tuple[Tuple[int, ...], str]:
    """Voxel the slices go through: center of mass of M1 (of M2, or the volume center, if empty)."""
    for name, mask in (("M1", m1), ("M2", m2)):
        if mask.any():
            center = tuple(int(c) for c in np.argwhere(mask).mean(axis=0))
            return center, f"center of mass of {name}"
    return tuple(s // 2 for s in m1.shape), "volume center"


def intensity_window(data: np.ndarray) -> float:
    """Upper display intensity of an MRI volume, clipped at a percentile of the foreground."""
    foreground = data[data > 0]
    if foreground.size == 0:
        return 1.0
    return float(np.percentile(foreground, T1C_WINDOW_PERCENTILE)) or float(data.max()) or 1.0


def add_slice_letters(ax: Axes, horizontal: int, vertical: int) -> None:
    """Anatomical letters inside of the four edges of a slice panel."""
    halo = [patheffects.withStroke(linewidth=1.5, foreground="black")]
    style = dict(transform=ax.transAxes, fontsize=7, color="white", path_effects=halo)
    ax.text(0.02, 0.5, AXIS_LETTERS[horizontal][0], ha="left", va="center", **style)
    ax.text(0.98, 0.5, AXIS_LETTERS[horizontal][1], ha="right", va="center", **style)
    ax.text(0.5, 0.02, AXIS_LETTERS[vertical][0], ha="center", va="bottom", **style)
    ax.text(0.5, 0.98, AXIS_LETTERS[vertical][1], ha="center", va="top", **style)


def draw_slice(
    ax: Axes, case: Case, vmax: float, overlay: np.ndarray, colors: list, axis: int, index: int
) -> None:
    """
    Slice of the T1c across a voxel axis with a label overlay (label i in colors[i]), shown
    as in scripts/visualize_respond_10.py and oriented like the projections.
    """
    horizontal, vertical = [a for a in range(3) if a != axis]
    extent_mm = np.asarray(case.t1c.shape) * case.spacing
    style = dict(
        origin="lower",
        extent=(0.0, extent_mm[horizontal], 0.0, extent_mm[vertical]),
        interpolation="nearest",
        aspect="equal",
    )
    ax.imshow(np.take(case.t1c, index, axis=axis).T, cmap="gray", vmin=0.0, vmax=vmax, **style)
    cmap = mcolors.ListedColormap(colors)
    norm = mcolors.BoundaryNorm(np.arange(len(colors) + 1) - 0.5, cmap.N)
    labels = np.take(overlay, index, axis=axis).T
    ax.imshow(labels, cmap=cmap, norm=norm, alpha=OVERLAY_ALPHA, **style)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    add_slice_letters(ax, horizontal, vertical)


def add_slice_legend(fig: Figure, legend_labels: List[str], title: str) -> None:
    """Legend of the slice grid: segmentation labels and the two difference classes."""
    colors = SEG_COLORS[1:] + DIFFERENCE_COLORS[1:]
    handles = [
        mpatches.Patch(color=color, label=label)
        for color, label in zip(colors, SEG_LABELS + legend_labels[:2])
    ]
    legend = fig.legend(
        handles=handles,
        loc="center",
        bbox_to_anchor=(0.5, 0.517),
        ncol=3,
        frameon=False,
        fontsize=8,
        title=title,
    )
    legend.get_title().set_fontsize(8)
    legend.get_title().set_color(COLOR_MUTED)


def draw_slice_grid(fig: Figure, case: Case, legend_labels: List[str]) -> None:
    """Rows 1-3: T1c slices per view with seg1, seg2 and the mask difference as columns."""
    extent_mm = np.asarray(case.t1c.shape) * case.spacing
    heights = [extent_mm[[a for a in range(3) if a != axis][1]] for _, axis in SLICE_VIEWS]
    grid = fig.add_gridspec(
        3,
        3,
        left=0.13,
        right=0.87,
        top=0.922,
        bottom=0.545,
        height_ratios=heights,
        wspace=0.03,
        hspace=0.03,
    )
    center, center_name = slice_center(case.m1, case.m2)
    vmax = intensity_window(case.t1c)
    difference = (case.m1 & ~case.m2) + 2 * (case.m2 & ~case.m1)
    columns = [
        ("Pre-op segmentation (seg1)", case.seg1, SEG_COLORS),
        ("Post-op segmentation (seg2)", case.seg2, SEG_COLORS),
        ("Difference (M1 only / M2 only)", difference, DIFFERENCE_COLORS),
    ]
    for row, (view, axis) in enumerate(SLICE_VIEWS):
        for column, (title, overlay, colors) in enumerate(columns):
            ax = fig.add_subplot(grid[row, column])
            draw_slice(ax, case, vmax, overlay, colors, axis, center[axis])
            if row == 0:
                ax.set_title(title, fontsize=9, color=COLOR_TEXT)
            if column == 0:
                ax.set_ylabel(view, fontsize=10, color=COLOR_TEXT)
    voxel = ", ".join(str(c) for c in center)
    add_slice_legend(fig, legend_labels, f"slices through the {center_name} (voxel {voxel})")


def add_orientation_letters(ax: Axes, horizontal: int, vertical: int) -> None:
    """Anatomical letters outside of the four edges of a projection panel."""
    style = dict(transform=ax.transAxes, fontsize=11, fontweight="bold", color=COLOR_TEXT)
    ax.text(-0.03, 0.5, AXIS_LETTERS[horizontal][0], ha="right", va="center", **style)
    ax.text(1.03, 0.5, AXIS_LETTERS[horizontal][1], ha="left", va="center", **style)
    ax.text(0.5, -0.03, AXIS_LETTERS[vertical][0], ha="center", va="top", **style)
    ax.text(0.5, 1.02, AXIS_LETTERS[vertical][1], ha="center", va="bottom", **style)


def add_scale_bar(ax: Axes, x_mm: float, y_mm: float) -> None:
    """Horizontal scale bar starting at (x_mm, y_mm) of a panel with mm data coordinates."""
    ax.plot(
        [x_mm, x_mm + SCALE_BAR_MM],
        [y_mm, y_mm],
        color=COLOR_TEXT,
        linewidth=2,
        solid_capstyle="butt",
    )
    ax.annotate(
        f"{SCALE_BAR_MM:g} mm",
        (x_mm + SCALE_BAR_MM / 2.0, y_mm),
        xytext=(0, 3),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=7,
        color=COLOR_TEXT,
    )


def draw_projection(
    ax: Axes, m1: np.ndarray, m2: np.ndarray, axis: int, spacing: np.ndarray, title: str
) -> None:
    """
    Projection along a voxel axis. Of the two remaining axes the lower one is horizontal and
    both increase to the right / top: R on the right, A on the right or top, S on top.
    """
    horizontal, vertical = [a for a in range(3) if a != axis]
    rgb = project_rgb(m1, m2, axis)
    extent = (0.0, rgb.shape[0] * spacing[horizontal], 0.0, rgb.shape[1] * spacing[vertical])
    ax.imshow(
        rgb.transpose(1, 0, 2),
        origin="lower",
        extent=extent,
        interpolation="nearest",
        aspect="equal",
    )
    ax.autoscale(False)
    style_frame(ax, title, pad=18.0)
    add_orientation_letters(ax, horizontal, vertical)
    # The bar sits in the crop margin, which is free of mask voxels.
    add_scale_bar(
        ax, 0.4 * CROP_MARGIN_VOX * spacing[horizontal], 0.4 * CROP_MARGIN_VOX * spacing[vertical]
    )


def add_projection_legend(fig: Figure, legend_labels: List[str]) -> None:
    """Shared legend of the three projection panels."""
    handles = [
        mpatches.Patch(color=color, label=label)
        for color, label in zip((COLOR_M1, COLOR_M2, COLOR_OVERLAP), legend_labels)
    ]
    legend = fig.legend(
        handles=handles,
        loc="center",
        bbox_to_anchor=(0.5, 0.273),
        ncol=3,
        frameon=False,
        fontsize=9,
        title="opacity: thickness along the projection "
        f"(saturates at its {ALPHA_PERCENTILE:g}th percentile per panel)",
    )
    legend.get_title().set_fontsize(8)
    legend.get_title().set_color(COLOR_MUTED)


def draw_projection_row(
    fig: Figure, m1: np.ndarray, m2: np.ndarray, spacing: np.ndarray, legend_labels: List[str]
) -> None:
    """Row 4: the three projections on a common crop and their shared legend."""
    grid = fig.add_gridspec(1, 3, left=0.08, right=0.92, top=0.457, bottom=0.311, wspace=0.25)
    union = m1 | m2
    if union.any():
        crop = crop_slices(union, CROP_MARGIN_VOX)
        m1, m2 = m1[crop], m2[crop]
    for column, (title, axis) in enumerate(VIEWS):
        ax = fig.add_subplot(grid[0, column])
        if union.any():
            draw_projection(ax, m1, m2, axis, spacing, title)
        else:
            draw_skipped(ax, title, "both masks are empty")
    add_projection_legend(fig, legend_labels)


def style_radial_axes(ax: Axes, radial: RadialResult) -> None:
    """Direction labels, grid and title of the Mollweide map."""
    ax.set_xticks(np.radians([-180.0, -90.0, 0.0, 90.0, 180.0]))
    ax.set_xticklabels(["P", "L", "A", "R", "P"], fontsize=11, fontweight="bold")
    # The longitude labels lie on the map, the halo keeps them readable on any color.
    halo = [patheffects.withStroke(linewidth=2.5, foreground="white")]
    for tick_label in ax.get_xticklabels():
        tick_label.set_path_effects(halo)
    ax.tick_params(axis="y", labelsize=7, labelcolor=COLOR_MUTED)
    ax.grid(True, color="#444444", linewidth=0.4, alpha=0.6)
    style = dict(transform=ax.transAxes, ha="center", fontsize=11, fontweight="bold")
    ax.text(0.5, 1.02, "S", va="bottom", color=COLOR_TEXT, **style)
    ax.text(0.5, -0.02, "I", va="top", color=COLOR_TEXT, **style)
    ax.set_title(
        f"{RADIAL_TITLE}\ngap fraction M1: {radial.gap_m1:.3f}, M2: {radial.gap_m2:.3f}",
        fontsize=10,
        color=COLOR_TEXT,
        pad=20,
    )


def add_radial_legend(ax: Axes, radial: RadialResult) -> None:
    """Legend of the overlays of the radial map, only for those that occur."""
    handles: List[Artist] = []
    if radial.unreliable.any():
        label = "mask entered more than once (unreliable)"
        handles.append(Line2D([], [], color="black", linewidth=UNRELIABLE_LINEWIDTH, label=label))
    if radial.delta.mask.any():
        handles.append(mpatches.Patch(color=COLOR_NO_HIT, label="neither mask hit"))
    if handles:
        ax.legend(
            handles=handles,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.09),
            ncol=2,
            frameon=False,
            fontsize=8,
        )


def draw_unreliable_contour(
    ax: Axes, lon: np.ndarray, lat: np.ndarray, unreliable: np.ndarray
) -> None:
    """Outline of the directions in which a ray enters either mask more than once."""
    if not unreliable.any():
        return
    # One wrapped column on either side closes the outlines across the P meridian.
    lon = np.concatenate([[lon[-1] - 2.0 * np.pi], lon, [lon[0] + 2.0 * np.pi]])
    field = np.concatenate([unreliable[:, -1:], unreliable, unreliable[:, :1]], axis=1)
    ax.contour(
        lon,
        lat,
        field.astype(float),
        levels=[0.5],
        colors="black",
        linewidths=UNRELIABLE_LINEWIDTH,
    )


def draw_radial(fig: Figure, ax: Axes, radial: RadialResult) -> None:
    """Mollweide map of r(M2) - r(M1) seen from the center: S at the top, A at the center."""
    n_theta, n_phi = radial.delta.shape
    lon, lat, order = mollweide_centers(n_theta, n_phi)
    delta = radial.delta[:, :-1][:, order]
    percentile = np.percentile(np.abs(delta.compressed()), DELTA_LIMIT_PERCENTILE)
    limit = max(DELTA_MIN_LIMIT_MM, float(percentile))
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad(COLOR_NO_HIT)
    mesh = ax.pcolormesh(
        cell_edges(lon, -np.pi, np.pi),
        cell_edges(lat, np.pi / 2.0, -np.pi / 2.0),
        delta,
        cmap=cmap,
        norm=mcolors.CenteredNorm(vcenter=0.0, halfrange=limit),
        shading="flat",
        rasterized=True,
    )
    draw_unreliable_contour(ax, lon, lat, radial.unreliable[:, :-1][:, order])
    style_radial_axes(ax, radial)
    add_radial_legend(ax, radial)
    colorbar = fig.colorbar(mesh, ax=ax, orientation="vertical", shrink=0.85, pad=0.04)
    colorbar.set_label("r(M2) − r(M1) [mm]", fontsize=9)
    colorbar.ax.tick_params(labelsize=8)


def stats_text(
    stats: Stats, spacing: np.ndarray, radial: Optional[RadialResult], labels: List[str]
) -> str:
    """Text box content: mask definitions, volumes, Dice, spacing and gap fractions."""
    gap_m1 = f"{radial.gap_m1:.3f}" if radial is not None else "n/a"
    gap_m2 = f"{radial.gap_m2:.3f}" if radial is not None else "n/a"
    rows = [
        ("Volume M1", f"{stats.vol_m1:.2f} mL"),
        ("Volume M2", f"{stats.vol_m2:.2f} mL"),
        ("M1 & M2", f"{stats.vol_overlap:.2f} mL"),
        ("M1 only", f"{stats.vol_m1_only:.2f} mL"),
        ("M2 only", f"{stats.vol_m2_only:.2f} mL"),
        ("Dice", f"{stats.dice:.3f}" if np.isfinite(stats.dice) else "n/a"),
        ("Spacing", " x ".join(f"{s:.2f}" for s in spacing) + " mm"),
        ("Gap fraction M1", gap_m1),
        ("Gap fraction M2", gap_m2),
    ]
    width = max(len(value) for _, value in rows) + 1
    lines = [f"M1: seg1 labels {labels[0]}", f"M2: seg2 labels {labels[1]}", ""]
    lines += [f"{name:<16}{value:>{width}}" for name, value in rows]
    return "\n".join(lines)


def draw_bottom_row(fig: Figure, radial: Optional[RadialResult], text: str) -> None:
    """Bottom row: the radial map (or its placeholder) and the statistics text box."""
    grid = fig.add_gridspec(
        1, 2, left=0.06, right=0.97, top=0.204, bottom=0.055, width_ratios=[2.6, 1.0], wspace=0.08
    )
    if radial is not None:
        draw_radial(fig, fig.add_subplot(grid[0, 0], projection="mollweide"), radial)
    else:
        reason = "needs both masks to be non-empty"
        draw_skipped(fig.add_subplot(grid[0, 0]), RADIAL_TITLE, reason)
    ax_text = fig.add_subplot(grid[0, 1])
    ax_text.axis("off")
    ax_text.text(
        0.5,
        0.5,
        text,
        transform=ax_text.transAxes,
        ha="center",
        va="center",
        fontsize=7,
        family="monospace",
        color=COLOR_TEXT,
        linespacing=1.4,
        bbox=dict(boxstyle="round,pad=0.8", facecolor="#f7f7f7", edgecolor=COLOR_FRAME),
    )


def build_page(
    case: Case, stats: Stats, radial: Optional[RadialResult], warnings: List[str]
) -> Figure:
    """Portrait A4 page: slice grid, projection row, radial map with the statistics."""
    fig = plt.figure(figsize=A4_PORTRAIT_INCH, dpi=FIGURE_DPI)
    fig.suptitle(case.case_id, fontsize=15, fontweight="bold", color=COLOR_TEXT, y=0.993)
    if warnings:
        fig.text(
            0.5,
            0.973,
            "\n".join(f"WARNING: {message}" for message in warnings),
            ha="center",
            va="top",
            fontsize=9,
            fontweight="bold",
            color=COLOR_WARNING,
        )
    labels = case.labels
    legend_labels = [f"M1 only (labels {labels[0]})", f"M2 only (labels {labels[1]})", "overlap"]
    draw_slice_grid(fig, case, legend_labels)
    draw_projection_row(fig, case.m1, case.m2, case.spacing, legend_labels)
    draw_bottom_row(fig, radial, stats_text(stats, case.spacing, radial, labels))
    return fig


def empty_mask_warnings(m1: np.ndarray, m2: np.ndarray, labels: List[str]) -> List[str]:
    warnings = []
    for name, seg, mask, label in (("M1", "seg1", m1, labels[0]), ("M2", "seg2", m2, labels[1])):
        if not mask.any():
            warnings.append(f"{name} is empty ({seg} has no voxels with labels {label}).")
    return warnings


def main() -> None:
    args = parse_args()
    case = load_case(args)
    m1, m2, spacing = case.m1, case.m2, case.spacing
    warnings = empty_mask_warnings(m1, m2, case.labels)
    for message in warnings:
        warn(message)

    stats = compute_stats(m1, m2, spacing)
    radial = compute_radial(m1, m2, spacing) if m1.any() and m2.any() else None
    fig = build_page(case, stats, radial, warnings)
    args.outfile.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.outfile)
    plt.close(fig)
    print(stats_text(stats, spacing, radial, case.labels))
    print(f"Saved {args.outfile}")


if __name__ == "__main__":
    main()
