"""
Per-patient summary figures of the respond longitudinal data: one row per MRI session with
axial / sagittal / coronal slices of the skull-stripped T1c (or FLAIR), plain and with the
tumor segmentation overlaid, a segmentation-quality tick-box column left of each image row,
a notes box, and below it the segmented core volume against the clinical day relative to the
post-op scan with the treatment events marked. The tick boxes and the notes box are
interactive PDF form fields, so they can be filled in any PDF viewer and read back with
pypdf (PdfReader.get_fields()).

Port of pde_solver_development/scripts/visualize_segmentations.py (2026-10-27) to this
codebase and to the lucas_long_followup data of scripts/process_respond_longitudinal.py.

Sessions: <data_dir>/<patient>/processed/session_labels.tsv (columns subject, session,
timepoint, processed, reference, label_source), dates from the session id ses-YYYYMMDD.
Drawn are the processed preop, postop and followup sessions; intraop sessions (scan on the
resection date) and unprocessed sessions are left out. Day 0 is the postop session (one per
patient; with several the one on the sheet's postop date, else the first, with a warning).

Images: each session's own skull-stripped t1c / flair and tumor_seg_corrected.nii.gz below
<patient>/processed/<session>/ (MODALITY_STRIPPED_SCHEMA, TUMORSEG_CORRECTED_SCHEMA, written
by scripts/process_respond_longitudinal.py; used for overlays, centres, videos and core
volumes); all sessions of
a patient lie on the preop grid (rigidly co-registered in atlas space). Every volume is
reoriented to LAS before drawing, so the slices are in radiological view (subject's right on
the image left, anterior / superior up; sagittal with anterior on the left) whatever the
stored orientation (RAS for this data); the x/y/z printed in the plain panels are LAS voxel
indices. Each row's slices go through the centre of mass of the exam's own necrotic +
enhancing segmentation (labels 1 + 3); an exam without such voxels uses the reference preop
session's centre (its core, falling back to the whole tumor, any labelled voxel and the grid
centre). Each image is drawn in gray from 0 to the -window_percentile of its non-zero voxels.

Form fields: left of each image row a checkbox group "Segmentation:" with the options Good /
Fake enhancing / Fake necrotic / Other / Unsure below each other (fields
<session>_segmentation_<option>, e.g. <session>_segmentation_fake_enhancing, on value /Yes),
above the volume panel a multiline text field "notes". Matplotlib draws the outlines, the widgets are
added to the saved PDF with pypdf (AcroForm, NeedAppearances).

Layout: fixed positions in inches (``Layout`` from the *_INCHES constants, no layout
engine): from the top the colour legend, the column titles, the image rows (tick-box
column, session label, six panels, video buttons), the legend again, the notes box and
the volume panel.

Videos: right of each row the buttons "Video: T1c", "Video: FLAIR" and "Video: trace"
(or "Video: ADC" for exams without a trace but with a registered ADC; exams with neither
get a note). Each video (<outdir>/videos/<patient>_<session>_<kind>.mp4, H.264, -fps
slices per second, -video_scale upscaling) scrolls through the axial slices of the
skull-stripped volume from inferior to superior over the brain mask's extent, every frame
the plain slice and the slice with the segmentation overlaid side by side in radiological
view with the same window and colours as the panels. With -video_links embed (default)
the mp4 files are embedded in the PDF and a button runs the JavaScript that opens the
attachment (Acrobat Reader; other viewers list the videos in their attachment panel),
with -video_links launch a button is a Launch action of videos/<file> next to the PDF.
Existing mp4 files are reused unless -override_videos; -no_videos draws no buttons.

Core volume (bottom panel): necrotic + enhancing (labels 1 + 3) in preop-labelled sessions,
enhancing alone (label 3) in postop and followup sessions, counted on the native
segmentation. A preop segmentation with cavity voxels (label 4) is drawn as it is and logged.

Treatment events (-dates_file, default <data_dir>/dates.xlsx): the resection on the sheet's
Resection date and the standard protocol anchored on the sheet's RT planning date t3: 30
fractions at t3 + 7 w + d (w = 0..5, d = 0..4) with 42 daily concomitant TMZ days from t3,
then up to 6 adjuvant TMZ cycles of 5 days every 28 days from 28 days after the last CRT
day, everything after the last drawn session dropped. Without a sheet row the resection is
placed 3 days before the postop scan and without an RT planning date the CRT starts at the
first followup after the postop scan (each with a warning), as the original did.

Outputs: <outdir>/<patient>_patient_summary_t1c.pdf and <patient>_patient_summary_flair.pdf
(-outdir, default <data_dir>/patient_summaries). A patient whose figure fails is logged and
skipped.

python scripts/visualize_respond_longitudinal.py
python scripts/visualize_respond_longitudinal.py -patients sub-GR149
"""

import argparse
import csv
import subprocess
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Dict, Iterable, List, NamedTuple, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import openpyxl
from loguru import logger
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch, Rectangle
from matplotlib.transforms import Bbox, ScaledTranslation, Transform
from nibabel.orientations import apply_orientation, axcodes2ornt, io_orientation, ornt_transform
from numpy.typing import NDArray
from PIL import Image, ImageDraw, ImageFont
from pypdf import PdfWriter
from pypdf.generic import (
    ArrayObject,
    BooleanObject,
    DecodedStreamObject,
    DictionaryObject,
    FloatObject,
    NameObject,
    NumberObject,
    TextStringObject,
)
from scipy.ndimage import center_of_mass

from predict_gbm.utils.constants import BRAIN_MASK_SCHEMA, MODALITY_STRIPPED_SCHEMA, TUMORSEG_CORRECTED_SCHEMA

# Data layout of scripts/process_respond_longitudinal.py.
DEFAULT_DATA_DIR = Path("/mnt/Drive4/lucas/respond_longitudinal/lucas_long_followup")
DEFAULT_PATIENT_DIR_GLOB = "sub-GR*"
PROCESSED_DIRNAME = "processed"
SESSION_LABELS_FILENAME = "session_labels.tsv"
DEFAULT_OUTDIR_NAME = "patient_summaries"
TIMEPOINT_PREOP = "preop"
TIMEPOINT_INTRAOP = "intraop"
TIMEPOINT_POSTOP = "postop"
TIMEPOINT_FOLLOWUP = "followup"
DRAWN_TIMEPOINTS = (TIMEPOINT_PREOP, TIMEPOINT_POSTOP, TIMEPOINT_FOLLOWUP)
# Surgery dates sheet (first worksheet, one row per subject).
DATES_FILENAME = "dates.xlsx"
DATES_ID_COLUMN = "ID"
DATES_RESECTION_COLUMN = "Resection"
DATES_POSTOP_COLUMN = "postop"
DATES_RT_PLANNING_COLUMN = "RT planning"

# Segmentation labels (BraTS convention of the processed data).
LABEL_NECROTIC, LABEL_EDEMA, LABEL_ENHANCING, LABEL_CAVITY = 1, 2, 3, 4
KNOWN_LABELS = (0, LABEL_NECROTIC, LABEL_EDEMA, LABEL_ENHANCING, LABEL_CAVITY)
PREOP_CORE_LABELS = (LABEL_NECROTIC, LABEL_ENHANCING)
POSTOP_CORE_LABELS = (LABEL_ENHANCING,)
CENTER_CORE_LABELS = (LABEL_NECROTIC, LABEL_ENHANCING)
WHOLE_LABELS = (LABEL_NECROTIC, LABEL_EDEMA, LABEL_ENHANCING)
# Palette and legend of scripts/visualize_respond_10.py (index = label).
SEG_COLORS = [
    (0, 0, 0, 0),
    (1, 127 / 255, 0, 1),
    (30 / 255, 144 / 255, 1, 1),
    (138 / 255, 43 / 255, 226 / 255, 1),
    (34 / 255, 139 / 255, 34 / 255, 1),
]
SEG_LABELS = ["Necrosis / non-enhancing", "Edema", "Enhancing tumor", "Resection cavity"]
SEG_CMAP = ListedColormap(SEG_COLORS)
SEGMENTATION_ALPHA = 0.6
CORE_CURVE_COLOR = (0.85, 0.25, 0.10, 1.0)
CORE_CURVE_LABEL = "segmented core (pre-op: necrotic + enhancing; post-op: enhancing)"
RESECTION_COLOR = (210 / 255, 43 / 255, 43 / 255, 1.0)

# Background modalities (file key of MODALITY_STRIPPED_SCHEMA -> panel name).
MODALITIES: Dict[str, str] = {"t1c": "T1c", "flair": "FLAIR"}
OUTPUT_FILENAME = "{patient}_patient_summary_{modality}.pdf"
TARGET_AXCODES = ("L", "A", "S")
VIEWS = ("axial", "sagittal", "coronal")
VIEW_INDEX = {"axial": 2, "sagittal": 0, "coronal": 1}
DEFAULT_WINDOW_PERCENTILE = 99.0
PANEL_INCHES = 3.0
PANEL_GAP_INCHES = 0.08  # between the panels of a row and between rows
MARGIN_INCHES = 0.15
LEGEND_INCHES = 0.35  # band of the colour legend (above the titles and below the last row)
TITLE_INCHES = 0.35  # band of the column titles
ROW_LABEL_INCHES = 0.55  # band of the rotated session label left of the first panel
SECTION_GAP_INCHES = 0.25
NOTES_TITLE_INCHES = 0.3
VOLUME_PANEL_INCHES = 4.0  # whole band: its legend, the axes, the x label
VOLUME_LEGEND_INCHES = 0.35
VOLUME_XLABEL_INCHES = 0.6
VOLUME_YLABEL_INCHES = 0.9

# Form fields: the tick-box row below each image row and the notes box above the volume
# panel (sizes in inches; the widget rectangles are taken from the drawn outlines).
FORM_COLUMN_INCHES = 1.6  # tick-box column left of each image row; must hold the longest option text
CHECKBOX_INCHES = 0.22
FORM_LABEL = "Segmentation:"
SEGMENTATION_OPTIONS = ("Good", "Fake enhancing", "Fake necrotic", "Other", "Unsure")
FORM_LINE_PITCH_INCHES = 0.32  # vertical distance between the label and the option lines
FORM_INDENT_INCHES = 0.05
CHECKBOX_TEXT_GAP_INCHES = 0.08
NOTES_INCHES = 1.6
NOTES_FIELD_NAME = "notes"
NOTES_TITLE = "Notes:"
FORM_FONT_SIZE = 10
FORM_LINE_WIDTH = 0.8
CHECKBOX_ON_STATE = "Yes"
FIELD_FONT_POINTS = 10
# Videos: one per row and kind in <outdir>/VIDEO_DIRNAME; (skull-stripped modality, button
# label) of every exam, then the first available diffusion image.
VIDEO_DIRNAME = "videos"
VIDEO_FILENAME = "{patient}_{session}_{kind}.mp4"
VIDEO_KINDS = (("t1c", "T1c"), ("flair", "FLAIR"))
DIFFUSION_KINDS = (("trace", "trace"), ("adc", "ADC"))
VIDEO_MARGIN_SLICES = 2  # slices beyond the brain mask at both ends
VIDEO_PANEL_GAP_PX = 4  # between the plain and the overlaid panel (before upscaling)
VIDEO_FONT_PX = 16
VIDEO_CRF = 23
DEFAULT_FPS = 8
DEFAULT_VIDEO_SCALE = 2
FFMPEG = "ffmpeg"
VIDEO_LINK_KINDS = ("embed", "launch")
VIDEO_HINT = {
    "embed": "Video buttons open the embedded mp4 files (Acrobat Reader); other viewers list them in the attachment panel.",
    "launch": f"Video buttons open the mp4 files in the {VIDEO_DIRNAME} folder next to this PDF (Launch action).",
}
# Video buttons, side by side right of each row.
BUTTON_COLUMN_INCHES = 3.1
BUTTON_WIDTH_INCHES = 0.9
BUTTON_HEIGHT_INCHES = 0.28
BUTTON_GAP_INCHES = 0.1
BUTTON_INDENT_INCHES = 0.1
BUTTON_FONT_SIZE = 9
BUTTON_COLOR = "tab:blue"
BUTTON_LABEL = "Video: {label}"
NO_DIFFUSION_TEXT = "(no trace / ADC)"

# Standard (Stupp) protocol as drawn by the original script, in days.
RESECTION_DAYS_BEFORE_POSTOP = 3
CRT_WEEKS = 6
FRACTIONS_PER_WEEK = 5
N_FRACTIONS = CRT_WEEKS * FRACTIONS_PER_WEEK
CONCOMITANT_DAYS = 7 * CRT_WEEKS
ADJUVANT_DELAY_DAYS = 28
ADJUVANT_DAYS_ON = 5
ADJUVANT_CYCLE_DAYS = 28
ADJUVANT_CYCLES = 6


class Session(NamedTuple):
    id: str
    timepoint: str
    date: date
    reference: bool


@dataclass
class Row:
    """A drawn session: its segmentation, core volume, slice centre and, once the centre is
    known, the oriented slices and display maxima of the background images."""

    session: Session
    day: int
    labels: NDArray
    core_ml: float
    center: Optional[Tuple[int, int, int]] = None
    center_source: Optional[str] = None
    slices: Dict[str, Dict[str, NDArray]] = field(default_factory=dict)
    vmax: Dict[str, float] = field(default_factory=dict)
    overlays: Dict[str, NDArray] = field(default_factory=dict)
    videos: Dict[str, Tuple[str, Path]] = field(default_factory=dict)  # kind -> (button label, mp4)


@dataclass
class TreatmentEvents:
    """Event days relative to the post-op scan."""

    resection_day: int
    rt_days: List[int]
    chemo_days: List[int]


@dataclass
class FormField:
    """A PDF annotation: kind 'checkbox' or 'text' (form widgets, name = field name) or
    'link' (a video button, name = file name of the video), and its rectangle in PDF points
    (x0, y0, x1, y1, origin bottom left)."""

    kind: str
    name: str
    rect: Tuple[float, float, float, float]


@dataclass(frozen=True)
class Layout:
    """Positions of the figure parts in inches from the bottom left: from the top the colour
    legend, the column titles, the image rows (tick-box column, session label, the panels,
    the video buttons if ``buttons``), the legend again, the notes box with its title and the
    volume panel (its legend, the axes, the x label)."""

    n_row: int
    buttons: bool = True

    @property
    def width(self) -> float:
        n_panel = 2 * len(VIEWS)
        width = 2 * MARGIN_INCHES + FORM_COLUMN_INCHES + ROW_LABEL_INCHES + n_panel * PANEL_INCHES + (n_panel - 1) * PANEL_GAP_INCHES
        if self.buttons:
            width += PANEL_GAP_INCHES + BUTTON_COLUMN_INCHES
        return width

    @property
    def button_left(self) -> float:
        """Left edge of the video-button column."""
        return self.panel_left(2 * len(VIEWS) - 1) + PANEL_INCHES + PANEL_GAP_INCHES

    @property
    def notes_bottom(self) -> float:
        return MARGIN_INCHES + VOLUME_PANEL_INCHES + SECTION_GAP_INCHES

    @property
    def bottom_legend_center(self) -> float:
        return self.notes_bottom + NOTES_INCHES + NOTES_TITLE_INCHES + SECTION_GAP_INCHES + LEGEND_INCHES / 2

    @property
    def rows_bottom(self) -> float:
        return self.bottom_legend_center + LEGEND_INCHES / 2

    def row_bottom(self, i: int) -> float:
        """Bottom edge of image row i (0 = top row)."""
        return self.rows_bottom + (self.n_row - 1 - i) * (PANEL_INCHES + PANEL_GAP_INCHES)

    @property
    def rows_top(self) -> float:
        return self.rows_bottom + self.n_row * PANEL_INCHES + (self.n_row - 1) * PANEL_GAP_INCHES

    @property
    def top_legend_center(self) -> float:
        return self.rows_top + TITLE_INCHES + LEGEND_INCHES / 2

    @property
    def height(self) -> float:
        return self.top_legend_center + LEGEND_INCHES / 2 + MARGIN_INCHES

    def panel_left(self, k: int) -> float:
        """Left edge of panel k (0 = first) of a row."""
        return MARGIN_INCHES + FORM_COLUMN_INCHES + ROW_LABEL_INCHES + k * (PANEL_INCHES + PANEL_GAP_INCHES)

    def axes(self, fig: plt.Figure, x0: float, y0: float, width: float, height: float) -> plt.Axes:
        """An axes at the given inch rectangle (x0, y0 = bottom left)."""
        return fig.add_axes([x0 / self.width, y0 / self.height, width / self.width, height / self.height])


def session_date(session: str) -> date:
    """Parses the date of a session directory name ses-YYYYMMDD."""
    return datetime.strptime(session.removeprefix("ses-"), "%Y%m%d").date()


def read_sessions(labels_file: Path) -> Tuple[str, List[Session]]:
    """
    The subject id and the processed sessions of a session_labels.tsv, sorted by date.
    Unprocessed sessions are left out with a warning.
    """
    with open(labels_file, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    if not rows:
        raise ValueError(f"{labels_file}: no sessions.")
    subjects = sorted({row["subject"] for row in rows})
    if len(subjects) != 1:
        raise ValueError(f"{labels_file}: expected one subject, found {subjects}.")
    sessions = []
    for row in rows:
        if row["processed"] != "1":
            logger.warning(f"{labels_file}: {row['session']} is not processed, left out.")
            continue
        sessions.append(
            Session(
                row["session"], row["timepoint"], session_date(row["session"]), row["reference"] == "1"
            )
        )
    return subjects[0], sorted(sessions, key=lambda s: s.date)


def read_sheet_dates(dates_file: Path) -> Dict[str, Dict[str, Optional[date]]]:
    """Subject id -> column name -> date (None for empty / non-date cells) of the sheet."""
    worksheet = openpyxl.load_workbook(dates_file, read_only=True, data_only=True).active
    rows = worksheet.iter_rows(values_only=True)
    header = [str(name).strip() if name is not None else "" for name in next(rows)]
    if DATES_ID_COLUMN not in header:
        raise ValueError(f"{dates_file}: no column {DATES_ID_COLUMN!r} in {header}.")
    sheet = {}
    for row in rows:
        subject = row[header.index(DATES_ID_COLUMN)]
        if subject is None:
            continue
        sheet[str(subject).strip()] = {
            name: (value.date() if isinstance(value, datetime) else None)
            for name, value in zip(header, row)
            if name
        }
    return sheet


def load_oriented(path: Path) -> Tuple[NDArray, nib.Nifti1Image]:
    """A volume reoriented to TARGET_AXCODES (float32) and its image."""
    image = nib.load(str(path))
    data = np.asarray(image.dataobj, dtype=np.float32)
    transform = ornt_transform(io_orientation(image.affine), axcodes2ornt(TARGET_AXCODES))
    return apply_orientation(data, transform), image


def load_segmentation(path: Path) -> Tuple[NDArray, float]:
    """The reoriented tumor segmentation as uint8 labels and its voxel volume in mL."""
    data, image = load_oriented(path)
    labels = np.rint(data).astype(np.int64)
    unknown = sorted(set(np.unique(labels).tolist()) - set(KNOWN_LABELS))
    if unknown:
        raise ValueError(f"{path}: label(s) {unknown} are not among {list(KNOWN_LABELS)}.")
    voxel_ml = float(np.prod(image.header.get_zooms()[:3])) / 1000.0
    return labels.astype(np.uint8), voxel_ml


def core_labels(session: Session) -> Tuple[int, ...]:
    """Necrotic + enhancing before the resection, enhancing alone after it."""
    return PREOP_CORE_LABELS if session.timepoint == TIMEPOINT_PREOP else POSTOP_CORE_LABELS


def display_window(image: NDArray, percentile: float) -> float:
    """Upper display limit: the percentile of the non-zero voxels, 1.0 for an empty volume."""
    inside = image[image > 0]
    return float(np.percentile(inside, percentile)) if inside.size else 1.0


def mask_center(mask: NDArray) -> Optional[Tuple[int, int, int]]:
    """Rounded voxel centre of mass of a mask, None for an empty one."""
    if not mask.any():
        return None
    centre = tuple(int(round(c)) for c in center_of_mass(mask))
    return centre[0], centre[1], centre[2]


def tumor_center(labels: NDArray) -> Tuple[Tuple[int, int, int], str]:
    """
    Voxel centre of mass of the tumor core (CENTER_CORE_LABELS), falling back to the whole
    tumor (WHOLE_LABELS), any labelled voxel and the grid centre; (centre, source).
    """
    for source, mask in (
        ("core", np.isin(labels, CENTER_CORE_LABELS)),
        ("whole", np.isin(labels, WHOLE_LABELS)),
        ("any", labels != 0),
    ):
        centre = mask_center(mask)
        if centre is not None:
            return centre, source
    return tuple(int(n // 2) for n in labels.shape[:3]), "grid"


def pad_to_square(plane: NDArray, size: int, fill: float = 0.0) -> NDArray:
    """The plane centred on a size x size canvas (isotropic voxels: the views share a scale)."""
    if plane.shape[0] > size or plane.shape[1] > size:
        raise ValueError(f"plane {plane.shape} exceeds the canvas {size}.")
    top = (size - plane.shape[0]) // 2
    left = (size - plane.shape[1]) // 2
    canvas = np.full((size, size), fill, dtype=plane.dtype)
    canvas[top : top + plane.shape[0], left : left + plane.shape[1]] = plane
    return canvas


def oriented_slices(volume: NDArray, centre: Tuple[int, int, int]) -> Dict[str, NDArray]:
    """
    The three views through a voxel of an LAS volume, each on a square canvas of the grid's
    largest extent: axial and coronal in radiological view (the subject's right on the left,
    anterior / superior up), the sagittal with anterior on the left and superior up.
    """
    x, y, z = centre
    size = int(max(volume.shape[:3]))
    return {
        "axial": pad_to_square(np.rot90(volume[:, :, z]), size),
        "coronal": pad_to_square(np.rot90(volume[:, y, :]), size),
        "sagittal": pad_to_square(np.rot90(volume[x, :, :])[:, ::-1], size),
    }


def load_rows(patient_dir: Path, sessions: Sequence[Session], postop: Session) -> List[Row]:
    """The drawn sessions' segmentations and core volumes; sessions with a missing file are
    left out with a warning."""
    rows = []
    for session in sessions:
        session_dir = patient_dir / PROCESSED_DIRNAME / session.id
        files = [MODALITY_STRIPPED_SCHEMA.format(base_dir=session_dir, modality=m) for m in MODALITIES]
        files.append(TUMORSEG_CORRECTED_SCHEMA.format(base_dir=session_dir))
        missing = [str(f) for f in files if not f.is_file()]
        if missing:
            logger.warning(f"{patient_dir.name}/{session.id} ({session.timepoint}): left out, missing {missing}")
            continue
        labels, voxel_ml = load_segmentation(files[-1])
        if session.timepoint == TIMEPOINT_PREOP and (labels == LABEL_CAVITY).any():
            logger.warning(
                f"{patient_dir.name}/{session.id}: preop segmentation has {int((labels == LABEL_CAVITY).sum())} "
                f"cavity voxels (label {LABEL_CAVITY}); drawn as they are, not counted as core."
            )
        core_ml = float(np.isin(labels, core_labels(session)).sum()) * voxel_ml
        rows.append(Row(session, (session.date - postop.date).days, labels, core_ml))
    if not rows:
        raise FileNotFoundError(f"{patient_dir.name}: no drawable session below {patient_dir / PROCESSED_DIRNAME}.")
    return rows


def assign_centers(patient: str, rows: List[Row]) -> None:
    """
    Every row's slice centre: the centre of mass of the exam's own core (CENTER_CORE_LABELS),
    else the reference preop session's centre (``tumor_center``, on the same grid), else the
    exam's own ``tumor_center`` fallbacks.
    """
    reference = next((r for r in rows if r.session.reference), None)
    if reference is None:
        logger.warning(f"{patient}: the reference preop session is not drawn, exams without core use their own fallbacks.")
    else:
        reference_center, reference_source = tumor_center(reference.labels)
    for row in rows:
        centre = mask_center(np.isin(row.labels, CENTER_CORE_LABELS))
        if centre is not None:
            row.center, row.center_source = centre, "exam core centre"
        elif reference is not None and row.labels.shape == reference.labels.shape:
            row.center, row.center_source = reference_center, f"preop {reference_source} centre, no core in this exam"
        else:
            centre, source = tumor_center(row.labels)
            row.center, row.center_source = centre, f"exam {source} centre, no core in this exam"


def load_slices(patient_dir: Path, rows: List[Row], window_percentile: float) -> None:
    """The oriented background slices and display maxima of every row (one volume in memory
    at a time) and the overlay slices of its segmentation."""
    for row in rows:
        session_dir = patient_dir / PROCESSED_DIRNAME / row.session.id
        for modality in MODALITIES:
            path = MODALITY_STRIPPED_SCHEMA.format(base_dir=session_dir, modality=modality)
            image, _ = load_oriented(path)
            if image.shape != row.labels.shape:
                raise ValueError(f"{path}: shape {image.shape} differs from the segmentation's {row.labels.shape}.")
            row.vmax[modality] = display_window(image, window_percentile)
            row.slices[modality] = oriented_slices(image, row.center)
        row.overlays = oriented_slices(row.labels, row.center)


def video_kinds(session_dir: Path) -> List[Tuple[str, str]]:
    """(skull-stripped modality, button label) of the videos of an exam: VIDEO_KINDS and the
    first of DIFFUSION_KINDS whose file the exam has."""
    kinds = list(VIDEO_KINDS)
    for kind, label in DIFFUSION_KINDS:
        if MODALITY_STRIPPED_SCHEMA.format(base_dir=session_dir, modality=kind).is_file():
            kinds.append((kind, label))
            break
    return kinds


def video_frame(image: NDArray, labels: NDArray, z: int, vmax: float, title: str, scale: int) -> NDArray:
    """One frame (uint8 RGB, even width and height): axial slice z of the LAS volume in
    radiological view, plain and with the segmentation overlaid (SEG_COLORS,
    SEGMENTATION_ALPHA) side by side, upscaled by scale, the title at the top left."""
    plane = np.rot90(image[:, :, z])
    seg = np.rot90(labels[:, :, z])
    gray = np.clip(plane / vmax, 0.0, 1.0) if vmax > 0 else np.zeros_like(plane)
    plain = np.repeat(gray[..., None], 3, axis=-1)
    overlaid = plain.copy()
    for label, color in enumerate(SEG_COLORS[1:], start=1):
        voxels = seg == label
        overlaid[voxels] = (1.0 - SEGMENTATION_ALPHA) * overlaid[voxels] + SEGMENTATION_ALPHA * np.array(color[:3])
    gap = np.zeros((plain.shape[0], VIDEO_PANEL_GAP_PX, 3), dtype=plain.dtype)
    frame = Image.fromarray(np.rint(np.concatenate([plain, gap, overlaid], axis=1) * 255).astype(np.uint8))
    frame = frame.resize((frame.width * scale, frame.height * scale), Image.NEAREST)
    ImageDraw.Draw(frame).text((6, 4), title, fill=(255, 255, 255), font=ImageFont.load_default(size=VIDEO_FONT_PX))
    if frame.width % 2 or frame.height % 2:  # yuv420p needs even dimensions
        padded = Image.new("RGB", (frame.width + frame.width % 2, frame.height + frame.height % 2))
        padded.paste(frame, (0, 0))
        frame = padded
    return np.asarray(frame)


def encode_video(frames: Iterable[NDArray], outfile: Path, fps: int) -> int:
    """Encodes RGB frames to outfile with ffmpeg (H.264, VIDEO_CRF, fps frames per second,
    written to a temporary file first); returns the number of frames."""
    tmp_file = outfile.with_name(outfile.name + ".tmp.mp4")
    process = None
    n_frames = 0
    for frame in frames:
        if process is None:
            height, width = frame.shape[:2]
            process = subprocess.Popen(
                [
                    FFMPEG, "-y", "-loglevel", "error",
                    "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", str(fps), "-i", "-",
                    "-c:v", "libx264", "-preset", "veryfast", "-crf", str(VIDEO_CRF), "-pix_fmt", "yuv420p",
                    "-movflags", "+faststart", str(tmp_file),
                ],
                stdin=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        process.stdin.write(frame.tobytes())
        n_frames += 1
    if process is None:
        raise ValueError(f"{outfile}: no frames to encode.")
    process.stdin.close()
    error = process.stderr.read().decode(errors="replace")
    if process.wait() != 0:
        tmp_file.unlink(missing_ok=True)
        raise RuntimeError(f"ffmpeg failed for {outfile}: {error.strip()}")
    tmp_file.replace(outfile)
    return n_frames


def write_videos(
    patient_dir: Path,
    rows: Sequence[Row],
    video_dir: Path,
    window_percentile: float,
    fps: int,
    scale: int,
    override: bool,
) -> None:
    """The videos of every row (``video_kinds``, recorded in row.videos): the axial slices
    over the brain mask's extent plus VIDEO_MARGIN_SLICES, inferior to superior, windowed like
    the panels. Existing files are reused unless override."""
    video_dir.mkdir(parents=True, exist_ok=True)
    for row in rows:
        session_dir = patient_dir / PROCESSED_DIRNAME / row.session.id
        mask, _ = load_oriented(BRAIN_MASK_SCHEMA.format(base_dir=session_dir))
        z_mask = np.where(mask.any(axis=(0, 1)))[0]
        if z_mask.size == 0:
            raise ValueError(f"{session_dir}: empty brain mask.")
        z_range = range(
            max(int(z_mask[0]) - VIDEO_MARGIN_SLICES, 0),
            min(int(z_mask[-1]) + VIDEO_MARGIN_SLICES, mask.shape[2] - 1) + 1,
        )
        for kind, label in video_kinds(session_dir):
            outfile = video_dir / VIDEO_FILENAME.format(patient=patient_dir.name, session=row.session.id, kind=kind)
            row.videos[kind] = (label, outfile)
            if outfile.is_file() and not override:
                continue
            image, _ = load_oriented(MODALITY_STRIPPED_SCHEMA.format(base_dir=session_dir, modality=kind))
            if image.shape != row.labels.shape:
                raise ValueError(f"{session_dir}/{kind}: shape {image.shape} differs from the segmentation's {row.labels.shape}.")
            vmax = display_window(image, window_percentile)
            heading = f"{row.session.id} {row.session.timepoint} (day {row.day})   {label}"
            n_frames = encode_video(
                (video_frame(image, row.labels, z, vmax, f"{heading}   z={z}", scale) for z in z_range), outfile, fps
            )
            logger.info(f"{patient_dir.name}/{row.session.id}: wrote {outfile.name} ({n_frames} slices).")


def session_blocks(days: Sequence[int], max_gap: int = 1) -> List[Tuple[int, int]]:
    """(first, last) day of each run of days at most max_gap apart, for sorted days."""
    blocks: List[Tuple[int, int]] = []
    for day in days:
        if blocks and day - blocks[-1][1] <= max_gap:
            blocks[-1] = (blocks[-1][0], day)
        else:
            blocks.append((day, day))
    return blocks


def treatment_events(
    patient: str, rows: Sequence[Row], postop: Session, sheet_row: Optional[Dict[str, Optional[date]]]
) -> TreatmentEvents:
    """
    The resection day and the protocol's radiotherapy and chemotherapy days relative to the
    post-op scan (module docstring), cut at the last drawn session.
    """
    horizon = max(row.day for row in rows)
    resection_date = sheet_row.get(DATES_RESECTION_COLUMN) if sheet_row else None
    if resection_date is not None:
        resection = (resection_date - postop.date).days
    else:
        resection = -RESECTION_DAYS_BEFORE_POSTOP
        logger.warning(f"{patient}: no resection date in the sheet, resection drawn {RESECTION_DAYS_BEFORE_POSTOP} days before the postop scan.")
    rt_planning = sheet_row.get(DATES_RT_PLANNING_COLUMN) if sheet_row else None
    if rt_planning is not None:
        crt_start, crt_source = (rt_planning - postop.date).days, "sheet RT planning date"
    else:
        first_followup = next((r for r in rows if r.session.timepoint == TIMEPOINT_FOLLOWUP and r.day > 0), None)
        if first_followup is None:
            logger.warning(f"{patient}: no RT planning date and no followup after the postop scan, only the resection is drawn.")
            return TreatmentEvents(resection, [], [])
        crt_start, crt_source = first_followup.day, f"first followup {first_followup.session.id} (no RT planning date in the sheet)"
        logger.warning(f"{patient}: CRT anchored on the {crt_source}.")
    if resection >= crt_start:
        logger.warning(f"{patient}: the resection (day {resection}) is not before the CRT start (day {crt_start}).")
    rt_all = [crt_start + 7 * (i // FRACTIONS_PER_WEEK) + i % FRACTIONS_PER_WEEK for i in range(N_FRACTIONS)]
    concomitant_all = [crt_start + i for i in range(CONCOMITANT_DAYS)]
    adjuvant: List[int] = []
    number, start = 1, max(rt_all[-1], concomitant_all[-1]) + ADJUVANT_DELAY_DAYS
    while start <= horizon and number <= ADJUVANT_CYCLES:
        adjuvant.extend(d for d in range(start, start + ADJUVANT_DAYS_ON) if d <= horizon)
        number += 1
        start += ADJUVANT_CYCLE_DAYS
    rt_days = [d for d in rt_all if d <= horizon]
    chemo_days = sorted([d for d in concomitant_all if d <= horizon] + adjuvant)
    logger.info(
        f"{patient}: resection day {resection}, CRT start day {crt_start} ({crt_source}), {len(rt_days)} fractions, "
        f"{number - 1} adjuvant cycle(s) started by the last session (day {horizon})."
    )
    return TreatmentEvents(resection, rt_days, chemo_days)


def mark_treatment_events(ax: plt.Axes, events: TreatmentEvents) -> None:
    """The resection as a line, the fractions as lines split by whether chemotherapy is given
    the same day, the chemotherapy-only days as one band per contiguous block."""
    ax.axvline(events.resection_day, color=RESECTION_COLOR, linewidth=1.5, label="resection")
    rt_days = np.asarray(events.rt_days, dtype=np.int64)
    chemo_days = np.asarray(events.chemo_days, dtype=np.int64)
    with_chemo = np.isin(rt_days, chemo_days)
    for days, label, color in (
        (rt_days[with_chemo], "radio- + chemotherapy", "blue"),
        (rt_days[~with_chemo], "radiotherapy only", "purple"),
    ):
        for index, day in enumerate(days):
            ax.axvline(int(day), color=color, linewidth=0.6, alpha=0.4, label=label if index == 0 else None)
    chemo_only = sorted(int(d) for d in chemo_days[~np.isin(chemo_days, rt_days)])
    for index, (start, end) in enumerate(session_blocks(chemo_only)):
        ax.axvspan(
            start - 0.5, end + 0.5, color="green", alpha=0.25, linewidth=0,
            label="chemotherapy only" if index == 0 else None,
        )


def slice_text(view: str, center: Tuple[int, int, int]) -> str:
    return f"{'xyz'[VIEW_INDEX[view]]}={center[VIEW_INDEX[view]]}"


def field_name(session: Session, option: str) -> str:
    """Checkbox field name of a session's option, e.g. ses-20210303_segmentation_fake_enhancing."""
    return f"{session.id}_segmentation_{option.lower().replace(' ', '_')}"


def points_bbox(fig: plt.Figure, bbox: Bbox) -> Tuple[float, float, float, float]:
    """A display bbox (pixels) as PDF points (x0, y0, x1, y1, origin bottom left)."""
    x0, y0, x1, y1 = bbox.extents * 72.0 / fig.dpi
    return float(x0), float(y0), float(x1), float(y1)


def inches_from(fig: plt.Figure, ax: plt.Axes, x: float, y: float) -> Transform:
    """Transform of inch coordinates with the origin at the point (x, y) of the axes (axes
    fractions); it follows the axes through the layout passes and the figure's dpi (the PDF
    backend renders at 72)."""
    return fig.dpi_scale_trans + ScaledTranslation(x, y, ax.transAxes)


def clipped_texts(fig: plt.Figure, ax: plt.Axes, tolerance: float = 0.5) -> List[str]:
    """The texts of the axes that reach beyond its box (after the final layout pass),
    measured with the canvas renderer at the figure's dpi."""
    renderer = fig.canvas.get_renderer()
    box = ax.get_window_extent()
    clipped = []
    for t in ax.texts:
        e = t.get_window_extent(renderer)
        if e.x0 < box.x0 - tolerance or e.x1 > box.x1 + tolerance or e.y0 < box.y0 - tolerance or e.y1 > box.y1 + tolerance:
            clipped.append(t.get_text())
    return clipped


def draw_form_column(fig: plt.Figure, ax: plt.Axes) -> None:
    """The tick-box column of a session, left of its images: the label and one square per
    option with its name below each other, in inches from the axes' left edge and centred
    vertically (the squares, in option order, become the checkbox widgets in
    ``widget_fields``). The texts are clipped to the axes so that they do not take part in
    the layout; ``clipped_texts`` reports the ones a too narrow column cuts off."""
    ax.set_axis_off()
    trans = inches_from(fig, ax, 0.0, 0.5)
    block = FORM_LINE_PITCH_INCHES * (1 + len(SEGMENTATION_OPTIONS))
    y = (block - FORM_LINE_PITCH_INCHES) / 2  # centre of the label line
    ax.text(
        FORM_INDENT_INCHES, y, FORM_LABEL, transform=trans, clip_on=True,
        fontsize=FORM_FONT_SIZE, fontweight="bold", va="center", ha="left",
    )
    for option in SEGMENTATION_OPTIONS:
        y -= FORM_LINE_PITCH_INCHES
        ax.add_patch(Rectangle(
            (FORM_INDENT_INCHES, y - CHECKBOX_INCHES / 2), CHECKBOX_INCHES, CHECKBOX_INCHES, transform=trans,
            fill=False, linewidth=FORM_LINE_WIDTH, edgecolor="black",
        ))
        ax.text(
            FORM_INDENT_INCHES + CHECKBOX_INCHES + CHECKBOX_TEXT_GAP_INCHES, y, option, transform=trans, clip_on=True,
            fontsize=FORM_FONT_SIZE, va="center", ha="left",
        )


def draw_notes_box(ax: plt.Axes) -> None:
    """The notes box: a framed empty axes titled NOTES_TITLE (the text widget covers it)."""
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(FORM_LINE_WIDTH)
    ax.set_title(NOTES_TITLE, loc="left", fontsize=FORM_FONT_SIZE, fontweight="bold")


def draw_button_row(fig: plt.Figure, ax: plt.Axes, row: Row) -> None:
    """The video buttons of a row side by side, right of its panels, in inches from the axes'
    left edge and centred vertically (the Rectangle patches, in the order of row.videos, get
    the link annotations in ``widget_fields``); a grey note where the exam has no diffusion
    image."""
    ax.set_axis_off()
    trans = inches_from(fig, ax, 0.0, 0.5)
    x = BUTTON_INDENT_INCHES
    for label, _ in row.videos.values():
        ax.add_patch(Rectangle(
            (x, -BUTTON_HEIGHT_INCHES / 2), BUTTON_WIDTH_INCHES, BUTTON_HEIGHT_INCHES, transform=trans,
            facecolor="white", edgecolor=BUTTON_COLOR, linewidth=FORM_LINE_WIDTH,
        ))
        ax.text(
            x + BUTTON_WIDTH_INCHES / 2, 0.0, BUTTON_LABEL.format(label=label), transform=trans, clip_on=True,
            color=BUTTON_COLOR, fontsize=BUTTON_FONT_SIZE, ha="center", va="center",
        )
        x += BUTTON_WIDTH_INCHES + BUTTON_GAP_INCHES
    if len(row.videos) == len(VIDEO_KINDS):
        ax.text(
            x, 0.0, NO_DIFFUSION_TEXT, transform=trans, clip_on=True, color="grey",
            fontsize=BUTTON_FONT_SIZE - 1, ha="left", va="center",
        )


def appearance_stream(writer: PdfWriter, width: float, height: float, content: str, resources: Optional[DictionaryObject] = None) -> NameObject:
    """An appearance XObject of the given size and content stream, added to the writer."""
    stream = DecodedStreamObject()
    stream.set_data(content.encode("latin-1"))
    stream.update(
        {
            NameObject("/Type"): NameObject("/XObject"),
            NameObject("/Subtype"): NameObject("/Form"),
            NameObject("/BBox"): ArrayObject([FloatObject(0), FloatObject(0), FloatObject(width), FloatObject(height)]),
            NameObject("/Resources"): resources if resources is not None else DictionaryObject(),
        }
    )
    return writer._add_object(stream)


def add_form_fields(
    pdf_file: Path,
    fields: Sequence[FormField],
    link_kind: Optional[str] = None,
    attachments: Optional[Dict[str, Path]] = None,
) -> None:
    """
    Adds the widgets as an AcroForm to the single-page PDF written by matplotlib: checkboxes
    (on state CHECKBOX_ON_STATE, a ZapfDingbats check mark as 'on' appearance, nothing when
    off) and multiline text fields (Helvetica, appearance generated by the viewer via
    NeedAppearances), and the video buttons as link annotations: with link_kind 'embed' the
    videos (attachments: file name -> file) are embedded in the PDF and a button runs the
    JavaScript that opens its attachment (exportDataObject), with 'launch' a button is a
    Launch action of VIDEO_DIRNAME/<file name> relative to the PDF. The outlines are already
    drawn in the page content.
    """
    writer = PdfWriter(clone_from=str(pdf_file))
    page = writer.pages[0]
    helvetica = writer._add_object(
        DictionaryObject(
            {
                NameObject("/Type"): NameObject("/Font"),
                NameObject("/Subtype"): NameObject("/Type1"),
                NameObject("/BaseFont"): NameObject("/Helvetica"),
                NameObject("/Encoding"): NameObject("/WinAnsiEncoding"),
            }
        )
    )
    zapf = writer._add_object(
        DictionaryObject(
            {
                NameObject("/Type"): NameObject("/Font"),
                NameObject("/Subtype"): NameObject("/Type1"),
                NameObject("/BaseFont"): NameObject("/ZapfDingbats"),
            }
        )
    )
    zapf_resources = DictionaryObject({NameObject("/Font"): DictionaryObject({NameObject("/ZaDb"): zapf})})
    field_refs = ArrayObject()
    annotations = page.get("/Annots", ArrayObject())
    for form_field in fields:
        x0, y0, x1, y1 = form_field.rect
        width, height = x1 - x0, y1 - y0
        if form_field.kind == "link":
            if link_kind == "embed":
                action = DictionaryObject(
                    {
                        NameObject("/S"): NameObject("/JavaScript"),
                        NameObject("/JS"): TextStringObject(
                            f'this.exportDataObject({{cName: "{form_field.name}", nLaunch: 2}});'
                        ),
                    }
                )
            elif link_kind == "launch":
                action = DictionaryObject(
                    {
                        NameObject("/S"): NameObject("/Launch"),
                        NameObject("/F"): TextStringObject(f"{VIDEO_DIRNAME}/{form_field.name}"),
                    }
                )
            else:
                raise ValueError(f"link kind {link_kind!r} is not one of {VIDEO_LINK_KINDS}.")
            link = DictionaryObject(
                {
                    NameObject("/Type"): NameObject("/Annot"),
                    NameObject("/Subtype"): NameObject("/Link"),
                    NameObject("/Rect"): ArrayObject([FloatObject(v) for v in form_field.rect]),
                    NameObject("/Border"): ArrayObject([NumberObject(0), NumberObject(0), NumberObject(0)]),
                    NameObject("/F"): NumberObject(4),  # print
                    NameObject("/P"): page.indirect_reference,
                    NameObject("/A"): action,
                }
            )
            annotations.append(writer._add_object(link))
            continue
        widget = DictionaryObject(
            {
                NameObject("/Type"): NameObject("/Annot"),
                NameObject("/Subtype"): NameObject("/Widget"),
                NameObject("/T"): TextStringObject(form_field.name),
                NameObject("/Rect"): ArrayObject([FloatObject(v) for v in form_field.rect]),
                NameObject("/F"): NumberObject(4),  # print
                NameObject("/P"): page.indirect_reference,
            }
        )
        if form_field.kind == "checkbox":
            size = 0.8 * height
            check = f"q BT /ZaDb {size:.2f} Tf 0 g {0.12 * width:.2f} {0.18 * height:.2f} Td (4) Tj ET Q"
            widget.update(
                {
                    NameObject("/FT"): NameObject("/Btn"),
                    NameObject("/V"): NameObject("/Off"),
                    NameObject("/AS"): NameObject("/Off"),
                    NameObject("/AP"): DictionaryObject(
                        {
                            NameObject("/N"): DictionaryObject(
                                {
                                    NameObject(f"/{CHECKBOX_ON_STATE}"): appearance_stream(writer, width, height, check, zapf_resources),
                                    NameObject("/Off"): appearance_stream(writer, width, height, ""),
                                }
                            )
                        }
                    ),
                }
            )
        else:
            widget.update(
                {
                    NameObject("/FT"): NameObject("/Tx"),
                    NameObject("/Ff"): NumberObject(4096),  # multiline
                    NameObject("/V"): TextStringObject(""),
                    NameObject("/DA"): TextStringObject(f"/Helv {FIELD_FONT_POINTS} Tf 0 g"),
                }
            )
        reference = writer._add_object(widget)
        field_refs.append(reference)
        annotations.append(reference)
    page[NameObject("/Annots")] = annotations
    if link_kind == "embed":
        # pypdf appends to the /EmbeddedFiles name tree, which the PDF spec requires sorted:
        # Acrobat binary-searches it, so unsorted names make some buttons fail silently.
        for name, video_file in sorted((attachments or {}).items()):
            writer.add_attachment(name, video_file.read_bytes())
    writer.root_object[NameObject("/AcroForm")] = DictionaryObject(
        {
            NameObject("/Fields"): field_refs,
            NameObject("/NeedAppearances"): BooleanObject(True),
            NameObject("/DA"): TextStringObject(f"/Helv {FIELD_FONT_POINTS} Tf 0 g"),
            NameObject("/DR"): DictionaryObject({NameObject("/Font"): DictionaryObject({NameObject("/Helv"): helvetica, NameObject("/ZaDb"): zapf})}),
        }
    )
    writer.write(str(pdf_file))


def render(
    outfile: Path,
    modality: str,
    rows: Sequence[Row],
    events: Optional[TreatmentEvents],
    yscale: Optional[Tuple[float, float]] = None,
    link_kind: Optional[str] = None,
) -> int:
    """The figure of one modality at the inch positions of ``Layout``: the colour legend,
    the slice grid with a tick-box column per session and, with link_kind (VIDEO_LINK_KINDS)
    and videos in the rows, the video buttons right of it, the legend again, the notes box
    and the core-volume panel; returns the number of annotations added."""
    buttons = link_kind is not None and any(row.videos for row in rows)
    layout = Layout(len(rows), buttons)
    fig = plt.figure(figsize=(layout.width, layout.height))
    handles = [Patch(color=c, label=l) for c, l in zip(SEG_COLORS[1:], SEG_LABELS)]
    for y in (layout.top_legend_center, layout.bottom_legend_center):
        fig.legend(
            handles=handles, loc="center", bbox_to_anchor=(0.5, y / layout.height),
            ncol=len(SEG_LABELS), fontsize=10, frameon=False,
        )
    if buttons:
        fig.text(
            1.0 - MARGIN_INCHES / layout.width, layout.top_legend_center / layout.height, VIDEO_HINT[link_kind],
            ha="right", va="center", fontsize=8, color="grey",
        )
    name = MODALITIES[modality]
    form_axes = []
    button_axes = []
    for i, row in enumerate(rows):
        y0 = layout.row_bottom(i)
        form_axes.append(layout.axes(fig, MARGIN_INCHES, y0, FORM_COLUMN_INCHES, PANEL_INCHES))
        if buttons:
            button_axes.append(layout.axes(fig, layout.button_left, y0, BUTTON_COLUMN_INCHES, PANEL_INCHES))
        for j, view in enumerate(VIEWS):
            plain = layout.axes(fig, layout.panel_left(2 * j), y0, PANEL_INCHES, PANEL_INCHES)
            overlaid = layout.axes(fig, layout.panel_left(2 * j + 1), y0, PANEL_INCHES, PANEL_INCHES)
            for ax in (plain, overlaid):
                ax.imshow(row.slices[modality][view], cmap="gray", vmin=0.0, vmax=row.vmax[modality], interpolation="none")
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)
            overlaid.imshow(
                row.overlays[view], cmap=SEG_CMAP, vmin=-0.5, vmax=len(SEG_COLORS) - 0.5,
                alpha=SEGMENTATION_ALPHA, interpolation="none",
            )
            plain.text(
                0.02, 0.02, slice_text(view, row.center), transform=plain.transAxes, color="white", fontsize=9, va="bottom"
            )
            if i == 0:
                plain.set_title(f"{view} {name}", fontsize=12, fontweight="bold", pad=8)
                overlaid.set_title(f"{view} {name} + segmentation", fontsize=12, fontweight="bold", pad=8)
            if j == 0:
                plain.set_ylabel(f"{row.session.id}\n{row.session.timepoint} (day {row.day})", fontsize=12, fontweight="bold")
    notes_ax = layout.axes(fig, MARGIN_INCHES, layout.notes_bottom, layout.width - 2 * MARGIN_INCHES, NOTES_INCHES)
    bottom = layout.axes(
        fig, MARGIN_INCHES + VOLUME_YLABEL_INCHES, MARGIN_INCHES + VOLUME_XLABEL_INCHES,
        layout.width - 2 * MARGIN_INCHES - VOLUME_YLABEL_INCHES, VOLUME_PANEL_INCHES - VOLUME_XLABEL_INCHES - VOLUME_LEGEND_INCHES,
    )
    days = [row.day for row in rows]
    volumes = [row.core_ml for row in rows]
    bottom.plot(days, volumes, "o-", color=CORE_CURVE_COLOR, markersize=6, label=CORE_CURVE_LABEL)
    for row in rows:
        bottom.annotate(
            row.session.id, (row.day, row.core_ml), xytext=(0, 7), textcoords="offset points",
            ha="center", va="bottom", fontsize=8, color=CORE_CURVE_COLOR,
        )
    low, high = bottom.get_ylim()
    bottom.set_ylim(low, high + 0.08 * (high - low))  # room for the label above the highest point
    if events is not None:
        mark_treatment_events(bottom, events)
    first, last = min(days), max(days)
    margin = 0.03 * max(last - first, 1)  # room for the session labels at both ends
    bottom.set_xlim(first - margin, last + margin)
    if yscale is not None:
        bottom.set_ylim(*yscale)
    bottom.set_xlabel("day relative to the post-op scan", fontsize=12)
    bottom.set_ylabel("core volume [mL]", fontsize=12)
    bottom.grid(alpha=0.3)
    bottom.legend(fontsize=9, loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=5, frameon=False)
    for ax in form_axes:
        draw_form_column(fig, ax)
    for ax, row in zip(button_axes, rows):
        draw_button_row(fig, ax, row)
    draw_notes_box(notes_ax)
    fig.savefig(str(outfile), format="pdf")
    clipped = [t for ax in form_axes + button_axes for t in clipped_texts(fig, ax)]
    if clipped:
        logger.warning(f"{outfile.name}: {len(clipped)} form / button text(s) cut off by their column width, e.g. '{clipped[0]}'.")
    fields = widget_fields(fig, form_axes, button_axes, notes_ax, rows)
    plt.close(fig)
    attachments = {video_file.name: video_file for row in rows for _, video_file in row.videos.values()}
    add_form_fields(outfile, fields, link_kind if buttons else None, attachments)
    return len(fields)

def widget_fields(
    fig: plt.Figure,
    form_axes: Sequence[plt.Axes],
    button_axes: Sequence[plt.Axes],
    notes_ax: plt.Axes,
    rows: Sequence[Row],
) -> List[FormField]:
    """The annotation rectangles of the drawn outlines: every Rectangle patch of a form column
    is a checkbox (option order), every one of a button row a video link (order of
    row.videos), the notes axes the text field."""
    fields: List[FormField] = []
    for ax, row in zip(form_axes, rows):
        boxes = [p for p in ax.patches if isinstance(p, Rectangle)]
        for option, box in zip(SEGMENTATION_OPTIONS, boxes):
            fields.append(FormField("checkbox", field_name(row.session, option), points_bbox(fig, box.get_window_extent())))
    for ax, row in zip(button_axes, rows):
        boxes = [p for p in ax.patches if isinstance(p, Rectangle)]
        for (_, video_file), box in zip(row.videos.values(), boxes):
            fields.append(FormField("link", video_file.name, points_bbox(fig, box.get_window_extent())))
    fields.append(FormField("text", NOTES_FIELD_NAME, points_bbox(fig, notes_ax.get_window_extent())))
    return fields


def postop_session(patient: str, sessions: Sequence[Session], sheet_row: Optional[Dict[str, Optional[date]]]) -> Session:
    """The day-0 session: the only postop one, with several the one on the sheet's postop
    date (else the first) with a warning."""
    postops = [s for s in sessions if s.timepoint == TIMEPOINT_POSTOP]
    if not postops:
        raise ValueError(f"{patient}: no session labelled {TIMEPOINT_POSTOP}.")
    if len(postops) == 1:
        return postops[0]
    sheet_date = sheet_row.get(DATES_POSTOP_COLUMN) if sheet_row else None
    chosen = next((s for s in postops if s.date == sheet_date), postops[0])
    logger.warning(f"{patient}: {len(postops)} postop sessions {[s.id for s in postops]}, day 0 is {chosen.id}.")
    return chosen


def summarize_patient(
    patient_dir: Path,
    outdir: Path,
    sheet: Dict[str, Dict[str, Optional[date]]],
    window_percentile: float,
    yscale: Optional[Tuple[float, float]],
    video_dir: Optional[Path] = None,
    link_kind: str = VIDEO_LINK_KINDS[0],
    fps: int = DEFAULT_FPS,
    video_scale: int = DEFAULT_VIDEO_SCALE,
    override_videos: bool = False,
) -> None:
    """Both figures of one patient and, with video_dir, the videos of its exams."""
    patient = patient_dir.name
    subject, sessions = read_sessions(patient_dir / PROCESSED_DIRNAME / SESSION_LABELS_FILENAME)
    sheet_row = sheet.get(subject)
    if sheet_row is None:
        logger.warning(f"{patient}: subject {subject} not in the dates sheet, protocol anchored on the sessions.")
    left_out = [s.id for s in sessions if s.timepoint not in DRAWN_TIMEPOINTS]
    if left_out:
        logger.info(f"{patient}: {len(left_out)} session(s) left out ({', '.join(left_out)}: not preop/postop/followup).")
    sessions = [s for s in sessions if s.timepoint in DRAWN_TIMEPOINTS]
    postop = postop_session(patient, sessions, sheet_row)
    rows = load_rows(patient_dir, sessions, postop)
    assign_centers(patient, rows)
    load_slices(patient_dir, rows, window_percentile)
    events = treatment_events(patient, rows, postop, sheet_row)
    for row in rows:
        x, y, z = row.center
        logger.info(
            f"{patient}/{row.session.id} {row.session.timepoint} (day {row.day}): core {row.core_ml:.2f} mL, "
            f"slices x={x} y={y} z={z} ({row.center_source})"
        )
    if video_dir is not None:
        write_videos(patient_dir, rows, video_dir, window_percentile, fps, video_scale, override_videos)
    for modality in MODALITIES:
        outfile = outdir / OUTPUT_FILENAME.format(patient=patient, modality=modality)
        n_fields = render(outfile, modality, rows, events, yscale, link_kind if video_dir is not None else None)
        logger.info(f"{patient}: wrote {outfile} ({n_fields} annotations)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-data_dir", type=str, default=str(DEFAULT_DATA_DIR), help="Directory with one directory per patient.")
    parser.add_argument("-patient_glob", type=str, default=DEFAULT_PATIENT_DIR_GLOB, help="Glob of the patient directory names in -data_dir.")
    parser.add_argument("-patients", type=str, nargs="+", default=None, help="Patient directory names to draw (e.g. sub-GR008). All if omitted.")
    parser.add_argument("-outdir", type=str, default=None, help=f"Output directory of the PDFs (default <data_dir>/{DEFAULT_OUTDIR_NAME}).")
    parser.add_argument("-dates_file", type=str, default=None, help=f"Surgery dates sheet (default <data_dir>/{DATES_FILENAME} if it exists).")
    parser.add_argument("-window_percentile", type=float, default=DEFAULT_WINDOW_PERCENTILE, help=f"Display window: black at 0, white at this percentile of the non-zero voxels (default {DEFAULT_WINDOW_PERCENTILE:g}).")
    parser.add_argument("-yscale", type=float, nargs=2, default=None, metavar=("LOW", "HIGH"), help="y limits in mL of the volume panel (default: autoscale).")
    parser.add_argument("-no_videos", action="store_true", help="Write no videos and draw no video buttons.")
    parser.add_argument("-video_links", type=str, choices=VIDEO_LINK_KINDS, default=VIDEO_LINK_KINDS[0], help="How a button reaches its video: 'embed' embeds the mp4 in the PDF (opened by Acrobat Reader, listed in the attachment panel elsewhere), 'launch' opens the file in the videos folder next to the PDF (default embed).")
    parser.add_argument("-override_videos", action="store_true", help="Rewrite existing mp4 files (default: reuse them).")
    parser.add_argument("-fps", type=int, default=DEFAULT_FPS, help=f"Slices per second of the videos (default {DEFAULT_FPS}).")
    parser.add_argument("-video_scale", type=int, default=DEFAULT_VIDEO_SCALE, help=f"Upscaling of the slices in the videos (default {DEFAULT_VIDEO_SCALE}).")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    data_dir = Path(args.data_dir)
    patient_dirs = sorted(p for p in data_dir.glob(args.patient_glob) if p.is_dir())
    if args.patients is not None:
        patient_dirs = [p for p in patient_dirs if p.name in args.patients]
        unknown = sorted(set(args.patients) - {p.name for p in patient_dirs})
        if unknown:
            raise FileNotFoundError(f"patient directories {unknown} not found in {data_dir}.")
    logger.info(f"Found {len(patient_dirs)} patients in {data_dir}.")
    dates_file = Path(args.dates_file) if args.dates_file else data_dir / DATES_FILENAME
    if dates_file.is_file():
        sheet = read_sheet_dates(dates_file)
        logger.info(f"Read dates of {len(sheet)} subjects from {dates_file}.")
    elif args.dates_file:
        raise FileNotFoundError(f"Dates file {dates_file} not found.")
    else:
        sheet = {}
        logger.warning(f"No {dates_file}: resection and CRT start are placed by the session rule for every patient.")
    outdir = Path(args.outdir) if args.outdir else data_dir / DEFAULT_OUTDIR_NAME
    outdir.mkdir(parents=True, exist_ok=True)
    yscale = tuple(args.yscale) if args.yscale else None
    video_dir = None if args.no_videos else outdir / VIDEO_DIRNAME
    failed = []
    for patient_dir in patient_dirs:
        try:
            summarize_patient(
                patient_dir, outdir, sheet, args.window_percentile, yscale,
                video_dir, args.video_links, args.fps, args.video_scale, args.override_videos,
            )
        except Exception:
            logger.exception(f"{patient_dir.name}: figure failed, skipping.")
            failed.append(patient_dir.name)
    logger.info(f"Done: {len(patient_dirs) - len(failed)} of {len(patient_dirs)} patients drawn, failed: {failed or 'none'}.")
