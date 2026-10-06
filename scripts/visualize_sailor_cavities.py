"""Cavity visualization of all processed SAILOR patients.

Runs scripts/visualize_cavity.py for every patient in the processed SAILOR directory, comparing
the pre-op tumor core (necrosis / non-enhancing + enhancing tumor of the pre-op segmentation)
with the resection cavity of the post-op segmentation in pre-op space, on the pre-op T1c. The
single pages are combined into one PDF (one page per patient) and deleted afterwards.

The pre-op and post-op session of a patient are taken from the session labels table (they
are not always ses-01 and ses-02). Patients without a pre-op or post-op session, or whose
inputs are not processed (yet), are skipped with a warning.

Usage:
    python visualize_sailor_cavities.py
    python visualize_sailor_cavities.py -patients sub-01 sub-03 -outfile /tmp/cavities.pdf
"""

import argparse
import csv
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

from loguru import logger
from pypdf import PdfWriter

from predict_gbm.utils.constants import (
    MODALITY_STRIPPED_SCHEMA,
    RECURRENCE_SCHEMA,
    TUMORSEG_SCHEMA,
)

VISUALIZE_CAVITY_SCRIPT = Path(__file__).resolve().parent / "visualize_cavity.py"
# Tumor core of the pre-op segmentation: 1 necrosis / non-enhancing, 3 enhancing tumor.
PREOP_CORE_LABELS = [1, 3]
# Resection cavity of the post-op segmentation.
POSTOP_CAVITY_LABELS = [4]
# Session labels (column "label" of the session labels table) of the two exams.
PREOP_LABEL = "preop"
POSTOP_LABEL = "postop"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cavity visualization (visualize_cavity.py) of all SAILOR patients as one PDF."
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
        help="Session labels table, defines the pre-op and post-op session of each patient.",
    )
    parser.add_argument(
        "-outfile",
        type=Path,
        default=Path("/mnt/Drive4/lucas/SAILOR/cavities.pdf"),
        help="Combined output PDF, one page per patient.",
    )
    parser.add_argument(
        "-patients",
        type=str,
        nargs="+",
        default=None,
        help="Patients to visualize (default: all patient directories in data_dir).",
    )
    return parser.parse_args()


def read_session_labels(session_labels_file: Path) -> Dict[str, Dict[str, List[str]]]:
    """Reads the session labels table as {patient: {label: [sessions, in table order]}}."""
    sessions: Dict[str, Dict[str, List[str]]] = {}
    with open(session_labels_file, newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            labels = sessions.setdefault(row["subject"], {})
            labels.setdefault(row["label"], []).append(row["session"])
    return sessions


def find_session(patient_id: str, labels: Dict[str, List[str]], label: str) -> Optional[str]:
    """First session of a patient with the given label, None (and a warning) if there is none."""
    sessions = labels.get(label, [])
    if not sessions:
        logger.warning(f"{patient_id}: no {label} session in the session labels, skipping.")
        return None
    if len(sessions) > 1:
        logger.info(f"{patient_id}: {len(sessions)} {label} sessions {sessions}, using the first.")
    return sessions[0]


def resolve_inputs(
    patient_dir: Path, labels: Dict[str, List[str]]
) -> Optional[Dict[str, Path]]:
    """
    Inputs of visualize_cavity.py for a patient: T1c and tumor segmentation of the pre-op exam
    and the tumor segmentation of the post-op exam in pre-op space. None if a session or a
    file is missing.
    """
    patient_id = patient_dir.name
    preop = find_session(patient_id, labels, PREOP_LABEL)
    postop = find_session(patient_id, labels, POSTOP_LABEL) if preop is not None else None
    if preop is None or postop is None:
        return None
    inputs = {
        "t1c": MODALITY_STRIPPED_SCHEMA.format(base_dir=patient_dir / preop, modality="t1c"),
        "seg1": TUMORSEG_SCHEMA.format(base_dir=patient_dir / preop),
        "seg2": RECURRENCE_SCHEMA.format(base_dir=patient_dir / postop),
    }
    missing = [str(f) for f in inputs.values() if not f.is_file()]
    if missing:
        logger.warning(f"{patient_id}: missing {missing}, skipping.")
        return None
    logger.info(f"{patient_id}: pre-op {preop}, post-op {postop}.")
    return inputs


def run_visualize_cavity(patient_id: str, inputs: Dict[str, Path], page_file: Path) -> bool:
    """Writes the page of a patient with visualize_cavity.py, returns whether it succeeded."""
    command = [sys.executable, str(VISUALIZE_CAVITY_SCRIPT)]
    for name, file in inputs.items():
        command += [f"--{name}", str(file)]
    command += ["--labels1", *map(str, PREOP_CORE_LABELS)]
    command += ["--labels2", *map(str, POSTOP_CAVITY_LABELS)]
    command += ["--id", patient_id, "--outfile", str(page_file)]
    result = subprocess.run(command, capture_output=True, text=True)
    for line in result.stderr.splitlines():
        logger.warning(f"{patient_id}: {line}")
    if result.returncode != 0 or not page_file.is_file():
        logger.error(f"{patient_id}: visualize_cavity.py failed (exit code {result.returncode}).")
        return False
    logger.debug(f"{patient_id}:\n{result.stdout}")
    return True


def merge_pages(page_files: List[Path], outfile: Path) -> None:
    """Combines the single-page PDFs into one PDF, in the given order."""
    writer = PdfWriter()
    for page_file in page_files:
        writer.append(str(page_file))
    outfile.parent.mkdir(parents=True, exist_ok=True)
    with open(outfile, "wb") as f:
        writer.write(f)
    writer.close()


def main() -> None:
    args = parse_args()
    session_labels = read_session_labels(args.session_labels_tsv)
    patient_ids = args.patients or sorted(d.name for d in args.data_dir.iterdir() if d.is_dir())
    logger.info(f"Visualizing {len(patient_ids)} patients of {args.data_dir}.")

    args.outfile.parent.mkdir(parents=True, exist_ok=True)
    # The single pages only live in this directory, which is deleted when the block is left.
    with tempfile.TemporaryDirectory(prefix="cavity_pages_", dir=args.outfile.parent) as tmp:
        page_files = []
        for patient_id in patient_ids:
            labels = session_labels.get(patient_id, {})
            inputs = resolve_inputs(args.data_dir / patient_id, labels)
            if inputs is None:
                continue
            page_file = Path(tmp) / f"{patient_id}.pdf"
            if run_visualize_cavity(patient_id, inputs, page_file):
                page_files.append(page_file)
        if not page_files:
            logger.error("No page was written, no output PDF.")
            sys.exit(1)
        merge_pages(page_files, args.outfile)
    skipped = len(patient_ids) - len(page_files)
    logger.info(f"Saved {len(page_files)} pages to {args.outfile} ({skipped} patients skipped).")


if __name__ == "__main__":
    main()
