"""
Converts the Sectra CD DICOM exports in /home/home/marco/data/dicoms_v2 to NIfTI, using the
same layout as the already converted dataset that scripts/preprocess_marco_healthy.py consumes:

    <outdir>/sub-<ID>/ses-<YYYYMMDD>/sub-<ID>_ses-<YYYYMMDD>_sequ-<SeriesNumber>_<modality>.nii.gz

with a dcm2niix json sidecar (and .bval/.bvec for diffusion series) next to every nifti.

Each export directory is one patient (the directory name is the subject id) and holds a
DICOMDIR-style tree under DICOM/ whose files carry no extension. The CD import merges every
exam of the patient into a single study whose StudyDate is the day the CD was imported, so
sessions are formed from the AcquisitionDate of the series instead (this is also what the
session dates of the existing dataset correspond to).

Series are grouped by SeriesInstanceUID and converted one by one with the codebase's
dicom_to_nifti. Non-MR objects (presentation states, reports) and derived reconstructions
(localizers/surveys, MPR/MIP/minIP, screen saves) are skipped unless -convert_all is given.
The modality suffix is derived from the series description (and ImageType / contrast agent
tag), following the suffixes of the existing dataset: t1, t1c, t2, t2star, flair, dwi, dti,
adc, perf, angio; anything unrecognised becomes "mr". Every decision is logged and written to
<outdir>/conversion_manifest.csv.

Some DICOM series contain more than one volume (e.g. Philips DWI series carry a derived ADC
volume): dcm2niix then writes extra files with a postfix such as "_ADC", which are kept
beside the primary nifti under the same name plus that postfix
(..._sequ-701_dwi_ADC.nii.gz). Identifying fields (patient name/id/birth date, institution,
...) are removed from the sidecars and PatientID is replaced by the subject id.

Must be run in an environment where the predict_gbm package imports (e.g. the brainles env).
"""

import csv
import json
import re
import shutil
import tempfile
import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pydicom
from loguru import logger

from predict_gbm.preprocessing import dicom_to_nifti

DEFAULT_INPUT_DIR = "/home/home/marco/data/dicoms_v2"
DEFAULT_OUTDIR = "/mnt/Drive4/lucas/niftis_v2"

# Subdirectory of a CD export holding the DICOM objects (the rest is viewer software).
DICOM_SUBDIR = "DICOM"

# The CD viewer exports contain files that are not DICOM at all; pydicom is forced to read
# them and yields datasets without these basic attributes, which identify a real image series.
REQUIRED_TAGS = ("SeriesInstanceUID", "Modality")

# ImageType values that mark derived reconstructions rather than acquired images.
DERIVED_IMAGE_TYPES = {
    "SECONDARY",
    "SCREEN SAVE",
    "REFORMATTED",
    "LOCALIZER",
    "MPR",
    "MAX_IP",
    "MIN_IP",
    "PROJECTION IMAGE",
}
# Series descriptions of scout/planning scans and reconstructions that are not worth
# converting even when their ImageType claims ORIGINAL (e.g. the Philips MPR planning scans).
SKIP_DESCRIPTION_PATTERN = re.compile(
    r"survey|localizer|localiser|scout|screen save|\bmpr\b|\bmip\b|minip|"
    r"reconstruction|smartplan|awplan",
    re.IGNORECASE,
)

# Modality classification from the series description, evaluated in this order; the first
# matching rule wins. t1 vs t1c is decided afterwards by the contrast rules below.
MODALITY_RULES: List[Tuple[str, re.Pattern]] = [
    ("adc", re.compile(r"adc|apparent diffusion", re.IGNORECASE)),
    ("dti", re.compile(r"dti", re.IGNORECASE)),
    ("dwi", re.compile(r"dwi|diff|\bb ?0\b|\bb ?1000\b|trace", re.IGNORECASE)),
    ("perf", re.compile(r"perf|dsc|dce", re.IGNORECASE)),
    ("angio", re.compile(r"tof|angio|\bmra\b|\bpca\b", re.IGNORECASE)),
    ("t2star", re.compile(r"haemo|hemo|\bham\b|t2\*|t2star|t2_star|\bswi\b|ffe.*t2|t2.*ffe", re.IGNORECASE)),
    ("flair", re.compile(r"flair", re.IGNORECASE)),
    ("t1", re.compile(r"t1|mprage|tfe|spgr|bravo|\btfl\b", re.IGNORECASE)),
    ("t2", re.compile(r"t2|tse|fse|propeller|cube", re.IGNORECASE)),
]
# ImageType values that identify ADC maps regardless of the description.
ADC_IMAGE_TYPES = {"ADC", "EADC"}
# Contrast administration in the series description ("KM" = Kontrastmittel).
CONTRAST_DESCRIPTION_PATTERN = re.compile(
    r"\bkm\b|\+ ?c\b|\bgd\b|gado|contrast|\bce\b|\bpost\b", re.IGNORECASE
)
# Tags carrying the b-value: standard, Philips private, GE private (first element).
BVALUE_TAGS = [(0x0018, 0x9087), (0x2001, 0x1003), (0x0043, 0x1039)]

# Sidecar keys that identify the patient, the exam or the site and are removed from the
# json files; PatientID is then set to the subject id like in the existing dataset.
PHI_SIDECAR_KEYS = {
    "PatientName",
    "PatientID",
    "PatientBirthDate",
    "PatientAge",
    "PatientSex",
    "PatientWeight",
    "PatientSize",
    "AccessionNumber",
    "ReferringPhysicianName",
    "PerformingPhysicianName",
    "OperatorsName",
    "InstitutionName",
    "InstitutionAddress",
    "InstitutionalDepartmentName",
    "StationName",
    "DeviceSerialNumber",
    "StudyID",
}

MANIFEST_COLUMNS = [
    "subject",
    "session",
    "series_number",
    "series_description",
    "image_type",
    "n_files",
    "decision",
    "modality_or_reason",
    "output",
]


@dataclass
class Series:
    """One DICOM series of a patient export, with the header fields used for the decisions."""

    uid: str
    number: str
    description: str
    protocol: str
    dicom_modality: str
    image_type: Tuple[str, ...]
    acquisition_date: Optional[str]
    date_source: str
    contrast_agent: str
    has_diffusion_weighting: bool
    files: List[Path] = field(default_factory=list)

    @property
    def label(self) -> str:
        return f"series {self.number} '{self.description}' ({len(self.files)} files)"


def read_bvalue(ds: pydicom.Dataset) -> Optional[float]:
    """Returns the b-value of a DICOM slice from the standard or vendor tags, or None."""
    for tag in BVALUE_TAGS:
        element = ds.get(tag)
        if element is None:
            continue
        value = element.value
        try:
            if isinstance(value, (list, tuple, pydicom.multival.MultiValue)):
                value = value[0]
            if isinstance(value, bytes):
                value = value.decode(errors="ignore").strip().split("\\")[0]
            return float(value)
        except (TypeError, ValueError, IndexError):
            continue
    return None


def series_date(ds: pydicom.Dataset) -> Tuple[Optional[str], str]:
    """
    Returns the exam date of a series as (YYYYMMDD, source tag). The CD import merges all exams
    into one study dated at import time, so AcquisitionDate is preferred; SeriesDate and
    ContentDate are tried next and StudyDate is the last resort.
    """
    for tag in ("AcquisitionDate", "SeriesDate", "ContentDate", "StudyDate"):
        value = str(ds.get(tag, "") or "").strip()
        if re.fullmatch(r"\d{8}", value):
            return value, tag
    return None, ""


def index_series(patient_dir: Path) -> List[Series]:
    """Reads the headers of every DICOM object of a patient export and groups them by series."""
    dicom_root = patient_dir / DICOM_SUBDIR
    if not dicom_root.is_dir():
        logger.warning(f"{patient_dir}: no {DICOM_SUBDIR}/ subdirectory, scanning whole directory.")
        dicom_root = patient_dir

    series_by_uid: Dict[str, Series] = {}
    n_unreadable = 0
    for f in sorted(p for p in dicom_root.rglob("*") if p.is_file()):
        try:
            ds = pydicom.dcmread(f, stop_before_pixels=True, force=True)
        except Exception as e:  # noqa: BLE001 - any unreadable file is skipped
            logger.debug(f"{f}: not readable as DICOM ({e}).")
            n_unreadable += 1
            continue
        if any(not str(ds.get(tag, "") or "").strip() for tag in REQUIRED_TAGS):
            logger.debug(f"{f}: no SeriesInstanceUID/Modality, not an image object.")
            n_unreadable += 1
            continue

        uid = str(ds.SeriesInstanceUID)
        series = series_by_uid.get(uid)
        if series is None:
            date, date_source = series_date(ds)
            series = Series(
                uid=uid,
                number=str(ds.get("SeriesNumber", "") or "").strip(),
                description=str(ds.get("SeriesDescription", "") or "").strip(),
                protocol=str(ds.get("ProtocolName", "") or "").strip(),
                dicom_modality=str(ds.get("Modality", "") or "").strip(),
                image_type=tuple(str(v).strip() for v in ds.get("ImageType", [])),
                acquisition_date=date,
                date_source=date_source,
                contrast_agent=str(ds.get("ContrastBolusAgent", "") or "").strip(),
                has_diffusion_weighting=False,
            )
            series_by_uid[uid] = series
        series.files.append(f)
        bvalue = read_bvalue(ds)
        if bvalue is not None and bvalue > 0:
            series.has_diffusion_weighting = True

    if n_unreadable:
        logger.info(f"{patient_dir.name}: {n_unreadable} non-image file(s) ignored.")

    def sort_key(s: Series):
        return (s.acquisition_date or "", int(s.number) if s.number.isdigit() else 0, s.number)

    return sorted(series_by_uid.values(), key=sort_key)


def unconvertible_reason(series: Series) -> Optional[str]:
    """Returns why a series can never be converted (not an MR image, no date), or None."""
    if series.dicom_modality != "MR":
        return f"non-MR object (Modality {series.dicom_modality})"
    if series.acquisition_date is None:
        return "no acquisition date"
    return None


def derived_reason(series: Series) -> Optional[str]:
    """Returns why a series is a derived/scout series that is skipped by default, or None."""
    derived = DERIVED_IMAGE_TYPES.intersection(series.image_type)
    if derived:
        return f"derived reconstruction (ImageType {'/'.join(sorted(derived))})"
    match = SKIP_DESCRIPTION_PATTERN.search(f"{series.description} {series.protocol}")
    if match:
        return f"scout/planning/reconstruction series ('{match.group(0)}' in description)"
    return None


def classify_modality(series: Series) -> str:
    """Maps a series to one of the modality suffixes of the existing dataset ('mr' if unknown)."""
    if ADC_IMAGE_TYPES.intersection(series.image_type):
        return "adc"

    text = f"{series.description} {series.protocol}"
    modality = "mr"
    for candidate, pattern in MODALITY_RULES:
        if pattern.search(text):
            modality = candidate
            break

    if modality == "mr" and series.has_diffusion_weighting:
        # e.g. Philips "Reg - Bewegungscorr." (motion corrected copy of the DWI series)
        return "dwi"
    if modality == "t1":
        if series.contrast_agent or CONTRAST_DESCRIPTION_PATTERN.search(text):
            return "t1c"
    return modality


def to_subject_label(name: str) -> str:
    """Strips everything but letters and digits so that sub-<label>_ses-<date> stays parseable."""
    return re.sub(r"[^A-Za-z0-9]", "", name)


def sanitize_sidecar(sidecar: Path, subject_id: str) -> None:
    """Removes identifying fields from a dcm2niix json sidecar and sets PatientID to the subject id."""
    with sidecar.open("r", encoding="utf-8") as f:
        meta = json.load(f)
    removed = [k for k in meta if k in PHI_SIDECAR_KEYS]
    for key in removed:
        del meta[key]
    meta["PatientID"] = subject_id
    with sidecar.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    if removed:
        logger.debug(f"{sidecar.name}: removed {sorted(removed)} from sidecar.")


def convert_series(
    series: Series, session_dir: Path, basename: str, subject_id: str, log_dir: Path
) -> Path:
    """
    Converts one series into session_dir/<basename>.nii.gz (+ sidecars). The DICOM files are
    staged into a temporary directory since dicom_to_nifti expects one series per directory,
    and dcm2niix writes into a temporary output directory so that additional volumes it splits
    off (postfixed files such as "_ADC") can be kept beside the primary nifti under
    <basename><postfix>.*. The dcm2niix log goes to log_dir. Returns the primary nifti.
    """
    with tempfile.TemporaryDirectory(prefix="convert_marco_") as tmp:
        stage_dir = Path(tmp) / "dicom"
        stage_dir.mkdir()
        for i, f in enumerate(series.files):
            shutil.copy2(f, stage_dir / f"{i:06d}.dcm")

        conversion_dir = Path(tmp) / "nifti"
        conversion_dir.mkdir()
        primary = conversion_dir / f"{basename}.nii.gz"
        try:
            dicom_to_nifti(input_dir=stage_dir, outfile=primary)
        except FileExistsError as e:
            # dcm2niix split the series into several volumes; the primary one exists and the
            # others keep their postfix (dicom_to_nifti only strips postfixes when unambiguous).
            logger.warning(f"{basename}: {e}")
        finally:
            # Keep the dcm2niix log also when the conversion failed.
            conversion_log = conversion_dir / f"{basename}_conversion.log"
            if conversion_log.exists():
                log_dir.mkdir(parents=True, exist_ok=True)
                shutil.move(str(conversion_log), str(log_dir / conversion_log.name))
        if not primary.exists():
            raise RuntimeError(
                f"dcm2niix produced no {primary.name}; created: "
                f"{sorted(p.name for p in conversion_dir.iterdir())}, see {log_dir}"
            )

        session_dir.mkdir(parents=True, exist_ok=True)
        for produced in sorted(conversion_dir.iterdir()):
            if produced.name.endswith(".json"):
                sanitize_sidecar(produced, subject_id)
            target = session_dir / produced.name
            if produced != primary and produced.name.endswith((".nii.gz", ".nii")):
                logger.warning(
                    f"{basename}: additional volume {produced.name} written next to the "
                    "primary nifti."
                )
            shutil.move(str(produced), str(target))
    return session_dir / primary.name


def process_patient(
    patient_dir: Path,
    subject_id: str,
    outdir: Path,
    convert_all: bool,
    override: bool,
    dry_run: bool,
) -> List[Dict[str, str]]:
    """Converts every eligible series of a patient export. Returns the manifest rows."""
    logger.info(f"Indexing {patient_dir} (subject {subject_id}).")
    all_series = index_series(patient_dir)
    logger.info(f"{subject_id}: {len(all_series)} series found.")

    subject_dir = outdir / f"sub-{subject_id}"
    log_dir = outdir / "logs" / f"sub-{subject_id}"
    rows: List[Dict[str, str]] = []
    planned_outputs: Dict[Path, Series] = {}
    for series in all_series:
        row = {
            "subject": subject_id,
            "session": series.acquisition_date or "",
            "series_number": series.number,
            "series_description": series.description,
            "image_type": "|".join(series.image_type),
            "n_files": str(len(series.files)),
            "decision": "",
            "modality_or_reason": "",
            "output": "",
        }
        rows.append(row)

        reason = unconvertible_reason(series)
        if reason is None and not convert_all:
            reason = derived_reason(series)
        if reason is not None:
            logger.info(f"{subject_id}: skipping {series.label}: {reason}.")
            row["decision"], row["modality_or_reason"] = "skipped", reason
            continue
        if series.date_source == "StudyDate":
            logger.warning(
                f"{subject_id}: {series.label} has no acquisition/series date, using StudyDate "
                f"{series.acquisition_date} (this is the CD import date, not the exam date)."
            )

        modality = classify_modality(series)
        session = f"ses-{series.acquisition_date}"
        basename = f"sub-{subject_id}_{session}_sequ-{series.number}_{modality}"
        session_dir = subject_dir / session
        outfile = session_dir / f"{basename}.nii.gz"
        row["modality_or_reason"], row["output"] = modality, str(outfile)

        if outfile in planned_outputs:
            other = planned_outputs[outfile]
            logger.error(
                f"{subject_id}: {series.label} maps to {outfile.name}, which {other.label} "
                "already produces in this session; skipping the series."
            )
            row["decision"], row["modality_or_reason"] = "skipped", f"duplicate output ({modality})"
            continue
        planned_outputs[outfile] = series

        logger.info(f"{subject_id}: {series.label} -> {session}/{basename} [{modality}]")
        if dry_run:
            row["decision"] = "planned"
            continue
        if outfile.exists() and not override:
            logger.info(f"{subject_id}: {outfile.name} exists, skipping conversion.")
            row["decision"] = "exists"
            continue
        try:
            convert_series(series, session_dir, basename, subject_id, log_dir)
            row["decision"] = "converted"
        except Exception as e:  # noqa: BLE001 - one failing series must not abort the patient
            logger.exception(f"{subject_id}: conversion of {series.label} failed: {e}")
            row["decision"], row["modality_or_reason"] = "failed", f"{modality}: {e}"
    return rows


def write_manifest(manifest_file: Path, rows: List[Dict[str, str]]) -> None:
    manifest_file.parent.mkdir(parents=True, exist_ok=True)
    with manifest_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    logger.info(f"Manifest written to {manifest_file}.")


def parse_subject_ids(entries: Optional[List[str]]) -> Dict[str, str]:
    """Parses '-subject_ids Tumorboard_example=GR999 ...' into {folder name: subject id}."""
    mapping = {}
    for entry in entries or []:
        folder, sep, subject_id = entry.partition("=")
        if not sep or not folder or not subject_id:
            raise ValueError(f"-subject_ids entries must look like FOLDER=ID, got {entry!r}.")
        mapping[folder] = subject_id
    return mapping


if __name__ == "__main__":
    # Examples:
    # python scripts/convert_marco.py -dry_run
    # nohup python -u scripts/convert_marco.py > tmp_convert_marco.out 2>&1 &
    # python scripts/convert_marco.py -patients GR067 -subject_ids Tumorboard_example=GR999
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-input_dir",
        type=str,
        default=DEFAULT_INPUT_DIR,
        help="Directory with one Sectra CD export (DICOMDIR + DICOM/) per patient.",
    )
    parser.add_argument(
        "-outdir",
        type=str,
        default=DEFAULT_OUTDIR,
        help="Directory to write sub-<ID>/ses-<date>/ niftis, logs and the manifest to.",
    )
    parser.add_argument(
        "-patients",
        type=str,
        nargs="*",
        default=None,
        help="Optional subset of patient directory names to convert. Defaults to all.",
    )
    parser.add_argument(
        "-subject_ids",
        type=str,
        nargs="*",
        default=None,
        help=(
            "Optional FOLDER=ID mappings for the subject id used in the output names "
            "(e.g. Tumorboard_example=GR999). Defaults to the folder name with everything but "
            "letters and digits removed."
        ),
    )
    parser.add_argument(
        "-convert_all",
        action="store_true",
        help="Also convert derived/scout series (MPR, MIP, localizers, screen saves) instead of skipping them.",
    )
    parser.add_argument(
        "-override",
        action="store_true",
        help="Reconvert series whose output nifti already exists.",
    )
    parser.add_argument(
        "-dry_run",
        action="store_true",
        help="Only index and classify the series and write the manifest, convert nothing.",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    outdir = Path(args.outdir)
    subject_id_map = parse_subject_ids(args.subject_ids)

    patient_dirs = sorted(p for p in input_dir.iterdir() if p.is_dir())
    if args.patients is not None:
        patient_dirs = [p for p in patient_dirs if p.name in args.patients]
        missing = set(args.patients) - {p.name for p in patient_dirs}
        if missing:
            logger.warning(f"Requested patients not found in {input_dir}: {sorted(missing)}.")
    logger.info(f"Converting {len(patient_dirs)} patient export(s) from {input_dir} to {outdir}.")

    manifest_rows: List[Dict[str, str]] = []
    for patient_dir in patient_dirs:
        subject_id = subject_id_map.get(patient_dir.name, to_subject_label(patient_dir.name))
        if subject_id != patient_dir.name:
            logger.warning(f"Using subject id '{subject_id}' for directory '{patient_dir.name}'.")
        manifest_rows.extend(
            process_patient(
                patient_dir=patient_dir,
                subject_id=subject_id,
                outdir=outdir,
                convert_all=args.convert_all,
                override=args.override,
                dry_run=args.dry_run,
            )
        )

    write_manifest(outdir / "conversion_manifest.csv", manifest_rows)
    decisions = {}
    for row in manifest_rows:
        decisions[row["decision"]] = decisions.get(row["decision"], 0) + 1
    logger.info(f"Done: {decisions}.")
