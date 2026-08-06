import json
import argparse
import numpy as np
import nibabel as nib
from pathlib import Path
from loguru import logger
from predict_gbm.utils.parsing import PatientDataset
from predict_gbm.preprocessing.conversion import dti_to_adc

# Fixed correction factor for Siemens scanners whose raw adc.nii.gz values are ~10x
# lower than the (correctly scaled) Avanto reference. Empirically confirmed on the
# SAILOR cohort: Avanto median ADC ~962 vs. ~79-93 for these models.
SIEMENS_NO_CORRECTION_MODELS = {"Avanto"}
SIEMENS_TENFOLD_MODELS = {"Symphony", "SymphonyVision", "SonataVision"}

# Philips scanners: real ADC = raw_pixel_value * PhilipsScaleSlope (from the adc.json
# sidecar). Confirmed against the Avanto reference for Intera/Achieva. Ingenia is a
# known-unreliable case even after this correction (see
# /mnt/Drive4/lucas/SAILOR/adc_comparison/notes.txt) and is still processed, but flagged.
PHILIPS_UNRELIABLE_MODELS = {"Ingenia"}


def get_adc_scale_factor(adc_json: Path) -> float | None:
    """
    Determines the scanner-dependent scale factor to apply to a raw adc.nii.gz so that
    its values are in real ADC units (1e-6 mm^2/s), based on the dcm2niix json sidecar.
    Returns None if the scanner model is not recognized.
    """
    with adc_json.open("r") as f:
        meta = json.load(f)

    manufacturer = meta.get("Manufacturer")
    model = meta.get("ManufacturersModelName")

    if model in SIEMENS_NO_CORRECTION_MODELS:
        return 1.0
    if model in SIEMENS_TENFOLD_MODELS:
        return 10.0
    if manufacturer == "Philips":
        if model in PHILIPS_UNRELIABLE_MODELS:
            logger.warning(
                f"{adc_json}: scanner model {model!r} has a known-unreliable ADC "
                "scaling correction; applying it anyway, but the result should not "
                "be trusted without further validation."
            )
        return float(meta["PhilipsScaleSlope"])

    logger.warning(
        f"{adc_json}: unrecognized scanner (Manufacturer={manufacturer!r}, "
        f"ManufacturersModelName={model!r}), cannot determine ADC scale factor."
    )
    return None


def scale_adc(infile: Path, outfile: Path, scale_factor: float) -> Path:
    """Copies a raw adc.nii.gz to outfile, applying the given scanner-dependent scale factor."""
    adc_nifti = nib.load(str(infile))
    adc_data = np.asarray(adc_nifti.dataobj, dtype=np.float32) * scale_factor

    outfile.parent.mkdir(parents=True, exist_ok=True)
    input_header = adc_nifti.header
    out_header = nib.Nifti1Header()
    out_header.set_data_dtype(np.float32)
    out_header.set_data_shape(adc_data.shape)
    out_header.set_zooms(input_header.get_zooms()[:3])
    out_header.set_xyzt_units(*input_header.get_xyzt_units())

    scaled_nifti = nib.Nifti1Image(adc_data, affine=adc_nifti.affine, header=out_header)
    nib.save(scaled_nifti, str(outfile))
    return outfile


def derive_adc(exam: dict, exam_dir: Path) -> Path | None:
    """
    Derives adc_derived.nii.gz for a single exam: from the raw DTI series via
    dti_to_adc if available, otherwise from the raw adc image with the scanner-dependent
    scale factor applied. Returns None if neither source is usable.
    """
    outfile = exam_dir / "adc_derived.nii.gz"

    dti_path = exam.get("dti")
    if dti_path is not None and dti_path.exists():
        try:
            return dti_to_adc(infile=dti_path, outfile=outfile)
        except Exception:
            logger.warning(
                f"{exam_dir}: dti_to_adc failed for {dti_path}, falling back to adc."
            )

    adc_path = exam.get("adc")
    if adc_path is None or not adc_path.exists():
        logger.warning(f"{exam_dir}: no dti or adc source available, skipping.")
        return None

    adc_json = adc_path.with_suffix("").with_suffix(".json")
    if not adc_json.exists():
        logger.warning(f"{exam_dir}: missing {adc_json.name}, cannot determine ADC scale factor.")
        return None

    scale_factor = get_adc_scale_factor(adc_json)
    if scale_factor is None:
        return None

    return scale_adc(adc_path, outfile, scale_factor)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-sailor_json",
        type=str,
        default="/mnt/Drive4/lucas/SAILOR/sailor.json",
        help="Path to the SAILOR dataset json.",
    )
    args = parser.parse_args()

    dataset = PatientDataset()
    dataset.load(args.sailor_json)
    logger.info(f"Loaded {len(dataset.patients)} patients from {args.sailor_json}.")

    for patient in dataset:
        patient_id = patient["patient_id"]

        for exam in patient:
            dir_name = exam["dir_name"]
            exam_dir = patient["patient_dir"] / dir_name

            try:
                adc_derived = derive_adc(exam, exam_dir)
            except Exception:
                logger.exception(f"{patient_id}/{dir_name}: ADC derivation failed, skipping.")
                adc_derived = None

            exam["adc_derived"] = adc_derived
            if adc_derived is not None:
                logger.info(f"{patient_id}/{dir_name}: wrote {adc_derived}.")

    dataset.save(args.sailor_json)
    logger.info(f"Finished deriving ADC images, updated {args.sailor_json}.")
