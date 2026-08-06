import os
import json
import argparse
from pathlib import Path
from loguru import logger
from predict_gbm.utils.parsing import PatientDataset
from predict_gbm.utils.constants import (
    MODALITY_STRIPPED_SCHEMA,
    TUMORSEG_SCHEMA,
    TISSUE_SEG_SCHEMA,
    RECURRENCE_SCHEMA,
)
from predict_gbm.preprocessing import (
    norm_ss_coregister,
    run_brats,
    run_tissue_seg_registration,
    register_recurrence,
)

# Modality keys as used in missing_modalities.json, mapped to the corresponding sailor.json exam key.
MISSING_MODALITY_KEYS = {
    "t1contrast": "t1c",
    "t1": "t1",
    "t2": "t2",
    "flair": "flair",
}


def resolve_modalities(exam: dict, missing: list) -> dict | None:
    """
    Resolves t1/t1c/t2/flair/adc paths for an exam, applying the fallback rules:
    t1c missing -> exam excluded; t1 missing -> use t1c; t2 missing -> use flair;
    flair missing -> use t2; adc -> use adc_derived.
    Returns None if the exam should be excluded (t1c missing, or t2/flair unresolvable).
    """
    if "t1contrast" in missing or exam.get("t1c") is None:
        return None
    t1c_path = exam["t1c"]

    t1_path = exam.get("t1")
    if "t1" in missing or t1_path is None:
        t1_path = t1c_path

    orig_t2 = exam.get("t2")
    orig_flair = exam.get("flair")

    t2_path = orig_flair if ("t2" in missing or orig_t2 is None) else orig_t2
    flair_path = orig_t2 if ("flair" in missing or orig_flair is None) else orig_flair

    if t2_path is None or flair_path is None:
        return None

    adc_path = exam.get("adc_derived")

    return {
        "t1": t1_path,
        "t1c": t1c_path,
        "t2": t2_path,
        "flair": flair_path,
        "adc": adc_path,
    }


def process_exam(modalities: dict, outdir: Path, cuda_device: str) -> None:
    """Runs atlas registration/skull stripping, tumor segmentation and tissue segmentation for a single exam."""
    outdir.mkdir(parents=True, exist_ok=True)

    additional_quantitative_modalities = (
        {"adc": modalities["adc"]} if modalities["adc"] is not None else None
    )

    t1c_stripped = MODALITY_STRIPPED_SCHEMA.format(base_dir=outdir, modality="t1c")
    if t1c_stripped.exists():
        logger.info(f"{outdir}: skull stripping already done, skipping.")
    else:
        norm_ss_coregister(
            t1_file=modalities["t1"],
            t1c_file=modalities["t1c"],
            t2_file=modalities["t2"],
            flair_file=modalities["flair"],
            skull_strip=True,
            outdir=outdir,
            additional_quantitative_modalities=additional_quantitative_modalities,
        )

    tumorseg_file = TUMORSEG_SCHEMA.format(base_dir=outdir)
    if tumorseg_file.exists():
        logger.info(f"{outdir}: tumor segmentation already done, skipping.")
    else:
        run_brats(
            t1_file=MODALITY_STRIPPED_SCHEMA.format(base_dir=outdir, modality="t1"),
            t1c_file=MODALITY_STRIPPED_SCHEMA.format(base_dir=outdir, modality="t1c"),
            t2_file=MODALITY_STRIPPED_SCHEMA.format(base_dir=outdir, modality="t2"),
            flair_file=MODALITY_STRIPPED_SCHEMA.format(base_dir=outdir, modality="flair"),
            outdir=outdir,
            cuda_device=cuda_device,
        )

    tissueseg_file = TISSUE_SEG_SCHEMA.format(base_dir=outdir)
    if tissueseg_file.exists():
        logger.info(f"{outdir}: tissue segmentation already done, skipping.")
    else:
        run_tissue_seg_registration(
            t1_file=MODALITY_STRIPPED_SCHEMA.format(base_dir=outdir, modality="t1c"),
            outdir=outdir,
        )


if __name__ == "__main__":
    # Example:
    # nohup python -u scripts/process_sailor.py -cuda_device 0 > tmp_process_sailor.out 2>&1 &
    parser = argparse.ArgumentParser()
    parser.add_argument("-cuda_device", type=str, default="0", help="GPU id to run on.")
    parser.add_argument(
        "-sailor_json",
        type=str,
        default="/mnt/Drive4/lucas/SAILOR/sailor.json",
        help="Path to the SAILOR dataset json.",
    )
    parser.add_argument(
        "-missing_modalities_json",
        type=str,
        default="/mnt/Drive4/lucas/SAILOR/missing_modalities.json",
        help="Path to the json listing missing modalities per exam.",
    )
    parser.add_argument(
        "-outdir",
        type=str,
        default="/mnt/Drive4/lucas/SAILOR/processed",
        help="Directory to save processed output to.",
    )
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device

    outdir_root = Path(args.outdir)
    outdir_root.mkdir(parents=True, exist_ok=True)

    with open(args.missing_modalities_json, "r") as f:
        missing_modalities = json.load(f)

    dataset = PatientDataset()
    dataset.load(args.sailor_json)
    logger.info(f"Loaded {len(dataset.patients)} patients from {args.sailor_json}.")

    for patient in dataset:
        patient_id = patient["patient_id"]
        exam_outdirs = {}  # dir_name -> (outdir, timepoint)

        for exam in patient:
            dir_name = exam["dir_name"]
            missing = missing_modalities.get(f"{patient_id}/{dir_name}", [])

            modalities = resolve_modalities(exam, missing)
            if modalities is None:
                logger.warning(
                    f"{patient_id}/{dir_name}: missing t1c or unresolvable t2/flair, skipping exam."
                )
                continue

            exam_outdir = outdir_root / patient_id / dir_name
            try:
                process_exam(modalities, exam_outdir, args.cuda_device)
                exam_outdirs[dir_name] = (exam_outdir, exam["timepoint"])
            except Exception:
                logger.exception(f"{patient_id}/{dir_name}: processing failed, skipping.")

        preop_entries = [
            outdir for outdir, timepoint in exam_outdirs.values() if timepoint == "preop"
        ]
        if not preop_entries:
            logger.warning(
                f"{patient_id}: no processed preop exam found, skipping recurrence registration."
            )
            continue
        preop_outdir = preop_entries[0]

        for dir_name, (exam_outdir, timepoint) in exam_outdirs.items():
            if timepoint not in ("postop", "followup"):
                continue

            recurrence_file = RECURRENCE_SCHEMA.format(base_dir=exam_outdir)
            if recurrence_file.exists():
                logger.info(f"{patient_id}/{dir_name}: recurrence registration already done, skipping.")
                continue

            try:
                register_recurrence(
                    t1c_pre_file=MODALITY_STRIPPED_SCHEMA.format(
                        base_dir=preop_outdir, modality="t1c"
                    ),
                    t1c_post_file=MODALITY_STRIPPED_SCHEMA.format(
                        base_dir=exam_outdir, modality="t1c"
                    ),
                    recurrence_seg_file=TUMORSEG_SCHEMA.format(base_dir=exam_outdir),
                    outdir=exam_outdir,
                )
            except Exception:
                logger.exception(
                    f"{patient_id}/{dir_name}: recurrence registration failed, skipping."
                )

    logger.info("Finished processing SAILOR dataset.")
