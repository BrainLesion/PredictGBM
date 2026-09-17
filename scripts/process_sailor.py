import os
import json
import shutil
import argparse
import ants
import torch
import nibabel as nib
import numpy as np
from pathlib import Path
from loguru import logger
from predict_gbm.utils.parsing import PatientDataset
from predict_gbm.utils.constants import (
    CONFIG_SCHEMA,
    CONFIG_STEP_REGISTER_RECURRENCE,
    LONGITUDINAL_AFFINE_SCHEMA,
    LONGITUDINAL_DISP_SCHEMA,
    LONGITUDINAL_WARP_SCHEMA,
    MODALITY_STRIPPED_SCHEMA,
    PathSchema,
    REGISTRATION_TRAFO_SCHEMA,
    TUMORSEG_SCHEMA,
    TISSUE_PBMAP_SCHEMA,
    RECURRENCE_SCHEMA,
)
from predict_gbm.preprocessing import (
    norm_ss_coregister,
    run_brats,
    run_tissue_seg,
    register_recurrence,
)
from predict_gbm.preprocessing.dirac import warp_image_to_preop

# Modality keys as used in missing_modalities.json, mapped to the corresponding sailor.json exam key.
MISSING_MODALITY_KEYS = {
    "t1contrast": "t1c",
    "t1": "t1",
    "t2": "t2",
    "flair": "flair",
}

# Radiotherapy dose map (Gy) per patient. Its voxel grid (and header) is that of the raw RT
# planning MRI (one of the post-op exams, e.g. ses-03 for sub-01), but its content lies in
# the world frame of the derivatives, i.e. MNI ICBM 2009c space (ONCOHabitats rigid+affine
# registration). It is treated as an additional quantitative modality named "dose".
DOSE_MAP_SCHEMA = PathSchema("{dose_map_dir}/{patient_id}/DoseMap.nii.gz")
DOSE_MODALITY = "dose"
# Max. deviation (mm) between the world-space bounding boxes of the dose map and an exam's
# raw t1c for the exam to count as the planning MRI the dose map was resampled onto.
DOSE_MAP_MATCH_TOLERANCE_MM = 10.0
# Skull-stripped, MNI-space T1c of the derivatives, used to register the derivatives' MNI
# frame (the frame the dose map content lies in) to the pipeline's atlas space. All sessions
# of a patient share the MNI frame, so a fixed session is used.
DOSE_MNI_T1C_SCHEMA = PathSchema("{dose_map_dir}/{patient_id}/{session}/T1c.nii.gz")
DOSE_MNI_SESSION = "ses-03"
# Affine mapping the derivatives' MNI frame into the pipeline atlas space of an exam.
DOSE_MNI_AFFINE_SCHEMA = REGISTRATION_TRAFO_SCHEMA / DOSE_MODALITY / "mni2009c_to_atlas.mat"


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


def process_exam(
    modalities: dict, outdir: Path, cuda_device: str, override: bool = False
) -> None:
    """
    Runs atlas registration/skull stripping, tumor segmentation and tissue segmentation for a single exam.
    If override is True, all steps are rerun even if their outputs already exist.
    """
    outdir.mkdir(parents=True, exist_ok=True)

    additional_quantitative_modalities = (
        {"adc": modalities["adc"]} if modalities["adc"] is not None else None
    )

    t1c_stripped = MODALITY_STRIPPED_SCHEMA.format(base_dir=outdir, modality="t1c")
    if t1c_stripped.exists() and not override:
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
    if tumorseg_file.exists() and not override:
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

    tissueseg_files = [
        TISSUE_PBMAP_SCHEMA.format(base_dir=outdir, tissue=tissue)
        for tissue in ("gm", "wm", "csf")
    ]
    #if all(f.exists() for f in tissueseg_files) and not override:
    if False:
        logger.info(f"{outdir}: tissue segmentation already done, skipping.")
    else:
        print("Tissue segmentation overiding old files by design")
        run_tissue_seg(
            t1_file=MODALITY_STRIPPED_SCHEMA.format(base_dir=outdir, modality="t1c"),
            outdir=outdir,
        )


def world_bbox(nifti_file: Path) -> np.ndarray:
    """Returns the world-space (min, max) corners, shape (2, 3), of a nifti's voxel grid."""
    img = nib.load(str(nifti_file))
    shape = np.array(img.shape[:3]) - 1
    corners = np.array(
        [[i, j, k, 1.0] for i in (0, shape[0]) for j in (0, shape[1]) for k in (0, shape[2])]
    )
    world = (img.affine @ corners.T).T[:, :3]
    return np.stack([world.min(axis=0), world.max(axis=0)])


def find_dose_map_exam(dose_file: Path, exam_t1c_files: dict) -> str | None:
    """
    Identifies the exam whose raw t1c grid the dose map was resampled onto, i.e. the RT
    planning MRI, by matching world-space bounding boxes. Only the grid is matched; the dose
    map content itself lies in the derivatives' MNI frame. exam_t1c_files maps exam dir_name
    to the raw t1c path. Returns the dir_name of the best match, or None if no exam matches
    within DOSE_MAP_MATCH_TOLERANCE_MM.
    """
    logger.warning(
        f"Automatically determining the exam the dose map {dose_file} is aligned with by "
        "matching world-space bounding boxes of the raw t1c images; verify the match."
    )
    dose_bbox = world_bbox(dose_file)
    best_name, best_dist = None, np.inf
    for dir_name, t1c_file in exam_t1c_files.items():
        dist = np.abs(world_bbox(t1c_file) - dose_bbox).max()
        logger.debug(f"{dir_name}: dose map bbox deviation {dist:.1f} mm.")
        if dist < best_dist:
            best_name, best_dist = dir_name, dist
    if best_dist > DOSE_MAP_MATCH_TOLERANCE_MM:
        logger.warning(
            f"No exam grid matches the dose map {dose_file} "
            f"(closest: {best_name} with {best_dist:.1f} mm deviation)."
        )
        return None
    logger.info(f"Dose map matched to exam {best_name} ({best_dist:.1f} mm deviation).")
    return best_name


def register_dose_map(
    dose_file: Path,
    mni_t1c_file: Path,
    exam_outdir: Path,
    preop_outdir: Path | None,
    override: bool = False,
    override_longitudinal: bool = False,
) -> None:
    """
    Transforms the dose map (content in the derivatives' MNI frame) to the atlas space of the
    exam in exam_outdir, and then to preop space using the exam's longitudinal
    (followup-to-preop) transform from register_recurrence.
    The MNI-to-atlas affine is obtained by registering the derivatives' skull-stripped MNI T1c
    (mni_t1c_file) to the exam's skull-stripped atlas t1c. It is applied to the dose map in
    world coordinates, so the dose map's own (raw planning MRI) voxel grid is resampled onto
    the atlas grid in a single interpolation. No brain masking is applied.
    Outputs: <exam_outdir>/skull_stripped/dose/mni2009c_to_atlas.mat (the affine),
    <exam_outdir>/skull_stripped/dose_skullstripped.nii.gz (atlas space) and
    <exam_outdir>/longitudinal/dose_warped_longitudinal.nii.gz (preop space).
    If preop_outdir is None (the exam is the preop exam itself) only the atlas step is run.
    If override_longitudinal is True, only the preop step is rerun (e.g. after the exam's
    longitudinal transform has been recomputed); the atlas step still follows override.
    """
    dose_atlas_file = MODALITY_STRIPPED_SCHEMA.format(
        base_dir=exam_outdir, modality=DOSE_MODALITY
    )
    if dose_atlas_file.exists() and not override:
        logger.info(f"{exam_outdir}: dose map atlas registration already done, skipping.")
    else:
        t1c_atlas = ants.image_read(
            str(MODALITY_STRIPPED_SCHEMA.format(base_dir=exam_outdir, modality="t1c"))
        )
        mni_t1c = ants.image_read(str(mni_t1c_file))
        affine_file = DOSE_MNI_AFFINE_SCHEMA.format(base_dir=exam_outdir)
        affine_file.parent.mkdir(parents=True, exist_ok=True)
        reg = ants.registration(
            fixed=t1c_atlas,
            moving=mni_t1c,
            type_of_transform="Affine",
            outprefix=str(affine_file.parent / "mni2009c_to_atlas_"),
        )
        shutil.copyfile(reg["fwdtransforms"][0], affine_file)
        for tmp in reg["fwdtransforms"] + reg["invtransforms"]:
            if Path(tmp).exists() and Path(tmp) != affine_file:
                os.remove(tmp)
        fixed_np, warped_np = t1c_atlas.numpy(), reg["warpedmovout"].numpy()
        corr = np.corrcoef(fixed_np.ravel(), warped_np.ravel())[0, 1]
        logger.info(
            f"{exam_outdir}: registered MNI T1c to atlas t1c, correlation {corr:.3f}, "
            f"saved affine to {affine_file}."
        )
        dose_atlas = ants.apply_transforms(
            fixed=t1c_atlas,
            moving=ants.image_read(str(dose_file)),
            transformlist=[str(affine_file)],
            interpolator="linear",
            defaultvalue=0,
        )
        ants.image_write(dose_atlas, str(dose_atlas_file))
        logger.info(f"Saved atlas-space dose map to {dose_atlas_file}.")

    if preop_outdir is None:
        logger.info(f"{exam_outdir}: dose map exam is the preop exam, no longitudinal warp.")
        return

    dose_preop_file = LONGITUDINAL_WARP_SCHEMA.format(
        base_dir=exam_outdir, modality=DOSE_MODALITY
    )
    if dose_preop_file.exists() and not (override or override_longitudinal):
        logger.info(f"{exam_outdir}: dose map preop registration already done, skipping.")
        return

    with open(CONFIG_SCHEMA.format(base_dir=exam_outdir), "r") as f:
        algorithm = json.load(f)[CONFIG_STEP_REGISTER_RECURRENCE]["registration_algorithm"]
    t1c_pre_file = MODALITY_STRIPPED_SCHEMA.format(base_dir=preop_outdir, modality="t1c")

    if algorithm == "dirac":
        warp_image_to_preop(
            image_file=dose_atlas_file,
            reference_file=t1c_pre_file,
            disp_field_file=LONGITUDINAL_DISP_SCHEMA.format(base_dir=exam_outdir),
            out_file=dose_preop_file,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            mode="bilinear",
        )
    else:
        trafo_schema = (
            LONGITUDINAL_DISP_SCHEMA if algorithm == "syn" else LONGITUDINAL_AFFINE_SCHEMA
        )
        dose_preop = ants.apply_transforms(
            fixed=ants.image_read(str(t1c_pre_file)),
            moving=ants.image_read(str(dose_atlas_file)),
            transformlist=[str(trafo_schema.format(base_dir=exam_outdir))],
            interpolator="linear",
            defaultvalue=0,
        )
        ants.image_write(dose_preop, str(dose_preop_file))
    logger.info(f"Saved preop-space dose map to {dose_preop_file}.")


def visualize_dose_maps(outdir_root: Path, pdf_file: Path, rows_per_page: int = 5) -> None:
    """
    Writes a pdf with one row per patient (dose exam) showing the axial, sagittal and coronal
    center slices of the atlas-space t1c with the atlas-space dose map overlaid (alpha 0.5).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    dose_files = sorted(
        outdir_root.glob(f"*/*/skull_stripped/{DOSE_MODALITY}_skullstripped.nii.gz")
    )
    if not dose_files:
        logger.warning(f"No dose maps found under {outdir_root}, nothing to visualize.")
        return
    with PdfPages(pdf_file) as pdf:
        for page_start in range(0, len(dose_files), rows_per_page):
            page_files = dose_files[page_start : page_start + rows_per_page]
            fig, axes = plt.subplots(
                len(page_files), 3, figsize=(9, 3 * len(page_files)), squeeze=False
            )
            for row, dose_file in zip(axes, page_files):
                exam_outdir = dose_file.parent.parent
                t1c = nib.load(
                    str(MODALITY_STRIPPED_SCHEMA.format(base_dir=exam_outdir, modality="t1c"))
                ).get_fdata()
                dose = nib.load(str(dose_file)).get_fdata()
                dose = np.ma.masked_where(dose <= 0, dose)
                cx, cy, cz = (np.array(t1c.shape) // 2).tolist()
                slices = {
                    "axial": (t1c[:, :, cz].T, dose[:, :, cz].T),
                    "sagittal": (t1c[cx, :, :].T, dose[cx, :, :].T),
                    "coronal": (t1c[:, cy, :].T, dose[:, cy, :].T),
                }
                for ax, (view, (anat, overlay)) in zip(row, slices.items()):
                    ax.imshow(anat, cmap="gray", origin="lower")
                    ax.imshow(
                        overlay, cmap="jet", alpha=0.5, origin="lower", vmin=0, vmax=dose.max()
                    )
                    ax.set_title(view, fontsize=8)
                    ax.axis("off")
                row[0].text(
                    -0.05,
                    0.5,
                    f"{exam_outdir.parent.name}/{exam_outdir.name}\nmax {float(dose.max()):.1f} Gy",
                    transform=row[0].transAxes,
                    fontsize=8,
                    ha="right",
                    va="center",
                )
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
    logger.info(f"Saved dose map visualization of {len(dose_files)} exams to {pdf_file}.")


if __name__ == "__main__":
    # Example:
    # nohup python -u scripts/process_sailor.py -cuda_device 0 > tmp_process_sailor.out 2>&1 &
    # Only (re)run the dose map step for a single patient:
    # python -u scripts/process_sailor.py -cuda_device 0 -patients sub-01 -dose_only
    # Only write the dose map check pdf (existing dose outputs are skipped, not recomputed):
    # python -u scripts/process_sailor.py -dose_only -visualize_dose
    # Recompute the followup-to-preop registrations (and the preop-space dose maps) only:
    # python -u scripts/process_sailor.py -cuda_device 0 -patients sub-01 -longitudinal_registration
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
    parser.add_argument(
        "-dose_map_dir",
        type=str,
        default="/mnt/Drive4/lucas/SAILOR/derivatives/mni2009c-n-s",
        help="Directory containing <patient_id>/DoseMap.nii.gz radiotherapy dose maps.",
    )
    parser.add_argument(
        "-patients",
        type=str,
        nargs="+",
        default=None,
        help="Patient ids to process (e.g. sub-01). Processes all patients if omitted.",
    )
    parser.add_argument(
        "-dose_exam",
        type=str,
        default=None,
        help="Exam dir_name (e.g. ses-02) whose transforms to use for the dose map, "
        "bypassing the automatic bounding box match. Only meaningful with a single patient.",
    )
    parser.add_argument(
        "-dose_only",
        action="store_true",
        help="Only run the dose map registration, using existing outputs of the other steps.",
    )
    parser.add_argument(
        "-longitudinal_registration",
        action="store_true",
        help="Only recompute the longitudinal (postop/followup t1c to preop) registration of "
        "every processed postop/followup exam, overwriting existing results, and apply the new "
        "transform to the modalities this script warps to preop space (t1c, tumor segmentation "
        "and dose map). Existing outputs of the other steps are reused.",
    )
    parser.add_argument(
        "-visualize_dose",
        action="store_true",
        help="After processing, write <outdir>/dose_maps.pdf with the atlas-space t1c and "
        "dose map overlay (center slices) of every exam holding a dose map.",
    )
    parser.add_argument(
        "-override",
        action="store_true",
        help="Rerun every step even if its output already exists, overwriting previous results.",
    )
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device
    # Modes reusing the existing per-exam outputs (skull stripping, tumor/tissue segmentation).
    reuse_exam_outputs = args.dose_only or args.longitudinal_registration

    outdir_root = Path(args.outdir)
    outdir_root.mkdir(parents=True, exist_ok=True)

    with open(args.missing_modalities_json, "r") as f:
        missing_modalities = json.load(f)

    dataset = PatientDataset()
    dataset.load(args.sailor_json)
    logger.info(f"Loaded {len(dataset.patients)} patients from {args.sailor_json}.")

    for patient in dataset:
        patient_id = patient["patient_id"]
        if args.patients is not None and patient_id not in args.patients:
            continue
        exam_outdirs = {}  # dir_name -> (outdir, timepoint)
        exam_t1c_files = {}  # dir_name -> raw t1c path (native frame of the exam)

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
            if reuse_exam_outputs:
                if MODALITY_STRIPPED_SCHEMA.format(base_dir=exam_outdir, modality="t1c").exists():
                    exam_outdirs[dir_name] = (exam_outdir, exam["timepoint"])
                    exam_t1c_files[dir_name] = Path(modalities["t1c"])
                else:
                    logger.warning(f"{patient_id}/{dir_name}: not processed yet, skipping exam.")
                continue

            try:
                process_exam(modalities, exam_outdir, args.cuda_device, args.override)
                exam_outdirs[dir_name] = (exam_outdir, exam["timepoint"])
                exam_t1c_files[dir_name] = Path(modalities["t1c"])
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
            if args.dose_only and not args.longitudinal_registration:
                continue
            if args.longitudinal_registration:
                logger.info(
                    f"{patient_id}/{dir_name}: recomputing longitudinal registration to preop."
                )
            elif recurrence_file.exists() and not args.override:
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

        # Dose map: derivatives MNI frame -> atlas space of the planning exam -> preop space
        dose_file = DOSE_MAP_SCHEMA.format(
            dose_map_dir=args.dose_map_dir, patient_id=patient_id
        )
        if not dose_file.exists():
            logger.warning(f"{patient_id}: no dose map found at {dose_file}, skipping.")
            continue
        mni_t1c_file = DOSE_MNI_T1C_SCHEMA.format(
            dose_map_dir=args.dose_map_dir, patient_id=patient_id, session=DOSE_MNI_SESSION
        )
        if not mni_t1c_file.exists():
            logger.warning(f"{patient_id}: no MNI T1c found at {mni_t1c_file}, skipping.")
            continue
        if args.dose_exam is not None:
            if Path(args.dose_exam) not in exam_outdirs:
                logger.warning(
                    f"{patient_id}: forced dose exam {args.dose_exam} not among processed "
                    f"exams {sorted(exam_outdirs)}, skipping."
                )
                continue
            logger.warning(f"{patient_id}: forcing dose map exam {args.dose_exam}.")
            dose_exam = Path(args.dose_exam)
        else:
            dose_exam = find_dose_map_exam(dose_file, exam_t1c_files)
        if dose_exam is None:
            logger.warning(f"{patient_id}: dose map matches no processed exam, skipping.")
            continue
        dose_outdir, dose_timepoint = exam_outdirs[dose_exam]
        if dose_timepoint != "preop" and not RECURRENCE_SCHEMA.format(
            base_dir=dose_outdir
        ).exists():
            logger.warning(
                f"{patient_id}/{dose_exam}: no longitudinal registration available, "
                "skipping dose map registration."
            )
            continue
        try:
            register_dose_map(
                dose_file=dose_file,
                mni_t1c_file=mni_t1c_file,
                exam_outdir=dose_outdir,
                preop_outdir=None if dose_timepoint == "preop" else preop_outdir,
                override=args.override,
                override_longitudinal=args.longitudinal_registration,
            )
        except Exception:
            logger.exception(f"{patient_id}/{dose_exam}: dose map registration failed.")

    if args.visualize_dose:
        visualize_dose_maps(outdir_root, outdir_root / "dose_maps.pdf")

    logger.info("Finished processing SAILOR dataset.")
