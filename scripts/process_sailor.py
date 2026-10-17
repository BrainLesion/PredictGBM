import os
import csv
import time
import json
import shutil
import argparse
import ants
import nibabel as nib
import numpy as np
from pathlib import Path
from typing import Dict
from loguru import logger
from brats import AdultGliomaPreAndPostTreatmentSegmenter
from brats.constants import AdultGliomaPreAndPostTreatmentAlgorithms
from predict_gbm.utils.parsing import PatientDataset
from predict_gbm.utils.constants import (
    LONGITUDINAL_DIR,
    LONGITUDINAL_WARP_SCHEMA,
    MODALITY_STRIPPED_SCHEMA,
    OUTPUT_BASE_SCHEMA,
    PathSchema,
    REGISTRATION_TRAFO_SCHEMA,
    TUMORSEG_SCHEMA,
    TISSUE_PBMAP_SCHEMA,
    TUMOR_SEGMENTATION_FOLDER,
    RECURRENCE_SCHEMA,
)
from predict_gbm.preprocessing import (
    longitudinal_transform_file,
    norm_ss_coregister,
    run_brats,
    run_tissue_seg,
    register_recurrence,
)
from predict_gbm.preprocessing.norm_ss_coregistration import SUPPORTED_ATLASES
from predict_gbm.utils.utils import update_config

# Atlas the modalities are registered to (and the tissue segmentation is based on). The
# outputs in <outdir> are in this atlas space, so do not mix atlases within one outdir.
DEFAULT_ATLAS = "sri24"

# Modality keys as used in missing_modalities.json, mapped to the corresponding sailor.json exam key.
MISSING_MODALITY_KEYS = {
    "t1contrast": "t1c",
    "t1": "t1",
    "t2": "t2",
    "flair": "flair",
}

# Radiotherapy dose map (Gy) per patient. Its voxel grid (and header) is that of a raw RT
# planning MRI, but its content lies in the world frame of the derivatives, i.e. MNI ICBM
# 2009c space (ONCOHabitats rigid+affine registration). It is treated as an additional
# quantitative modality named "dose".
DOSE_MAP_SCHEMA = PathSchema("{dose_map_dir}/{patient_id}/DoseMap.nii.gz")
DOSE_MODALITY = "dose"
# Skull-stripped, MNI-space T1c of a derivatives session. The dose map is brought into the
# pipeline atlas space of the "dose exam" (the first followup exam with a derivatives
# counterpart, see find_dose_map_exam) by affinely registering the derivatives T1c of that
# same visit to the exam's atlas-space t1c. Since the derivatives were registered to MNI
# rigid+affine but the pipeline registers rigidly, this affine is patient specific.
DOSE_MNI_T1C_SCHEMA = PathSchema("{dose_map_dir}/{patient_id}/{session}/T1c.nii.gz")
# Table linking raw sessions to derivatives (MNI) sessions: columns subject, raw session,
# mni session ("no" if the raw session has no derivatives counterpart).
DOSE_RAW_MNI_LINK_SCHEMA = PathSchema("{dose_map_dir}/raw-mni-link.tsv")
# Affine mapping the derivatives' MNI frame into the pipeline atlas space of an exam.
DOSE_MNI_AFFINE_SCHEMA = REGISTRATION_TRAFO_SCHEMA / DOSE_MODALITY / "mni2009c_to_atlas.mat"
# Treatment label of a derivatives session (CRT, TMZ, no, unknown), added to the session
# labels table written to the output directory.
TREATMENT_SCHEMA = PathSchema("{dose_map_dir}/{patient_id}/{session}/treatment.txt")
SESSION_LABELS_FILENAME = "session_labels.tsv"

# Tissue probability maps co-transformed to preop space alongside t1c and flair.
LONGITUDINAL_TISSUES = ("gm", "wm", "csf")
# Modalities of both exams that drive the four-modality DIRAC longitudinal registration.
DIRAC_MODALITIES = ("t1", "t1c", "t2", "flair")

# Alternative tumor segmentation of the postop exams with the BraTS 2024 post-treatment
# algorithm (-brats24_postop). The BraTS25 pre-and-post algorithm used by the pipeline tends
# to label early postop resection cavities as non-enhancing tumor core (label 1) instead of
# resection cavity (label 4), which BraTS24_1 gets right. Stored next to the pipeline
# outputs, in atlas space and warped to preop space with the exam's longitudinal transform.
BRATS24_ALGORITHM = AdultGliomaPreAndPostTreatmentAlgorithms.BraTS24_1
BRATS24_TUMORSEG_SCHEMA = (
    OUTPUT_BASE_SCHEMA / TUMOR_SEGMENTATION_FOLDER / "tumor_seg_24_1.nii.gz"
)
BRATS24_RECURRENCE_SCHEMA = (
    OUTPUT_BASE_SCHEMA / LONGITUDINAL_DIR / "recurrence_preop_24_1.nii.gz"
)
CONFIG_STEP_BRATS24_POSTOP = "run_brats24_postop"


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
    modalities: dict,
    outdir: Path,
    cuda_device: str,
    atlas: str = DEFAULT_ATLAS,
    is_preop: bool = False,
    override: bool = False,
) -> None:
    """
    Runs atlas registration/skull stripping, tumor segmentation and tissue segmentation for a single exam.
    The modalities are registered to the given atlas (see SUPPORTED_ATLASES), which is also used
    for the atlas-based tissue segmentation. is_preop selects the pre-treatment branch of the
    tumor segmentation (no resection cavity label); post-op and followup exams use the
    post-treatment branch. If override is True, all steps are rerun even if their outputs
    already exist.
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
            atlas=atlas,
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
            is_preop=is_preop,
        )

    tissueseg_files = [
        TISSUE_PBMAP_SCHEMA.format(base_dir=outdir, tissue=tissue)
        for tissue in ("gm", "wm", "csf")
    ]
    if all(f.exists() for f in tissueseg_files) and not override:
        logger.info(f"{outdir}: tissue segmentation already done, skipping.")
    else:
        run_tissue_seg(
            t1_file=MODALITY_STRIPPED_SCHEMA.format(base_dir=outdir, modality="t1c"),
            outdir=outdir,
            atlas=atlas,
        )


def dirac_modality_files(preop_outdir: Path, exam_outdir: Path) -> Dict[str, Path]:
    """
    Skull-stripped t1, t1c, t2 and flair of the preop and the followup exam as the
    {modality}_pre_file / {modality}_post_file arguments of register_recurrence, which the
    four-modality DIRAC model needs.
    """
    files: Dict[str, Path] = {}
    for modality in DIRAC_MODALITIES:
        files[f"{modality}_pre_file"] = MODALITY_STRIPPED_SCHEMA.format(
            base_dir=preop_outdir, modality=modality
        )
        files[f"{modality}_post_file"] = MODALITY_STRIPPED_SCHEMA.format(
            base_dir=exam_outdir, modality=modality
        )
    return files


def longitudinal_modalities(exam_outdir: Path) -> dict:
    """
    Additional modalities of an exam that register_recurrence co-transforms to preop space:
    the skull-stripped flair and the gm/wm/csf tissue probability maps. Outputs are stored as
    <exam_outdir>/longitudinal/<name>_warped_longitudinal.nii.gz.
    """
    modalities = {
        "flair": MODALITY_STRIPPED_SCHEMA.format(base_dir=exam_outdir, modality="flair")
    }
    for tissue in LONGITUDINAL_TISSUES:
        modalities[f"{tissue}_pbmap"] = TISSUE_PBMAP_SCHEMA.format(
            base_dir=exam_outdir, tissue=tissue
        )
    return modalities


def write_session_labels(
    session_labels_file: Path, raw_mni_link: dict, dose_map_dir: str, outfile: Path
) -> None:
    """
    Copies session_labels.tsv to outfile with an additional column treatment_label, read from
    the derivatives treatment.txt of the session linked to each raw session via raw_mni_link
    ({(patient_id, raw session): derivatives session}). Sessions without a derivatives
    counterpart or without a treatment.txt get an empty label.
    """
    with open(session_labels_file, newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        fieldnames = list(reader.fieldnames) + ["treatment_label"]
        rows = list(reader)
    n_labeled = 0
    for row in rows:
        mni_session = raw_mni_link.get((row["subject"], row["session"]))
        label = ""
        if mni_session is not None:
            treatment_file = TREATMENT_SCHEMA.format(
                dose_map_dir=dose_map_dir, patient_id=row["subject"], session=mni_session
            )
            if treatment_file.exists():
                label = treatment_file.read_text().strip()
        row["treatment_label"] = label
        n_labeled += bool(label)
    with open(outfile, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    logger.info(
        f"Wrote {outfile} ({len(rows)} sessions, {n_labeled} with a treatment label)."
    )


def load_raw_mni_link(link_file: Path) -> dict:
    """
    Reads raw-mni-link.tsv and returns {(patient_id, raw dir_name): derivatives session} for
    the raw sessions that have a derivatives counterpart.
    """
    link = {}
    with open(link_file, newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if row["mni session"] != "no":
                link[(row["subject"], row["raw session"])] = row["mni session"]
    return link


def find_dose_map_exam(
    patient_id: str, exam_outdirs: dict, raw_mni_link: dict
) -> tuple[str, str] | None:
    """
    Selects the exam the dose map is registered through: the first (by dir_name) processed
    followup exam that has a derivatives counterpart in raw_mni_link. The dose map content
    lies in the derivatives' MNI frame shared by all sessions, so any exam would do; the
    first followup is closest in time to the radiotherapy. exam_outdirs maps dir_name to
    (outdir, timepoint). Returns (dir_name, derivatives session) or None if no exam qualifies.
    """
    for dir_name in sorted(exam_outdirs):
        _, timepoint = exam_outdirs[dir_name]
        # dir_name may be a Path (PatientDataset yields Paths); the link table uses strings.
        mni_session = raw_mni_link.get((patient_id, str(dir_name)))
        if timepoint == "followup" and mni_session is not None:
            logger.info(
                f"{patient_id}: dose map exam {dir_name} (derivatives session {mni_session})."
            )
            return dir_name, mni_session
    return None


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
    of the same visit (mni_t1c_file) to the exam's skull-stripped atlas t1c. It is applied to
    the dose map in world coordinates, so the dose map's own (raw planning MRI) voxel grid is
    resampled onto the atlas grid in a single interpolation. No brain masking is applied.
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

    warp_to_preop(
        image_file=dose_atlas_file,
        exam_outdir=exam_outdir,
        preop_outdir=preop_outdir,
        out_file=dose_preop_file,
        label_map=False,
    )
    logger.info(f"Saved preop-space dose map to {dose_preop_file}.")


def warp_to_preop(
    image_file: Path,
    exam_outdir: Path,
    preop_outdir: Path,
    out_file: Path,
    label_map: bool = False,
) -> None:
    """
    Warps an atlas-space image of the exam in exam_outdir to preop space with the exam's
    longitudinal (followup-to-preop) transform from register_recurrence, resolved with
    longitudinal_transform_file (displacement field for dirac/syn, .mat for affine/rigid).
    Label maps are warped with nearest neighbour interpolation, images linearly.
    """
    t1c_pre_file = MODALITY_STRIPPED_SCHEMA.format(base_dir=preop_outdir, modality="t1c")
    warped = ants.apply_transforms(
        fixed=ants.image_read(str(t1c_pre_file)),
        moving=ants.image_read(str(image_file)),
        transformlist=[str(longitudinal_transform_file(exam_outdir))],
        interpolator="nearestNeighbor" if label_map else "linear",
        defaultvalue=0,
    )
    ants.image_write(warped, str(out_file))


def segment_postop_brats24(
    exam_outdir: Path,
    preop_outdir: Path,
    cuda_device: str,
    override: bool = False,
) -> None:
    """
    Segments a postop exam with the BraTS 2024 post-treatment algorithm (BRATS24_ALGORITHM)
    on the exam's skull-stripped atlas-space modalities and warps the result to preop space
    with the exam's longitudinal transform. The pipeline's own segmentation (tumor_seg.nii.gz,
    recurrence_preop.nii.gz) is left untouched.
    Outputs: <exam_outdir>/tumor_segmentation/tumor_seg_24_1.nii.gz (atlas space) and
    <exam_outdir>/longitudinal/recurrence_preop_24_1.nii.gz (preop space).
    """
    seg_file = BRATS24_TUMORSEG_SCHEMA.format(base_dir=exam_outdir)
    if seg_file.exists() and not override:
        logger.info(f"{exam_outdir}: BraTS24 postop segmentation already done, skipping.")
    else:
        start_time = time.time()
        logger.info(f"{exam_outdir}: starting {BRATS24_ALGORITHM.value} postop segmentation.")
        segmenter = AdultGliomaPreAndPostTreatmentSegmenter(
            algorithm=BRATS24_ALGORITHM,
            cuda_devices=cuda_device,
        )
        seg_file.parent.mkdir(parents=True, exist_ok=True)
        segmenter.infer_single(
            t1n=str(MODALITY_STRIPPED_SCHEMA.format(base_dir=exam_outdir, modality="t1")),
            t1c=str(MODALITY_STRIPPED_SCHEMA.format(base_dir=exam_outdir, modality="t1c")),
            t2w=str(MODALITY_STRIPPED_SCHEMA.format(base_dir=exam_outdir, modality="t2")),
            t2f=str(MODALITY_STRIPPED_SCHEMA.format(base_dir=exam_outdir, modality="flair")),
            output_file=str(seg_file),
        )
        update_config(
            exam_outdir,
            CONFIG_STEP_BRATS24_POSTOP,
            {"algorithm": BRATS24_ALGORITHM.value, "timepoint": "postop"},
        )
        logger.info(
            f"{exam_outdir}: finished BraTS24 postop segmentation in "
            f"{time.time() - start_time:.2f} seconds. Saved output to {seg_file}."
        )

    recurrence_file = BRATS24_RECURRENCE_SCHEMA.format(base_dir=exam_outdir)
    if recurrence_file.exists() and not override:
        logger.info(f"{exam_outdir}: BraTS24 preop-space segmentation already done, skipping.")
        return
    warp_to_preop(
        image_file=seg_file,
        exam_outdir=exam_outdir,
        preop_outdir=preop_outdir,
        out_file=recurrence_file,
        label_map=True,
    )
    logger.info(f"{exam_outdir}: saved preop-space BraTS24 segmentation to {recurrence_file}.")


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
    # Full processing (skull stripping/atlas registration, tumor + tissue segmentation,
    # longitudinal registration and dose maps) in one call; steps with existing outputs are
    # skipped unless -override is given:
    # nohup python -u scripts/process_sailor.py -cuda_device 0 > tmp_process_sailor.out 2>&1 &
    # Same, but into a separate directory with a different atlas:
    # python -u scripts/process_sailor.py -cuda_device 0 -atlas sri24 -outdir /mnt/Drive4/lucas/SAILOR/processed_sri24
    # Only (re)run the dose map step for a single patient:
    # python -u scripts/process_sailor.py -cuda_device 0 -patients sub-01 -dose_only
    # Only write the dose map check pdf (existing dose outputs are skipped, not recomputed):
    # python -u scripts/process_sailor.py -dose_only -visualize_dose
    # Recompute the followup-to-preop registrations (and the preop-space dose maps) only:
    # python -u scripts/process_sailor.py -cuda_device 0 -patients sub-01 -longitudinal_registration
    # Additionally segment the postop exams with the BraTS 2024 post-treatment algorithm and
    # warp the result to preop space (tumor_seg_24_1 / recurrence_preop_24_1, see BRATS24_*):
    # nohup python -u scripts/process_sailor.py -cuda_device 0 -brats24_postop > tmp_process_sailor_brats24.out 2>&1 &
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
        "-session_labels_tsv",
        type=str,
        default="/mnt/Drive4/lucas/SAILOR/session_labels.tsv",
        help="Session labels table; copied to <outdir> with an added treatment_label column.",
    )
    parser.add_argument(
        "-outdir",
        type=str,
        default="/mnt/Drive4/lucas/SAILOR/processed",
        help="Directory to save processed output to.",
    )
    parser.add_argument(
        "-atlas",
        type=str,
        default=DEFAULT_ATLAS,
        choices=SUPPORTED_ATLASES,
        help="Atlas to register the modalities to. All outputs in -outdir must use the same "
        "atlas; use a fresh -outdir (or -override) when changing it.",
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
        "-dose_only",
        action="store_true",
        help="Only run the dose map registration, using existing outputs of the other steps.",
    )
    parser.add_argument(
        "-longitudinal_registration",
        action="store_true",
        help="Only recompute the longitudinal (postop/followup t1c to preop) registration of "
        "every processed postop/followup exam, overwriting existing results, and apply the new "
        "transform to the modalities this script warps to preop space (t1c, flair, tissue "
        "probability maps, tumor segmentation and dose map). Existing outputs of the other "
        "steps are reused.",
    )
    parser.add_argument(
        "-brats24_postop",
        "--brats24_postop",
        action="store_true",
        help="Only segment the postop exams (session label 'postop') with the BraTS 2024 "
        "post-treatment algorithm, saved as tumor_segmentation/tumor_seg_24_1.nii.gz, and "
        "warp the result to preop space with the existing longitudinal transform, saved as "
        "longitudinal/recurrence_preop_24_1.nii.gz. The pipeline's own segmentations are kept; "
        "existing outputs of this step are skipped unless -override is given.",
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
    reuse_exam_outputs = (
        args.dose_only or args.longitudinal_registration or args.brats24_postop
    )

    outdir_root = Path(args.outdir)
    outdir_root.mkdir(parents=True, exist_ok=True)

    with open(args.missing_modalities_json, "r") as f:
        missing_modalities = json.load(f)
    raw_mni_link = load_raw_mni_link(
        DOSE_RAW_MNI_LINK_SCHEMA.format(dose_map_dir=args.dose_map_dir)
    )

    dataset = PatientDataset()
    dataset.load(args.sailor_json)
    logger.info(f"Loaded {len(dataset.patients)} patients from {args.sailor_json}.")

    for patient in dataset:
        patient_id = patient["patient_id"]
        if args.patients is not None and patient_id not in args.patients:
            continue
        # Patients need a usable preop and postop exam (t1c + t2/flair resolvable), the
        # longitudinal outputs are defined relative to both; sub-06 and sub-25 have no postop.
        usable_timepoints = {
            exam["timepoint"]
            for exam in patient
            if resolve_modalities(
                exam, missing_modalities.get(f"{patient_id}/{exam['dir_name']}", [])
            )
            is not None
        }
        missing_timepoints = [t for t in ("preop", "postop") if t not in usable_timepoints]
        if missing_timepoints:
            logger.warning(
                f"{patient_id}: no usable {' and '.join(missing_timepoints)} exam, skipping patient."
            )
            continue
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
            if reuse_exam_outputs:
                if MODALITY_STRIPPED_SCHEMA.format(base_dir=exam_outdir, modality="t1c").exists():
                    exam_outdirs[dir_name] = (exam_outdir, exam["timepoint"])
                else:
                    logger.warning(f"{patient_id}/{dir_name}: not processed yet, skipping exam.")
                continue

            try:
                process_exam(
                    modalities=modalities,
                    outdir=exam_outdir,
                    cuda_device=args.cuda_device,
                    atlas=args.atlas,
                    is_preop=exam["timepoint"] == "preop",
                    override=args.override,
                )
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

        if args.brats24_postop:
            for dir_name, (exam_outdir, timepoint) in exam_outdirs.items():
                if timepoint != "postop":
                    continue
                if not RECURRENCE_SCHEMA.format(base_dir=exam_outdir).exists():
                    logger.warning(
                        f"{patient_id}/{dir_name}: no longitudinal registration available, "
                        "skipping BraTS24 postop segmentation."
                    )
                    continue
                try:
                    segment_postop_brats24(
                        exam_outdir=exam_outdir,
                        preop_outdir=preop_outdir,
                        cuda_device=args.cuda_device,
                        override=args.override,
                    )
                except Exception:
                    logger.exception(
                        f"{patient_id}/{dir_name}: BraTS24 postop segmentation failed, skipping."
                    )
            continue

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
                # DIRAC (four-modality model): t1, t1c, t2 and flair of both exams.
                register_recurrence(
                    recurrence_seg_file=TUMORSEG_SCHEMA.format(base_dir=exam_outdir),
                    outdir=exam_outdir,
                    additional_modalities=longitudinal_modalities(exam_outdir),
                    **dirac_modality_files(preop_outdir, exam_outdir),
                )
            except Exception:
                logger.exception(
                    f"{patient_id}/{dir_name}: recurrence registration failed, skipping."
                )

        # Dose map: derivatives MNI frame -> atlas space of the dose exam -> preop space
        dose_file = DOSE_MAP_SCHEMA.format(
            dose_map_dir=args.dose_map_dir, patient_id=patient_id
        )
        if not dose_file.exists():
            logger.warning(f"{patient_id}: no dose map found at {dose_file}, skipping.")
            continue
        dose_match = find_dose_map_exam(patient_id, exam_outdirs, raw_mni_link)
        if dose_match is None:
            logger.warning(
                f"{patient_id}: no processed followup exam with a derivatives counterpart, "
                "skipping dose map registration."
            )
            continue
        dose_exam, mni_session = dose_match
        mni_t1c_file = DOSE_MNI_T1C_SCHEMA.format(
            dose_map_dir=args.dose_map_dir, patient_id=patient_id, session=mni_session
        )
        if not mni_t1c_file.exists():
            logger.warning(f"{patient_id}: no MNI T1c found at {mni_t1c_file}, skipping.")
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

    write_session_labels(
        session_labels_file=Path(args.session_labels_tsv),
        raw_mni_link=raw_mni_link,
        dose_map_dir=args.dose_map_dir,
        outfile=outdir_root / SESSION_LABELS_FILENAME,
    )

    if args.visualize_dose:
        visualize_dose_maps(outdir_root, outdir_root / "dose_maps.pdf")

    logger.info("Finished processing SAILOR dataset.")
