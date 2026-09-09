import os
import re
import argparse
import shutil
from pathlib import Path
from typing import Dict, Optional

import ants
import nibabel as nib
import numpy as np
from loguru import logger
from brainles_preprocessing.preprocessor import AtlasCentricPreprocessor
from brainles_preprocessing.registration import ANTsRegistrator
from brainles_preprocessing.brain_extraction.synthstrip import SynthStripExtractor
from brainles_preprocessing.modality import CenterModality

from predict_gbm.preprocessing import register_recurrence
from predict_gbm.preprocessing.norm_ss_coregistration import normalize
from predict_gbm.utils.constants import (
    ATLAS_UNSTRIPPED_SCHEMA,
    BRAIN_MASK_SCHEMA,
    LONGITUDINAL_WARP_SCHEMA,
    MODALITY_STRIPPED_SCHEMA,
    RECURRENCE_SCHEMA,
    REGISTRATION_TRAFO_SCHEMA,
)

ATLAS = "sri24"

# Intermediate outputs (skull stripping, transforms, longitudinal registration) are written to
# this subdirectory of each patient's output folder and removed once the patient is done; the
# final images live at the top level.
WORK_FOLDER = "work"

SESSION_PATTERN = re.compile(r"(ses-\d+)")


def parse_session(nifti_file: Path) -> str:
    """Extracts the session tag (e.g. 'ses-20221213') from a file name."""
    match = SESSION_PATTERN.search(nifti_file.name)
    if match is None:
        raise ValueError(f"{nifti_file.name}: no session tag found.")
    return match.group(1)


def resolve_patient_files(patient_dir: Path) -> Optional[Dict[str, Path]]:
    """
    Resolves the three inputs of a patient folder: the skull-stripped pre-op t1c in SRI space
    (*_space-sri_t1c.nii.gz), the dose map in post-op space (*_RTDOSE_*) and the non
    skull-stripped post-op t1c (*_t1c.nii.gz without the space-sri tag). Returns None if any
    of them is missing or ambiguous.
    """
    # macOS AppleDouble copies ("._*") sit next to the images, so hidden files are excluded.
    def find(pattern: str):
        return sorted(
            f for f in patient_dir.glob(pattern) if not f.name.startswith(".")
        )

    preop_candidates = find("*_space-sri_t1c.nii.gz")
    dose_candidates = find("*_RTDOSE_*")
    # The post-op t1c naming is close to the pre-op one, so the space-sri files are excluded.
    postop_candidates = [f for f in find("*_t1c.nii.gz") if "space-sri" not in f.name]

    files = {}
    for key, candidates in (
        ("preop_t1c", preop_candidates),
        ("dose", dose_candidates),
        ("postop_t1c", postop_candidates),
    ):
        if len(candidates) != 1:
            logger.warning(
                f"{patient_dir.name}: expected exactly one {key} file, found "
                f"{[f.name for f in candidates]}, skipping patient."
            )
            return None
        files[key] = candidates[0]
    return files


def register_postop_to_sri(postop_t1c_file: Path, work_dir: Path) -> Path:
    """
    Registers the post-op t1c to SRI24 space and skull strips it, without intensity
    normalization (same steps as norm_ss_coregister, but with the t1c as the only, center
    modality and a raw instead of a normalized output). The atlas registration transform is
    saved so the dose map can be warped with it afterwards. Returns the path to the
    skull-stripped image in SRI space.
    """
    stripped_file = MODALITY_STRIPPED_SCHEMA.format(base_dir=work_dir, modality="t1c")
    center = CenterModality(
        modality_name="t1c",
        input_path=str(postop_t1c_file),
        raw_bet_output_path=str(stripped_file),
        bet_mask_output_path=str(BRAIN_MASK_SCHEMA.format(base_dir=work_dir)),
    )
    registrator = ANTsRegistrator(transformation_params={"defaultvalue": 0})
    LONGITUDINAL_WARP_SCHEMA,reprocessor = AtlasCentricPreprocessor(
        center_modality=center,
        moving_modalities=[],
        registrator=registrator,
        brain_extractor=SynthStripExtractor(),
        atlas_image_path=ATLAS_UNSTRIPPED_SCHEMA.format(atlas=ATLAS),
    )
    preprocessor.run(
        save_dir_transformations=REGISTRATION_TRAFO_SCHEMA.format(base_dir=work_dir)
    )
    return stripped_file


def warp_dose_to_sri(
    dose_file: Path, reference_sri_file: Path, work_dir: Path, outfile: Path
) -> None:
    """
    Warps the dose map into SRI space, reusing the atlas registration transform of the post-op
    t1c saved by register_postop_to_sri. The registered post-op t1c serves as the fixed
    reference (the sri24 atlas nifti itself is 4D), so the dose ends up on the exact grid of
    the t1c it is later warped to pre-op space with. The dose is a continuous map, so it is
    interpolated linearly. The dose still covers the skull, so it is masked with the brain
    mask from the skull stripping afterwards.
    """
    transform_dir = REGISTRATION_TRAFO_SCHEMA.format(base_dir=work_dir) / "t1c"
    transform_files = [str(f) for f in sorted(transform_dir.glob("*.mat"))]
    if not transform_files:
        raise FileNotFoundError(
            f"No atlas registration transform found in {transform_dir}."
        )

    dose_warped = ants.apply_transforms(
        fixed=ants.image_read(str(reference_sri_file)),
        moving=ants.image_read(str(dose_file)),
        transformlist=transform_files,
        interpolator="linear",
        defaultvalue=0,
    )

    brain_mask = ants.image_read(str(BRAIN_MASK_SCHEMA.format(base_dir=work_dir)))
    dose_masked = dose_warped * (brain_mask > 0)
    ants.image_write(dose_masked, str(outfile))


def register_postop_to_preop(
    preop_t1c_file: Path, postop_sri_t1c_file: Path, dose_sri_file: Path, work_dir: Path
) -> None:
    """
    Registers the post-op t1c (in SRI space) to the pre-op t1c with the DIRAC algorithm of
    register_recurrence and warps the SRI-space dose map into pre-op space alongside it.

    The DIRAC instance optimization expects intensity-normalized images (its NCC loss
    diverges on raw scanner intensities), so the registration is driven by percentile
    normalized copies of the two t1c images while the raw post-op t1c and the dose map are
    warped as additional modalities, keeping the stored outputs non-normalized.
    register_recurrence always warps a tumor segmentation, so an all-zero placeholder is
    passed and its warped copy is removed afterwards.
    """
    preop_norm_file = work_dir / "preop_t1c_normalized.nii.gz"
    postop_norm_file = work_dir / "postop_t1c_normalized.nii.gz"
    normalize(img_file=preop_t1c_file, outfile=preop_norm_file)
    normalize(img_file=postop_sri_t1c_file, outfile=postop_norm_file)

    postop_img = nib.load(str(postop_sri_t1c_file))
    placeholder_seg_file = work_dir / "empty_seg_placeholder.nii.gz"
    nib.save(
        nib.Nifti1Image(np.zeros(postop_img.shape, dtype=np.uint8), postop_img.affine),
        str(placeholder_seg_file),
    )

    try:
        register_recurrence(
            t1c_pre_file=preop_norm_file,
            t1c_post_file=postop_norm_file,
            recurrence_seg_file=placeholder_seg_file,
            outdir=work_dir,
            registration_algorithm="dirac",
            additional_modalities={
                "t1c_raw": postop_sri_t1c_file,
                "rtdose": dose_sri_file,
            },
        )
    finally:
        placeholder_seg_file.unlink(missing_ok=True)

    RECURRENCE_SCHEMA.format(base_dir=work_dir).unlink(missing_ok=True)


def process_patient(patient_dir: Path, outdir: Path, override: bool = False) -> None:
    """
    Processes a single patient: post-op t1c to SRI space (skull-stripped, no normalization),
    dose map to SRI space with the same transform and masked with the brain mask, post-op t1c
    from SRI to pre-op space via DIRAC, and the dose map to pre-op space with the DIRAC
    transform. Each step is skipped if its outputs already exist, unless override is True.
    The work directory with the intermediate outputs is removed once the patient is done.
    """
    files = resolve_patient_files(patient_dir)
    if files is None:
        return

    sub = patient_dir.name
    ses_post = parse_session(files["postop_t1c"])
    ses_dose = parse_session(files["dose"])

    outdir.mkdir(parents=True, exist_ok=True)
    work_dir = outdir / WORK_FOLDER

    postop_sri_out = outdir / f"{sub}_{ses_post}_space-sri_t1c.nii.gz"
    dose_sri_out = outdir / f"{sub}_{ses_dose}_space-sri_rtdose.nii.gz"
    postop_preop_out = outdir / f"{sub}_{ses_post}_space-preop_t1c.nii.gz"
    dose_preop_out = outdir / f"{sub}_{ses_dose}_space-preop_rtdose.nii.gz"

    # 1./2. Post-op t1c to SRI space (skull-stripped, no normalization) and dose map to SRI
    # space with the same transform. The two steps are run together, since the dose warp needs
    # the transform and brain mask from the work directory, which is removed once the patient
    # is done.
    if postop_sri_out.exists() and dose_sri_out.exists() and not override:
        logger.info(f"{sub}: SRI registration already done, skipping.")
    else:
        stripped_file = register_postop_to_sri(files["postop_t1c"], work_dir)
        shutil.copyfile(str(stripped_file), str(postop_sri_out))
        warp_dose_to_sri(files["dose"], postop_sri_out, work_dir, dose_sri_out)

    # 3./4. Post-op t1c (SRI space) to pre-op space via DIRAC, dose map alongside it.
    if postop_preop_out.exists() and dose_preop_out.exists() and not override:
        logger.info(f"{sub}: registration to preop already done, skipping.")
    else:
        register_postop_to_preop(
            preop_t1c_file=files["preop_t1c"],
            postop_sri_t1c_file=postop_sri_out,
            dose_sri_file=dose_sri_out,
            work_dir=work_dir,
        )
        shutil.copyfile(
            str(LONGITUDINAL_WARP_SCHEMA.format(base_dir=work_dir, modality="t1c_raw")),
            str(postop_preop_out),
        )
        shutil.copyfile(
            str(LONGITUDINAL_WARP_SCHEMA.format(base_dir=work_dir, modality="rtdose")),
            str(dose_preop_out),
        )

    if work_dir.exists():
        shutil.rmtree(work_dir)

    logger.info(f"{sub}: finished, output saved to {outdir}.")


if __name__ == "__main__":
    # Processes the patients in /mnt/Drive4/lucas/register_copy/. Each patient folder contains a
    # skull-stripped pre-op t1c in SRI space (*_space-sri_t1c.nii.gz), a dose map in post-op
    # space (*_RTDOSE_*) and a non skull-stripped post-op t1c (*_t1c.nii.gz). Per patient, the
    # post-op t1c is registered to SRI24 space and skull-stripped (no normalization), the dose
    # map is warped to SRI space with the same transform, the post-op t1c is then registered
    # from SRI to pre-op space with DIRAC and the dose map is warped along with it.
    #
    # Example:
    # nohup python -u scripts/preprocess_luca.py -cuda_device 0 > tmp_preprocess_luca.out 2>&1 &
    parser = argparse.ArgumentParser()
    parser.add_argument("-cuda_device", type=str, default="0", help="GPU id to run on.")
    parser.add_argument(
        "-datadir",
        type=str,
        default="/mnt/Drive4/lucas/register_copy",
        help="Directory containing one folder per patient.",
    )
    parser.add_argument(
        "-outdir",
        type=str,
        default="/mnt/Drive4/lucas/register_copy/processed",
        help="Directory to save processed output to, one folder per patient.",
    )
    parser.add_argument(
        "-patients",
        type=str,
        nargs="*",
        default=None,
        help="Optional subset of patient directory names to process. Defaults to all.",
    )
    parser.add_argument(
        "-override",
        action="store_true",
        help="Rerun every step even if its output already exists, overwriting previous results.",
    )
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device

    datadir = Path(args.datadir)
    outdir_root = Path(args.outdir)
    outdir_root.mkdir(parents=True, exist_ok=True)

    patient_dirs = sorted(
        p for p in datadir.iterdir() if p.is_dir() and p.name.startswith("sub-")
    )
    if args.patients:
        selected = set(args.patients)
        patient_dirs = [p for p in patient_dirs if p.name in selected]
        missing = selected - {p.name for p in patient_dirs}
        if missing:
            logger.warning(f"Patients not found in {datadir}: {sorted(missing)}.")
    logger.info(f"Processing {len(patient_dirs)} patients from {datadir}.")

    for patient_dir in patient_dirs:
        try:
            process_patient(
                patient_dir=patient_dir,
                outdir=outdir_root / patient_dir.name,
                override=args.override,
            )
        except Exception:
            logger.exception(f"{patient_dir.name}: processing failed, skipping patient.")

    logger.info(f"Finished processing. Results saved to {outdir_root}.")
