import os
import shutil
import argparse
from pathlib import Path
from loguru import logger
from predict_gbm.preprocessing import run_brats
from predict_gbm.utils.constants import TUMORSEG_SCHEMA

if __name__ == "__main__":
    # Example:
    # nohup python -u scripts/tumorseg_rhuh_synthstrip.py -cuda_device 0 > tmp_tumorseg.out 2>&1 &
    parser = argparse.ArgumentParser()
    parser.add_argument("-cuda_device", type=str, default="0", help="GPU id to run on.")
    parser.add_argument(
        "-data_dir",
        type=str,
        default="/mnt/Drive2/lucas/cavity_data_synthstrip",
        help="Root directory containing one folder per patient.",
    )
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device

    data_dir = Path(args.data_dir)
    patient_dirs = sorted(p for p in data_dir.iterdir() if p.is_dir())

    for patient_dir in patient_dirs:
        patient_id = patient_dir.name

        for session in ["preop", "postop"]:
            tumorseg_outfile = patient_dir / f"tumorseg_{session}.nii.gz"
            if tumorseg_outfile.exists():
                logger.info(f"{patient_id} ({session}): output already exists, skipping.")
                continue

            skull_strip_dir = patient_dir / f"{session}_skull_stripping" / "skull_stripped"
            t1_file = skull_strip_dir / "t1_bet_normalized.nii.gz"
            t1c_file = skull_strip_dir / "t1c_bet_normalized.nii.gz"
            t2_file = skull_strip_dir / "t2_bet_normalized.nii.gz"
            flair_file = skull_strip_dir / "flair_bet_normalized.nii.gz"

            if not all(f.exists() for f in [t1_file, t1c_file, t2_file, flair_file]):
                logger.warning(
                    f"{patient_id} ({session}): missing skull-stripped modalities, skipping."
                )
                continue

            logger.info(f"{patient_id} ({session}): running tumor segmentation.")
            work_dir = patient_dir / f"tumorseg_{session}_work"
            work_dir.mkdir(parents=True, exist_ok=True)

            try:
                run_brats(
                    t1_file=t1_file,
                    t1c_file=t1c_file,
                    t2_file=t2_file,
                    flair_file=flair_file,
                    outdir=work_dir,
                    cuda_device=args.cuda_device,
                )
                seg_file = TUMORSEG_SCHEMA.format(base_dir=work_dir)
                shutil.move(str(seg_file), str(tumorseg_outfile))
            except Exception:
                logger.exception(f"{patient_id} ({session}): tumor segmentation failed.")
            finally:
                shutil.rmtree(work_dir, ignore_errors=True)

    logger.info("Finished tumor segmentation for all patients.")
