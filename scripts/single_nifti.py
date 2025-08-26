import argparse
from pathlib import Path
from predict_gbm import NiftiProcessor


if __name__ == "__main__":
    # Example:
    # nohup python -u scripts/single_nifti.py -cuda_device 0 > tmp_test_nifti.out 2>&1 &
    parser = argparse.ArgumentParser()
    parser.add_argument("-cuda_device", type=str, default="0", help="GPU id to run on.")
    args = parser.parse_args()

    dcm2niix_location = Path("/home/home/lucas/bin/dcm2niix")
    model_id = "sbtc"

    patient_root_dir = Path(
        "/mnt/Drive2/lucas/datasets/RHUH-GBM/Images/NIfTI/RHUH-GBM/RHUH-0001"
    )
    root_dir_preop = patient_root_dir / "0"
    root_dir_followup = patient_root_dir / "2"

    processor = NiftiProcessor(
        patient_id="RHUH-0001",
        model_id="sbtc",
        t1_preop_file=root_dir_preop / "RHUH-0001_0_t1.nii.gz",
        t1c_preop_file=root_dir_preop / "RHUH-0001_0_t1ce.nii.gz",
        t2_preop_file=root_dir_preop / "RHUH-0001_0_t2.nii.gz",
        flair_preop_file=root_dir_preop / "RHUH-0001_0_flair.nii.gz",
        t1_followup_file=root_dir_followup / "RHUH-0001_2_t1.nii.gz",
        t1c_followup_file=root_dir_followup / "RHUH-0001_2_t1ce.nii.gz",
        t2_followup_file=root_dir_followup / "RHUH-0001_2_t2.nii.gz",
        flair_followup_file=root_dir_followup / "RHUH-0001_2_flair.nii.gz",
        outdir=patient_root_dir,
        cuda_device=args.cuda_device,
        tumorseg_file=root_dir_preop / "RHUH-0001_0_segmentations.nii.gz",  # Optional
        recurrenceseg_file=root_dir_followup
        / "RHUH-0001_2_segmentations.nii.gz",  # Optional,
        is_skull_stripped=True,
    )
    processor.run()
