import argparse
from pathlib import Path
from predict_gbm.preprocessing import NiftiPreprocessor

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-cuda_device", type=str, default="0", help="GPU id to run on.")
    args = parser.parse_args()

    patient_root_dir = Path(
        "/mnt/Drive2/lucas/datasets/RHUH-GBM/Images/NIfTI/RHUH-GBM/RHUH-0001"
    )

    processor = NiftiPreprocessor(
        t1_file=patient_root_dir / "RHUH-0001_0_t1.nii.gz",
        t1c_file=patient_root_dir / "RHUH-0001_0_t1ce.nii.gz",
        t2_file=patient_root_dir / "RHUH-0001_0_t2.nii.gz",
        flair_file=patient_root_dir / "RHUH-0001_0_flair.nii.gz",
        pre_treatment=True,
        outdir=patient_root_dir,
        cuda_device=args.cuda_device,
        is_skull_stripped=True,
        is_coregistered=False,
    )
    processor.run()
