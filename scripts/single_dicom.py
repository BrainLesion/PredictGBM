import argparse
from pathlib import Path
from predict_gbm import DicomProcessor


if __name__ == "__main__":
    # Example:
    # nohup python -u scripts/single_dicom.py -cuda_device 0 > tmp_test_dicom.out 2>&1 &
    parser = argparse.ArgumentParser()
    parser.add_argument("-cuda_device", type=str, default="0", help="GPU id to run on.")
    args = parser.parse_args()

    model_id = "sbtc"

    patient_root_dir = Path(
        "/mnt/Drive2/lucas/datasets/RHUH-GBM/Images/DICOM/RHUH-GBM/RHUH-0001"
    )
    root_dir_preop = patient_root_dir / "01-25-2015-NA-RM CEREBRAL6NEURNAV-21029"
    root_dir_followup = patient_root_dir / "11-05-2016-NA-RM CEREBRAL-42953"

    processor = DicomProcessor(
        patient_id="RHUH-0001",
        model_id="sbtc",
        t1_preop_dir=root_dir_preop / "5.000000-Ax T1 FSE-08383",
        t1c_preop_dir=root_dir_preop / "12.000000-Ax T1 3d NEURONAVEGADOR-55128",
        t2_preop_dir=root_dir_preop / "6.000000-Ax T2 FRFSE-46501",
        flair_preop_dir=root_dir_preop / "3.000000-Ax T2 FLAIR-62646",
        t1_followup_dir=root_dir_followup / "6.000000-Ax T1 FSE-26698",
        t1c_followup_dir=root_dir_followup / "13.000000-sag SPGR 3D Isotropico-59281",
        t2_followup_dir=root_dir_followup / "4.000000-Ax T2 FSE Prop.-99802",
        flair_followup_dir=root_dir_followup / "3.000000-Ax T2 FLAIR-90513",
        outdir=patient_root_dir,
        cuda_device=args.cuda_device,
    )
    processor.run()
