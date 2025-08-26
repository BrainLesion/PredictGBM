import os
import argparse
from pathlib import Path
from predict_gbm.preprocessing import dicom_to_nifti
from predict_gbm.preprocessing import (
    norm_ss_coregister,
    register_recurrence,
)
from predict_gbm.preprocessing import DicomPreprocessor
from predict_gbm.preprosessing import run_tissue_seg_registration
from predict_gbm.preprocessing import run_brats
from predict_gbm.prediction import predict_tumor_growth
from predict_gbm.evaluation import evaluate_tumor_model


if __name__ == "__main__":
    # Example:
    # nohup python -u scripts/stepwise_processing.py -cuda_device 0 > tmp_test_dicom.out 2>&1 &
    parser = argparse.ArgumentParser()
    parser.add_argument("-cuda_device", type=str, default="0", help="GPU id to run on.")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device

    model_id = "test_model"
    patient_id = "RHUH-0001"
    outdir = Path("stepwise")
    outdir.mkdir(parents=True, exist_ok=True)

    patient_dir = Path(
        "/mnt/Drive2/lucas/datasets/RHUH-GBM/Images/DICOM/RHUH-GBM/RHUH-0001"
    )
    preop_dir = patient_dir / "01-25-2015-NA-RM CEREBRAL6NEURNAV-21029"
    followup_dir = patient_dir / "11-05-2016-NA-RM CEREBRAL-42953"

    t1_preop_dir = preop_dir / "5.000000-Ax T1 FSE-08383"
    t1c_preop_dir = preop_dir / "12.000000-Ax T1 3d NEURONAVEGADOR-55128"
    t2_preop_dir = preop_dir / "6.000000-Ax T2 FRFSE-46501"
    flair_preop_dir = preop_dir / "3.000000-Ax T2 FLAIR-62646"

    t1_followup_dir = followup_dir / "6.000000-Ax T1 FSE-26698"
    t1c_followup_dir = followup_dir / "13.000000-sag SPGR 3D Isotropico-59281"
    t2_followup_dir = followup_dir / "4.000000-Ax T2 FSE Prop.-99802"
    flair_followup_dir = followup_dir / "3.000000-Ax T2 FLAIR-90513"

    outdir_preop = outdir / "preop"
    outdir_preop.mkdir(parents=True, exist_ok=True)

    outdir_followup = outdir / "followup"
    outdir_followup.mkdir(parents=True, exist_ok=True)

    # 1. PREPROCESSING
    # a) All in one using DicomPreprocessor and NiftiPreprocessor
    preprocessor = DicomPreprocessor(
        t1_dir=t1_preop_dir,
        t1c_dir=t1c_preop_dir,
        t2_dir=t2_preop_dir,
        flair_dir=flair_preop_dir,
        outdir=outdir_preop,
        pre_treatment=True,
        cuda_device=args.cuda_device,
    )
    preprocessor.run()

    # b) Explicit pre-processing steps
    # DICOM to NIfTI conversion
    dicom_conversion_outdir = outdir_followup / "converted"
    dicom_conversion_outdir.mkdir(parents=True, exist_ok=True)
    modality_to_dir = {
        "t1": t1_followup_dir,
        "t1c": t1c_followup_dir,
        "t2": t2_followup_dir,
        "flair": flair_followup_dir,
    }

    for modality, modality_dir in modality_to_dir.items():
        dicom_to_nifti(
            input_dir=modality_dir,
            outfile=dicom_conversion_outdir / f"{modality}.nii.gz",
        )

    # Normalization, skull stripping, atlas co-registration
    skull_strip_followup_outdir = outdir_followup / "skull_stripped"
    skull_strip_followup_outdir.mkdir(parents=True, exist_ok=True)
    norm_ss_coregister(
        t1_file=dicom_conversion_outdir / "t1.nii.gz",
        t1c_file=dicom_conversion_outdir / "t1c.nii.gz",
        t2_file=dicom_conversion_outdir / "t2.nii.gz",
        flair_file=dicom_conversion_outdir / "flair.nii.gz",
        skull_strip=True,
        outdir=skull_strip_followup_outdir,
    )

    # Tumor segmentation
    tumorseg_outdir = outdir_followup / "tumor_segmentation"
    tumorseg_outdir.mkdir(parents=True, exist_ok=True)
    run_brats(
        t1_file=skull_strip_followup_outdir / "t1_bet_normalized.nii.gz",
        t1c_file=skull_strip_followup_outdir / "t1c_bet_normalized.nii.gz",
        t2_file=skull_strip_followup_outdir / "t2_bet_normalized.nii.gz",
        flair_file=skull_strip_followup_outdir / "flair_bet_normalized.nii.gz",
        outdir=tumorseg_outdir,
        pre_treatment=False,
        cuda_device=args.cuda_device,
    )

    # Tissue segmentation
    tissueseg_outdir = outdir_followup / "tissue_segmentation"
    tissueseg_outdir.mkdir(parents=True, exist_ok=True)
    run_tissue_seg_registration(
        t1_file=skull_strip_followup_outdir / "t1c_bet_normalized.nii.gz",
        outdir=tissueseg_outdir,
    )

    # Longitudinal registration
    longitudinal_outdir = outdir_followup / "longitudinal"
    longitudinal_outdir.mkdir(parents=True, exist_ok=True)
    register_recurrence(
        t1c_pre_file=outdir_preop
        / f"{patient_id}/ses-preop/skull_stripped/t1c_bet_normalized.nii.gz",
        t1c_post_file=skull_strip_followup_outdir / "t1c_bet_normalized.nii.gz",
        recurrence_seg_file=tumorseg_outdir / "tumor_seg.nii.gz",
        outdir=longitudinal_outdir,
    )

    # 2. PREDICTION
    predict_tumor_growth(
        tumorseg_file=outdir_preop
        / f"{patient_id}/ses-preop/tumor_segmentation/tumor_seg.nii.gz",
        gm_file=outdir_preop
        / f"{patient_id}/ses-preop/tissue_segmentation/gm_pbmap.nii.gz",
        wm_file=outdir_preop
        / f"{patient_id}/ses-preop/tissue_segmentation/wm_pbmap.nii.gz",
        csf_file=outdir_preop
        / f"{patient_id}/ses-preop/tissue_segmentation/csf_pbmap.nii.gz",
        model_id=model_id,
        outdir=outdir_preop,
    )

    # 3. EVALUATION
    outdir_eval = outdir_followup / "eval"
    outdir_eval.mkdir(parents=True, exist_ok=True)
    pred_file = outdir_preop / f"growth_models/{model_id}/{model_id}_pred.nii.gz"
    results = evaluate_tumor_model(
        t1c_file=outdir_preop
        / f"{patient_id}/ses-preop/skull_stripped/t1c_bet_normalized.nii.gz",
        tumorseg_file=outdir_preop
        / f"{patient_id}/ses-preop/tumor_segmentation/tumor_seg.nii.gz",
        recurrence_file=tumorseg_outdir / "tumor_seg.nii.gz",
        pred_file=pred_file,
        ctv_margin=15,
    )
    print(results)
