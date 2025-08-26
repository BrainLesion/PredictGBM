import argparse
import numpy as np
from pathlib import Path
from predict_gbm.utils.parsing import PatientDataset
from predict_gbm.utils.constants import PREDICT_GBM_DIR
from predict_gbm.prediction import predict_tumor_growth
from predict_gbm.evaluation import evaluate_tumor_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-cuda_device", type=str, default="0", help="GPU id to run on.")
    args = parser.parse_args()

    model_id = "test_model"
    outdir = Path("predict_eval")
    outdir.mkdir(parents=True, exist_ok=True)
    predict_gbm_rootdir = "/path/to/predict_gbm/dataset/rootdir"

    predict_gbm_dataset = PatientDataset()
    predict_gbm_dataset.load(PREDICT_GBM_DIR)
    predict_gbm_dataset.set_root_dir(predict_gbm_rootdir)

    all_results = []
    for patient in predict_gbm_dataset:
        print(patient["patient_id"])

        derivatives = patient["derivatives"]

        predict_tumor_growth(
            tumorseg_file=derivatives["tumor_seg"],
            gm_file=derivatives["gm_pbmap"],
            wm_file=derivatives["wm_pbmap"],
            csf_file=derivatives["csf_pbmap"],
            model_id=model_id,
            outdir=outdir,
        )

        pred_file = outdir / f"growth_models/{model_id}/{model_id}_pred.nii.gz"
        results = evaluate_tumor_model(
            tumorseg_file=derivatives["tumor_seg"],
            recurrence_file=derivatives["recurrence_seg"],
            pred_file=pred_file,
            brain_mask_file=derivatives["brain_mask"],
            ctv_margin=15,
        )
        all_results.append(results)

    coverages_standard = [r["recurrence_coverage_standard"] for r in all_results]
    coverages_model = [r["recurrence_coverage_model"] for r in all_results]

    mean_coverage_standard = 100 * np.mean(coverages_standard)
    mean_coverage_model = 100 * np.mean(coverages_model)

    print("Finished evaluation.")
    print(f"Standard plan coverge: {mean_coverage_standard:.2f}")
    print(f"Model plan coverge: {mean_coverage_model:.2f}")
