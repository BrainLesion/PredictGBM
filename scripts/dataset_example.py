from pathlib import Path
from predict_gbm.utils.parsing import PatientDataset
from predict_gbm.utils.constants import RHUH_GBM_DIR, PREDICT_GBM_DIR


if __name__ == "__main__":

    # Load the dataset from JSON
    dataset = PatientDataset()
    dataset.load(RHUH_GBM_DIR)
    print(f"Loaded {len(dataset.patients)} patients.")

    # Save the dataset to JSON
    tmp_file = "tmp/dataset.json"
    dataset.save(tmp_file)

    # Set the root directory of all stored paths
    new_root = Path("/new/root")
    dataset.set_root_dir(new_root)
    print(f"New root directory: {dataset.root_dir}")
    print(f"First patient directory: {dataset.patients[0]['patient_dir']}")

    # Iterate over patients and exams
    for patient in dataset:
        print(patient["patient_id"])
        for exam in patient:
            if exam["timepoint"] == "preop":
                print(exam)

    # Get a specific patient's exam
    patient_id = dataset.patients[0]["patient_id"]
    exams = dataset.get_patient_exams(patient_id=patient_id, timepoint="preop")

    # Derivatives datasets and getting a specific patient's derivatives
    dataset_derivatives = PatientDataset()
    dataset_derivatives.load(PREDICT_GBM_DIR)
    patient_id = dataset_derivatives.patients[0]["patient_id"]
    derivatives = dataset_derivatives.get_patient_derivatives(patient_id=patient_id)
    print(derivatives)
