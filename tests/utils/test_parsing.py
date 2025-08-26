import tempfile
import unittest
import warnings
from pathlib import Path
from predict_gbm.utils.parsing import PatientDataset
from predict_gbm.utils.constants import RHUH_GBM_DIR, PREDICT_GBM_DIR


# Silence third-party warnings that clutter test output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


class TestPatientDataset(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_savefile = Path(self.temp_dir.name) / "dataset.json"

        self.exam_dataset_dir = RHUH_GBM_DIR
        self.derivatives_dataset_dir = PREDICT_GBM_DIR

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_loading_saving_exam_dataset(self):
        dataset = PatientDataset()
        dataset.load(self.exam_dataset_dir)
        num_patients = len(dataset)

        dataset.save(self.temp_savefile)
        self.assertTrue(self.temp_savefile.exists())

        dataset2 = PatientDataset()
        dataset2.load(self.temp_savefile)
        num_patients2 = len(dataset2)
        self.assertTrue(num_patients == num_patients2)

    def test_loading_saving_derivatives_dataset(self):
        dataset = PatientDataset()
        dataset.load(self.derivatives_dataset_dir)
        num_patients = len(dataset)

        dataset.save(self.temp_savefile)
        self.assertTrue(self.temp_savefile.exists())

        dataset2 = PatientDataset()
        dataset2.load(self.temp_savefile)
        num_patients2 = len(dataset2)
        self.assertTrue(num_patients == num_patients2)

    def test_root_change_exam_dataset(self):
        dataset = PatientDataset()
        dataset.load(self.exam_dataset_dir)
        new_root = Path("/new/root")
        dataset.set_root_dir(new_root)
        self.assertTrue(str(new_root) in str(dataset.patients[0]["patient_dir"]))
        self.assertTrue(str(new_root) in str(dataset.patients[0]["exams"][0]["t1c"]))

    def test_root_change_derivatives_dataset(self):
        dataset = PatientDataset()
        dataset.load(self.derivatives_dataset_dir)
        new_root = Path("/new/root")
        dataset.set_root_dir(new_root)
        self.assertTrue(str(new_root) in str(dataset.patients[0]["patient_dir"]))
        self.assertTrue(
            str(new_root) in str(dataset.patients[0]["derivatives"]["tumor_seg"])
        )

    def test_iteration(self):
        dataset = PatientDataset()
        dataset.load(self.exam_dataset_dir)
        counter = 0
        for patient in dataset:
            counter += 1
            for exam in patient:
                continue
        self.assertTrue(counter == len(dataset))

    def test_get_patient(self):
        dataset = PatientDataset()
        dataset.load(self.exam_dataset_dir)
        patient_id = dataset.patients[0]["patient_id"]
        patient = dataset.get_patient(patient_id=patient_id)
        self.assertTrue(patient["patient_id"] == patient_id)

    def test_remove_patient(self):
        dataset = PatientDataset()
        dataset.load(self.exam_dataset_dir)
        patient_id = dataset.patients[0]["patient_id"]
        num_patients = len(dataset)
        dataset.remove_patient(patient_id=patient_id)
        self.assertTrue(len(dataset) + 1 == num_patients)

    def test_get_patient_exams(self):
        dataset = PatientDataset()
        dataset.load(self.exam_dataset_dir)
        patient_id = dataset.patients[0]["patient_id"]
        exam = dataset.get_patient_exams(patient_id=patient_id, timepoint="preop")[0]
        self.assertTrue(exam["timepoint"] == "preop")

    def test_get_patient_derivatives(self):
        dataset = PatientDataset()
        dataset.load(self.derivatives_dataset_dir)
        patient_id = dataset.patients[0]["patient_id"]
        derivatives = dataset.get_patient_derivatives(patient_id=patient_id)
        self.assertTrue(isinstance(derivatives, dict))


if __name__ == "__main__":
    unittest.main()
