import json
from pathlib import Path
from loguru import logger
from typing_extensions import Required, NotRequired
from typing import Dict, Iterator, List, Literal, Optional, TypedDict, Union


class Exam(TypedDict):
    """
    TypedDict representing a single exam with modality names as keys.
    """

    t1c: Required[Path]
    timepoint: Required[Literal["preop", "postop", "followup"]]

    t1: NotRequired[Path]
    t2: NotRequired[Path]
    flair: NotRequired[Path]
    pet: NotRequired[Path]
    diffusion: NotRequired[Path]
    perfusion: NotRequired[Path]
    tumorseg: NotRequired[Path]


class PatientDict(TypedDict):
    """TypedDict helper describing the expected keys for a patient."""

    patient_id: Required[str]
    patient_dir: Required[Path]

    exams: NotRequired[List[Exam]]
    derivatives: NotRequired[List[Dict]]


class Patient(dict):
    """
    Iterable dict representing a patient with exams and derivatives
    """

    def __iter__(self) -> Iterator[Exam]:
        return iter(self.get("exams", []))


class PatientDataset:
    """Base class for datasets consisting of patients."""

    def __init__(self, dataset_id: str = "", root_dir: Union[str, Path] = "."):
        root_dir = Path(root_dir).resolve()

        self.dataset_id = dataset_id
        self.root_dir = Path(root_dir).resolve()
        self.patients: List[Patient] = []

    def __len__(self) -> int:
        return len(self.patients)

    def __iter__(self) -> Iterator[Patient]:
        """Iterate directly over the stored patients."""
        return iter(self.patients)

    def _convert_path(self, path: Optional[Path]) -> Optional[str]:
        """Helper function that converts Path objects to str while leaving None values as is."""
        return str(path) if path is not None else None

    def _substitute_root(
        self, substitute_path: Path, old_root: Path, new_root: Path
    ) -> Path:
        """Helper function that replaces part of a Path object with a specified subpath."""
        substitu_str = str(substitute_path)
        old_root_str = str(old_root).strip("/")
        new_root_str = str(new_root).strip("/")
        return Path(substitu_str.replace(old_root_str, new_root_str))

    def save(self, out: Union[str, Path]) -> None:
        """Saves dataset by converting it to a dict and saving as json. Path objects are converted to strings for readability."""
        out_path = Path(out)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        dataset_dict = {
            "dataset_id": self.dataset_id,
            "root_dir": str(self.root_dir),
            "patients": [],
        }

        for patient in self.patients:
            patient_dict = {
                "patient_id": patient["patient_id"],
                "patient_dir": self._convert_path(patient["patient_dir"]),
            }

            if "exams" in patient:
                exams = []
                for exam in patient["exams"]:
                    exam_dict = {}
                    for key, value in exam.items():
                        if key == "timepoint":
                            exam_dict[key] = value
                        else:
                            exam_dict[key] = self._convert_path(value)
                    exams.append(exam_dict)
                patient_dict["exams"] = exams

            if "derivatives" in patient:
                derivatives = {
                    key: self._convert_path(value)
                    for key, value in patient["derivatives"].items()
                }
                patient_dict["derivatives"] = derivatives

            dataset_dict["patients"].append(patient_dict)

        with out_path.open("w") as f:
            json.dump(dataset_dict, f, indent=2)

    def load(self, path: Union[str, Path]) -> None:
        """Loads dataset from a json object as created by the save method."""
        load_path = Path(path)
        with load_path.open("r") as f:
            data = json.load(f)

        self.dataset_id = data.get("dataset_id", self.dataset_id)
        self.root_dir = Path(data.get("root_dir", self.root_dir)).resolve()
        self.patients = []

        for patient in data.get("patients", []):
            patient_dict = {
                "patient_id": patient["patient_id"],
                "patient_dir": Path(patient["patient_dir"]),
            }

            if "exams" in patient:
                exams = []
                for exam in patient["exams"]:
                    exam_dict = {}
                    for key, value in exam.items():
                        if key == "timepoint":
                            exam_dict[key] = value
                        else:
                            exam_dict[key] = Path(value) if value is not None else None
                    exams.append(exam_dict)
                patient_dict["exams"] = exams

            if "derivatives" in patient:
                derivatives = {
                    key: Path(value) if value is not None else None
                    for key, value in patient["derivatives"].items()
                }
                patient_dict["derivatives"] = derivatives

            self.patients.append(Patient(patient_dict))

    def set_root_dir(self, new_root_dir: Union[Path, str]) -> None:
        """Updates the root directory and adjusts all patient and exam paths by string substituation."""
        new_root = Path(new_root_dir).resolve()
        old_root = self.root_dir.resolve()
        self.root_dir = new_root

        for patient in self.patients:
            # Update patient_dir
            patient["patient_dir"] = self._substitute_root(
                patient["patient_dir"], old_root, new_root
            )

            # Update exams
            if "exams" in patient:
                for exam in patient["exams"]:
                    for key, value in exam.items():
                        if key == "timepoint" or value is None:
                            continue
                        exam[key] = self._substitute_root(value, old_root, new_root)

            # Update derivatives
            if "derivatives" in patient:
                for key, value in patient["derivatives"].items():
                    if value is None:
                        continue
                    patient["derivatives"][key] = self._substitute_root(
                        value, old_root, new_root
                    )

    def remove_patient(self, patient_id: str) -> bool:
        """
        Remove a patient from the dataset.
        """
        for i, patient in enumerate(self.patients):
            if patient["patient_id"] == patient_id:
                del self.patients[i]
                logger.info(f"Removed patient {patient_id} from dataset.")
                return True

        logger.warning(f"Patient {patient_id} not found in dataset.")
        return False

    def get_patient(self, patient_id: str) -> Patient | None:
        """Retrieves a patient from its id."""
        for patient in self.patients:
            if patient["patient_id"] == patient_id:
                return patient
        logger.info(f"No patient with patient id {patient_id} found.")
        return None

    def get_patient_exams(
        self, patient_id: str, timepoint: Optional[str] = None
    ) -> List[Exam]:
        """
        Retrieves exams for a specific patient, optionally filtered by timepoint (None, preop, postop, followup).
        """
        patient = self.get_patient(patient_id)
        if patient is None or "exams" not in patient.keys():
            return []

        if timepoint is None:
            return patient["exams"].copy()

        filtered_exams = []
        for exam in patient["exams"]:
            if exam["timepoint"] == timepoint:
                filtered_exams.append(exam)
        return filtered_exams

    def get_patient_derivatives(self, patient_id: str) -> Dict:
        """Retrieves derivatives for a specific patient."""
        patient = self.get_patient(patient_id)
        if patient is None or "derivatives" not in patient.keys():
            return dict()
        return patient["derivatives"]
