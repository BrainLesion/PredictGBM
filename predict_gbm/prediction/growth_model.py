import sys
import shutil
from pathlib import Path
from loguru import logger
from typing import Dict, Optional
from brats.utils.data_handling import InferenceSetup
from predict_gbm.prediction.docker_funcs import run_container
from predict_gbm.utils.constants import (
    DOCKER_OUTPUT_SCHEMA,
    PREDICTION_OUTPUT_SCHEMA,
    GROWTH_MODEL_DIR,
)


def load_algorithms(model_dir: Path = GROWTH_MODEL_DIR) -> Dict[str, Path]:
    """
    Parses model_dir for growth model images in the form of *.tar
    """
    if not model_dir.is_dir():
        raise ValueError(
            f"Provided path to model directory {model_dir} is not a directory."
        )

    available_models = {
        model_file.stem: model_file
        for model_file in model_dir.iterdir()
        if model_file.suffix == ".tar"
    }
    return available_models


class TumorGrowthModel:
    """A class that utilizes Docker images of tumor growth models to make tumor growth predictions."""

    def __init__(
        self, algorithm: str, cuda_device: Optional[str] = "0", force_cpu: bool = False
    ):
        self.algorithm_list = load_algorithms()

        if algorithm not in self.algorithm_list.keys():
            raise ValueError(f"algorithm not in {self.algorithm_list.keys()}.")

        self.algorithm = algorithm
        self.model_file = self.algorithm_list[algorithm]
        self.cuda_device = cuda_device
        self.force_cpu = force_cpu

    def _standardize_input_files(
        self, tmp_data_dir: Path, subject_id: int, inputs: Dict[str, Path]
    ) -> None:
        """Standardize the input images for a single subject to match requirements of all algorithms and save them in @tmp_data_dir/Patient-@subject_id.
        Example:
                Patient-00000 \n
                ┣ 00000-gm.nii.gz \n
                ┣ 00000-wm.nii.gz \n
                ┣ 00000-csf.nii.gz \n
                ┣ 00000-tumorseg.nii.gz \n
                ┗ 00000-pet.nii.gz \n

        Args:
            tmp_data_dir: Temporary folder to cache patient images
            subject_id: Subject ID to be used for the folder and filenames
            inputs: Dictionary with the input images
            subject_modality_separator: Separator between the subject ID and the modality
        """

        subject_folder = tmp_data_dir / f"Patient-{subject_id}"
        subject_folder.mkdir(parents=True, exist_ok=True)
        try:
            for modality, path in inputs.items():
                shutil.copy(
                    str(path), str(subject_folder / f"{subject_id}-{modality}.nii.gz")
                )
        except FileNotFoundError as e:
            logger.error(f"Error while standardizing files: {e}")
            sys.exit(1)

    def _process_output(self, tmp_outdir: Path, subject_id: str, outdir: Path) -> None:
        """Moves the output of the docker model to the specified directory.

        Args:
            tmp_outdir: Folder with the algorithm output
            subject_id: Subject ID of the output
            outdir: Path to the desired output file
        """
        docker_outfile = tmp_outdir / DOCKER_OUTPUT_SCHEMA.format(subject_id=subject_id)
        prediction_outfile = PREDICTION_OUTPUT_SCHEMA.format(
            base_dir=outdir.resolve(), algo_id=self.algorithm
        )
        prediction_outfile.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(docker_outfile, prediction_outfile)

    def predict_single(
        self,
        gm: Path,
        wm: Path,
        csf: Path,
        tumorseg: Path,
        outdir: Path,
        pet: Optional[Path] = None,
        log_file: Optional[Path] = None,
    ) -> None:
        """Predict tumor growth on a single subject with the provided images and save the result to the output file.

        Args:
            gm: Path to gm probability map
            wm: Path to wm probability map
            csf: Path to csf probability map
            tumorseg: Path to tumor segmentation
            pet: Path to pet image
            outdir: Path to save the segmentation
            log_file: Save logs to this file
        """
        inputs = {"gm": gm, "wm": wm, "csf": csf, "tumorseg": tumorseg}
        if pet is not None:
            inputs["pet"] = pet

        with InferenceSetup(log_file=log_file) as (tmp_data_dir, tmp_outdir):

            # the id here is arbitrary
            subject_id = "00000"

            self._standardize_input_files(
                tmp_data_dir=tmp_data_dir,
                subject_id=subject_id,
                inputs=inputs,
            )

            run_container(
                algorithm=self.algorithm,
                model_file=self.model_file,
                data_dir=tmp_data_dir,
                outdir=tmp_outdir,
                cuda_device=self.cuda_device,
                force_cpu=self.force_cpu,
            )

            self._process_output(
                tmp_outdir=tmp_outdir,
                subject_id=subject_id,
                outdir=outdir,
            )
            logger.info(
                f"Finisehd growth prediction. Saved output to: {outdir.resolve()}"
            )
