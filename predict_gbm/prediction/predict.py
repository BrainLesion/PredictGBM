from pathlib import Path
from loguru import logger
from typing import Optional
from predict_gbm.base import BasePipe
from predict_gbm.prediction.growth_model import TumorGrowthModel
from predict_gbm.utils.constants import (
    TISSUE_PBMAP_SCHEMA,
    TUMORSEG_SCHEMA,
)


def predict_tumor_growth(
    tumorseg_file: Path,
    gm_file: Path,
    wm_file: Path,
    csf_file: Path,
    model_id: str,
    outdir: Path,
    cuda_device: Optional[str] = "0",
) -> None:
    """
    Predict tumor cell concentration with  model_id as growth model.

    Parameters:
        tumorseg_file (Path): Path to the tumor segmentation NIfTI file.
        gm_file (Path): Path to the grey matter probability map NIfTI file.
        wm_file (Path): Path to the white matter probability map NIfTI file.
        csf_file (Path): Path to the cerebral spinal fluid probability map NIfTI file.
        outdir (Path): Base directory for the model output
        model_id (str): Identifier for the model. Used to load the model.
        cuda_device (str): GPU device to use.

    Returns:
        None
    """
    logger.info(f"Starting growth prediction with {model_id}.")

    model_kwargs = {
        "gm": gm_file,
        "wm": wm_file,
        "csf": csf_file,
        "tumorseg": tumorseg_file,
        "outdir": outdir,
    }

    model = TumorGrowthModel(algorithm=model_id, cuda_device=cuda_device)
    model.predict_single(**model_kwargs)
    logger.info(f"Finished growth prediction, output saved to {outdir}.")


class PredictTumorGrowthPipe(BasePipe):
    """
    Predict tumor cell concentration from a preprocessed exam using model_id as growth model.
    Operates on PredictGBM's fixed directory structure.

    Parameters:
        preop_dir (Path): Directory to the preoperative exam that has been preprocessed. Should contain the folder with the output.
        model_id (str): Identifier for the model. Used to load the model.
        cuda_device (str): GPU device to use.
        outdir (optional, Path): Base directory for the model output
    """

    def __init__(
        self,
        preop_dir: Path,
        model_id: str,
        cuda_device: Optional[str] = "0",
        outdir: Optional[Path] = None,
    ) -> None:
        super().__init__(preop_dir=preop_dir, cuda_device=cuda_device)
        self.model_id = model_id
        self.outdir = outdir or preop_dir

    def run(self) -> None:  # pragma: no cover - wrapper tested via pipeline
        logger.info(
            f"Starting growth prediction on {self.preop_dir} with {self.model_id}."
        )

        model_kwargs = {
            "gm": TISSUE_PBMAP_SCHEMA.format(base_dir=self.preop_dir, tissue="gm"),
            "wm": TISSUE_PBMAP_SCHEMA.format(base_dir=self.preop_dir, tissue="wm"),
            "csf": TISSUE_PBMAP_SCHEMA.format(base_dir=self.preop_dir, tissue="csf"),
            "tumorseg": TUMORSEG_SCHEMA.format(base_dir=self.preop_dir),
            "outdir": self.outdir,
        }

        model = TumorGrowthModel(algorithm=self.model_id, cuda_device=self.cuda_device)
        model.predict_single(**model_kwargs)
        logger.info(f"Finished growth prediction, output saved to {self.outdir}.")
