from pathlib import Path
from typing import Any, Union


# MODALITIES
RESERVED_MODALITY_NAMES = {"t1", "t1c", "t2", "flair"}


class PathSchema:
    """
    A simple helper class that wraps a format string to generate Path objects. Accepts both string and Path objects and
    supports the "/" operator to join additional format strings or paths in the same fasion as the Path object.

    Attributes:
        schema (str): The schema string containing placeholders for formatting.
    """

    def __init__(self, schema: Union[str, Path]) -> None:
        if isinstance(schema, Path):
            self.schema = str(schema)
        else:
            self.schema = schema

    def format(self, **kwargs: Any) -> Path:
        """
        Format the stored schema with the given keyword arguments and return a Path object.

        Parameters:
            **kwargs: Keyword arguments used to replace placeholders in the schema string.

        Returns:
            Path: A Path object constructed from the formatted string.
        """
        return Path(self.schema.format(**kwargs))

    def __truediv__(self, other: Union[str, Path, "PathSchema"]) -> "PathSchema":
        """
        Allow the use of the "/" operator to join another format string, Path, or PathSchema.

        Parameters:
            other (Union[str, Path, PathSchema]): The string, Path, or PathSchema to join with the current schema.

        Returns:
            PathSchema: A new PathSchema instance with the joined schema.
        """
        if isinstance(other, PathSchema):
            other_str = other.schema
        else:
            other_str = str(other)
        new_schema = str(Path(self.schema) / other_str)
        return PathSchema(new_schema)


# LABELS
TISSUE_LABELS = {"csf": 1.0, "gm": 2.0, "wm": 3.0}
TUMOR_LABELS = {"necrotic": 1, "edema": 2, "enhancing": 3}
RECURRENCE_LABELS = {"necrotic": 1, "edema": 2, "enhancing": 3, "cavity": 4}


# BASIC DIRECTORIES
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"


# ATLAS
ATLAS_DIR = DATA_DIR / "atlas"

ATLAS_UNSTRIPPED_SCHEMA = PathSchema(
    ATLAS_DIR / "{atlas}" / "{atlas}_t1_1mm_unstripped.nii.gz"
)
ATLAS_STRIPPED_SCHEMA = PathSchema(
    ATLAS_DIR / "{atlas}" / "{atlas}_t1_1mm_skull_stripped.nii.gz"
)
ATLAS_TISSUE_PBMAP_SCHEMA = PathSchema(
    ATLAS_DIR / "{atlas}" / "{atlas}_{tissue}_pbmap.nii.gz"
)


# DATASETS
DATASET_DIR = DATA_DIR / "datasets"

RHUH_GBM_DIR = DATASET_DIR / "rhuh.json"
RHUH_NIFTI_DIR = DATASET_DIR / "rhuh_nifti.json"
UPENN_GBM_DIR = DATASET_DIR / "upenn_gbm.json"
LUMIERE_DIR = DATASET_DIR / "lumiere.json"
GLIODIL_DIR = DATASET_DIR / "gliodil.json"
IVYGAP_DIR = DATASET_DIR / "ivygap.json"
CPTAC_DIR = DATASET_DIR / "cptac.json"
TCGA_GBM_DIR = DATASET_DIR / "tcga_gbm.json"
TCGA_LGG_DIR = DATASET_DIR / "tcga_lgg.json"
PREDICT_GBM_DIR = DATASET_DIR / "predict_gbm.json"

# MODELS
GROWTH_MODEL_DIR = DATA_DIR / "models"

# DIRAC longitudinal registration checkpoints (cLapIRN, Mok & Chung, BraTS-Reg 2022), stored
# next to the growth models under their original filenames: the four-modality checkpoint
# (t1, t1c, flair, t2 of both exams) from the BraTS-Reg 2022 challenge container and the
# t1c-only checkpoint from the public DIRAC release.
DIRAC_CHECKPOINT_FOUR_MODALITIES = (
    DATA_DIR
    / "models"
    / "Brats_NCC_disp_fea6b5_AdaIn64_all4_fbcon_occ01_inv5_a0015_saug_mean_keypt10_affine_ffixed_stagelvl3_112000.pth"
)
DIRAC_CHECKPOINT_T1CE = (
    DATA_DIR
    / "models"
    / "Brats_NCC_disp_fea6b5_AdaIn64_t1ce_fbcon_occ01_inv5_a0015_aug_mean_fffixed_github_stagelvl3_64000.pth"
)


# OUTPUT DIRECTOIES
OUTPUT_FOLDER = "predict_gbm"
CONVERSION_FOLDER = "nifti_conversion"
SKULL_STRIP_FOLDER = "skull_stripped"
TISSUE_SEGMENTATION_FOLDER = "tissue_segmentation"
TUMOR_SEGMENTATION_FOLDER = "tumor_segmentation"
LONGITUDINAL_DIR = "longitudinal"
MODEL_OUTPUT_DIR = "growth_models"
VISUALIZATION_OUTPUT_DIR = "visualization"


# SCHEMATA - PREPROCESSING OUTPUT
OUTPUT_BASE_SCHEMA = PathSchema("{base_dir}")

PATIENT_PREOP_OUTPUT_SCHEMA = (
    OUTPUT_BASE_SCHEMA / OUTPUT_FOLDER / "{patient_id}" / "ses-preop"
)
PATIENT_FOLLOWUP_OUTPUT_SCHEMA = (
    OUTPUT_BASE_SCHEMA / OUTPUT_FOLDER / "{patient_id}" / "ses-followup"
)

MODALITY_CONVERTED_SCHEMA = OUTPUT_BASE_SCHEMA / CONVERSION_FOLDER / "{modality}.nii.gz"

BRAIN_MASK_SCHEMA = OUTPUT_BASE_SCHEMA / SKULL_STRIP_FOLDER / "brain_mask.nii.gz"
BRAINLES_LOGFILE_SCHEMA = OUTPUT_BASE_SCHEMA / SKULL_STRIP_FOLDER / "brainles.log"
MODALITY_STRIPPED_SCHEMA = (
    OUTPUT_BASE_SCHEMA / SKULL_STRIP_FOLDER / "{modality}_skullstripped.nii.gz"
)
REGISTRATION_TRAFO_SCHEMA = OUTPUT_BASE_SCHEMA / SKULL_STRIP_FOLDER

TUMORSEG_SCHEMA = OUTPUT_BASE_SCHEMA / TUMOR_SEGMENTATION_FOLDER / "tumor_seg.nii.gz"
TUMORSEG_EDEMA_SCHEMA = (
    OUTPUT_BASE_SCHEMA / TUMOR_SEGMENTATION_FOLDER / "peritumoral_edema.nii.gz"
)
TUMORSEG_CORE_SCHEMA = (
    OUTPUT_BASE_SCHEMA
    / TUMOR_SEGMENTATION_FOLDER
    / "enhancing_non_enhancing_tumor.nii.gz"
)
HEALTHY_BRAIN_MASK_SCHEMA = (
    OUTPUT_BASE_SCHEMA / TUMOR_SEGMENTATION_FOLDER / "healthy_brain_mask.nii.gz"
)

TISSUE_SEG_BASE_SCHEMA = OUTPUT_BASE_SCHEMA / TISSUE_SEGMENTATION_FOLDER
TISSUE_PBMAP_SCHEMA = (
    OUTPUT_BASE_SCHEMA / TISSUE_SEGMENTATION_FOLDER / "{tissue}_pbmap.nii.gz"
)
REGISTRATION_MASK_SCHEMA = (
    OUTPUT_BASE_SCHEMA / TISSUE_SEGMENTATION_FOLDER / "registration_mask.nii.gz"
)

RECURRENCE_SCHEMA = OUTPUT_BASE_SCHEMA / LONGITUDINAL_DIR / "recurrence_preop.nii.gz"
# The follow-up-to-pre-op transform of every registration algorithm (dirac, syn, affine,
# rigid), stored as one ANTs/ITK displacement field (vectors in physical LPS mm) on the pre-op
# grid; apply it with ants.apply_transforms(fixed=<pre-op image>, transformlist=[<this file>]).
LONGITUDINAL_DISP_SCHEMA = (
    OUTPUT_BASE_SCHEMA / LONGITUDINAL_DIR / "longitudinal_trafo.nii.gz"
)
LONGITUDINAL_WARP_SCHEMA = (
    OUTPUT_BASE_SCHEMA / LONGITUDINAL_DIR / "{modality}_warped_longitudinal.nii.gz"
)

STANDARD_PLAN_SCHEMA = (
    OUTPUT_BASE_SCHEMA / TUMOR_SEGMENTATION_FOLDER / "standard_plan.nii.gz"
)


# CONFIG
# Central record of per-step parameters, written by update_config (predict_gbm.utils.utils)
# into a single predict_gbm_config.json per outdir. Step names below are the top-level keys.
CONFIG_FILENAME = "predict_gbm_config.json"
CONFIG_SCHEMA = OUTPUT_BASE_SCHEMA / CONFIG_FILENAME

CONFIG_STEP_NORM_SS_COREGISTER = "norm_ss_coregister"
CONFIG_STEP_REGISTER_RECURRENCE = "register_recurrence"
CONFIG_STEP_TISSUE_SEG = "run_tissue_seg"
CONFIG_STEP_TUMOR_SEG = "run_brats"

# Skull stripping is currently always performed via SynthStrip in norm_ss_coregister; this
# constant is the single source of truth for that name so config and code cannot drift apart.
SKULL_STRIP_ALGORITHM_SYNTHSTRIP = "SynthStrip"


# LONGITUDINAL REGISTRATION
# Sanity check of the ANTs affine pre-registration in the DIRAC branch of register_recurrence
# (warning only, no fallback). Limits on the linear part A of the fitted pre-op-to-follow-up
# matrix and on its translation after removing the rotation-centre term.
LONGITUDINAL_AFFINE_MAX_ABS_LOG_DET = 0.2
LONGITUDINAL_AFFINE_SINGULAR_VALUE_RANGE = (0.8, 1.25)
LONGITUDINAL_AFFINE_MAX_ROTATION_DEG = 15.0
LONGITUDINAL_AFFINE_MAX_TRANSLATION_MM = 20.0


# SCHEMATA - MODEL PREDICTIONS
DOCKER_OUTPUT_SCHEMA = "{subject_id}.nii.gz"
PREDICTION_OUTPUT_SCHEMA = (
    OUTPUT_BASE_SCHEMA / MODEL_OUTPUT_DIR / "{algo_id}" / "{algo_id}_pred.nii.gz"
)
MODEL_PLAN_SCHEMA = (
    OUTPUT_BASE_SCHEMA / MODEL_OUTPUT_DIR / "{algo_id}" / "{algo_id}_plan.nii.gz"
)
METRICS_SCHEMA = OUTPUT_BASE_SCHEMA / MODEL_OUTPUT_DIR / "{algo_id}_metrics.json"


# SCHEMATA - VISUALIZATION
TUMOR_VISUALIZATION_SCHEMA = (
    OUTPUT_BASE_SCHEMA / VISUALIZATION_OUTPUT_DIR / "axial_coronal_sliced_tumor.pdf"
)
RECURRENCE_VISUALIZATION_SCHEMA = (
    OUTPUT_BASE_SCHEMA
    / VISUALIZATION_OUTPUT_DIR
    / "axial_coronal_sliced_recurrence.pdf"
)
