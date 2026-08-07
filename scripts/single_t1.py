import argparse
from pathlib import Path
from loguru import logger
from brainles_preprocessing.preprocessor import AtlasCentricPreprocessor
from brainles_preprocessing.registration import ANTsRegistrator
from brainles_preprocessing.brain_extraction.synthstrip import SynthStripExtractor
from brainles_preprocessing.normalization.percentile_normalizer import (
    PercentileNormalizer,
)
from predict_gbm.preprocessing import run_tissue_seg
from predict_gbm.preprocessing.norm_ss_coregistration import (
    SUPPORTED_ATLASES,
    initialize_center_modality,
)
from predict_gbm.utils.constants import (
    ATLAS_UNSTRIPPED_SCHEMA,
    MODALITY_STRIPPED_SCHEMA,
)

if __name__ == "__main__":
    # Example:
    # python scripts/single_t1.py /path/to/t1.nii.gz -outdir /path/to/outdir
    parser = argparse.ArgumentParser()
    parser.add_argument("t1_file", type=str, help="Path to the t1 nifti to process.")
    parser.add_argument(
        "-outdir", type=str, required=True, help="Directory where outputs are saved."
    )
    parser.add_argument(
        "-atlas",
        type=str,
        default="sri24",
        choices=sorted(SUPPORTED_ATLASES),
        help="Atlas to register the t1 into.",
    )
    args = parser.parse_args()

    t1_file = Path(args.t1_file)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Starting single-T1 preprocessing for {t1_file}.")

    # Normalization, skull stripping, atlas co-registration (same steps as norm_ss_coregister,
    # but with the t1 as the only, center modality since no other modalities are available).
    percentile_normalizer = PercentileNormalizer(
        lower_percentile=0.1,
        upper_percentile=99.9,
        lower_limit=0,
        upper_limit=1,
    )
    center = initialize_center_modality(
        modality_file=t1_file,
        modality_name="t1",
        normalizer=percentile_normalizer,
        outdir=outdir,
        skull_strip=True,
    )
    atlas_schema = ATLAS_UNSTRIPPED_SCHEMA
    registrator = ANTsRegistrator(transformation_params={"defaultvalue": 0})
    preprocessor = AtlasCentricPreprocessor(
        center_modality=center,
        moving_modalities=[],
        registrator=registrator,
        brain_extractor=SynthStripExtractor(),
        atlas_image_path=atlas_schema.format(atlas=args.atlas),
    )
    preprocessor.run()

    # Tissue segmentation
    t1_stripped_file = MODALITY_STRIPPED_SCHEMA.format(base_dir=outdir, modality="t1")
    run_tissue_seg(
        t1_file=t1_stripped_file,
        outdir=outdir,
        algorithm="antsAtroposN4",
    )

    logger.info(f"Finished single-T1 preprocessing. Output saved to {outdir}.")
