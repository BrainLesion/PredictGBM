def test_public_imports():
    """Ensure documented import paths are available."""
    from predict_gbm import DicomProcessor, NiftiProcessor
    from predict_gbm.prediction import predict_tumor_growth
    from predict_gbm.evaluation import evaluate_tumor_model
    from predict_gbm.preprocessing import (
        DicomPreprocessor,
        NiftiPreprocessor,
        run_brats,
        run_tissue_seg_registration,
        dicom_to_nifti,
        norm_ss_coregister,
        register_recurrence,
    )

    # Access the imported names to silence linter warnings
    assert DicomProcessor and NiftiProcessor
    assert predict_tumor_growth and evaluate_tumor_model
    assert DicomPreprocessor and NiftiPreprocessor
    assert run_brats and run_tissue_seg_registration
    assert dicom_to_nifti and norm_ss_coregister and register_recurrence
