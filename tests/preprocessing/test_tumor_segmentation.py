import tempfile
import unittest
import warnings
import numpy as np
import nibabel as nib
from pathlib import Path
from unittest.mock import MagicMock, patch
from tests.helpers import generate_mock_nifti
from predict_gbm.preprocessing import tumor_segmentation as ts
from predict_gbm.utils.constants import (
    TUMOR_LABELS,
    TUMORSEG_CORE_SCHEMA,
    TUMORSEG_EDEMA_SCHEMA,
    TUMORSEG_SCHEMA,
    TUMOR_SEGMENTATION_FOLDER,
)
from brats.constants import (
    AdultGliomaPreTreatmentAlgorithms,
    AdultGliomaPostTreatmentAlgorithms,
)

# Silence third-party warnings that clutter test output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


class TestTumorSegmentation(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.base_dir = Path(self.temp_dir.name)
        self.tumor_dir = self.base_dir / TUMOR_SEGMENTATION_FOLDER
        self.tumor_dir.mkdir(parents=True, exist_ok=True)

        seg_arr = np.zeros((2, 2, 2), dtype=np.int32)
        seg_arr[0, 0, 0] = TUMOR_LABELS["necrotic"]
        seg_arr[0, 1, 0] = TUMOR_LABELS["edema"]
        seg_arr[1, 0, 0] = TUMOR_LABELS["enhancing"]
        seg_img = nib.Nifti1Image(seg_arr, np.eye(4))
        self.seg_file = self.tumor_dir / "tumor_seg.nii.gz"
        nib.save(seg_img, self.seg_file)

        mock_img = generate_mock_nifti(shape=(2, 2, 2))
        self.t1_file = self.base_dir / "t1.nii.gz"
        self.t1c_file = self.base_dir / "t1c.nii.gz"
        self.t2_file = self.base_dir / "t2.nii.gz"
        self.flair_file = self.base_dir / "flair.nii.gz"
        for f in [self.t1_file, self.t1c_file, self.t2_file, self.flair_file]:
            nib.save(mock_img, f)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_split_segmentation_creates_correct_masks(self):
        ts.split_segmentation(self.seg_file, self.base_dir)

        core_file = TUMORSEG_CORE_SCHEMA.format(base_dir=self.base_dir)
        edema_file = TUMORSEG_EDEMA_SCHEMA.format(base_dir=self.base_dir)
        self.assertTrue(core_file.exists())
        self.assertTrue(edema_file.exists())

        core_data = np.asanyarray(nib.load(core_file).dataobj)
        edema_data = np.asanyarray(nib.load(edema_file).dataobj)

        expected_core = np.zeros((2, 2, 2), dtype=np.int32)
        expected_core[0, 0, 0] = 1
        expected_core[1, 0, 0] = 1

        expected_edema = np.zeros((2, 2, 2), dtype=np.int32)
        expected_edema[0, 1, 0] = 1

        self.assertTrue(np.array_equal(core_data, expected_core))
        self.assertTrue(np.array_equal(edema_data, expected_edema))

    @patch("predict_gbm.preprocessing.tumor_segmentation.split_segmentation")
    @patch(
        "predict_gbm.preprocessing.tumor_segmentation.AdultGliomaPreTreatmentSegmenter"
    )
    def test_run_brats_pre_treatment(self, mock_seg_cls, mock_split):
        mock_segmenter = MagicMock()
        mock_seg_cls.return_value = mock_segmenter

        ts.run_brats(
            self.t1_file,
            self.t1c_file,
            self.t2_file,
            self.flair_file,
            self.base_dir,
            pre_treatment=True,
        )

        mock_seg_cls.assert_called_once_with(
            algorithm=AdultGliomaPreTreatmentAlgorithms.BraTS23_1,
            cuda_devices="0",
        )

        seg_outfile = str(TUMORSEG_SCHEMA.format(base_dir=self.base_dir))
        mock_segmenter.infer_single.assert_called_once_with(
            t1n=str(self.t1_file),
            t1c=str(self.t1c_file),
            t2w=str(self.t2_file),
            t2f=str(self.flair_file),
            output_file=seg_outfile,
        )
        mock_split.assert_called_once_with(seg_outfile, self.base_dir)

    @patch("predict_gbm.preprocessing.tumor_segmentation.split_segmentation")
    @patch(
        "predict_gbm.preprocessing.tumor_segmentation.AdultGliomaPostTreatmentSegmenter"
    )
    def test_run_brats_post_treatment(self, mock_seg_cls, mock_split):
        mock_segmenter = MagicMock()
        mock_seg_cls.return_value = mock_segmenter

        ts.run_brats(
            self.t1_file,
            self.t1c_file,
            self.t2_file,
            self.flair_file,
            self.base_dir,
            pre_treatment=False,
            cuda_device="1",
        )

        mock_seg_cls.assert_called_once_with(
            algorithm=AdultGliomaPostTreatmentAlgorithms.BraTS24_1,
            cuda_devices="1",
        )

        seg_outfile = str(TUMORSEG_SCHEMA.format(base_dir=self.base_dir))
        mock_segmenter.infer_single.assert_called_once_with(
            t1n=str(self.t1_file),
            t1c=str(self.t1c_file),
            t2w=str(self.t2_file),
            t2f=str(self.flair_file),
            output_file=seg_outfile,
        )
        mock_split.assert_called_once_with(seg_outfile, self.base_dir)


if __name__ == "__main__":
    unittest.main()
