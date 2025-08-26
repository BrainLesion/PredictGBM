import unittest
import warnings
import numpy as np
from predict_gbm.evaluation.metrics import (
    recurrence_coverage,
    missed_voxels,
    roc_auc,
)

# Silence third-party warnings that clutter test output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


class TestMetrics(unittest.TestCase):
    def test_recurrence_coverage(self):
        rec = np.array([[1, 0], [0, 1]], dtype=np.int32)
        plan = np.array([[1, 0], [0, 1]], dtype=np.int32)
        self.assertAlmostEqual(recurrence_coverage(rec, plan), 1.0)

    def test_missed_voxels(self):
        rec = np.array([[1, 0], [0, 1]], dtype=np.int32)
        plan = np.array([[1, 0], [0, 0]], dtype=np.int32)
        self.assertEqual(missed_voxels(rec, plan), 1)

    def test_roc_auc(self):
        pred = np.array([[0.1, 0.4], [0.35, 0.8]], dtype=np.float32)
        seg = np.array([[0, 3], [0, 3]], dtype=np.int32)
        mask = np.ones_like(seg, dtype=bool)
        self.assertAlmostEqual(roc_auc(pred, seg, mask), 1.0)
        pred2 = np.array([[0.1, 0.2]], dtype=np.float32)
        seg2 = np.array([[0, 0]], dtype=np.int32)
        mask2 = np.ones_like(seg2, dtype=bool)
        self.assertEqual(roc_auc(pred2, seg2, mask2), 0.0)


if __name__ == "__main__":
    unittest.main()
