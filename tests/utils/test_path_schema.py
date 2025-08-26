import unittest
import warnings
from pathlib import Path
from predict_gbm.utils.constants import PathSchema

# Silence third-party warnings that clutter test output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


class TestPathSchema(unittest.TestCase):
    def test_format_returns_path(self):
        schema = PathSchema("/tmp/{patient_id}/image.nii.gz")
        schema_formatted = schema.format(patient_id="123")
        self.assertEqual(schema_formatted, Path("/tmp/123/image.nii.gz"))

    def test_truediv_joins_schema(self):
        schema = PathSchema("/tmp") / "data" / "{patient_id}"
        schema_formatted = schema.format(patient_id="001")
        self.assertEqual(schema_formatted, Path("/tmp/data/001"))

    def test_truediv_with_pathschema(self):
        schema1 = PathSchema("/tmp/{root}")
        schema2 = PathSchema("sub/{name}.nii.gz")
        combined = (schema1 / schema2).format(root="r", name="n")
        self.assertEqual(combined, Path("/tmp/r/sub/n.nii.gz"))
