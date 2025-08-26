import numpy as np
import nibabel as nib
from pathlib import Path
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid


class MockContainer:
    def __init__(self, status_code):
        self.status_code = status_code

    def attach(self, stdout=True, stderr=True, stream=True, logs=True):
        return iter([b"line1", b"line2"])

    def wait(self):
        return {"StatusCode": self.status_code}


def generate_mock_dicom_series(
    outdir: Path,
    shape=(16, 16, 16),
    random=False,
    dtype=np.uint16,
    seed=0,
):
    """Create a small DICOM series for testing."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    rows, cols, num_slices = shape
    rng = np.random.default_rng(seed)

    volume = (
        rng.integers(0, np.iinfo(dtype).max, size=shape, dtype=dtype)
        if random
        else np.zeros(shape, dtype=dtype)
    )

    file_meta = FileMetaDataset()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    study_uid = generate_uid()
    series_uid = generate_uid()

    for k in range(num_slices):
        ds = FileDataset("", {}, file_meta=file_meta, preamble=b"\0" * 128)
        ds.Rows, ds.Columns = rows, cols
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.PixelRepresentation = 0
        ds.BitsAllocated = ds.BitsStored = volume.dtype.itemsize * 8
        ds.HighBit = ds.BitsStored - 1
        ds.StudyInstanceUID = study_uid
        ds.SeriesInstanceUID = series_uid
        ds.SOPInstanceUID = generate_uid()
        ds.InstanceNumber = k + 1
        ds.PixelData = volume[:, :, k].tobytes()
        ds.save_as(outdir / f"slice_{k+1:03d}.dcm")

    return outdir


def generate_mock_nifti(shape=(16, 16, 16), seed=0):
    rng = np.random.default_rng(seed)
    arr = rng.normal(size=shape).astype(np.float32)
    return nib.Nifti1Image(arr, np.eye(4))
