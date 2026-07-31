import os
import shlex
import subprocess
from pathlib import Path
from loguru import logger


def _strip_nifti_suffix(filename: str) -> str:
    """Return filename without a trailing .nii.gz or .nii extension."""
    if filename.endswith(".nii.gz"):
        return filename[: -len(".nii.gz")]
    if filename.endswith(".nii"):
        return filename[: -len(".nii")]
    return filename


def remove_postfixes(export_dir: Path) -> None:
    """Remove postfixes created by dcm2niix (e.g. '_real') from filenames in the given directory."""
    for f in export_dir.iterdir():
        if f.is_file() and ("_" in f.name) and not f.name.endswith(".log"):
            parts = f.name.split(".")
            modality = parts[0].split("_")[0]
            new_name = ".".join([modality] + parts[1:])
            new_path = export_dir / new_name
            # prevent overiding, in case a dicom has to be converted manually
            while os.path.isfile(new_path):
                new_path = str(new_path).split(".")[0] + "_a" + ".nii.gz"
            f.rename(new_path)
            logger.info(f"Renamed postfix file {f} to {new_path}.")


def dicom_to_nifti(
    input_dir: Path, outfile: Path, dcm2niix_location: Path = "dcm2niix"
) -> None:
    """Convert DICOM files to NIfTI format using dcm2niix."""
    try:
        outfile.parent.mkdir(parents=True, exist_ok=True)
        output_basename = _strip_nifti_suffix(outfile.name)
        cmd_readable = (
            str(dcm2niix_location)
            + " -d 9 -f "
            + output_basename
            + " -z y -o"
            + ' "'
            + str(outfile.parent)
            + '" "'
            + str(input_dir)
            + '"'
        )

        logger.info(f"Running: {cmd_readable}")
        cmd = shlex.split(cmd_readable)

        log_file = outfile.parent / f"{output_basename}_conversion.log"
        with open(log_file, "w", encoding="utf-8") as logf:
            subprocess.run(cmd, stdout=logf, stderr=logf, check=False)

        remove_postfixes(outfile.parent)
        logger.debug(f"Nifti conversion complete for {input_dir}.")

    except Exception as e:
        logger.error(
            f"Error while trying to convert {input_dir} via {cmd_readable}: {e}"
        )
