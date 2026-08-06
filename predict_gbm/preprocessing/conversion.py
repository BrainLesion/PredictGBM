import shlex
import subprocess
import numpy as np
import nibabel as nib
from pathlib import Path
from typing import Literal
from loguru import logger
from collections.abc import Iterable
from dipy.core.gradients import gradient_table
from dipy.io.gradients import read_bvals_bvecs
from dipy.reconst.dti import TensorModel
from dipy.segment.mask import median_otsu

# Floor applied to the DWI signal before the log transform in tensor fitting, so that
# noise-only (near-zero) voxels do not produce -inf/nan. Well below any real signal.
_SIGNAL_EPSILON = 1e-6


def _strip_nifti_suffix(filename: str) -> str:
    """Return filename without a trailing .nii.gz or .nii extension."""
    if filename.endswith(".nii.gz"):
        return filename[: -len(".nii.gz")]
    if filename.endswith(".nii"):
        return filename[: -len(".nii")]
    return filename


def remove_postfixes(files: Iterable[Path], basename: str) -> None:
    """Remove postfixes created by dcm2niix (e.g. '_real') from filenames in the given directory."""
    if "." in basename:
        raise ValueError(
            f"basename must not contain '.': {basename!r}. Postfix detection "
            f"splits filenames at the first '.', so a dotted basename would "
            f"silently skip all files."
        )

    for f in files:
        if not f.is_file() or f.suffix == ".log":
            continue
        stem, dot, ext = f.name.partition(".")  # 'x_e1', '.', 'nii.gz'
        if not stem.startswith(f"{basename}_"):
            continue
        new_path = f.with_name(basename + dot + ext)
        if new_path.exists():
            raise FileExistsError(
                f"Cannot strip postfix from {f.name}: {new_path.name} already "
                f"exists. Likely a multi-echo or multi-series input."
            )
        f.rename(new_path)
        logger.info(f"Renamed {f.name} -> {new_path.name}")


def dicom_to_nifti(
    input_dir: Path, outfile: Path, dcm2niix_location: str | Path = "dcm2niix"
) -> Path:
    """Convert a single DICOM series to NIfTI via dcm2niix.
    """
    outfile.parent.mkdir(parents=True, exist_ok=True)
    basename = _strip_nifti_suffix(outfile.name)
    if "." in basename:
        raise ValueError(
            f"outfile stem must not contain '.': {outfile.name!r} "
            f"(stem {basename!r})"
        )

    cmd = [
        str(dcm2niix_location),
        "-d", "9",
        "-b", "y",
        "-z", "y",
        "-f", basename,
        "-o", str(outfile.parent),
        str(input_dir),
    ]
    logger.info(f"Running: {shlex.join(cmd)}")

    before = set(outfile.parent.iterdir())
    log_file = outfile.parent / f"{basename}_conversion.log"
    with open(log_file, "w", encoding="utf-8") as logf:
        proc = subprocess.run(cmd, stdout=logf, stderr=logf, check=False)
    if proc.returncode != 0:
        raise RuntimeError(
            f"dcm2niix exited {proc.returncode} for {input_dir}; see {log_file}"
        )

    created = {p for p in outfile.parent.iterdir() if p not in before}
    remove_postfixes(created - {log_file}, basename)

    if not outfile.exists():
        raise RuntimeError(
            f"dcm2niix produced no {outfile.name} from {input_dir}; created: "
            f"{sorted(p.name for p in created)}. See {log_file}."
        )
    logger.debug(f"Nifti conversion complete for {input_dir}.")
    return outfile


def dti_to_adc(
    infile: Path,
    outfile: Path,
    bval: Path | None = None,
    bvec: Path | None = None,
    bval_max: float = 1200.0,
    b0_threshold: float = 50.0,
    fit_method: Literal["WLS", "OLS"] = "WLS",
) -> Path:
    """
    Fit a diffusion tensor to a raw 4D DWI series and derive an ADC (mean diffusivity) map.

    Only volumes with b <= bval_max are used for the fit (b0s, i.e. b <= b0_threshold, are
    always included regardless of bval_max). This is a modeling choice, not a convenience
    filter: DTI assumes monoexponential signal decay, and including higher shells makes the
    fit absorb kurtosis curvature into D, biasing the recovered diffusivity downward. The
    tensor is fit over the full image with dipy's TensorModel, without a brain mask, by
    design.

    MD is computed as trace(D)/3 directly from the fitted diffusion tensor (not by
    eigendecomposing and averaging eigenvalues, and without clipping individual
    eigenvalues, which would bias MD upward at low SNR). The result is scaled to units of
    1e-6 mm^2/s (i.e. the mm^2/s value times 1e6), assuming b-values in s/mm^2. Non-finite
    voxels are set to 0. Negative MD indicates a fit failure and is not clamped; the
    affected fraction is logged.

    Parameters:
        infile (Path): Path to the 4D raw DWI NIfTI (one volume per acquired gradient
            direction/b-value, including b0s).
        outfile (Path): Output path for the 3D MD/ADC map. Compression follows the
            '.nii'/'.nii.gz' suffix.
        bval (Path | None): Path to the .bval sidecar. Defaults to a file with the same
            stem as infile and a '.bval' suffix.
        bvec (Path | None): Path to the .bvec sidecar. Defaults to a file with the same
            stem as infile and a '.bvec' suffix.
        bval_max (float): Maximum b-value (s/mm^2) included in the tensor fit.
        b0_threshold (float): b-value (s/mm^2) at or below which a volume is treated as a
            b0. b0s are always included in the fit regardless of bval_max.
        fit_method (Literal["WLS", "OLS"]): Tensor fit method passed to dipy's TensorModel.

    Returns:
        Path: outfile.
    """
    infile = Path(infile)
    outfile = Path(outfile)
    stem = _strip_nifti_suffix(infile.name)
    bval = Path(bval) if bval is not None else infile.parent / f"{stem}.bval"
    bvec = Path(bvec) if bvec is not None else infile.parent / f"{stem}.bvec"

    if not bval.exists():
        raise FileNotFoundError(f"Missing bval sidecar: {bval}")
    if not bvec.exists():
        raise FileNotFoundError(f"Missing bvec sidecar: {bvec}")

    dwi_nifti = nib.load(str(infile))
    dwi_data = np.asarray(dwi_nifti.dataobj, dtype=np.float32)
    if dwi_data.ndim != 4:
        raise ValueError(
            f"Expected a 4D DWI series, got array with ndim={dwi_data.ndim} for {infile}."
        )

    bvals, bvecs = read_bvals_bvecs(str(bval), str(bvec))
    n_volumes = dwi_data.shape[-1]
    if bvals.shape[0] != n_volumes or bvecs.shape[0] != n_volumes:
        raise ValueError(
            f"Gradient count does not match volume count for {infile}: "
            f"{bvals.shape[0]} bvals, {bvecs.shape[0]} bvecs, {n_volumes} volumes."
        )

    # Shell selection: fit only b <= bval_max, always including b0s (b <= b0_threshold).
    b0_mask = bvals <= b0_threshold
    selected_mask = (bvals <= bval_max) | b0_mask
    n_b0 = int(np.count_nonzero(b0_mask))
    n_directions = int(np.count_nonzero(selected_mask & ~b0_mask))

    if n_directions < 6 or n_b0 == 0:
        acquired_shells = sorted({round(float(b)) for b in bvals})
        raise ValueError(
            f"Shell selection for bval_max={bval_max}, b0_threshold={b0_threshold} on "
            f"{infile} yields {n_directions} non-b0 direction(s) and {n_b0} b0(s); at "
            "least 6 non-b0 directions and 1 b0 are required. Acquired shells "
            f"(s/mm^2): {acquired_shells}."
        )

    selected_shells = sorted({round(float(b)) for b in bvals[selected_mask]})
    logger.info(
        f"Fitting tensor ({fit_method}) for {infile} using shells {selected_shells} "
        f"s/mm^2: {n_b0} b0(s), {n_directions} direction(s), "
        f"{int(np.count_nonzero(selected_mask))} volume(s) total."
    )

    selected_data = np.maximum(dwi_data[..., selected_mask], _SIGNAL_EPSILON)
    gtab = gradient_table(
        bvals[selected_mask], bvecs=bvecs[selected_mask], b0_threshold=b0_threshold
    )
    tensor_model = TensorModel(gtab, fit_method=fit_method)
    tensor_fit = tensor_model.fit(selected_data)

    # MD = trace(D)/3 from the fitted tensor directly, not from averaged eigenvalues.
    D = tensor_fit.quadratic_form
    md = (D[..., 0, 0] + D[..., 1, 1] + D[..., 2, 2]) / 3.0
    md *= 1e6  # mm^2/s -> 1e-6 mm^2/s, assuming b-values in s/mm^2

    finite_mask = np.isfinite(md)
    md = np.where(finite_mask, md, 0.0).astype(np.float32)

    negative_fraction = float(np.count_nonzero(md < 0)) / md.size
    if negative_fraction > 0:
        logger.warning(
            f"{negative_fraction:.4%} of voxels have negative MD (fit failure) in "
            f"{infile}; left unclamped."
        )

    outfile.parent.mkdir(parents=True, exist_ok=True)
    input_header = dwi_nifti.header
    out_header = nib.Nifti1Header()
    out_header.set_data_dtype(np.float32)
    out_header.set_data_shape(md.shape)
    out_header.set_zooms(input_header.get_zooms()[:3])
    out_header.set_xyzt_units(*input_header.get_xyzt_units())

    md_nifti = nib.Nifti1Image(md, affine=dwi_nifti.affine, header=out_header)
    nib.save(md_nifti, str(outfile))

    logger.info(f"ADC/MD map saved to {outfile}.")
    return outfile
