from pathlib import Path
import shutil
import subprocess
import sys

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F


def run_dirac_inference(t1c_pre_file: Path, t1c_post_file: Path, workdir: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    model_path = (
        repo_root
        / "predict_gbm"
        / "data"
        / "models"
        / (
            "Brats_NCC_disp_fea6b5_AdaIn64_t1ce_fbcon_occ01_inv5_a0015_aug_mean_fffixed_github_stagelvl3_64000.pth"
        )
    )
    if not model_path.exists():
        raise FileNotFoundError(f"Missing DIRAC model checkpoint: {model_path}")

    infer_case_dir = workdir / "dirac_infer_case"
    if infer_case_dir.exists():
        shutil.rmtree(infer_case_dir)
    infer_case_dir.mkdir(parents=True, exist_ok=True)

    shutil.copyfile(
        str(t1c_pre_file), str(infer_case_dir / "t1c_bet_normalized.nii.gz")
    )
    shutil.copyfile(
        str(t1c_post_file), str(infer_case_dir / "t1c_bet_normalized_followup.nii.gz")
    )

    cmd = [
        sys.executable,
        "-m",
        "predict_gbm.preprocessing.brats_infer_dirac",
        "--modelname",
        str(model_path),
        "--datapath",
        str(workdir),
        "--output_seg",
        "True",
        "--save_transform",
        "True",
    ]
    subprocess.run(cmd, cwd=str(repo_root), check=True)

    for suffix in [
        "followup_to_preop_disp_voxel",
        "preop_to_followup_disp_voxel",
        "xy_seg",
        "yx_seg",
    ]:
        src = infer_case_dir / f"dirac_infer_case_{suffix}.nii.gz"
        if not src.exists():
            raise FileNotFoundError(f"Expected DIRAC inference output missing: {src}")

        shutil.copyfile(str(src), str(workdir / src.name))

    shutil.rmtree(infer_case_dir)


def make_identity_grid(d, h, w, device, dtype):
    xs = torch.linspace(-1, 1, w, device=device, dtype=dtype)
    ys = torch.linspace(-1, 1, h, device=device, dtype=dtype)
    zs = torch.linspace(-1, 1, d, device=device, dtype=dtype)
    zz, yy, xx = torch.meshgrid(zs, ys, xs, indexing="ij")
    return torch.stack((xx, yy, zz), dim=-1)[None]


def voxel_disp_to_norm(disp, d, h, w):
    dx, dy, dz = disp[:, 0], disp[:, 1], disp[:, 2]
    sx = 2.0 / max(w - 1, 1)
    sy = 2.0 / max(h - 1, 1)
    sz = 2.0 / max(d - 1, 1)
    return torch.stack((dx * sx, dy * sy, dz * sz), dim=-1)


def warp(img, disp, mode="bilinear"):
    _, _, d, h, w = img.shape
    grid0 = make_identity_grid(d, h, w, img.device, img.dtype)
    grid = grid0 + voxel_disp_to_norm(disp, d, h, w)
    return F.grid_sample(
        img, grid, mode=mode, padding_mode="border", align_corners=True
    )


def warp_field(field, disp):
    _, _, d, h, w = field.shape
    grid0 = make_identity_grid(d, h, w, field.device, field.dtype)
    grid = grid0 + voxel_disp_to_norm(disp, d, h, w)
    return F.grid_sample(
        field, grid, mode="bilinear", padding_mode="border", align_corners=True
    )


def resize_disp_voxel(disp, size):
    _, _, d0, h0, w0 = disp.shape
    d1, h1, w1 = size
    resized = F.interpolate(disp, size=size, mode="trilinear", align_corners=True)
    sx = (w1 - 1) / max(w0 - 1, 1)
    sy = (h1 - 1) / max(h0 - 1, 1)
    sz = (d1 - 1) / max(d0 - 1, 1)
    resized[:, 0] *= sx
    resized[:, 1] *= sy
    resized[:, 2] *= sz
    return resized


def pad_tensor_to_min_size(x, min_size, mode="constant", value=0.0):
    _, _, d, h, w = x.shape
    min_d, min_h, min_w = min_size
    pad_d, pad_h, pad_w = max(min_d - d, 0), max(min_h - h, 0), max(min_w - w, 0)
    pad_d0, pad_d1 = pad_d // 2, pad_d - (pad_d // 2)
    pad_h0, pad_h1 = pad_h // 2, pad_h - (pad_h // 2)
    pad_w0, pad_w1 = pad_w // 2, pad_w - (pad_w // 2)
    pad = (pad_w0, pad_w1, pad_h0, pad_h1, pad_d0, pad_d1)
    x_pad = (
        F.pad(x, pad, mode=mode, value=float(value))
        if mode == "constant"
        else F.pad(x, pad, mode=mode)
    )
    crop_slices = (
        slice(pad_d0, pad_d0 + d),
        slice(pad_h0, pad_h0 + h),
        slice(pad_w0, pad_w0 + w),
    )
    return x_pad, crop_slices


def crop_to_slices(x, crop_slices):
    d_slice, h_slice, w_slice = crop_slices
    return x[:, :, d_slice, h_slice, w_slice]


def ncc_loss(i, j, mask=None, win=3, eps=1e-5):
    pad = win // 2
    filt = torch.ones((1, 1, win, win, win), device=i.device, dtype=i.dtype)

    def conv(x):
        return F.conv3d(x, filt, padding=pad)

    if mask is None:
        mask = torch.ones_like(i)
    mask = mask.to(dtype=i.dtype)
    i2, j2, ij = i * i, j * j, i * j
    w_sum = conv(mask)
    i_sum, j_sum = conv(mask * i), conv(mask * j)
    i2_sum, j2_sum, ij_sum = conv(mask * i2), conv(mask * j2), conv(mask * ij)
    u_i, u_j = i_sum / (w_sum + eps), j_sum / (w_sum + eps)
    cross = ij_sum - u_j * i_sum - u_i * j_sum + u_i * u_j * w_sum
    i_var = i2_sum - 2 * u_i * i_sum + u_i * u_i * w_sum
    j_var = j2_sum - 2 * u_j * j_sum + u_j * u_j * w_sum
    ncc = cross * cross / (i_var * j_var + eps)
    valid = (w_sum > 0).to(dtype=i.dtype)
    return -(ncc * valid).sum()


def smoothness(disp, valid_mask=None):
    dx = disp[:, :, :, :, 1:] - disp[:, :, :, :, :-1]
    dy = disp[:, :, :, 1:, :] - disp[:, :, :, :-1, :]
    dz = disp[:, :, 1:, :, :] - disp[:, :, :-1, :, :]
    if valid_mask is None:
        return dx.pow(2).sum() + dy.pow(2).sum() + dz.pow(2).sum()
    valid_mask = valid_mask.to(dtype=disp.dtype)
    mx = valid_mask[:, :, :, :, 1:] * valid_mask[:, :, :, :, :-1]
    my = valid_mask[:, :, :, 1:, :] * valid_mask[:, :, :, :-1, :]
    mz = valid_mask[:, :, 1:, :, :] * valid_mask[:, :, :-1, :, :]
    return (dx.pow(2) * mx).sum() + (dy.pow(2) * my).sum() + (dz.pow(2) * mz).sum()


def inv_consistency(d_fwd, d_bwd, m_fwd=None, m_bwd=None, valid_mask=None):
    bwd_warped, fwd_warped = warp_field(d_bwd, d_fwd), warp_field(d_fwd, d_bwd)
    err_fwd = ((d_fwd + bwd_warped) ** 2).sum(dim=1, keepdim=True)
    err_bwd = ((d_bwd + fwd_warped) ** 2).sum(dim=1, keepdim=True)
    w_fwd, w_bwd = torch.ones_like(err_fwd), torch.ones_like(err_bwd)
    if m_fwd is not None:
        w_fwd = w_fwd * (1.0 - m_fwd.to(dtype=err_fwd.dtype))
    if m_bwd is not None:
        w_bwd = w_bwd * (1.0 - m_bwd.to(dtype=err_bwd.dtype))
    if valid_mask is not None:
        vm = valid_mask.to(dtype=err_fwd.dtype)
        w_fwd, w_bwd = w_fwd * vm, w_bwd * vm
    return (err_fwd * w_fwd).sum() + (err_bwd * w_bwd).sum()


def load_image_for_grid_sample(path, device):
    img = nib.load(path).get_fdata().astype(np.float32)
    return torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).unsqueeze(0).to(device)


def load_mask_for_grid_sample(path, device):
    mask = nib.load(path).get_fdata().astype(np.float32)
    mask = (mask > 0.5).astype(np.float32)
    return torch.from_numpy(mask).permute(2, 0, 1).unsqueeze(0).unsqueeze(0).to(device)


def load_dirac_voxel_disp_for_grid_sample(path, device):
    disp = nib.load(path).get_fdata().astype(np.float32)  # (H,W,D,3)

    # DIRAC voxel output components follow image axes: [axis0(H), axis1(W), axis2(D)].
    # grid_sample expects channels [dx(W), dy(H), dz(D)] for tensor shape (1,3,D,H,W).
    comp_axis0 = torch.from_numpy(disp[..., 0]).permute(2, 0, 1)  # (D,H,W)
    comp_axis1 = torch.from_numpy(disp[..., 1]).permute(2, 0, 1)  # (D,H,W)
    comp_axis2 = torch.from_numpy(disp[..., 2]).permute(2, 0, 1)  # (D,H,W)

    disp_grid = torch.stack((comp_axis1, comp_axis0, comp_axis2), dim=0).unsqueeze(0)
    return disp_grid.to(device)


def grid_sample_disp_to_dirac_voxel(disp):
    dx = disp[0, 0].permute(1, 2, 0).cpu().numpy()
    dy = disp[0, 1].permute(1, 2, 0).cpu().numpy()
    dz = disp[0, 2].permute(1, 2, 0).cpu().numpy()
    return np.stack((dy, dx, dz), axis=-1).astype(np.float32)


def save_nifti(data, reference_path: Path, output_path: Path):
    reference = nib.load(str(reference_path))
    nib.save(
        nib.Nifti1Image(
            data.astype(np.float32), affine=reference.affine, header=reference.header
        ),
        str(output_path),
    )


def resolve_dirac_disp_field(workdir: Path, suffix: str) -> Path:
    candidates = sorted(workdir.glob(f"*_{suffix}.nii.gz"))
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise FileNotFoundError(
            f"Could not find DIRAC displacement field '*_{suffix}.nii.gz' in {workdir}. "
            "Run BRATS_infer_DIRAC.py first or provide the expected file in this directory."
        )

    raise FileNotFoundError(
        f"Expected exactly one '*_{suffix}.nii.gz' in {workdir}, found: {candidates}"
    )


def dirac_instance_optimization(
    B,
    Fup,
    disp_fb_init,
    disp_bf_init,
    m_fb_fixed=None,
    m_bf_fixed=None,
    lambdas_reg=(0.25, 0.3, 0.3, 0.35, 0.35),
    lambdas_inv=(1.0, 2.0, 4.0, 8.0, 10.0),
    lrs=(1e-2, 5e-3, 5e-3, 3e-3, 3e-3),
    iters=(150, 100, 100, 100, 50),
):
    if m_fb_fixed is None:
        m_fb_fixed = torch.zeros_like(B)
    if m_bf_fixed is None:
        m_bf_fixed = torch.zeros_like(B)
    _, _, d_full, h_full, w_full = B.shape
    dmin, hmin, wmin, g_min, g_max = 80, 80, 80, 32, 64
    n_levels = len(lrs)
    scale_min = min(dmin / d_full, hmin / h_full, wmin / w_full, 1.0)
    scales = [
        scale_min + (1.0 - scale_min) * i / max(n_levels - 1, 1)
        for i in range(n_levels)
    ]
    pyr_sizes_base = [
        (
            max(1, int(round(d_full * s))),
            max(1, int(round(h_full * s))),
            max(1, int(round(w_full * s))),
        )
        for s in scales
    ]
    pyr_sizes = [
        (max(d, dmin), max(h, hmin), max(w, wmin)) for (d, h, w) in pyr_sizes_base
    ]
    grid_sizes = [
        int(round(g_min + (g_max - g_min) * i / max(n_levels - 1, 1)))
        for i in range(n_levels)
    ]
    disp_fb_full, disp_bf_full = (
        disp_fb_init.clone().detach(),
        disp_bf_init.clone().detach(),
    )

    for lvl, (lr, n_iter, lam_reg, lam_inv) in enumerate(
        zip(lrs, iters, lambdas_reg, lambdas_inv)
    ):
        d_base, h_base, w_base = pyr_sizes_base[lvl]
        d, h, w = pyr_sizes[lvl]
        g = grid_sizes[lvl]
        B_l_base = F.interpolate(
            B, size=(d_base, h_base, w_base), mode="trilinear", align_corners=True
        )
        F_l_base = F.interpolate(
            Fup, size=(d_base, h_base, w_base), mode="trilinear", align_corners=True
        )
        mfb_l_base = F.interpolate(
            m_fb_fixed, size=(d_base, h_base, w_base), mode="nearest"
        )
        mbf_l_base = F.interpolate(
            m_bf_fixed, size=(d_base, h_base, w_base), mode="nearest"
        )
        B_l, crop_slices = pad_tensor_to_min_size(
            B_l_base, min_size=(dmin, hmin, wmin), mode="replicate"
        )
        F_l, _ = pad_tensor_to_min_size(
            F_l_base, min_size=(dmin, hmin, wmin), mode="replicate"
        )
        mfb_l, _ = pad_tensor_to_min_size(
            mfb_l_base, min_size=(dmin, hmin, wmin), mode="constant", value=1.0
        )
        mbf_l, _ = pad_tensor_to_min_size(
            mbf_l_base, min_size=(dmin, hmin, wmin), mode="constant", value=1.0
        )
        valid_mask = torch.zeros_like(B_l)
        d_slice, h_slice, w_slice = crop_slices
        valid_mask[:, :, d_slice, h_slice, w_slice] = 1.0
        disp_fb_l_base = resize_disp_voxel(disp_fb_full, size=(d_base, h_base, w_base))
        disp_bf_l_base = resize_disp_voxel(disp_bf_full, size=(d_base, h_base, w_base))
        disp_fb_l, crop_slices_disp = pad_tensor_to_min_size(
            disp_fb_l_base, min_size=(dmin, hmin, wmin), mode="replicate"
        )
        disp_bf_l, _ = pad_tensor_to_min_size(
            disp_bf_l_base, min_size=(dmin, hmin, wmin), mode="replicate"
        )
        if crop_slices_disp != crop_slices:
            raise RuntimeError("Padding crop mismatch")
        cp_fb = (
            F.interpolate(
                disp_fb_l, size=(g, g, g), mode="trilinear", align_corners=True
            )
            .detach()
            .requires_grad_(True)
        )
        cp_bf = (
            F.interpolate(
                disp_bf_l, size=(g, g, g), mode="trilinear", align_corners=True
            )
            .detach()
            .requires_grad_(True)
        )
        opt = torch.optim.Adam([cp_fb, cp_bf], lr=lr)

        for _ in range(n_iter):
            disp_fb = F.interpolate(
                cp_fb, size=(d, h, w), mode="trilinear", align_corners=True
            )
            disp_bf = F.interpolate(
                cp_bf, size=(d, h, w), mode="trilinear", align_corners=True
            )
            F_warp, B_warp = warp(F_l, disp_fb), warp(B_l, disp_bf)
            Ls = ncc_loss(B_l, F_warp, mask=(1 - mfb_l)) + ncc_loss(
                F_l, B_warp, mask=(1 - mbf_l)
            )
            Lr = smoothness(disp_fb, valid_mask) + smoothness(disp_bf, valid_mask)
            Linv = inv_consistency(disp_fb, disp_bf, mfb_l, mbf_l, valid_mask)
            loss = (1 - lam_reg) * Ls + lam_reg * Lr + lam_inv * Linv
            opt.zero_grad()
            loss.backward()
            opt.step()

        disp_fb_level = F.interpolate(
            cp_fb.detach(), size=(d, h, w), mode="trilinear", align_corners=True
        )
        disp_bf_level = F.interpolate(
            cp_bf.detach(), size=(d, h, w), mode="trilinear", align_corners=True
        )
        disp_fb_base = crop_to_slices(disp_fb_level, crop_slices)
        disp_bf_base = crop_to_slices(disp_bf_level, crop_slices)
        disp_fb_full = resize_disp_voxel(disp_fb_base, size=(d_full, h_full, w_full))
        disp_bf_full = resize_disp_voxel(disp_bf_base, size=(d_full, h_full, w_full))

    return (
        disp_fb_full.detach(),
        disp_bf_full.detach(),
        m_fb_fixed.detach(),
        m_bf_fixed.detach(),
    )


def optimize_warp_field(
    t1c_pre_file: Path,
    t1c_post_file: Path,
    followup_to_preop_disp: Path,
    preop_to_followup_disp: Path,
    optimized_followup_to_preop_disp: Path,
    preop_mask_file: Path | None = None,
    followup_mask_file: Path | None = None,
    device: torch.device | None = None,
):
    device = device or torch.device("cpu" if not torch.cuda.is_available() else "cuda")
    preop = load_image_for_grid_sample(t1c_pre_file, device)
    followup = load_image_for_grid_sample(t1c_post_file, device)
    disp_fb = load_dirac_voxel_disp_for_grid_sample(followup_to_preop_disp, device)
    disp_bf = load_dirac_voxel_disp_for_grid_sample(preop_to_followup_disp, device)
    preop_mask = (
        load_mask_for_grid_sample(preop_mask_file, device)
        if preop_mask_file is not None
        else None
    )
    followup_mask = (
        load_mask_for_grid_sample(followup_mask_file, device)
        if followup_mask_file is not None
        else None
    )
    disp_fb_opt, _, _, _ = dirac_instance_optimization(
        B=preop,
        Fup=followup,
        disp_fb_init=disp_fb,
        disp_bf_init=disp_bf,
        m_fb_fixed=preop_mask,
        m_bf_fixed=followup_mask,
    )
    fb_voxel = grid_sample_disp_to_dirac_voxel(disp_fb_opt)
    save_nifti(fb_voxel, t1c_pre_file, optimized_followup_to_preop_disp)


def apply_longitudinal_warp(
    t1c_pre_file: Path,
    t1c_post_file: Path,
    recurrence_seg_file: Path,
    optimized_followup_to_preop_disp: Path,
    warped_post_out: Path,
    recurrence_out: Path,
):
    device = torch.device("cpu" if not torch.cuda.is_available() else "cuda")
    followup = load_image_for_grid_sample(t1c_post_file, device)
    tumor_seg = load_image_for_grid_sample(recurrence_seg_file, device)
    disp_fb_opt = load_dirac_voxel_disp_for_grid_sample(
        optimized_followup_to_preop_disp, device
    )
    followup_warped = warp(followup, disp_fb_opt, mode="bilinear")
    tumor_warped = warp(tumor_seg, disp_fb_opt, mode="nearest")
    save_nifti(
        followup_warped[0, 0].permute(1, 2, 0).cpu().numpy(),
        t1c_pre_file,
        warped_post_out,
    )
    save_nifti(
        tumor_warped[0, 0].permute(1, 2, 0).cpu().numpy(), t1c_pre_file, recurrence_out
    )
