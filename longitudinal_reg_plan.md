# Longitudinal registration: adopting the BraTS-Reg 2022 winner pipeline

Status: plan only, nothing implemented. All decisions settled on 2026-10-06. Branch
`update_long_reg`.

Constraints: no commits except the single checkpoint commit (D12), no pushes, no ruff, no
formatting-only edits. Existing tests may be adapted, no new test files or test cases.
Check scripts and their outputs live in the scratchpad, never in the repository or in dataset
directories.

Reference code: `/mnt/Drive4/lucas/BraTS-Reg_algorithms/cwmokab_extracted/` (`my_script_pytorch.py`,
`bratsreg_model_selected.py`). "Container" below means that code, and only its inference path; the
training script of the challenge model is not available. X is the pre-op (baseline) image, Y the
follow-up; `F_Y_X` is the field on the pre-op grid that resamples the follow-up onto it.


## 1. Target behaviour

`register_recurrence(..., registration_algorithm="dirac")` becomes:

1. ANTs affine registration of follow-up t1c to pre-op t1c.
2. The 4-modality challenge network (t1, t1c, flair, t2 of both exams), run on the affinely
   pre-aligned images, conditioning 0.4, both directions.
3. Affine and network field combined into one field per direction.
4. The container's instance optimization on t1c + t2, starting from the combined fields.
5. The final follow-up-to-pre-op field written as one ANTs displacement field, and used to warp
   the follow-up images and the segmentation.

Files left in `<follow-up exam dir>/longitudinal/` for every algorithm (dirac, syn, affine, rigid):

- `longitudinal_trafo.nii.gz`: the combined transform, an ANTs/ITK displacement field in mm
- `<modality>_warped_longitudinal.nii.gz`: t1c plus every entry of `additional_modalities`
- `recurrence_preop.nii.gz`: the warped segmentation

No `dirac_infer_case_*` files, no `followup_to_preop_disp_voxel_optimized.nii.gz`, no
`longitudinal_trafo.mat`. The `register_recurrence` entry in `predict_gbm_config.json` stays.


## 2. Decisions

### 2.1 From the request

- R1. Default atlas becomes `sri24`. DIRAC warns when the input grid differs from the
  240x240x155 BraTS-Reg grid.
- R2. The 4-modality checkpoint is the default. A flag selects the current t1c checkpoint. The
  unused t1c+t2 checkpoint is deleted.
- R3. The container's instance optimization replaces the repo's, with the container's
  similarity modalities and conditioning 0.4.
- R4. ANTs affine pre-registration; only the combined transform is stored.

Correction to R3: the container's similarity term uses **t1c and t2**, not t1c and flair
(`my_script_pytorch.py:297-299`, channels 1 and 3 of [t1, t1c, flair, t2]). The plan follows the
container: t1c + t2 by default, t1c alone with the single-modality checkpoint.

### 2.2 Settled in discussion

| Id | Topic | Outcome |
|---|---|---|
| C1 | Format of `longitudinal_trafo.nii.gz` | ANTs/ITK displacement field for all four algorithms; all warping through `ants.apply_transforms`. |
| C2 | Orientation | The DIRAC code reorients to LPS voxel order internally (the orientation of the BraTS-Reg data the container runs on) and maps the result back; stored outputs keep the pipeline orientation. |
| C3 | Current instance optimization | Removed (residual control grid, fixed masks, anti-aliased pyramid, `regularize` / `multi_scale_ncc` arguments). |
| C4 | Missing t1, t2 or flair with the 4-modality model | Raise an error that names the single-modality flag. No automatic fallback. |
| C5 | Old transform files | No backwards compatibility and no migration for existing DIRAC trafos or `.mat` files. |
| C6 | Atlas change | Accepted for the whole pipeline, not only for registration. |
| D1 | `recurrence_preop.nii.gz` | Kept. |
| D2 | Scope of the affine pre-registration | Added to DIRAC only. `syn` already runs rigid + affine internally; `affine` and `rigid` stay single linear registrations. All four store one dense combined field. |
| D3 | Conditioning for the t1c checkpoint | 0.4, same as the 4-modality model. |
| D4 | Passing the extra modalities | Six optional file parameters on `register_recurrence` (`t1_pre_file`, `t2_pre_file`, `flair_pre_file`, `t1_post_file`, `t2_post_file`, `flair_post_file`) and `dirac_single_modality: bool = False`. |
| D5 | Images warped and stored | Unchanged: t1c always, others through `additional_modalities` / `longitudinal_transform_all`. |
| D6 | Masks with DIRAC | As in the container: none. `fixed_mask_file` / `moving_mask_file` are not passed to the affine step and stay ignored by DIRAC, with the existing warning. |
| D7 | Affine fit | One ANTs `Affine` fit on t1c (pre-op fixed); the other direction uses the inverted matrix. The fitted matrix is decomposed and a warning is logged when it is implausible (section 3 step 2); warning only, no fallback. Thresholds live in `constants.py`. |
| D8 | Network inference | In-process; the subprocess is removed. |
| D9 | Atlas warning | Triggered by the input grid: warn when it is not 240x240x155. A trailing singleton dimension, as in the SRI24 atlas files (`(240, 240, 155, 1)`), is dropped before the comparison. |
| D10 | Interpolation conventions | Ported call by call from the container's inference code, which mixes `align_corners=True` and `False`; see section 3.1. |
| D11 | Atlas parameter on the pipeline classes | Not added; only the defaults of `norm_ss_coregister` and the two tissue-segmentation functions change. |
| D12 | Checkpoints in git | Kept under their original filenames (`..._all4_..._112000.pth` and `..._t1ce_..._64000.pth`); the `cLapIRN_NCC_*` names used at first were reverted on 2026-10-07. One commit containing only these two files, added with `git add -f`; `.gitignore` is not edited. Nothing is pushed and every other change stays uncommitted. The t1c+t2 file was never tracked and is only deleted locally. |


## 3. New DIRAC flow in detail

All tensors use the container layout `(1, C, n0, n1, n2)` with the array axes of the LPS-reoriented
image; field channel 0 belongs to the last array axis. Fields are in normalized coordinates until
the final conversion.

1. **Checks.** Required files exist (C4). Images are loaded as 3-D volumes; a trailing singleton
   dimension is dropped first (D9). Pre-op and follow-up images share shape and affine, otherwise
   raise. Warn when the grid is not 240x240x155 (R1, D9): the container runs the model on
   BraTS-Reg data of that size, squeezed to 160x160x80, so another grid changes the voxel aspect
   ratio the model sees.
2. **Affine (D6, D7).** `ants.registration(fixed=t1c_pre, moving=t1c_post,
   type_of_transform="Affine")` without masks, written into a temporary directory. The resulting
   ITK transform maps pre-op physical points to follow-up physical points; it is what resamples
   the follow-up onto the pre-op grid. Read it with `ants.read_transform` and build
   `affine_pre_to_post_mm` (4x4, physical LPS millimetres) from the matrix `A` (first nine
   parameters, row-major), the translation `t` (last three) and the rotation centre `c` (fixed
   parameters): `p_post = A (p_pre - c) + c + t`, so the offset column is `t + c - A c`. Pass the
   matrix to the DIRAC code, so `dirac.py` stays free of ANTs.

   **Sanity check (D7).** Decompose `affine_pre_to_post_mm` and log one warning per violated
   condition, naming the value and the threshold:
   - `|log det A| > 0.2` (volume change beyond about 20 %);
   - any singular value of `A` outside `[0.8, 1.25]`;
   - rotation angle `> 15°`, taken from the rotation `R = U V^T` of the polar decomposition
     (`A = U S V^T`, sign of the last column of `U` flipped if `det R < 0`),
     `angle = arccos((tr R - 1) / 2)`;
   - translation `> 20 mm`, where the translation is `t` after removing the centre term, that
     is the offset column plus `(A - I) c` (the displacement of the rotation centre `c`), not the
     offset column itself.

   Warning only: the matrix is used unchanged and there is no fallback to rigid or identity. The
   four thresholds are constants in `constants.py`.
3. **Load and reorient (C2).** Load the modalities with nibabel, reorient arrays to LPS
   (`nibabel.orientations`), keep the reoriented affine. Min-max normalize each volume as the
   container does (a no-op on pipeline images). Stack in the order [t1, t1c, flair, t2].
4. **Affine as sampling grids.** From `affine_pre_to_post_mm` derive two normalized 3x4 matrices
   for `F.affine_grid`:
   - `theta_sample_post_on_pre_grid`: samples the follow-up on the pre-op grid. Feeds the `F_Y_X`
     direction (the container's `best_affine_theta_yx`).
   - `theta_sample_pre_on_post_grid`: from the inverted matrix; samples the pre-op on the
     follow-up grid. Feeds the `F_X_Y` direction (the container's `best_affine_theta_xy`).

   The conversion uses the index-to-physical (LPS) matrices of the reoriented images, the
   `(N-1)/2` scaling of the `align_corners=True` convention and the axis reversal that
   `F.affine_grid` expects.
5. **Network** (`my_script_pytorch.py:257-267`). Sample the image being warped through its affine
   grid at 160x160x80; resize the other image. `reg_code` 0.4 (D3). Run both directions.
   Single-modality flag: 2 input channels, t1c only, `DIRAC_CHECKPOINT_T1CE`.
6. **Combine** (`:269-275`). `CompositionTransform_unit(affine_field, network_field)` per
   direction, at 160x160x80.
7. **Instance optimization** (`:277-433`), ported as is:
   - Similarity images: t1c + t2 (t1c only with the flag), `NCC_weight(win=3, eps=1e-8)`.
   - Five levels. Image sizes: per-axis linear from 80 to the native size (for 240x240x155: 80^3,
     120x120x98, 160x160x117, 200x200x136, native).
   - Control grids 32^3, 40^3, 48^3, 56^3, 64^3 holding the total field. The first is the combined
     field downsampled; later ones come from the previous level.
   - Iterations 150/100/100/100/50; Adam learning rates 1e-2/5e-3/5e-3/3e-3/3e-3; smoothness
     weights 0.25/0.3/0.3/0.35/0.35; inverse-consistency weights 1/2/4/8/10.
   - Occlusion masks recomputed every iteration from the current fields: forward-backward error,
     smoothed twice with a 5^3 average, threshold = mean error inside the foreground + 0.015.
   - Smoothness on the total field in voxel units of the level (`(N-1)/2` scaling).
   - Result: the dense field of the lowest-loss iteration at the last level; between levels the
     last iterate is handed on (what the container effectively does).
8. **Convert.** `F_Y_X` to voxel displacement (`:435-439` convention), then to physical LPS
   millimetres with the linear part of the index-to-physical matrix. Reorder the spatial axes back
   to the stored orientation; the vector components are physical and need no change. The result,
   `disp_pre_to_post_mm`, follows the same direction convention as `affine_pre_to_post_mm`.
9. **Write and warp (C1).** Build the ANTs vector image with the pre-op geometry
   (`ants.from_numpy(..., has_components=True)`) and write `longitudinal_trafo.nii.gz`. Warp t1c
   and the additional modalities (linear) and the segmentation (nearest neighbour) with
   `ants.apply_transforms`. Free GPU memory (`torch.cuda.empty_cache()`).

`F_X_Y` is optimized alongside but not stored, as today.

Naming: the explicit direction names above apply to the DIRAC code path only. Ported container
code keeps the container's names (`F_X_Y`, `F_Y_X`) so it stays comparable with the reference; the
hand-off in steps 2 and 4 documents the mapping. Names in the ANTs branch and existing parameter
names are untouched.

### 3.1 `align_corners` per call (D10)

The container mixes both conventions. Each call is ported with the setting it has there. Plain
line numbers refer to `my_script_pytorch.py`, `model:` to `bratsreg_model_selected.py`.

| Call | Container location | `align_corners` | In the port |
|---|---|---|---|
| Un-warped image to the network grid, `F.interpolate(X_ori, size=imgshape)` | 261-262 | False (default) | Ported. |
| Affine sampling grids and identity grid, `F.affine_grid` | 214-216, 252-254, 270 | True | Ported; `theta` comes from ANTs (step 4). |
| Warped image to the network grid, `F.grid_sample(X_ori, affine_grid)`, border padding | 258-259 | True | Ported. |
| Network: `grid_1` and `SpatialTransform_unit` | model: 21-22, 440-452 | True | Already identical in `dirac.py`. |
| Network: field upsampling between its levels, `torch.nn.Upsample(scale_factor=2)` | model: 174, 316 | False | Already identical in `dirac.py` (lines 612-614, 886-888). |
| Combining affine and network field, `CompositionTransform_unit` | 271-273; model: 470-481 | True | Class already in `dirac.py`. |
| Image pyramid of the instance optimization, `F.interpolate(X, size=size)` | 301-305 | False (default) | Ported. |
| Combined field down to the first control grid, `F.interpolate(deep_F_X_Y, size=theta_size[0])` | 332-333 | False (default) | Ported. |
| Control grid to the next level, `F.interpolate(best_theta_xy, size=theta_size[level])` | 335-336 | False (default) | Ported. |
| Control grid up to the dense field, `F.interpolate(self.disp_xy, size=self.imgshape)` in `L2R_Instance_opt_grid_bidirection.forward` | model: 818-819 | True | Ported. |
| Identity grids of the instance optimization (`grid_1` of that class, `grid_unit` in the loop) | model: 815; 352-353 | True | Ported. |
| Warping images and fields in the loop, `SpatialTransform_unit` | 355-356; model: 821-822 | True | Ported. |
| Smoothness scaling `(N-1)/2` | 407-412 | True convention | Ported. |
| Final conversion to voxel units `(N-1)/2` | 435-439 | True convention | Ported. |

Not ported: the container's own affine stage (154-255) is replaced by ANTs, and its image output
(488-501, `apply_deformation`) by `ants.apply_transforms`.


## 4. Changes per file

### `predict_gbm/data/models/` (D12)

- `Brats_NCC_disp_fea6b5_AdaIn64_all4_fbcon_occ01_inv5_a0015_saug_mean_keypt10_affine_ffixed_stagelvl3_112000.pth`:
  copy of the file of the same name from the extracted container directory.
- `Brats_NCC_disp_fea6b5_AdaIn64_t1ce_fbcon_occ01_inv5_a0015_aug_mean_fffixed_github_stagelvl3_64000.pth`:
  the checkpoint in use today, tracked under its original name, so `main` and `sailor` keep
  loading it unchanged from this working tree.
- Delete `Brats_NCC_disp_fea7b5_AdaIn64_t1cet2_fbcon_occ01_inv5_a0015_aug_mean_ffixed_stagelvl3_98000.pth`
  locally. It was never tracked; a copy stays in `/mnt/Drive2/lucas/models/DIRAC/Model/`.
- Commit the two checkpoints with `git add -f`, as one commit that contains nothing else. No
  push. (First committed as `cLapIRN_NCC_four_modalities.pth` / `cLapIRN_NCC_t1ce.pth`; renamed
  back to the original names on 2026-10-07 by amending that commit.)

### `predict_gbm/utils/constants.py`

- Remove `LONGITUDINAL_AFFINE_SCHEMA`; update the comment on `LONGITUDINAL_DISP_SCHEMA` (ANTs
  displacement field, all algorithms).
- Add the two checkpoint filenames as constants (today hard-coded in `dirac.py:1868` and `:2017`),
  with a comment giving the original filenames.
- Add the four thresholds of the affine sanity check (section 3 step 2, D7): maximum
  `|log det A|` 0.2, singular-value range `[0.8, 1.25]`, maximum rotation angle 15°, maximum
  translation 20 mm.

### `predict_gbm/preprocessing/norm_ss_coregistration.py`

- `norm_ss_coregister`: default `atlas="sri24"`, docstring.
- `ANTS_REGISTRATION_TRANSFORMS`: drop the schema column; every algorithm writes
  `LONGITUDINAL_DISP_SCHEMA`.
- `register_recurrence`:
  - New parameters (D4), validation (C4), docstring.
  - One temporary directory for all intermediates, including the ANTs output prefix.
  - ANTs branch (syn, affine, rigid): after `ants.registration`, build the combined field with
    `ants.apply_transforms(..., compose=<prefix>)` and move it to `longitudinal_trafo.nii.gz`.
    Masks are used here as today.
  - DIRAC branch: affine step without masks (D6), `affine_pre_to_post_mm` built as in section 3
    step 2, call into `dirac.py`, write the field. The two existing mask warnings stay.
  - One shared block that warps t1c, segmentation and additional modalities from the stored
    field, replacing the two separate implementations.
  - Config entry gains the model used (`four_modalities` or `t1ce`) and the affine
    pre-registration.
- Imports: remove the five DIRAC helpers that disappear.

### `predict_gbm/preprocessing/tissue_segmentation.py`

- `run_tissue_seg` and `run_tissue_seg_atlas_registration`: default `atlas="sri24"`, docstrings.
  Both defaults must change together with `norm_ss_coregister`, since the tissue maps come from
  the same atlas.

### `predict_gbm/preprocessing/dirac.py`

- Header comment at the top of the file: `Copyright (c) 2022 Tony Mok`, followed by the MIT
  permission notice copied verbatim from the DIRAC repository's `LICENSE` (the permission
  paragraph with its condition sentence and the warranty disclaimer).
- Keep: `generate_grid_unit`, `imgnorm`, the network building blocks and the three network
  classes. They already take `in_channel`; the 4-modality checkpoint has the same 228 state-dict
  keys and differs only in the three first convolutions. Adjust the unused `down_y` slice to the
  container's `[:, in_channel // 2:in_channel]`.
- Remove (C3, D8): `Validation_Brats`, `run_inference` and the `__main__` block,
  `run_dirac_inference`, `resolve_dirac_disp_field`, `optimize_warp_field`,
  `apply_longitudinal_warp`, `warp_image_to_preop`, `dirac_instance_optimization` and all helpers
  only it uses (lines 1362-1852), `load_4D`, `save_img`, `save_flow`, `save_nifti` if unused
  afterwards.
- Add, ported from the container with explicit devices instead of `.cuda()`: `NCC_weight`,
  `smoothloss`, `L2R_Instance_opt_grid_bidirection`.
- Add: image loading that drops a trailing singleton dimension (D9), orientation helpers (to LPS
  and back), the conversion from `affine_pre_to_post_mm` to the two `theta_sample_*` matrices, the
  network step, the instance optimization (keeping the name `dirac_instance_optimization`, new
  signature), and one entry point that runs section 3 steps 3 to 8 and returns
  `disp_pre_to_post_mm`.
- Type hints on everything added; loguru logging, one INFO line per level.
- Update the module docstring.

### `predict_gbm/preprocessing/preprocess.py`

- `RegisterRecurrencePipe`: new argument `dirac_single_modality`; build the t1, t2, flair paths
  for both exams from `MODALITY_STRIPPED_SCHEMA` and pass them on. The `is_coregistered` branch is
  unchanged.

### `predict_gbm/pipeline.py`

- `DicomProcessor`, `NiftiProcessor`: new argument `dirac_single_modality`, forwarded to
  `RegisterRecurrencePipe`; docstrings.

### `scripts/stepwise_processing.py`

- Pass the t1, t2, flair files of both exams to `register_recurrence`.

### Tests (adapted, none added)

- `tests/preprocessing/test_norm_ss_coregister.py`
  - Default-atlas tests (lines 181-193 and 233): expect `sri24`.
  - Two DIRAC tests (lines 293-336, 381-430): patch the new entry point and the ANTs calls instead
    of the removed helpers; pass the modality files or the flag.
  - Two syn tests: account for the extra `apply_transforms` call that builds the combined field.
  - Linear test (lines 489-544): expect `longitudinal_trafo.nii.gz`, no `.mat`; drop the
    `LONGITUDINAL_AFFINE_SCHEMA` import.
- `tests/preprocessing/test_dirac.py`: same two cases against the new function. The
  zero-iteration case can no longer expect the input back exactly (the control grid band-limits
  the field) and the masks are no longer inputs or outputs.
- `tests/preprocessing/test_preprocess.py`, `tests/test_pipeline_registration_algorithm.py`:
  adjust for the new arguments where a call or assertion is affected.


## 5. Deviations from the container that remain

- ANTs affine (mutual information, full resolution, one fit) instead of the container's
  low-resolution gradient-field affine fitted per direction (R4, D7).
- Inputs are the pipeline's percentile-normalized images (top 0.1 % clipped), not raw intensities.
- The top pyramid level is the native grid instead of a hard-coded 240x240x155, so other atlases
  run (with the R1 warning). On SRI24 this is identical.
- Skull stripping is SynthStrip, not the BraTS preprocessing.
- PyTorch 2.8 instead of 1.10.


## 6. Checks during implementation (scratchpad only)

Verification of the result, such as a comparison with the container or with the previous
implementation, is done separately after the implementation and is not part of this plan. The
checks below only establish that the pieces do what the plan says. `BraTSReg_002`
(`/mnt/Drive2/lucas/models/DIRAC/Dataset/test/BraTSReg_002/`) may be used for them.

- **Affine hand-off, both directions**, on full-resolution t1c:
  - follow-up sampled through `theta_sample_post_on_pre_grid` matches
    `ants.apply_transforms(fixed=pre, moving=post, transformlist=[mat])`;
  - pre-op sampled through `theta_sample_pre_on_post_grid` matches
    `ants.apply_transforms(fixed=post, moving=pre, transformlist=[mat], whichtoinvert=[True])`.

  A wrong direction, a missing rotation centre or a wrong axis order shows up here as a large
  difference.
- **Field conversion:** `ants.apply_transforms` with the written field matches the torch warp of
  the same field; ANTsPy array order matches nibabel's.
- **Orientation:** the same case stored as RAS, LAS and LPS gives the same result after mapping
  back.
- **Trailing singleton dimension:** a `(240, 240, 155, 1)` input loads, passes the grid check
  without a warning and registers like its 3-D version.
- **Linear algorithms:** `compose` produces a dense field for a purely linear transform list
  (affine, rigid). If not, build the field from the matrix with the affine helper.
- **Network:** same tensors into the container's model class and the repo's give the same field.
- **Adapted tests:** `python -m pytest tests/preprocessing/test_dirac.py
  tests/preprocessing/test_norm_ss_coregister.py tests/preprocessing/test_preprocess.py
  tests/test_pipeline_registration_algorithm.py` with the `brainles` interpreter.
- **Smoke run:** `register_recurrence` on one case into a scratch output directory; the directory
  holds exactly the files of section 1.

Report at the end: diff summary, decisions applied, results of the tests and checks.

GPU: shared machine; check `nvidia-smi` first and never share a GPU with another user's job.


## 7. Things to know

- **Stored syn transform is incomplete today.** `register_recurrence` copies only
  `fwdtransforms[0]`, the deformable part; the affine part is dropped from
  `longitudinal_trafo.nii.gz`. The warped images are correct. The combined field fixes this.
- **Format change (C5).** Existing `longitudinal_trafo.nii.gz` files from DIRAC runs (voxel
  units, array-axis components) and existing `.mat` files are not in the new format.
- **SAILOR script.** `scripts/process_sailor.py` on the `sailor` branch calls
  `register_recurrence` and deletes DIRAC intermediates; it needs the new arguments when the
  branches meet, and the cleanup becomes unnecessary.
- **Default atlas (C6).** Tissue maps, growth-model inputs and evaluation move to the SRI24 grid.
  The repo code has no fixed-shape assumption there; the growth-model containers were not checked.
- **SRI24 atlas files differ in stored orientation.** The unstripped file and the tissue maps are
  RAS, the skull-stripped file is LAS; they agree in physical space. Pipeline output is therefore
  RAS or LAS depending on `skull_strip`. C2 makes DIRAC independent of this.
- **Checkpoints (D12).** The commit is local. Pushing the branch puts the 4-modality checkpoint,
  which the authors have not released publicly, on GitHub.
- **Licence header.** The MIT licence is that of the public `cwmok/DIRAC` repository, which
  contains the network classes, `NCC_weight`, `smoothloss` and the transform units.
  `L2R_Instance_opt_grid_bidirection` and the instance-optimization loop exist only in the
  container code, which carries no licence file.
- **Evaluation data.** The BraTS-Reg training set is no longer at
  `/mnt/Drive4/lucas/BraTSReg_Training_Data_v2`; relevant for the separate verification.
- **CUDA.** The network classes call `.cuda()` in their constructors, so DIRAC needs a GPU, as
  today. Not changed here.
- **Runtime and memory.** The instance optimization now runs two channels at full resolution in
  both directions with masks recomputed every iteration; expect it to be slower and to need more
  GPU memory than today. The smoke run will give a first number.


## 8. Implementation order

1. Checkpoint files and their commit (D12).
2. Constants, atlas defaults, default-atlas tests.
3. `dirac.py`: licence header, remove the old code, port the instance optimization, add loading,
   orientation and affine helpers and the entry point; component checks.
4. `register_recurrence`: affine step, combined field, shared warping, temporary directory.
5. `preprocess.py`, `pipeline.py`, `scripts/stepwise_processing.py`.
6. Adapt and run the tests.
7. Smoke run and report.
