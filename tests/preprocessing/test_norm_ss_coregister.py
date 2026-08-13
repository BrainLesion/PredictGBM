import tempfile
from unittest.mock import MagicMock, patch
import unittest
import warnings
import nibabel as nib
from pathlib import Path
from tests.helpers import generate_mock_nifti
from predict_gbm.preprocessing.norm_ss_coregistration import (
    normalize,
    initialize_center_modality,
    initialize_moving_modalities,
    norm_ss_coregister,
    register_recurrence,
)
from brainles_preprocessing.modality import Modality, CenterModality
from brainles_preprocessing.normalization.percentile_normalizer import (
    PercentileNormalizer,
)
from brainles_preprocessing.brain_extraction.synthstrip import SynthStripExtractor
from predict_gbm.utils.constants import (
    ATLAS_STRIPPED_SCHEMA,
    ATLAS_UNSTRIPPED_SCHEMA,
    LONGITUDINAL_AFFINE_SCHEMA,
    LONGITUDINAL_DISP_SCHEMA,
    LONGITUDINAL_WARP_SCHEMA,
)

# Silence third-party warnings that clutter test output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


class TestNormSsCoregistration(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.outdir = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def _save_mock_nifti(self, name: str) -> Path:
        img = generate_mock_nifti()
        file_path = self.outdir / name
        nib.save(img, str(file_path))
        return file_path

    def _get_normalizer(self):
        return PercentileNormalizer(
            lower_percentile=0.1,
            upper_percentile=99.9,
            lower_limit=0,
            upper_limit=1,
        )

    def test_normalize(self):
        img_file = self._save_mock_nifti("img.nii.gz")
        outfile = self.outdir / "norm.nii.gz"

        normalize(img_file, outfile)
        normalized_data = nib.load(outfile).get_fdata()

        self.assertTrue(outfile.exists())
        self.assertTrue(normalized_data.min() >= 0)
        self.assertTrue(normalized_data.max() <= 1)
        self.assertEqual(nib.load(outfile).shape, nib.load(img_file).shape)

    def test_initialize_center_modality_paths(self):
        img_file = self._save_mock_nifti("t1c.nii.gz")
        normalizer = self._get_normalizer()

        center = initialize_center_modality(
            modality_file=img_file,
            modality_name="t1c",
            normalizer=normalizer,
            outdir=self.outdir,
            skull_strip=True,
        )

        center_no_ss = initialize_center_modality(
            modality_file=img_file,
            modality_name="t1c",
            normalizer=normalizer,
            outdir=self.outdir,
            skull_strip=False,
        )

        self.assertTrue(isinstance(center, CenterModality))
        self.assertTrue(isinstance(center_no_ss, CenterModality))

    def test_initialize_moving_modalities_returns_list(self):
        t1 = self._save_mock_nifti("t1.nii.gz")
        t2 = self._save_mock_nifti("t2.nii.gz")
        fl = self._save_mock_nifti("flair.nii.gz")
        normalizer = self._get_normalizer()

        mods = initialize_moving_modalities(
            modality_files=[t1, t2, fl],
            modality_names=["t1", "t2", "flair"],
            normalizer=normalizer,
            outdir=self.outdir,
            skull_strip=False,
        )
        self.assertEqual(len(mods), 3)
        self.assertTrue(all([isinstance(m, Modality) for m in mods]))

    def test_initialize_moving_modalities_normalize_false_skull_strip_true(self):
        adc = self._save_mock_nifti("adc.nii.gz")

        mods = initialize_moving_modalities(
            modality_files=[adc],
            modality_names=["adc"],
            outdir=self.outdir,
            skull_strip=True,
            normalize=False,
        )
        self.assertEqual(len(mods), 1)
        self.assertIsNotNone(mods[0].raw_bet_output_path)
        self.assertIsNone(mods[0].normalized_bet_output_path)

    def test_initialize_moving_modalities_normalize_false_skull_strip_false(self):
        adc = self._save_mock_nifti("adc.nii.gz")

        mods = initialize_moving_modalities(
            modality_files=[adc],
            modality_names=["adc"],
            outdir=self.outdir,
            skull_strip=False,
            normalize=False,
        )
        self.assertEqual(len(mods), 1)
        self.assertIsNotNone(mods[0].raw_skull_output_path)
        self.assertIsNone(mods[0].normalized_skull_output_path)

    @patch("predict_gbm.preprocessing.norm_ss_coregistration.AtlasCentricPreprocessor")
    def test_norm_ss_coregister_uses_synthstrip(self, preprocessor_cls_mock):
        t1 = self._save_mock_nifti("t1.nii.gz")
        t1c = self._save_mock_nifti("t1c.nii.gz")
        t2 = self._save_mock_nifti("t2.nii.gz")
        fl = self._save_mock_nifti("flair.nii.gz")

        norm_ss_coregister(t1, t1c, t2, fl, self.outdir)

        preprocessor_cls_mock.assert_called_once()
        brain_extractor = preprocessor_cls_mock.call_args.kwargs["brain_extractor"]
        self.assertIsInstance(brain_extractor, SynthStripExtractor)

    @patch("predict_gbm.preprocessing.norm_ss_coregistration.AtlasCentricPreprocessor")
    def test_norm_ss_coregister_processes_additional_and_quantitative_modalities(
        self, preprocessor_cls_mock
    ):
        t1 = self._save_mock_nifti("t1.nii.gz")
        t1c = self._save_mock_nifti("t1c.nii.gz")
        t2 = self._save_mock_nifti("t2.nii.gz")
        fl = self._save_mock_nifti("flair.nii.gz")
        swi = self._save_mock_nifti("swi.nii.gz")
        adc = self._save_mock_nifti("adc.nii.gz")

        norm_ss_coregister(
            t1,
            t1c,
            t2,
            fl,
            self.outdir,
            additional_modalities={"swi": swi},
            additional_quantitative_modalities={"adc": adc},
        )

        moving_modalities = preprocessor_cls_mock.call_args.kwargs["moving_modalities"]
        by_name = {m.modality_name: m for m in moving_modalities}
        self.assertEqual(set(by_name.keys()), {"t1", "t2", "flair", "swi", "adc"})

        # additional_modalities are normalized like t1/t2/flair
        self.assertIsNotNone(by_name["swi"].normalized_bet_output_path)
        self.assertIsNone(by_name["swi"].raw_bet_output_path)

        # additional_quantitative_modalities are not normalized
        self.assertIsNotNone(by_name["adc"].raw_bet_output_path)
        self.assertIsNone(by_name["adc"].normalized_bet_output_path)

    @patch("predict_gbm.preprocessing.norm_ss_coregistration.AtlasCentricPreprocessor")
    def test_norm_ss_coregister_defaults_to_brats_mni152_atlas(
        self, preprocessor_cls_mock
    ):
        t1 = self._save_mock_nifti("t1.nii.gz")
        t1c = self._save_mock_nifti("t1c.nii.gz")
        t2 = self._save_mock_nifti("t2.nii.gz")
        fl = self._save_mock_nifti("flair.nii.gz")

        norm_ss_coregister(t1, t1c, t2, fl, self.outdir)

        atlas_image_path = preprocessor_cls_mock.call_args.kwargs["atlas_image_path"]
        self.assertEqual(
            atlas_image_path, ATLAS_UNSTRIPPED_SCHEMA.format(atlas="brats_mni152")
        )

    @patch("predict_gbm.preprocessing.norm_ss_coregistration.AtlasCentricPreprocessor")
    def test_norm_ss_coregister_can_switch_to_mni152_atlas(self, preprocessor_cls_mock):
        t1 = self._save_mock_nifti("t1.nii.gz")
        t1c = self._save_mock_nifti("t1c.nii.gz")
        t2 = self._save_mock_nifti("t2.nii.gz")
        fl = self._save_mock_nifti("flair.nii.gz")

        norm_ss_coregister(t1, t1c, t2, fl, self.outdir, atlas="mni152")

        atlas_image_path = preprocessor_cls_mock.call_args.kwargs["atlas_image_path"]
        self.assertEqual(atlas_image_path, ATLAS_UNSTRIPPED_SCHEMA.format(atlas="mni152"))

    @patch("predict_gbm.preprocessing.norm_ss_coregistration.AtlasCentricPreprocessor")
    def test_norm_ss_coregister_can_switch_to_sri24_atlas(self, preprocessor_cls_mock):
        t1 = self._save_mock_nifti("t1.nii.gz")
        t1c = self._save_mock_nifti("t1c.nii.gz")
        t2 = self._save_mock_nifti("t2.nii.gz")
        fl = self._save_mock_nifti("flair.nii.gz")

        norm_ss_coregister(t1, t1c, t2, fl, self.outdir, atlas="sri24")

        atlas_image_path = preprocessor_cls_mock.call_args.kwargs["atlas_image_path"]
        self.assertEqual(atlas_image_path, ATLAS_UNSTRIPPED_SCHEMA.format(atlas="sri24"))

    @patch("predict_gbm.preprocessing.norm_ss_coregistration.AtlasCentricPreprocessor")
    def test_norm_ss_coregister_uses_stripped_atlas_when_skull_strip_false(
        self, preprocessor_cls_mock
    ):
        t1 = self._save_mock_nifti("t1.nii.gz")
        t1c = self._save_mock_nifti("t1c.nii.gz")
        t2 = self._save_mock_nifti("t2.nii.gz")
        fl = self._save_mock_nifti("flair.nii.gz")

        norm_ss_coregister(t1, t1c, t2, fl, self.outdir, skull_strip=False)

        atlas_image_path = preprocessor_cls_mock.call_args.kwargs["atlas_image_path"]
        self.assertEqual(
            atlas_image_path, ATLAS_STRIPPED_SCHEMA.format(atlas="brats_mni152")
        )

    def test_norm_ss_coregister_rejects_unsupported_atlas(self):
        t1 = self._save_mock_nifti("t1.nii.gz")
        t1c = self._save_mock_nifti("t1c.nii.gz")
        t2 = self._save_mock_nifti("t2.nii.gz")
        fl = self._save_mock_nifti("flair.nii.gz")

        with self.assertRaises(ValueError):
            norm_ss_coregister(t1, t1c, t2, fl, self.outdir, atlas="unknown")

    def test_norm_ss_coregister_rejects_reserved_name_in_additional_modalities(self):
        t1 = self._save_mock_nifti("t1.nii.gz")
        t1c = self._save_mock_nifti("t1c.nii.gz")
        t2 = self._save_mock_nifti("t2.nii.gz")
        fl = self._save_mock_nifti("flair.nii.gz")
        other = self._save_mock_nifti("other.nii.gz")

        with self.assertRaises(ValueError):
            norm_ss_coregister(
                t1, t1c, t2, fl, self.outdir, additional_modalities={"t2": other}
            )

    def test_norm_ss_coregister_rejects_reserved_name_in_quantitative_modalities(self):
        t1 = self._save_mock_nifti("t1.nii.gz")
        t1c = self._save_mock_nifti("t1c.nii.gz")
        t2 = self._save_mock_nifti("t2.nii.gz")
        fl = self._save_mock_nifti("flair.nii.gz")
        other = self._save_mock_nifti("other.nii.gz")

        with self.assertRaises(ValueError):
            norm_ss_coregister(
                t1,
                t1c,
                t2,
                fl,
                self.outdir,
                additional_quantitative_modalities={"flair": other},
            )

    def test_norm_ss_coregister_rejects_overlap_between_modality_dicts(self):
        t1 = self._save_mock_nifti("t1.nii.gz")
        t1c = self._save_mock_nifti("t1c.nii.gz")
        t2 = self._save_mock_nifti("t2.nii.gz")
        fl = self._save_mock_nifti("flair.nii.gz")
        swi_norm = self._save_mock_nifti("swi_norm.nii.gz")
        swi_quant = self._save_mock_nifti("swi_quant.nii.gz")

        with self.assertRaises(ValueError):
            norm_ss_coregister(
                t1,
                t1c,
                t2,
                fl,
                self.outdir,
                additional_modalities={"swi": swi_norm},
                additional_quantitative_modalities={"swi": swi_quant},
            )

    @patch("predict_gbm.preprocessing.norm_ss_coregistration.apply_longitudinal_warp")
    @patch("predict_gbm.preprocessing.norm_ss_coregistration.optimize_warp_field")
    @patch("predict_gbm.preprocessing.norm_ss_coregistration.resolve_dirac_disp_field")
    @patch("predict_gbm.preprocessing.norm_ss_coregistration.run_dirac_inference")
    def test_register_recurrence_runs_three_step_dirac_pipeline(
        self,
        run_dirac_mock,
        resolve_mock,
        optimize_mock,
        apply_mock,
    ):
        pre = self._save_mock_nifti("pre.nii.gz")
        post = self._save_mock_nifti("post.nii.gz")
        seg = self._save_mock_nifti("seg.nii.gz")

        longitudinal_dir = self.outdir / "longitudinal"
        longitudinal_dir.mkdir(parents=True, exist_ok=True)
        initial_fwd = longitudinal_dir / "followup_to_preop_disp_voxel.nii.gz"
        initial_bwd = longitudinal_dir / "preop_to_followup_disp_voxel.nii.gz"
        initial_fwd.write_text("dummy")
        initial_bwd.write_text("dummy")

        optimized = longitudinal_dir / "followup_to_preop_disp_voxel_optimized.nii.gz"

        resolve_mock.side_effect = [
            initial_fwd,
            initial_bwd,
            longitudinal_dir / "dirac_infer_case_yx_seg.nii.gz",
            longitudinal_dir / "dirac_infer_case_xy_seg.nii.gz",
        ]

        def _optimize_side_effect(*args, **kwargs):
            optimized.write_text("optimized")

        optimize_mock.side_effect = _optimize_side_effect

        register_recurrence(pre, post, seg, self.outdir)

        run_dirac_mock.assert_called_once()
        optimize_mock.assert_called_once()
        apply_mock.assert_called_once()
        self.assertTrue(
            (self.outdir / "longitudinal" / "longitudinal_trafo.nii.gz").exists()
        )

    @patch("predict_gbm.preprocessing.norm_ss_coregistration.ants.image_write")
    @patch("predict_gbm.preprocessing.norm_ss_coregistration.ants.apply_transforms")
    @patch("predict_gbm.preprocessing.norm_ss_coregistration.ants.registration")
    @patch("predict_gbm.preprocessing.norm_ss_coregistration.ants.image_read")
    def test_register_recurrence_syn_uses_ants_registration(
        self,
        image_read_mock,
        registration_mock,
        apply_transforms_mock,
        image_write_mock,
    ):
        pre = self._save_mock_nifti("pre.nii.gz")
        post = self._save_mock_nifti("post.nii.gz")
        seg = self._save_mock_nifti("seg.nii.gz")

        pre_img = MagicMock()
        pre_img.clone.return_value = pre_img
        post_img = MagicMock()
        seg_img = MagicMock()
        image_read_mock.side_effect = [pre_img, post_img, seg_img]

        trafo_file = self.outdir / "warp.nii.gz"
        trafo_file.write_text("warp")
        registration_mock.return_value = {
            "fwdtransforms": [str(trafo_file)],
            "warpedmovout": MagicMock(),
        }
        apply_transforms_mock.return_value = MagicMock()

        register_recurrence(
            pre,
            post,
            seg,
            self.outdir,
            registration_algorithm="syn",
        )

        registration_mock.assert_called_once()
        apply_transforms_mock.assert_called_once()
        self.assertTrue(
            (self.outdir / "longitudinal" / "longitudinal_trafo.nii.gz").exists()
        )

    @patch("predict_gbm.preprocessing.norm_ss_coregistration.warp_image_to_preop")
    @patch("predict_gbm.preprocessing.norm_ss_coregistration.apply_longitudinal_warp")
    @patch("predict_gbm.preprocessing.norm_ss_coregistration.optimize_warp_field")
    @patch("predict_gbm.preprocessing.norm_ss_coregistration.resolve_dirac_disp_field")
    @patch("predict_gbm.preprocessing.norm_ss_coregistration.run_dirac_inference")
    def test_register_recurrence_dirac_warps_additional_modalities(
        self,
        run_dirac_mock,
        resolve_mock,
        optimize_mock,
        apply_mock,
        warp_image_mock,
    ):
        pre = self._save_mock_nifti("pre.nii.gz")
        post = self._save_mock_nifti("post.nii.gz")
        seg = self._save_mock_nifti("seg.nii.gz")
        swi_post = self._save_mock_nifti("swi_post.nii.gz")

        longitudinal_dir = self.outdir / "longitudinal"
        longitudinal_dir.mkdir(parents=True, exist_ok=True)
        initial_fwd = longitudinal_dir / "followup_to_preop_disp_voxel.nii.gz"
        initial_bwd = longitudinal_dir / "preop_to_followup_disp_voxel.nii.gz"
        initial_fwd.write_text("dummy")
        initial_bwd.write_text("dummy")

        optimized = longitudinal_dir / "followup_to_preop_disp_voxel_optimized.nii.gz"

        resolve_mock.side_effect = [
            initial_fwd,
            initial_bwd,
            longitudinal_dir / "dirac_infer_case_yx_seg.nii.gz",
            longitudinal_dir / "dirac_infer_case_xy_seg.nii.gz",
        ]

        def _optimize_side_effect(*args, **kwargs):
            optimized.write_text("optimized")

        optimize_mock.side_effect = _optimize_side_effect

        register_recurrence(
            pre, post, seg, self.outdir, additional_modalities={"swi": swi_post}
        )

        warp_image_mock.assert_called_once()
        call_kwargs = warp_image_mock.call_args.kwargs
        self.assertEqual(call_kwargs["image_file"], swi_post)
        self.assertEqual(
            call_kwargs["out_file"],
            LONGITUDINAL_WARP_SCHEMA.format(base_dir=self.outdir, modality="swi"),
        )

    @patch("predict_gbm.preprocessing.norm_ss_coregistration.ants.image_write")
    @patch("predict_gbm.preprocessing.norm_ss_coregistration.ants.apply_transforms")
    @patch("predict_gbm.preprocessing.norm_ss_coregistration.ants.registration")
    @patch("predict_gbm.preprocessing.norm_ss_coregistration.ants.image_read")
    def test_register_recurrence_syn_warps_additional_modalities(
        self,
        image_read_mock,
        registration_mock,
        apply_transforms_mock,
        image_write_mock,
    ):
        pre = self._save_mock_nifti("pre.nii.gz")
        post = self._save_mock_nifti("post.nii.gz")
        seg = self._save_mock_nifti("seg.nii.gz")
        swi_post = self._save_mock_nifti("swi_post.nii.gz")

        pre_img = MagicMock()
        pre_img.clone.return_value = pre_img
        post_img = MagicMock()
        seg_img = MagicMock()
        swi_img = MagicMock()
        image_read_mock.side_effect = [pre_img, post_img, seg_img, swi_img]

        trafo_file = self.outdir / "warp.nii.gz"
        trafo_file.write_text("warp")
        registration_mock.return_value = {
            "fwdtransforms": [str(trafo_file)],
            "warpedmovout": MagicMock(),
        }
        apply_transforms_mock.return_value = MagicMock()

        register_recurrence(
            pre,
            post,
            seg,
            self.outdir,
            registration_algorithm="syn",
            additional_modalities={"swi": swi_post},
        )

        self.assertEqual(apply_transforms_mock.call_count, 2)
        written_paths = [str(c.args[1]) for c in image_write_mock.call_args_list]
        expected = str(
            LONGITUDINAL_WARP_SCHEMA.format(base_dir=self.outdir, modality="swi")
        )
        self.assertIn(expected, written_paths)

    def test_register_recurrence_rejects_t1c_in_additional_modalities(self):
        pre = self._save_mock_nifti("pre.nii.gz")
        post = self._save_mock_nifti("post.nii.gz")
        seg = self._save_mock_nifti("seg.nii.gz")

        with self.assertRaises(ValueError):
            register_recurrence(
                pre, post, seg, self.outdir, additional_modalities={"t1c": post}
            )

    @patch("predict_gbm.preprocessing.norm_ss_coregistration.ants.image_write")
    @patch("predict_gbm.preprocessing.norm_ss_coregistration.ants.apply_transforms")
    @patch("predict_gbm.preprocessing.norm_ss_coregistration.ants.registration")
    @patch("predict_gbm.preprocessing.norm_ss_coregistration.ants.image_read")
    def test_register_recurrence_linear_uses_ants_registration(
        self,
        image_read_mock,
        registration_mock,
        apply_transforms_mock,
        image_write_mock,
    ):
        for algorithm, expected_transform in (
            ("affine", "Affine"),
            ("rigid", "Rigid"),
        ):
            with self.subTest(algorithm=algorithm):
                image_read_mock.reset_mock()
                registration_mock.reset_mock()
                apply_transforms_mock.reset_mock()

                pre = self._save_mock_nifti("pre.nii.gz")
                post = self._save_mock_nifti("post.nii.gz")
                seg = self._save_mock_nifti("seg.nii.gz")

                pre_img = MagicMock()
                pre_img.clone.return_value = pre_img
                image_read_mock.side_effect = [pre_img, MagicMock(), MagicMock()]

                trafo_file = self.outdir / "0GenericAffine.mat"
                trafo_file.write_text("affine")
                registration_mock.return_value = {
                    "fwdtransforms": [str(trafo_file)],
                    "warpedmovout": MagicMock(),
                }
                apply_transforms_mock.return_value = MagicMock()

                register_recurrence(
                    pre,
                    post,
                    seg,
                    self.outdir,
                    registration_algorithm=algorithm,
                )

                self.assertEqual(
                    registration_mock.call_args.kwargs["type_of_transform"],
                    expected_transform,
                )
                apply_transforms_mock.assert_called_once()
                # Linear transforms are stored as .mat, not as a displacement field.
                self.assertTrue(
                    LONGITUDINAL_AFFINE_SCHEMA.format(base_dir=self.outdir).exists()
                )
                self.assertFalse(
                    LONGITUDINAL_DISP_SCHEMA.format(base_dir=self.outdir).exists()
                )

    def test_register_recurrence_invalid_algorithm_raises(self):
        pre = self._save_mock_nifti("pre.nii.gz")
        post = self._save_mock_nifti("post.nii.gz")
        seg = self._save_mock_nifti("seg.nii.gz")

        with self.assertRaises(ValueError):
            register_recurrence(
                pre,
                post,
                seg,
                self.outdir,
                registration_algorithm="invalid",
            )


if __name__ == "__main__":
    unittest.main()
