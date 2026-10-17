import tempfile
from unittest.mock import MagicMock, patch
import unittest
import warnings
import nibabel as nib
import numpy as np
from pathlib import Path
from tests.helpers import generate_mock_nifti
from predict_gbm.preprocessing.norm_ss_coregistration import (
    normalize,
    initialize_center_modality,
    initialize_moving_modalities,
    norm_ss_coregister,
    register_recurrence,
    longitudinal_transform_file,
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

    @staticmethod
    def _apply_transforms_side_effect(*args, **kwargs):
        """Mocked ants.apply_transforms: the compose call writes the combined field file and
        returns its path, warping calls return an image mock."""
        if kwargs.get("compose"):
            composed = Path(kwargs["compose"] + "comptx.nii.gz")
            composed.write_text("field")
            return str(composed)
        return MagicMock()

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
    def test_norm_ss_coregister_defaults_to_sri24_atlas(self, preprocessor_cls_mock):
        t1 = self._save_mock_nifti("t1.nii.gz")
        t1c = self._save_mock_nifti("t1c.nii.gz")
        t2 = self._save_mock_nifti("t2.nii.gz")
        fl = self._save_mock_nifti("flair.nii.gz")

        norm_ss_coregister(t1, t1c, t2, fl, self.outdir)

        atlas_image_path = preprocessor_cls_mock.call_args.kwargs["atlas_image_path"]
        self.assertEqual(atlas_image_path, ATLAS_UNSTRIPPED_SCHEMA.format(atlas="sri24"))

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
    def test_norm_ss_coregister_can_switch_to_brats_mni152_atlas(
        self, preprocessor_cls_mock
    ):
        t1 = self._save_mock_nifti("t1.nii.gz")
        t1c = self._save_mock_nifti("t1c.nii.gz")
        t2 = self._save_mock_nifti("t2.nii.gz")
        fl = self._save_mock_nifti("flair.nii.gz")

        norm_ss_coregister(t1, t1c, t2, fl, self.outdir, atlas="brats_mni152")

        atlas_image_path = preprocessor_cls_mock.call_args.kwargs["atlas_image_path"]
        self.assertEqual(
            atlas_image_path, ATLAS_UNSTRIPPED_SCHEMA.format(atlas="brats_mni152")
        )

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
        self.assertEqual(atlas_image_path, ATLAS_STRIPPED_SCHEMA.format(atlas="sri24"))

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

    @patch("predict_gbm.preprocessing.norm_ss_coregistration.run_dirac_registration")
    @patch(
        "predict_gbm.preprocessing.norm_ss_coregistration.read_ants_affine_pre_to_post_mm"
    )
    @patch("predict_gbm.preprocessing.norm_ss_coregistration.ants.registration")
    def test_register_recurrence_runs_dirac_pipeline(
        self,
        registration_mock,
        read_affine_mock,
        run_dirac_mock,
    ):
        pre = self._save_mock_nifti("pre.nii.gz")
        post = self._save_mock_nifti("post.nii.gz")
        seg = self._save_mock_nifti("seg.nii.gz")
        others = {
            f"{modality}_{exam}_file": self._save_mock_nifti(f"{modality}_{exam}.nii.gz")
            for modality in ("t1", "t2", "flair")
            for exam in ("pre", "post")
        }

        registration_mock.return_value = {
            "fwdtransforms": [str(self.outdir / "affine_0GenericAffine.mat")]
        }
        read_affine_mock.return_value = (np.eye(4), np.zeros(3))
        # Zero displacement: the warped images must equal the follow-up images.
        run_dirac_mock.return_value = np.zeros((16, 16, 16, 3), dtype=np.float32)

        # The four-modality model needs t1, t2 and flair of both exams (C4).
        with self.assertRaisesRegex(ValueError, "dirac_single_modality"):
            register_recurrence(pre, post, seg, self.outdir)
        registration_mock.assert_not_called()

        register_recurrence(pre, post, seg, self.outdir, **others)

        registration_mock.assert_called_once()
        self.assertEqual(
            registration_mock.call_args.kwargs["type_of_transform"], "Affine"
        )
        self.assertNotIn("mask", registration_mock.call_args.kwargs)
        run_dirac_mock.assert_called_once()
        dirac_kwargs = run_dirac_mock.call_args.kwargs
        self.assertEqual(set(dirac_kwargs["pre_files"]), {"t1", "t1c", "flair", "t2"})
        self.assertEqual(dirac_kwargs["pre_files"]["t1c"], pre)
        self.assertEqual(dirac_kwargs["post_files"]["flair"], others["flair_post_file"])
        self.assertFalse(dirac_kwargs["single_modality"])

        longitudinal_dir = self.outdir / "longitudinal"
        self.assertEqual(
            sorted(p.name for p in longitudinal_dir.iterdir()),
            [
                "longitudinal_trafo.nii.gz",
                "recurrence_preop.nii.gz",
                "t1c_warped_longitudinal.nii.gz",
            ],
        )
        trafo = nib.load(LONGITUDINAL_DISP_SCHEMA.format(base_dir=self.outdir))
        self.assertEqual(trafo.shape[:3], (16, 16, 16))
        warped = nib.load(
            LONGITUDINAL_WARP_SCHEMA.format(base_dir=self.outdir, modality="t1c")
        ).get_fdata()
        np.testing.assert_allclose(warped, nib.load(post).get_fdata(), atol=1e-4)

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

        affine_file = self.outdir / "0GenericAffine.mat"
        affine_file.write_text("affine")
        warp_file = self.outdir / "1Warp.nii.gz"
        warp_file.write_text("warp")
        registration_mock.return_value = {
            "fwdtransforms": [str(warp_file), str(affine_file)]
        }
        apply_transforms_mock.side_effect = self._apply_transforms_side_effect

        register_recurrence(
            pre,
            post,
            seg,
            self.outdir,
            registration_algorithm="syn",
        )

        registration_mock.assert_called_once()
        # One call composes affine + warp into the stored field, two warp t1c and the
        # segmentation with that field.
        self.assertEqual(apply_transforms_mock.call_count, 3)
        compose_kwargs = apply_transforms_mock.call_args_list[0].kwargs
        self.assertEqual(compose_kwargs["transformlist"], [str(warp_file), str(affine_file)])
        disp_file = LONGITUDINAL_DISP_SCHEMA.format(base_dir=self.outdir)
        self.assertTrue(disp_file.exists())
        for call in apply_transforms_mock.call_args_list[1:]:
            self.assertEqual(call.kwargs["transformlist"], [str(disp_file)])

    @patch("predict_gbm.preprocessing.norm_ss_coregistration.run_dirac_registration")
    @patch(
        "predict_gbm.preprocessing.norm_ss_coregistration.read_ants_affine_pre_to_post_mm"
    )
    @patch("predict_gbm.preprocessing.norm_ss_coregistration.ants.registration")
    def test_register_recurrence_dirac_warps_additional_modalities(
        self,
        registration_mock,
        read_affine_mock,
        run_dirac_mock,
    ):
        pre = self._save_mock_nifti("pre.nii.gz")
        post = self._save_mock_nifti("post.nii.gz")
        seg = self._save_mock_nifti("seg.nii.gz")
        swi_post = self._save_mock_nifti("swi_post.nii.gz")

        registration_mock.return_value = {
            "fwdtransforms": [str(self.outdir / "affine_0GenericAffine.mat")]
        }
        read_affine_mock.return_value = (np.eye(4), np.zeros(3))
        run_dirac_mock.return_value = np.zeros((16, 16, 16, 3), dtype=np.float32)

        # The t1c-only model needs no further modalities.
        register_recurrence(
            pre,
            post,
            seg,
            self.outdir,
            additional_modalities={"swi": swi_post},
            dirac_single_modality=True,
        )

        dirac_kwargs = run_dirac_mock.call_args.kwargs
        self.assertTrue(dirac_kwargs["single_modality"])
        self.assertEqual(dirac_kwargs["pre_files"]["t1c"], pre)
        self.assertEqual(dirac_kwargs["post_files"]["t1c"], post)
        swi_warped_file = LONGITUDINAL_WARP_SCHEMA.format(
            base_dir=self.outdir, modality="swi"
        )
        self.assertTrue(swi_warped_file.exists())
        np.testing.assert_allclose(
            nib.load(swi_warped_file).get_fdata(),
            nib.load(swi_post).get_fdata(),
            atol=1e-4,
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
        registration_mock.return_value = {"fwdtransforms": [str(trafo_file)]}
        apply_transforms_mock.side_effect = self._apply_transforms_side_effect

        register_recurrence(
            pre,
            post,
            seg,
            self.outdir,
            registration_algorithm="syn",
            additional_modalities={"swi": swi_post},
        )

        # compose + t1c + segmentation + swi
        self.assertEqual(apply_transforms_mock.call_count, 4)
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
                registration_mock.return_value = {"fwdtransforms": [str(trafo_file)]}
                apply_transforms_mock.side_effect = self._apply_transforms_side_effect

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
                # t1c + segmentation, no compose call: the .mat is stored as is.
                self.assertEqual(apply_transforms_mock.call_count, 2)
                affine_file = LONGITUDINAL_AFFINE_SCHEMA.format(base_dir=self.outdir)
                self.assertTrue(affine_file.exists())
                self.assertEqual(affine_file.read_text(), "affine")
                self.assertFalse(
                    LONGITUDINAL_DISP_SCHEMA.format(base_dir=self.outdir).exists()
                )
                for call in apply_transforms_mock.call_args_list:
                    self.assertEqual(call.kwargs["transformlist"], [str(affine_file)])
                self.assertEqual(longitudinal_transform_file(self.outdir), affine_file)

    @patch("predict_gbm.preprocessing.norm_ss_coregistration.ants.image_write")
    @patch("predict_gbm.preprocessing.norm_ss_coregistration.ants.apply_transforms")
    @patch("predict_gbm.preprocessing.norm_ss_coregistration.ants.registration")
    @patch("predict_gbm.preprocessing.norm_ss_coregistration.ants.image_read")
    def test_register_recurrence_rerun_replaces_other_transform_format(
        self,
        image_read_mock,
        registration_mock,
        apply_transforms_mock,
        image_write_mock,
    ):
        """An affine run after a syn run (and vice versa) leaves exactly one transform file."""
        pre = self._save_mock_nifti("pre.nii.gz")
        post = self._save_mock_nifti("post.nii.gz")
        seg = self._save_mock_nifti("seg.nii.gz")
        pre_img = MagicMock()
        pre_img.clone.return_value = pre_img
        apply_transforms_mock.side_effect = self._apply_transforms_side_effect
        affine_file = LONGITUDINAL_AFFINE_SCHEMA.format(base_dir=self.outdir)
        disp_file = LONGITUDINAL_DISP_SCHEMA.format(base_dir=self.outdir)

        for algorithm, expected_file, other_file in (
            ("syn", disp_file, affine_file),
            ("affine", affine_file, disp_file),
            ("syn", disp_file, affine_file),
        ):
            with self.subTest(algorithm=algorithm):
                image_read_mock.side_effect = [pre_img, MagicMock(), MagicMock()]
                mat_file = self.outdir / "0GenericAffine.mat"
                mat_file.write_text("affine")
                fwdtransforms = [str(mat_file)]
                if algorithm == "syn":
                    warp_file = self.outdir / "1Warp.nii.gz"
                    warp_file.write_text("warp")
                    fwdtransforms = [str(warp_file), str(mat_file)]
                registration_mock.return_value = {"fwdtransforms": fwdtransforms}

                register_recurrence(
                    pre, post, seg, self.outdir, registration_algorithm=algorithm
                )

                self.assertTrue(expected_file.exists())
                self.assertFalse(other_file.exists())
                self.assertEqual(longitudinal_transform_file(self.outdir), expected_file)

    def test_longitudinal_transform_file_defaults_to_displacement_field(self):
        self.assertEqual(
            longitudinal_transform_file(self.outdir),
            LONGITUDINAL_DISP_SCHEMA.format(base_dir=self.outdir),
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
