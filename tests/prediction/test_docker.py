import tempfile
import unittest
import warnings
import subprocess
import nibabel as nib
from pathlib import Path
from unittest.mock import MagicMock, patch
from tests.helpers import MockContainer, generate_mock_nifti
from predict_gbm.prediction import docker_funcs

# Silence third-party warnings that clutter test output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


class TestDockerFuncs(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.data_dir = Path(self.temp_dir.name) / "data"
        self.out_dir = Path(self.temp_dir.name) / "out"
        self.data_dir.mkdir()
        self.out_dir.mkdir()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_is_cuda_available(self):
        with patch("predict_gbm.prediction.docker_funcs.subprocess.run") as run_mock:
            run_mock.return_value = None
            self.assertTrue(docker_funcs._is_cuda_available())
        with patch(
            "predict_gbm.prediction.docker_funcs.subprocess.run",
            side_effect=Exception,
        ):
            self.assertFalse(docker_funcs._is_cuda_available())

    def test_handle_device_requests(self):
        with patch(
            "predict_gbm.prediction.docker_funcs._is_cuda_available", return_value=False
        ):
            self.assertEqual(
                docker_funcs._handle_device_requests("0", force_cpu=False), []
            )
        with patch(
            "predict_gbm.prediction.docker_funcs._is_cuda_available", return_value=True
        ):
            reqs = docker_funcs._handle_device_requests("0", force_cpu=False)
            self.assertEqual(len(reqs), 1)
            self.assertEqual(reqs[0].device_ids, ["0"])
        with patch(
            "predict_gbm.prediction.docker_funcs._is_cuda_available", return_value=True
        ):
            self.assertEqual(
                docker_funcs._handle_device_requests("0", force_cpu=True), []
            )

    def test_get_volume_mappings(self):
        mappings = docker_funcs._get_volume_mappings(self.data_dir, self.out_dir)
        self.assertEqual(
            set(mappings.keys()), {self.data_dir.resolve(), self.out_dir.resolve()}
        )
        self.assertEqual(mappings[self.data_dir.resolve()]["bind"], "/mlcube_io0")
        self.assertEqual(mappings[self.out_dir.resolve()]["bind"], "/mlcube_io1")

    def test_ensure_image(self):
        model_file = Path("model.tar")
        # Image exists
        with patch("predict_gbm.prediction.docker_funcs.subprocess.run") as run_mock:
            run_mock.return_value = None
            tag = docker_funcs._ensure_image("algo", model_file)
            self.assertEqual(tag, "algo:latest")
            run_mock.assert_called_once()
        # Image missing
        with patch("predict_gbm.prediction.docker_funcs.subprocess.run") as run_mock:
            run_mock.side_effect = [
                subprocess.CalledProcessError(returncode=1, cmd="inspect"),
                None,
            ]
            tag = docker_funcs._ensure_image("algo", model_file)
            self.assertEqual(tag, "algo:latest")
            self.assertEqual(run_mock.call_count, 2)

    def test_get_wandb_apikey(self):
        mock_api = MagicMock()
        mock_api.api_key = "key"
        with patch(
            "predict_gbm.prediction.docker_funcs.InternalApi", return_value=mock_api
        ):
            self.assertEqual(docker_funcs._get_wandb_apikey(), "key")
        mock_api.api_key = None
        with patch(
            "predict_gbm.prediction.docker_funcs.InternalApi", return_value=mock_api
        ):
            self.assertEqual(docker_funcs._get_wandb_apikey(), "")

    def test_observe_docker_output(self):
        container = MockContainer(status_code=0)
        output = docker_funcs._observe_docker_output(container)
        self.assertIn("line1", output)

        container_fail = MockContainer(status_code=1)
        with self.assertRaises(RuntimeError):
            docker_funcs._observe_docker_output(container_fail)

    def test_sanity_check_output(self):
        img = generate_mock_nifti()
        nib.save(img, self.out_dir / "out.nii.gz")
        docker_funcs._sanity_check_output(self.data_dir, self.out_dir, "")
        empty_dir = Path(self.temp_dir.name) / "empty"
        empty_dir.mkdir()
        with self.assertRaises(RuntimeError):
            docker_funcs._sanity_check_output(self.data_dir, empty_dir, "")


if __name__ == "__main__":
    unittest.main()
