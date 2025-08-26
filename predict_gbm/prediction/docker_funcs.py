import time
import docker
import subprocess
from pathlib import Path
from loguru import logger
from rich.console import Console
from wandb.apis import InternalApi
from docker.errors import DockerException
from typing import Dict, List


try:
    client = docker.from_env()
except DockerException as e:
    logger.error(
        f"Failed to connect to docker daemon. Please make sure docker is installed and running. Error: {e}"
    )
    # not aborting since this happens during read the docs builds. not a great solution tbf


def _is_cuda_available() -> bool:
    """Check if CUDA is available on the system by trying to run nvidia-smi."""
    try:
        # Attempt to run `nvidia-smi` to check for CUDA.
        # This command should run successfully if NVIDIA drivers are installed and GPUs are present.
        subprocess.run(
            ["nvidia-smi"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
        return True
    except Exception:
        return False


def _get_wandb_apikey():
    """Retrieves wandb API key and returns it. If no key was found returns an empty string."""
    api_key = InternalApi().api_key
    if api_key is None:
        api_key = ""
        logger.warning("Could not find a wandb API key. Models using wandb might fail.")
    return api_key


def _handle_device_requests(
    cuda_device: str, force_cpu: bool
) -> List[docker.types.DeviceRequest]:
    """Handle the device requests for the docker container (request cuda or cpu).

    Args:
        cuda_device (str): The CUDA devices to use
        force_cpu (bool): Whether to force CPU execution
    """
    cuda_available = _is_cuda_available()
    if not cuda_available or force_cpu:
        # empty device requests => run on CPU
        logger.info("Forcing CPU execution")
        return []
    # request gpu with chosen devices
    return [
        docker.types.DeviceRequest(device_ids=[cuda_device], capabilities=[["gpu"]])
    ]


def _get_volume_mappings(data_dir: Path, outdir: Path) -> Dict:
    """Get the volume mappings for the docker container.

    Args:
        data_dir: The path to the input data
        outdir: The path to save the output

    Returns:
        Dict: The volume mappings
    """
    # TODO: add support for recommended "ro" mount mode for input data
    # data = mlcube_io0, output = mlcube_io1
    return {
        volume.resolve(): {
            "bind": f"/mlcube_io{i}",
            "mode": "rw",
        }
        for i, volume in enumerate([data_dir, outdir])
    }


def _ensure_image(algorithm: str, model_file: Path) -> str:
    """
    Checks if algorithm:latest image is present. If not loads model_file into docker. Returns the image tag.

    Args:
        algorithm: Algorithm name
        model_file: Path to the growth model docker image
    """
    image_tag = f"{algorithm}:latest"

    try:
        # Check if the docker image exists by trying to inspect it
        subprocess.run(
            ["docker", "inspect", image_tag],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        logger.info(f"Image '{image_tag}' found. Skipping loading the image.")

    except subprocess.CalledProcessError:
        logger.info(
            f"Image '{image_tag}' not found. Loading image from '{model_file}'..."
        )
        try:
            # Load the docker image
            subprocess.run(["docker", "load", "-i", model_file], check=True)
            logger.info(f"Image '{image_tag}' loaded successfully from '{model_file}'.")
        except subprocess.CalledProcessError as e:
            raise e

    return image_tag


def _observe_docker_output(container: docker.models.containers.Container) -> str:
    """Observe the output of a running docker container and display a spinner. On Errors log container output.

    Args:
        container (docker.models.containers.Container): The container to observe
    """
    # capture the output
    container_output = container.attach(
        stdout=True, stderr=True, stream=True, logs=True
    )

    # Display spinner while the container is running
    with Console().status("Running inference..."):
        # Wait for the container to finish
        exit_code = container.wait()
        container_output = "\n\r".join(
            [line.decode("utf-8", errors="replace") for line in container_output]
        )
        # Check if the container exited with an error
        if exit_code["StatusCode"] != 0:
            logger.error(f">> {container_output}")
            raise RuntimeError(
                "Container finished with an error. See logs above for details."
            )

    return container_output


def _sanity_check_output(data_dir: Path, outdir: Path, container_output: str) -> None:
    """Sanity check that the number of output files matches the number of input files and the output is not empty.

    Args:
        data_dir: The path to the input data
        outdir: The path to the output data
        container_output: The output of the docker container
    """
    outputs = list(outdir.iterdir())
    if len(outputs) < 1:
        logger.error(f"Docker container output: \n\r{container_output}")
        raise RuntimeError(
            f"Expected 1 or more output files but got {len(outputs)}. Check docker container log for more."
        )


def run_container(
    algorithm: str,
    model_file: Path,
    data_dir: Path,
    outdir: Path,
    cuda_device: str,
    force_cpu: bool,
) -> None:
    """Run a docker container for the provided algorithm.

    Args:
        algorithm: Name of the algorithm
        model_file: Path to the growth model docker image.
        data_dir: The path to the input data
        outdir: The path to save the output
        cuda_device: The CUDA devices to use
        force_cpu: Whether to force CPU execution
    """
    # ensure output folder exists
    outdir.mkdir(parents=True, exist_ok=True)

    volume_mappings = _get_volume_mappings(data_dir=data_dir, outdir=outdir)
    logger.debug(f"Volume mappings: {volume_mappings}")

    # device setup
    device_requests = _handle_device_requests(
        cuda_device=cuda_device, force_cpu=force_cpu
    )
    logger.debug(f"GPU Device requests: {device_requests}")

    # load image if necessary
    image_tag = _ensure_image(algorithm, model_file)

    # get wandb api key
    wandb_apikey = _get_wandb_apikey()

    # Run the container
    logger.info(f"{'Starting growth prediction'}")
    start_time = time.time()
    container = client.containers.run(
        image=image_tag,
        volumes=volume_mappings,
        device_requests=device_requests,
        detach=True,
        shm_size="20gb",
        environment={"WANDB_API_KEY": wandb_apikey},
        # user=f"{os.getuid()}:{os.getgid()}"  # this line disables running as root
    )
    container_output = _observe_docker_output(container=container)

    # Remove container and check output
    container.remove()
    _sanity_check_output(
        data_dir=data_dir, outdir=outdir, container_output=container_output
    )

    logger.debug(f"Docker container output: \n\r{container_output}")

    logger.info(f"Finished inference in {time.time() - start_time:.2f} seconds")
