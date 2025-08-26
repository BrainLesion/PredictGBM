from pathlib import Path
from typing import Any, Optional


class BasePipe:
    """
    Abstract base class for pipeline components.

    Parameters:
        preop_dir (Path): Path to the output directory containing the pre-operative pipeline outputs.
        followup_dir (optional, Path): Path to the output directory containing the follow-up pipeline outputs.
        cuda_device (optional, str): The gpu device to use.
    """

    def __init__(
        self,
        preop_dir,
        followup_dir: Optional[Path] = None,
        cuda_device: Optional[str] = "0",
    ) -> None:
        self.preop_dir = preop_dir
        self.followup_dir = followup_dir
        self.cuda_device = cuda_device

    def run(self) -> Any:  # pragma: no cover - interface only
        """Execute the pipe component."""
        raise NotImplementedError

    def __call__(self) -> Any:
        """Allow calling the instance directly to execute :meth:`run`."""
        return self.run()
