import sys
from loguru import logger
from .pipeline import DicomProcessor, NiftiProcessor

logger.remove()
logger.add(sys.stdout, level="INFO")

__all__ = [
    "DicomProcessor",
    "NiftiProcessor",
    "logger",
]
