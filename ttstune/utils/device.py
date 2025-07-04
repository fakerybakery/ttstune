"""Device management utilities."""

import torch
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def get_device(device: Optional[str] = None) -> torch.device:
    """Get the appropriate device for training/inference.

    Args:
        device: Specific device to use. If None, auto-detect best available.

    Returns:
        torch.device: The device to use.
    """
    if device is not None:
        return torch.device(device)

    if torch.cuda.is_available():
        device_name = "cuda"
        logger.info(f"CUDA available. Using GPU: {torch.cuda.get_device_name()}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device_name = "mps"
        logger.info("MPS (Apple Silicon) available. Using MPS.")
    else:
        device_name = "cpu"
        logger.info("Using CPU.")

    return torch.device(device_name)


def setup_device(device: Optional[str] = None) -> torch.device:
    """Setup and configure the device for training.

    Args:
        device: Specific device to use. If None, auto-detect best available.

    Returns:
        torch.device: The configured device.
    """
    device_obj = get_device(device)

    if device_obj.type == "cuda":
        # Set CUDA optimizations
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

        # Print GPU memory info
        total_memory = torch.cuda.get_device_properties(device_obj).total_memory / 1e9
        logger.info(f"GPU Memory: {total_memory:.1f} GB")

        # Clear cache
        torch.cuda.empty_cache()

    return device_obj


def get_device_info() -> dict:
    """Get detailed device information.

    Returns:
        dict: Device information including type, memory, etc.
    """
    info = {
        "device_type": "cpu",
        "device_name": "CPU",
        "memory_gb": None,
        "cuda_available": torch.cuda.is_available(),
        "mps_available": hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available(),
    }

    if torch.cuda.is_available():
        info.update(
            {
                "device_type": "cuda",
                "device_name": torch.cuda.get_device_name(),
                "memory_gb": torch.cuda.get_device_properties(0).total_memory / 1e9,
                "cuda_version": torch.version.cuda,
                "cudnn_version": torch.backends.cudnn.version(),
            }
        )
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        info.update({"device_type": "mps", "device_name": "Apple Silicon (MPS)"})

    return info
