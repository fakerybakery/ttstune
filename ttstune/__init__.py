"""TTSTune - Configuration-driven TTS model fine-tuning framework."""

from .config import (
    DatasetConfig,
    DatasetType,
    ModelConfig,
    ModelType,
    TrainingConfig,
    TTSTuneConfig,
    WandbConfig,
    create_example_config,
    load_config,
)
from .trainers.base import MultiComponentTrainer, TTSTuneTrainer
from .trainers.chatterbox import ChatterboxTrainer
from .utils import (
    CheckpointManager,
    WandbLogger,
    get_device,
    get_logger,
    setup_device,
    setup_logging,
)

__version__ = "0.1.0"

__all__ = [
    "ChatterboxTrainer",
    "CheckpointManager",
    "DatasetConfig",
    "DatasetType",
    "ModelConfig",
    "ModelType",
    "MultiComponentTrainer",
    # Config
    "TTSTuneConfig",
    # Trainers
    "TTSTuneTrainer",
    "TrainingConfig",
    "WandbConfig",
    "WandbLogger",
    "create_example_config",
    "get_device",
    "get_logger",
    "load_config",
    "setup_device",
    # Utils
    "setup_logging",
]
