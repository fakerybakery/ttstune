"""TTSTune - Configuration-driven TTS model fine-tuning framework."""

from ttstune.config import (
    TTSTuneConfig,
    ModelConfig,
    DatasetConfig,
    TrainingConfig,
    WandbConfig,
    ModelType,
    DatasetType,
    load_config,
    create_example_config,
)

from ttstune.trainers.base import TTSTuneTrainer, MultiComponentTrainer
from ttstune.trainers.chatterbox import ChatterboxTrainer

from ttstune.utils import (
    setup_logging,
    get_logger,
    setup_device,
    get_device,
    CheckpointManager,
    WandbLogger,
)

__version__ = "0.1.0"

__all__ = [
    # Config
    "TTSTuneConfig",
    "ModelConfig",
    "DatasetConfig",
    "TrainingConfig",
    "WandbConfig",
    "ModelType",
    "DatasetType",
    "load_config",
    "create_example_config",
    # Trainers
    "TTSTuneTrainer",
    "MultiComponentTrainer",
    "ChatterboxTrainer",
    # Utils
    "setup_logging",
    "get_logger",
    "setup_device",
    "get_device",
    "CheckpointManager",
    "WandbLogger",
]
