"""Configuration system for TTSTune."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import yaml
from enum import Enum


class ModelType(str, Enum):
    """Supported model types."""

    CHATTERBOX = "chatterbox"
    F5_TTS = "f5_tts"
    CSM_1B = "csm_1b"
    ORPHEUS = "orpheus"
    STYLETTS2 = "styletts2"


class DatasetType(str, Enum):
    """Supported dataset types."""

    WAV_TXT = "wav_txt"  # 1.wav, 1.txt format
    HF_DATASET = "hf_dataset"  # Hugging Face dataset
    METADATA_CSV = "metadata_csv"  # CSV with audio_path, text columns
    METADATA_JSON = "metadata_json"  # JSON lines format


@dataclass
class WandbConfig:
    """Weights & Biases configuration."""

    enabled: bool = False
    project: Optional[str] = None
    entity: Optional[str] = None
    name: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    notes: Optional[str] = None
    config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DatasetConfig:
    """Dataset configuration."""

    dataset_path: Optional[str] = None
    dataset_type: DatasetType = DatasetType.WAV_TXT
    dataset_name: Optional[str] = None  # For HF datasets
    dataset_config_name: Optional[str] = None
    train_split_name: str = "train"
    eval_split_name: Optional[str] = "validation"
    text_column_name: str = "text"
    audio_column_name: str = "audio"
    eval_split_size: float = 0.01
    preprocessing_num_workers: Optional[int] = None
    ignore_verifications: bool = False

    # Audio processing
    max_audio_duration_s: Optional[float] = None
    min_audio_duration_s: Optional[float] = None
    max_audio_length: Optional[float] = None  # Alias for max_audio_duration_s
    min_audio_length: Optional[float] = None  # Alias for min_audio_duration_s
    sample_rate: Optional[int] = None

    # Text processing
    max_text_length: Optional[int] = None
    min_text_length: Optional[int] = None

    # HF dataset specific settings
    streaming: bool = False
    trust_remote_code: bool = False
    use_auth_token: Optional[str] = None


@dataclass
class ModelConfig:
    """Model configuration."""

    model_type: ModelType
    base_model: str  # Model name or path
    local_model_dir: Optional[str] = None
    cache_dir: Optional[str] = None

    # Freezing options
    freeze_components: List[str] = field(default_factory=list)

    # Model-specific parameters
    model_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainingConfig:
    """Training configuration."""

    output_dir: str
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 8
    gradient_accumulation_steps: int = 1
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_steps: int = 0
    warmup_ratio: float = 0.0
    logging_steps: int = 10
    eval_steps: Optional[int] = None
    save_steps: int = 500
    save_total_limit: int = 3
    eval_strategy: str = "steps"
    save_strategy: str = "steps"
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    early_stopping_patience: Optional[int] = None

    # Advanced training
    fp16: bool = False
    bf16: bool = False
    gradient_checkpointing: bool = False
    dataloader_num_workers: int = 0
    dataloader_pin_memory: bool = True
    remove_unused_columns: bool = False

    # Optimizer and scheduler
    optim: str = "adamw_torch"
    lr_scheduler_type: str = "linear"

    # Resume training
    resume_from_checkpoint: Optional[str] = None

    # Additional training options
    eval_on_start: bool = False
    report_to: List[str] = field(default_factory=lambda: ["tensorboard"])
    do_train: bool = True
    do_eval: bool = True
    gradient_clip_norm: Optional[float] = None
    label_names: Optional[List[str]] = None

    # Custom training parameters
    custom_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AdvancedConfig:
    """Advanced configuration options."""

    # Component-specific learning rates
    component_learning_rates: Dict[str, float] = field(default_factory=dict)

    # Training strategies
    progressive_training: bool = False

    # Data augmentation
    data_augmentation: Dict[str, Any] = field(default_factory=dict)

    # Custom preprocessing
    preprocessing: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TTSTuneConfig:
    """Main TTSTune configuration."""

    model: ModelConfig
    dataset: DatasetConfig
    training: TrainingConfig
    wandb: WandbConfig = field(default_factory=WandbConfig)
    advanced: AdvancedConfig = field(default_factory=AdvancedConfig)

    # Global settings
    seed: int = 42
    device: Optional[str] = None  # Auto-detect if None

    @classmethod
    def from_yaml(cls, yaml_path: Union[str, Path]) -> "TTSTuneConfig":
        """Load configuration from YAML file."""
        with open(yaml_path, "r", encoding="utf-8") as f:
            config_dict = yaml.safe_load(f)

        return cls.from_dict(config_dict)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "TTSTuneConfig":
        """Create configuration from dictionary."""
        # Convert string enums
        if "model" in config_dict and "model_type" in config_dict["model"]:
            config_dict["model"]["model_type"] = ModelType(
                config_dict["model"]["model_type"]
            )

        if "dataset" in config_dict and "dataset_type" in config_dict["dataset"]:
            config_dict["dataset"]["dataset_type"] = DatasetType(
                config_dict["dataset"]["dataset_type"]
            )

        # Type conversion for training config
        if "training" in config_dict:
            training_dict = config_dict["training"]
            # Convert numeric fields that might be strings
            numeric_fields = [
                "learning_rate",
                "weight_decay",
                "warmup_ratio",
                "eval_split_size",
                "gradient_clip_norm",
            ]
            for field in numeric_fields:
                if field in training_dict and isinstance(training_dict[field], str):
                    try:
                        training_dict[field] = float(training_dict[field])
                    except (ValueError, TypeError):
                        pass  # Keep original value if conversion fails

        # Type conversion for dataset config
        if "dataset" in config_dict:
            dataset_dict = config_dict["dataset"]
            # Convert numeric fields that might be strings
            numeric_fields = [
                "eval_split_size",
                "max_audio_length",
                "min_audio_length",
                "max_audio_duration_s",
                "min_audio_duration_s",
                "sample_rate",
                "max_text_length",
                "min_text_length",
            ]
            for field in numeric_fields:
                if field in dataset_dict and isinstance(dataset_dict[field], str):
                    try:
                        if field in [
                            "sample_rate",
                            "max_text_length",
                            "min_text_length",
                        ]:
                            dataset_dict[field] = int(dataset_dict[field])
                        else:
                            dataset_dict[field] = float(dataset_dict[field])
                    except (ValueError, TypeError):
                        pass  # Keep original value if conversion fails

        # Type conversion for advanced config
        if (
            "advanced" in config_dict
            and "component_learning_rates" in config_dict["advanced"]
        ):
            component_lrs = config_dict["advanced"]["component_learning_rates"]
            for component, lr in component_lrs.items():
                if isinstance(lr, str):
                    try:
                        component_lrs[component] = float(lr)
                    except (ValueError, TypeError):
                        pass  # Keep original value if conversion fails

        # Create nested configs
        model_config = ModelConfig(**config_dict.get("model", {}))
        dataset_config = DatasetConfig(**config_dict.get("dataset", {}))
        training_config = TrainingConfig(**config_dict.get("training", {}))
        wandb_config = WandbConfig(**config_dict.get("wandb", {}))
        advanced_config = AdvancedConfig(**config_dict.get("advanced", {}))

        # Create main config
        main_config_dict = {
            k: v
            for k, v in config_dict.items()
            if k not in ["model", "dataset", "training", "wandb", "advanced"]
        }

        return cls(
            model=model_config,
            dataset=dataset_config,
            training=training_config,
            wandb=wandb_config,
            advanced=advanced_config,
            **main_config_dict,
        )

    def to_yaml(self, yaml_path: Union[str, Path]) -> None:
        """Save configuration to YAML file."""
        config_dict = self.to_dict()

        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""

        def convert_dataclass(obj):
            if hasattr(obj, "__dataclass_fields__"):
                result = {}
                for field_name, field_def in obj.__dataclass_fields__.items():
                    value = getattr(obj, field_name)
                    if isinstance(value, Enum):
                        result[field_name] = value.value
                    elif hasattr(value, "__dataclass_fields__"):
                        result[field_name] = convert_dataclass(value)
                    elif isinstance(value, list):
                        result[field_name] = [
                            (
                                convert_dataclass(item)
                                if hasattr(item, "__dataclass_fields__")
                                else item
                            )
                            for item in value
                        ]
                    elif isinstance(value, dict):
                        result[field_name] = {
                            k: (
                                convert_dataclass(v)
                                if hasattr(v, "__dataclass_fields__")
                                else v
                            )
                            for k, v in value.items()
                        }
                    else:
                        result[field_name] = value
                return result
            return obj

        return convert_dataclass(self)

    def validate(self) -> None:
        """Validate configuration."""
        # Basic validation
        if self.model.model_type not in ModelType:
            raise ValueError(f"Unsupported model type: {self.model.model_type}")

        if self.dataset.dataset_type not in DatasetType:
            raise ValueError(f"Unsupported dataset type: {self.dataset.dataset_type}")

        # Dataset validation
        if self.dataset.dataset_type == DatasetType.WAV_TXT:
            if not self.dataset.dataset_path:
                raise ValueError("dataset_path is required for wav_txt dataset type")
        elif self.dataset.dataset_type == DatasetType.HF_DATASET:
            if not self.dataset.dataset_name:
                raise ValueError("dataset_name is required for hf_dataset dataset type")

        # Training validation
        if self.training.per_device_train_batch_size < 1:
            raise ValueError("per_device_train_batch_size must be >= 1")

        if self.training.learning_rate <= 0:
            raise ValueError("learning_rate must be > 0")

        # Wandb validation
        if self.wandb.enabled and not self.wandb.project:
            raise ValueError("wandb.project is required when wandb.enabled is True")


def load_config(config_path: Union[str, Path]) -> TTSTuneConfig:
    """Load and validate configuration from file."""
    config = TTSTuneConfig.from_yaml(config_path)
    config.validate()
    return config


def create_example_config(
    model_type: ModelType = ModelType.CHATTERBOX,
) -> TTSTuneConfig:
    """Create an example configuration."""
    return TTSTuneConfig(
        model=ModelConfig(
            model_type=model_type,
            base_model="ResembleAI/chatterbox",
            freeze_components=["voice_encoder", "s3gen"],
        ),
        dataset=DatasetConfig(
            dataset_path="./dataset",
            dataset_type=DatasetType.WAV_TXT,
            eval_split_size=0.01,
        ),
        training=TrainingConfig(
            output_dir="./outputs",
            num_train_epochs=10,
            per_device_train_batch_size=4,
            learning_rate=1e-4,
            save_steps=1000,
            eval_steps=500,
            early_stopping_patience=5,
        ),
        wandb=WandbConfig(
            enabled=True, project="ttstune-training", tags=["chatterbox", "finetuning"]
        ),
    )
