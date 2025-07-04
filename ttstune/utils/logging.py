"""Logging utilities with wandb integration."""

import logging
import sys
from pathlib import Path
from typing import Optional, Dict, Any
import wandb
from ..config import WandbConfig


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    format_string: Optional[str] = None,
) -> None:
    """Setup logging configuration.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file to write logs to
        format_string: Custom format string for logs
    """
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Create formatter
    formatter = logging.Formatter(format_string, datefmt="%Y-%m-%d %H:%M:%S")

    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))

    # Clear existing handlers
    root_logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File handler if specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the specified name.

    Args:
        name: Logger name

    Returns:
        logging.Logger: Configured logger
    """
    return logging.getLogger(name)


class WandbLogger:
    """Weights & Biases logger wrapper."""

    def __init__(self, config: WandbConfig, training_config: Dict[str, Any]):
        """Initialize wandb logger.

        Args:
            config: Wandb configuration
            training_config: Training configuration to log
        """
        self.config = config
        self.enabled = config.enabled
        self.run = None

        if self.enabled:
            self._init_wandb(training_config)

    def _init_wandb(self, training_config: Dict[str, Any]) -> None:
        """Initialize wandb run."""
        try:
            wandb_config = {**training_config, **self.config.config}

            self.run = wandb.init(
                project=self.config.project,
                entity=self.config.entity,
                name=self.config.name,
                tags=self.config.tags,
                notes=self.config.notes,
                config=wandb_config,
                reinit=True,
            )

            logging.getLogger(__name__).info(f"Initialized wandb run: {self.run.name}")

        except Exception as e:
            logging.getLogger(__name__).warning(f"Failed to initialize wandb: {e}")
            self.enabled = False

    def log(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        """Log metrics to wandb.

        Args:
            metrics: Dictionary of metrics to log
            step: Optional step number
        """
        if self.enabled and self.run:
            try:
                wandb.log(metrics, step=step)
            except Exception as e:
                logging.getLogger(__name__).warning(f"Failed to log to wandb: {e}")

    def log_artifact(
        self,
        artifact_path: str,
        artifact_type: str = "model",
        name: Optional[str] = None,
    ) -> None:
        """Log an artifact to wandb.

        Args:
            artifact_path: Path to the artifact
            artifact_type: Type of artifact (model, dataset, etc.)
            name: Optional name for the artifact
        """
        if self.enabled and self.run:
            try:
                artifact = wandb.Artifact(
                    name=name or Path(artifact_path).name, type=artifact_type
                )
                artifact.add_file(artifact_path)
                self.run.log_artifact(artifact)
            except Exception as e:
                logging.getLogger(__name__).warning(
                    f"Failed to log artifact to wandb: {e}"
                )

    def finish(self) -> None:
        """Finish the wandb run."""
        if self.enabled and self.run:
            try:
                wandb.finish()
                logging.getLogger(__name__).info("Finished wandb run")
            except Exception as e:
                logging.getLogger(__name__).warning(f"Failed to finish wandb run: {e}")

    def watch(self, model, log_freq: int = 100) -> None:
        """Watch model parameters and gradients.

        Args:
            model: PyTorch model to watch
            log_freq: Frequency of logging gradients
        """
        if self.enabled and self.run:
            try:
                wandb.watch(model, log_freq=log_freq)
            except Exception as e:
                logging.getLogger(__name__).warning(
                    f"Failed to watch model in wandb: {e}"
                )


def setup_wandb_logging(
    config: WandbConfig, training_config: Dict[str, Any]
) -> WandbLogger:
    """Setup wandb logging.

    Args:
        config: Wandb configuration
        training_config: Training configuration to log

    Returns:
        WandbLogger: Configured wandb logger
    """
    return WandbLogger(config, training_config)
