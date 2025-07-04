"""Abstract base trainer for TTSTune."""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Dict, Any, Union, Tuple
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import Trainer, TrainingArguments, set_seed
from transformers.trainer_callback import TrainerCallback

from ttstune.config import TTSTuneConfig, ModelType
from ttstune.utils import (
    setup_device,
    setup_logging,
    get_logger,
    CheckpointManager,
    WandbLogger,
    setup_wandb_logging,
    create_dataset,
    create_data_collator,
)

logger = get_logger(__name__)


class TTSTuneCallback(TrainerCallback):
    """Custom callback for TTSTune training."""

    def __init__(self, wandb_logger: Optional[WandbLogger] = None):
        """Initialize callback.

        Args:
            wandb_logger: Optional wandb logger for custom logging
        """
        self.wandb_logger = wandb_logger

    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        """Called when logging occurs."""
        if self.wandb_logger and logs:
            # Filter and log metrics to wandb
            wandb_logs = {k: v for k, v in logs.items() if isinstance(v, (int, float))}
            if wandb_logs:
                self.wandb_logger.log(wandb_logs, step=state.global_step)


class TTSTuneTrainer(ABC):
    """Abstract base class for TTS trainers."""

    def __init__(self, config: TTSTuneConfig):
        """Initialize trainer.

        Args:
            config: TTSTune configuration
        """
        self.config = config
        self.device = None
        self.model = None
        self.tokenizer = None
        self.train_dataset = None
        self.eval_dataset = None
        self.data_collator = None
        self.trainer = None
        self.checkpoint_manager = None
        self.wandb_logger = None

        self._setup()

    def _setup(self) -> None:
        """Setup trainer components."""
        # Setup logging
        log_file = str(Path(self.config.training.output_dir) / "training.log")
        setup_logging(log_file=log_file)

        # Set seed
        set_seed(self.config.seed)

        # Setup device
        self.device = setup_device(self.config.device)
        logger.info(f"Using device: {self.device}")

        # Setup checkpoint manager
        self.checkpoint_manager = CheckpointManager(
            output_dir=self.config.training.output_dir,
            save_total_limit=self.config.training.save_total_limit,
        )

        # Setup wandb logging
        if self.config.wandb.enabled:
            training_config_dict = self.config.training.to_dict()
            self.wandb_logger = setup_wandb_logging(
                self.config.wandb, training_config_dict
            )

    @abstractmethod
    def load_model(self) -> None:
        """Load and setup the model."""
        pass

    @abstractmethod
    def create_datasets(self) -> None:
        """Create train and eval datasets."""
        pass

    @abstractmethod
    def create_data_collator(self) -> None:
        """Create data collator for the model."""
        pass

    @abstractmethod
    def compute_metrics(self, eval_preds) -> Dict[str, float]:
        """Compute evaluation metrics.

        Args:
            eval_preds: Evaluation predictions

        Returns:
            Dictionary of metrics
        """
        pass

    def create_training_arguments(self) -> TrainingArguments:
        """Create Hugging Face training arguments from config.

        Returns:
            TrainingArguments: Configured training arguments
        """
        training_config = self.config.training

        return TrainingArguments(
            output_dir=training_config.output_dir,
            num_train_epochs=training_config.num_train_epochs,
            per_device_train_batch_size=training_config.per_device_train_batch_size,
            per_device_eval_batch_size=training_config.per_device_eval_batch_size,
            gradient_accumulation_steps=training_config.gradient_accumulation_steps,
            learning_rate=training_config.learning_rate,
            weight_decay=training_config.weight_decay,
            warmup_steps=training_config.warmup_steps,
            warmup_ratio=training_config.warmup_ratio,
            logging_steps=training_config.logging_steps,
            eval_steps=training_config.eval_steps,
            save_steps=training_config.save_steps,
            save_total_limit=training_config.save_total_limit,
            evaluation_strategy=training_config.eval_strategy,
            save_strategy=training_config.save_strategy,
            load_best_model_at_end=training_config.load_best_model_at_end,
            metric_for_best_model=training_config.metric_for_best_model,
            greater_is_better=training_config.greater_is_better,
            fp16=training_config.fp16,
            bf16=training_config.bf16,
            gradient_checkpointing=training_config.gradient_checkpointing,
            dataloader_num_workers=training_config.dataloader_num_workers,
            dataloader_pin_memory=training_config.dataloader_pin_memory,
            remove_unused_columns=training_config.remove_unused_columns,
            optim=training_config.optim,
            lr_scheduler_type=training_config.lr_scheduler_type,
            resume_from_checkpoint=training_config.resume_from_checkpoint,
            do_train=True,
            do_eval=self.eval_dataset is not None,
            report_to="none",  # We handle wandb manually
            seed=self.config.seed,
        )

    def create_trainer(self) -> Trainer:
        """Create Hugging Face trainer.

        Returns:
            Trainer: Configured trainer
        """
        training_args = self.create_training_arguments()

        # Create callbacks
        callbacks = []

        # Add early stopping if configured
        if self.config.training.early_stopping_patience:
            from transformers import EarlyStoppingCallback

            callbacks.append(
                EarlyStoppingCallback(
                    early_stopping_patience=self.config.training.early_stopping_patience
                )
            )

        # Add custom callback
        callbacks.append(TTSTuneCallback(self.wandb_logger))

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            data_collator=self.data_collator,
            compute_metrics=self.compute_metrics if self.eval_dataset else None,
            callbacks=callbacks,
        )

        # Watch model with wandb if enabled
        if self.wandb_logger:
            self.wandb_logger.watch(self.model)

        return trainer

    def train(self) -> Dict[str, Any]:
        """Run training.

        Returns:
            Training results
        """
        logger.info("Starting training...")

        # Load model
        logger.info("Loading model...")
        self.load_model()

        # Create datasets
        logger.info("Creating datasets...")
        self.create_datasets()

        # Create data collator
        logger.info("Creating data collator...")
        self.create_data_collator()

        # Create trainer
        logger.info("Creating trainer...")
        self.trainer = self.create_trainer()

        # Train
        logger.info("Starting training loop...")
        train_result = self.trainer.train(
            resume_from_checkpoint=self.config.training.resume_from_checkpoint
        )

        # Save model
        logger.info("Saving final model...")
        self.save_model()

        # Log metrics
        metrics = train_result.metrics
        self.trainer.log_metrics("train", metrics)
        self.trainer.save_metrics("train", metrics)
        self.trainer.save_state()

        logger.info("Training completed!")
        return train_result

    def evaluate(self) -> Dict[str, Any]:
        """Run evaluation.

        Returns:
            Evaluation results
        """
        if not self.trainer:
            raise RuntimeError("Trainer not initialized. Run train() first.")

        if not self.eval_dataset:
            logger.warning("No evaluation dataset available.")
            return {}

        logger.info("Running evaluation...")
        metrics = self.trainer.evaluate()

        # Log metrics
        self.trainer.log_metrics("eval", metrics)
        self.trainer.save_metrics("eval", metrics)

        return metrics

    @abstractmethod
    def save_model(self) -> None:
        """Save the trained model in the appropriate format."""
        pass

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load model from checkpoint.

        Args:
            checkpoint_path: Path to checkpoint
        """
        if not self.model:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        logger.info(f"Loading checkpoint from {checkpoint_path}")
        self.checkpoint_manager.load_checkpoint(
            checkpoint_path, self.model, device=self.device
        )

    def cleanup(self) -> None:
        """Cleanup resources."""
        if self.wandb_logger:
            self.wandb_logger.finish()

        # Clear CUDA cache if using GPU
        if self.device and self.device.type == "cuda":
            torch.cuda.empty_cache()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()


class MultiComponentTrainer(TTSTuneTrainer):
    """Base class for trainers that handle multiple model components."""

    def __init__(self, config: TTSTuneConfig):
        """Initialize multi-component trainer."""
        super().__init__(config)
        self.component_trainers = {}

    @abstractmethod
    def get_component_configs(self) -> Dict[str, Any]:
        """Get configurations for individual components.

        Returns:
            Dictionary mapping component names to their configs
        """
        pass

    @abstractmethod
    def train_component(self, component_name: str) -> Dict[str, Any]:
        """Train a specific component.

        Args:
            component_name: Name of component to train

        Returns:
            Training results for the component
        """
        pass

    def train(self) -> Dict[str, Any]:
        """Train all components sequentially.

        Returns:
            Combined training results
        """
        results = {}
        component_configs = self.get_component_configs()

        for component_name, component_config in component_configs.items():
            logger.info(f"Training component: {component_name}")

            # Train component
            component_results = self.train_component(component_name)
            results[component_name] = component_results

            logger.info(f"Completed training component: {component_name}")

        return results
