"""Chatterbox trainer implementation."""

import logging
import os
from pathlib import Path
from typing import Dict, Any, Optional, List
import torch
from dataclasses import dataclass
import subprocess
import sys

from ttstune.config import TTSTuneConfig, ModelType, DatasetConfig, TrainingConfig
from ttstune.utils import get_logger
from ttstune.trainers.base import MultiComponentTrainer

logger = get_logger(__name__)


@dataclass
class ChatterboxComponentConfig:
    """Configuration for a Chatterbox component."""

    name: str
    script_path: str
    model_args: Dict[str, Any]
    data_args: Dict[str, Any]
    training_args: Dict[str, Any]
    enabled: bool = True


class ChatterboxTrainer(MultiComponentTrainer):
    """Trainer for Chatterbox TTS model that handles T3 and S3Gen components."""

    def __init__(self, config: TTSTuneConfig):
        """Initialize Chatterbox trainer.

        Args:
            config: TTSTune configuration
        """
        if config.model.model_type != ModelType.CHATTERBOX:
            raise ValueError(
                f"Expected model_type 'chatterbox', got '{config.model.model_type}'"
            )

        super().__init__(config)

        # Determine which components to train based on frozen components
        self.train_t3 = "t3" not in config.model.freeze_components
        self.train_s3gen = "s3gen" not in config.model.freeze_components

        if not self.train_t3 and not self.train_s3gen:
            raise ValueError("At least one component (t3 or s3gen) must be trainable")

        logger.info(
            f"Training components: T3={self.train_t3}, S3Gen={self.train_s3gen}"
        )

    def get_component_configs(self) -> Dict[str, ChatterboxComponentConfig]:
        """Get configurations for individual components.

        Returns:
            Dictionary mapping component names to their configs
        """
        components = {}

        if self.train_t3:
            components["t3"] = self._create_t3_config()

        if self.train_s3gen:
            components["s3gen"] = self._create_s3gen_config()

        return components

    def _create_t3_config(self) -> ChatterboxComponentConfig:
        """Create configuration for T3 training."""
        # Convert TTSTune config to T3 trainer format
        model_args = {
            "model_name_or_path": self.config.model.base_model,
            "local_model_dir": self.config.model.local_model_dir,
            "cache_dir": self.config.model.cache_dir,
            "freeze_voice_encoder": "voice_encoder"
            in self.config.model.freeze_components,
            "freeze_s3gen": True,  # Always freeze S3Gen during T3 training
        }

        data_args = self._convert_dataset_config()
        training_args = self._convert_training_config("t3")

        return ChatterboxComponentConfig(
            name="t3",
            script_path=str(Path(__file__).parent / "trainer_t3.py"),
            model_args=model_args,
            data_args=data_args,
            training_args=training_args,
        )

    def _create_s3gen_config(self) -> ChatterboxComponentConfig:
        """Create configuration for S3Gen training."""
        model_args = {
            "model_name_or_path": self.config.model.base_model,
            "local_model_dir": self.config.model.local_model_dir,
            "cache_dir": self.config.model.cache_dir,
            "freeze_speaker_encoder": "speaker_encoder"
            in self.config.model.freeze_components,
            "freeze_s3_tokenizer": "s3_tokenizer"
            in self.config.model.freeze_components,
        }

        data_args = self._convert_dataset_config()
        training_args = self._convert_training_config("s3gen")

        return ChatterboxComponentConfig(
            name="s3gen",
            script_path=str(Path(__file__).parent / "trainer_s3gen.py"),
            model_args=model_args,
            data_args=data_args,
            training_args=training_args,
        )

    def _convert_dataset_config(self) -> Dict[str, Any]:
        """Convert TTSTune dataset config to component format."""
        dataset_config = self.config.dataset

        if dataset_config.dataset_type.value == "wav_txt":
            return {
                "dataset_dir": dataset_config.dataset_path,
                "metadata_file": None,
                "dataset_name": None,
                "eval_split_size": dataset_config.eval_split_size,
                "preprocessing_num_workers": dataset_config.preprocessing_num_workers,
                "ignore_verifications": dataset_config.ignore_verifications,
            }
        elif dataset_config.dataset_type.value == "hf_dataset":
            return {
                "dataset_dir": None,
                "metadata_file": None,
                "dataset_name": dataset_config.dataset_name,
                "dataset_config_name": dataset_config.dataset_config_name,
                "train_split_name": dataset_config.train_split_name,
                "eval_split_name": dataset_config.eval_split_name,
                "text_column_name": dataset_config.text_column_name,
                "audio_column_name": dataset_config.audio_column_name,
                "eval_split_size": dataset_config.eval_split_size,
                "preprocessing_num_workers": dataset_config.preprocessing_num_workers,
                "ignore_verifications": dataset_config.ignore_verifications,
            }
        else:
            # For other formats, use metadata file approach
            return {
                "dataset_dir": None,
                "metadata_file": dataset_config.dataset_path,
                "dataset_name": None,
                "eval_split_size": dataset_config.eval_split_size,
                "preprocessing_num_workers": dataset_config.preprocessing_num_workers,
                "ignore_verifications": dataset_config.ignore_verifications,
            }

    def _convert_training_config(self, component: str) -> Dict[str, Any]:
        """Convert TTSTune training config to component format."""
        training_config = self.config.training

        # Create component-specific output directory
        output_dir = Path(training_config.output_dir) / component

        # Determine if evaluation is actually possible
        # load_best_model_at_end requires evaluation to be enabled
        # If eval_split_size is very small or 0, evaluation might not be possible
        eval_split_size = self.config.dataset.eval_split_size
        load_best_model_at_end = training_config.load_best_model_at_end

        # Disable load_best_model_at_end if evaluation is unlikely to work
        # Very small eval_split_size might not create a meaningful evaluation set
        if eval_split_size <= 0 or eval_split_size < 0.01:  # Less than 1%
            load_best_model_at_end = False
            logger.warning(
                f"Disabling load_best_model_at_end for {component} because eval_split_size "
                f"({eval_split_size}) is too small to create a meaningful evaluation set (< 1%)"
            )

        # Also check if we have an explicit eval split name but it might not exist
        # In that case, we'll let the individual trainer handle it and disable load_best_model_at_end
        # if no eval dataset is created
        if (
            hasattr(self.config.dataset, "eval_split_name")
            and self.config.dataset.eval_split_name
            and eval_split_size <= 0
        ):
            # If we have an eval split name but eval_split_size is 0, we're relying on the dataset
            # having that split. If it doesn't exist, evaluation will fail.
            # We'll be conservative and disable load_best_model_at_end
            load_best_model_at_end = False
            logger.warning(
                f"Disabling load_best_model_at_end for {component} because eval_split_size is 0 "
                f"and we're relying on dataset split '{self.config.dataset.eval_split_name}' which may not exist"
            )

        return {
            "output_dir": str(output_dir),
            "num_train_epochs": training_config.num_train_epochs,
            "per_device_train_batch_size": training_config.per_device_train_batch_size,
            "per_device_eval_batch_size": training_config.per_device_eval_batch_size,
            "gradient_accumulation_steps": training_config.gradient_accumulation_steps,
            "learning_rate": training_config.learning_rate,
            "weight_decay": training_config.weight_decay,
            "warmup_steps": training_config.warmup_steps,
            "warmup_ratio": training_config.warmup_ratio,
            "logging_steps": training_config.logging_steps,
            "eval_steps": training_config.eval_steps,
            "save_steps": training_config.save_steps,
            "save_total_limit": training_config.save_total_limit,
            "evaluation_strategy": training_config.eval_strategy,
            "save_strategy": training_config.save_strategy,
            "load_best_model_at_end": load_best_model_at_end,
            "metric_for_best_model": training_config.metric_for_best_model,
            "greater_is_better": training_config.greater_is_better,
            "early_stopping_patience": training_config.early_stopping_patience,
            "fp16": training_config.fp16,
            "bf16": training_config.bf16,
            "gradient_checkpointing": training_config.gradient_checkpointing,
            "dataloader_num_workers": training_config.dataloader_num_workers,
            "dataloader_pin_memory": training_config.dataloader_pin_memory,
            "remove_unused_columns": training_config.remove_unused_columns,
            "optim": training_config.optim,
            "lr_scheduler_type": training_config.lr_scheduler_type,
            "resume_from_checkpoint": training_config.resume_from_checkpoint,
            "do_train": True,
            "do_eval": True,
            "seed": self.config.seed,
        }

    def train_component(self, component_name: str) -> Dict[str, Any]:
        """Train a specific component using its dedicated trainer.

        Args:
            component_name: Name of component to train (t3 or s3gen)

        Returns:
            Training results for the component
        """
        components = self.get_component_configs()

        if component_name not in components:
            raise ValueError(f"Component '{component_name}' not found in configuration")

        component_config = components[component_name]

        # Create arguments list for subprocess
        args = [sys.executable, component_config.script_path]

        logger.info(f"Preparing {component_name} training...")
        logger.info(f"Python executable: {sys.executable}")
        logger.info(f"Script path: {component_config.script_path}")
        logger.info(f"Working directory: {os.getcwd()}")

        # Add model arguments
        for key, value in component_config.model_args.items():
            if value is not None:
                if isinstance(value, bool):
                    if value:
                        args.append(f"--{key}")
                else:
                    args.extend([f"--{key}", str(value)])

        # Add data arguments
        for key, value in component_config.data_args.items():
            if value is not None:
                if isinstance(value, bool):
                    if value:
                        args.append(f"--{key}")
                else:
                    args.extend([f"--{key}", str(value)])

        # Add training arguments
        for key, value in component_config.training_args.items():
            if value is not None:
                if isinstance(value, bool):
                    if value:
                        args.append(f"--{key}")
                else:
                    args.extend([f"--{key}", str(value)])

        # Log the full command for debugging
        logger.info(f"Full command: {' '.join(args)}")
        logger.info(f"Total arguments: {len(args)}")

        # Log key arguments for visibility
        key_args = {}
        for i, arg in enumerate(args):
            if arg.startswith("--"):
                key = arg[2:]  # Remove --
                if i + 1 < len(args) and not args[i + 1].startswith("--"):
                    key_args[key] = args[i + 1]
                else:
                    key_args[key] = True

        logger.info(f"Key arguments: {key_args}")
        logger.info(f"Starting {component_name} training subprocess...")

        # Run the component trainer with real-time output
        try:
            logger.info(f"Launching {component_name} subprocess...")

            # Start the process with real-time output streaming
            process = subprocess.Popen(
                args,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,  # Combine stderr with stdout
                text=True,
                bufsize=1,  # Line buffered
                universal_newlines=True,
            )

            logger.info(f"{component_name} process started with PID: {process.pid}")

            # Stream output in real-time
            output_lines = []
            while True:
                output = process.stdout.readline()
                if output == "" and process.poll() is not None:
                    break
                if output:
                    line = output.strip()
                    output_lines.append(line)
                    # Log progress lines and important messages
                    if any(
                        keyword in line.lower()
                        for keyword in [
                            "downloading",
                            "progress",
                            "epoch",
                            "step",
                            "loss",
                            "loading",
                            "generating",
                            "extracting",
                            "error",
                            "warning",
                        ]
                    ):
                        logger.info(f"[{component_name}] {line}")
                    else:
                        logger.debug(f"[{component_name}] {line}")

            # Wait for process to complete
            return_code = process.poll()

            if return_code == 0:
                logger.info(f"{component_name} training completed successfully")
                return {
                    "success": True,
                    "returncode": return_code,
                    "stdout": "\n".join(output_lines),
                    "stderr": "",
                }
            else:
                logger.error(
                    f"{component_name} training failed with return code {return_code}"
                )
                # Log last few lines for debugging
                if output_lines:
                    logger.error(f"{component_name} last output lines:")
                    for line in output_lines[-10:]:  # Last 10 lines
                        logger.error(f"[{component_name}] {line}")

                return {
                    "success": False,
                    "returncode": return_code,
                    "stdout": "\n".join(output_lines),
                    "stderr": "",
                    "error": f"Process failed with return code {return_code}",
                }

        except Exception as e:
            logger.error(f"Exception while running {component_name} training: {e}")
            logger.error(f"Command was: {' '.join(args)}")

            return {
                "success": False,
                "returncode": -1,
                "stdout": "",
                "stderr": str(e),
                "error": str(e),
            }

    def load_model(self) -> None:
        """Load model (not used in multi-component approach)."""
        # This is handled by individual component trainers
        pass

    def create_datasets(self) -> None:
        """Create datasets (not used in multi-component approach)."""
        # This is handled by individual component trainers
        pass

    def create_data_collator(self) -> None:
        """Create data collator (not used in multi-component approach)."""
        # This is handled by individual component trainers
        pass

    def compute_metrics(self, eval_preds) -> Dict[str, float]:
        """Compute metrics (not used in multi-component approach)."""
        # This is handled by individual component trainers
        return {}

    def save_model(self) -> None:
        """Save the final combined model."""
        logger.info("Assembling final Chatterbox model from trained components...")

        output_dir = Path(self.config.training.output_dir)
        final_model_dir = output_dir / "final_model"
        final_model_dir.mkdir(parents=True, exist_ok=True)

        # Copy trained components to final model directory
        components_to_copy = []

        if self.train_t3:
            t3_dir = output_dir / "t3"
            t3_model_file = t3_dir / "t3_cfg.safetensors"
            if t3_model_file.exists():
                import shutil

                shutil.copy2(t3_model_file, final_model_dir / "t3_cfg.safetensors")
                components_to_copy.append("t3_cfg.safetensors")
                logger.info("Copied trained T3 model")

        if self.train_s3gen:
            s3gen_dir = output_dir / "s3gen"
            s3gen_model_file = s3gen_dir / "s3gen.safetensors"
            if s3gen_model_file.exists():
                import shutil

                shutil.copy2(s3gen_model_file, final_model_dir / "s3gen.safetensors")
                components_to_copy.append("s3gen.safetensors")
                logger.info("Copied trained S3Gen model")

        # Copy other required files from the first available component directory
        other_files = ["ve.safetensors", "tokenizer.json", "conds.pt"]

        for component in ["t3", "s3gen"]:
            component_dir = output_dir / component
            if component_dir.exists():
                for file_name in other_files:
                    src_file = component_dir / file_name
                    dst_file = final_model_dir / file_name
                    if src_file.exists() and not dst_file.exists():
                        import shutil

                        shutil.copy2(src_file, dst_file)
                        logger.info(f"Copied {file_name} from {component} directory")

        # Create a model info file
        model_info = {
            "model_type": "chatterbox",
            "base_model": self.config.model.base_model,
            "trained_components": components_to_copy,
            "training_config": self.config.training.to_dict(),
            "dataset_config": self.config.dataset.to_dict(),
        }

        import json

        with open(final_model_dir / "model_info.json", "w") as f:
            json.dump(model_info, f, indent=2)

        logger.info(f"Final Chatterbox model saved to {final_model_dir}")

        # Log to wandb if enabled
        if self.wandb_logger:
            self.wandb_logger.log_artifact(
                str(final_model_dir), artifact_type="model", name="chatterbox_finetuned"
            )
