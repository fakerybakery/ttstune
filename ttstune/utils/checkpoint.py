"""Checkpoint management utilities."""

import json
import logging
import shutil
from pathlib import Path
from typing import Any, Optional

import torch
from safetensors.torch import load_file, save_file

logger = logging.getLogger(__name__)


class CheckpointManager:
    """Manages model checkpoints and saves/loads model state."""

    def __init__(self, output_dir: str, save_total_limit: int = 3) -> None:
        """Initialize checkpoint manager.

        Args:
            output_dir: Directory to save checkpoints
            save_total_limit: Maximum number of checkpoints to keep

        """
        self.output_dir = Path(output_dir)
        self.save_total_limit = save_total_limit
        self.checkpoints: list[Path] = []

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load existing checkpoints
        self._discover_checkpoints()

    def _discover_checkpoints(self) -> None:
        """Discover existing checkpoints in output directory."""
        checkpoint_pattern = "checkpoint-*"
        checkpoints = list(self.output_dir.glob(checkpoint_pattern))

        # Sort by step number
        def get_step(checkpoint_path: Path) -> int:
            try:
                return int(checkpoint_path.name.split("-")[1])
            except (IndexError, ValueError):
                return 0

        self.checkpoints = sorted(checkpoints, key=get_step)
        logger.info(f"Found {len(self.checkpoints)} existing checkpoints")

    def save_checkpoint(
        self,
        model: torch.nn.Module,
        step: int,
        optimizer: Optional[torch.optim.Optimizer] = None,
        lr_scheduler: Optional[Any] = None,
        metrics: Optional[dict[str, float]] = None,
        extra_data: Optional[dict[str, Any]] = None,
    ) -> Path:
        """Save a model checkpoint.

        Args:
            model: Model to save
            step: Training step number
            optimizer: Optional optimizer state
            lr_scheduler: Optional learning rate scheduler state
            metrics: Optional metrics to save
            extra_data: Optional extra data to save

        Returns:
            Path: Path to saved checkpoint

        """
        checkpoint_dir = self.output_dir / f"checkpoint-{step}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Save model state dict using safetensors
        model_path = checkpoint_dir / "model.safetensors"
        model_state_dict = model.state_dict()
        save_file(model_state_dict, model_path)

        # Save training state
        training_state = {
            "step": step,
            "metrics": metrics or {},
            "extra_data": extra_data or {},
        }

        if optimizer is not None:
            training_state["optimizer"] = optimizer.state_dict()

        if lr_scheduler is not None:
            training_state["lr_scheduler"] = lr_scheduler.state_dict()

        # Save training state as JSON + torch
        state_path = checkpoint_dir / "training_state.json"
        torch_state_path = checkpoint_dir / "training_state.pt"

        # Save JSON-serializable parts
        json_state = {
            "step": training_state["step"],
            "metrics": training_state["metrics"],
            "extra_data": training_state["extra_data"],
        }

        with open(state_path, "w") as f:
            json.dump(json_state, f, indent=2)

        # Save torch-specific parts
        torch_state = {
            k: v
            for k, v in training_state.items()
            if k not in ["step", "metrics", "extra_data"]
        }

        if torch_state:
            torch.save(torch_state, torch_state_path)

        # Add to checkpoints list and manage limit
        self.checkpoints.append(checkpoint_dir)
        self._cleanup_old_checkpoints()

        logger.info(f"Saved checkpoint at step {step} to {checkpoint_dir}")
        return checkpoint_dir

    def load_checkpoint(
        self,
        checkpoint_path: str,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        lr_scheduler: Optional[Any] = None,
        device: Optional[torch.device] = None,
    ) -> dict[str, Any]:
        """Load a model checkpoint.

        Args:
            checkpoint_path: Path to checkpoint directory
            model: Model to load state into
            optimizer: Optional optimizer to load state into
            lr_scheduler: Optional scheduler to load state into
            device: Device to map tensors to

        Returns:
            Dict containing loaded training state

        """
        checkpoint_dir = Path(checkpoint_path)

        if not checkpoint_dir.exists():
            msg = f"Checkpoint not found: {checkpoint_dir}"
            raise FileNotFoundError(msg)

        # Load model state
        model_path = checkpoint_dir / "model.safetensors"
        if model_path.exists():
            model_state_dict = load_file(
                model_path,
                device=str(device) if device else "cpu",
            )
            model.load_state_dict(model_state_dict)
            logger.info(f"Loaded model state from {model_path}")
        else:
            logger.warning(f"Model state not found in {checkpoint_dir}")

        # Load training state
        training_state = {}

        # Load JSON state
        state_path = checkpoint_dir / "training_state.json"
        if state_path.exists():
            with open(state_path) as f:
                training_state.update(json.load(f))

        # Load torch state
        torch_state_path = checkpoint_dir / "training_state.pt"
        if torch_state_path.exists():
            torch_state = torch.load(torch_state_path, map_location=device)
            training_state.update(torch_state)

        # Load optimizer state
        if optimizer is not None and "optimizer" in training_state:
            try:
                optimizer.load_state_dict(training_state["optimizer"])
                logger.info("Loaded optimizer state")
            except Exception as e:
                logger.warning(f"Failed to load optimizer state: {e}")

        # Load scheduler state
        if lr_scheduler is not None and "lr_scheduler" in training_state:
            try:
                lr_scheduler.load_state_dict(training_state["lr_scheduler"])
                logger.info("Loaded learning rate scheduler state")
            except Exception as e:
                logger.warning(f"Failed to load scheduler state: {e}")

        logger.info(f"Loaded checkpoint from {checkpoint_dir}")
        return training_state

    def get_latest_checkpoint(self) -> Optional[Path]:
        """Get the latest checkpoint directory.

        Returns:
            Optional[Path]: Path to latest checkpoint or None if no checkpoints

        """
        if not self.checkpoints:
            return None
        return self.checkpoints[-1]

    def _cleanup_old_checkpoints(self) -> None:
        """Remove old checkpoints to stay within save_total_limit."""
        while len(self.checkpoints) > self.save_total_limit:
            old_checkpoint = self.checkpoints.pop(0)
            try:
                shutil.rmtree(old_checkpoint)
                logger.info(f"Removed old checkpoint: {old_checkpoint}")
            except Exception as e:
                logger.warning(f"Failed to remove checkpoint {old_checkpoint}: {e}")

    def save_final_model(
        self,
        model: torch.nn.Module,
        tokenizer: Optional[Any] = None,
        config: Optional[Any] = None,
    ) -> Path:
        """Save the final trained model.

        Args:
            model: Final trained model
            tokenizer: Optional tokenizer to save
            config: Optional model config to save

        Returns:
            Path: Path to saved model directory

        """
        final_model_dir = self.output_dir / "final_model"
        final_model_dir.mkdir(parents=True, exist_ok=True)

        # Save model
        model_path = final_model_dir / "model.safetensors"
        model_state_dict = model.state_dict()
        save_file(model_state_dict, model_path)

        # Save tokenizer if provided
        if tokenizer is not None and hasattr(tokenizer, "save_pretrained"):
            try:
                tokenizer.save_pretrained(final_model_dir)
                logger.info("Saved tokenizer")
            except Exception as e:
                logger.warning(f"Failed to save tokenizer: {e}")

        # Save config if provided
        if config is not None:
            try:
                if hasattr(config, "save_pretrained"):
                    config.save_pretrained(final_model_dir)
                elif hasattr(config, "to_dict"):
                    config_path = final_model_dir / "config.json"
                    with open(config_path, "w") as f:
                        json.dump(config.to_dict(), f, indent=2)
                logger.info("Saved model config")
            except Exception as e:
                logger.warning(f"Failed to save config: {e}")

        logger.info(f"Saved final model to {final_model_dir}")
        return final_model_dir
