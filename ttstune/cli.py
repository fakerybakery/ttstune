"""TTSTune CLI."""

import click
from pathlib import Path
from typing import Optional

from .config import load_config, create_example_config, ModelType, TTSTuneConfig
from .trainers.chatterbox import ChatterboxTrainer
from .utils import get_logger

logger = get_logger(__name__)


@click.group()
def main() -> None:
    """TTSTune - Configuration-driven TTS model fine-tuning framework."""
    pass


@main.command()
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to YAML configuration file",
)
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
def train(config: Path, verbose: bool) -> None:
    """Train a TTS model using configuration file."""
    try:
        # Load configuration
        click.echo(f"Loading configuration from {config}")
        tts_config = load_config(config)

        # Setup logging level
        if verbose:
            import logging

            logging.getLogger().setLevel(logging.DEBUG)

        # Print configuration summary
        click.echo(f"Model: {tts_config.model.model_type.value}")
        click.echo(f"Base model: {tts_config.model.base_model}")
        click.echo(f"Dataset: {tts_config.dataset.dataset_type.value}")
        click.echo(f"Output directory: {tts_config.training.output_dir}")

        # Create trainer based on model type
        if tts_config.model.model_type == ModelType.CHATTERBOX:
            trainer = ChatterboxTrainer(tts_config)
        else:
            raise ValueError(f"Unsupported model type: {tts_config.model.model_type}")

        # Train the model
        with trainer:
            click.echo("Starting training...")
            results = trainer.train()

            click.echo("Training completed successfully!")

            # Print summary
            if isinstance(results, dict) and "train_loss" in results:
                click.echo(f"Final training loss: {results['train_loss']:.4f}")

            # Run evaluation if configured
            if tts_config.dataset.eval_split_size > 0:
                click.echo("Running evaluation...")
                eval_results = trainer.evaluate()
                if eval_results and "eval_loss" in eval_results:
                    click.echo(f"Evaluation loss: {eval_results['eval_loss']:.4f}")

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        if verbose:
            import traceback

            traceback.print_exc()
        raise click.Abort()


@main.command()
@click.option(
    "--model-type",
    type=click.Choice([t.value for t in ModelType]),
    default=ModelType.CHATTERBOX.value,
    help="Type of model to create config for",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    default="config.yaml",
    help="Output path for the configuration file",
)
def create_config(model_type: str, output: Path) -> None:
    """Create an example configuration file."""
    try:
        # Create example configuration
        model_type_enum = ModelType(model_type)
        config = create_example_config(model_type_enum)

        # Save to file
        config.to_yaml(output)

        click.echo(f"Created example configuration for {model_type} at {output}")
        click.echo("\nNext steps:")
        click.echo(
            "1. Edit the configuration file to match your dataset and requirements"
        )
        click.echo("2. Run training with: ttstune train --config config.yaml")

    except Exception as e:
        click.echo(f"Error creating config: {e}", err=True)
        raise click.Abort()


@main.command()
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to YAML configuration file",
)
def validate_config(config: Path) -> None:
    """Validate a configuration file."""
    try:
        # Load and validate configuration
        tts_config = load_config(config)

        click.echo("✓ Configuration is valid!")
        click.echo("\nConfiguration summary:")
        click.echo(f"  Model type: {tts_config.model.model_type.value}")
        click.echo(f"  Base model: {tts_config.model.base_model}")
        click.echo(f"  Dataset type: {tts_config.dataset.dataset_type.value}")

        if tts_config.dataset.dataset_path:
            dataset_path = Path(tts_config.dataset.dataset_path)
            if dataset_path.exists():
                click.echo(f"  Dataset path: {dataset_path} ✓")
            else:
                click.echo(f"  Dataset path: {dataset_path} ✗ (not found)")

        click.echo(f"  Output directory: {tts_config.training.output_dir}")
        click.echo(f"  Training epochs: {tts_config.training.num_train_epochs}")
        click.echo(f"  Batch size: {tts_config.training.per_device_train_batch_size}")
        click.echo(f"  Learning rate: {tts_config.training.learning_rate}")

        if tts_config.wandb.enabled:
            click.echo(f"  Wandb project: {tts_config.wandb.project}")

    except Exception as e:
        click.echo(f"✗ Configuration validation failed: {e}", err=True)
        raise click.Abort()


@main.command()
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to YAML configuration file",
)
@click.option(
    "--checkpoint",
    type=click.Path(exists=True, path_type=Path),
    help="Path to checkpoint directory to evaluate",
)
def evaluate(config: Path, checkpoint: Optional[Path]) -> None:
    """Evaluate a trained model."""
    try:
        # Load configuration
        tts_config = load_config(config)

        # Create trainer
        if tts_config.model.model_type == ModelType.CHATTERBOX:
            trainer = ChatterboxTrainer(tts_config)
        else:
            raise ValueError(f"Unsupported model type: {tts_config.model.model_type}")

        with trainer:
            # Load checkpoint if provided
            if checkpoint:
                click.echo(f"Loading checkpoint from {checkpoint}")
                trainer.load_checkpoint(str(checkpoint))

            # Run evaluation
            click.echo("Running evaluation...")
            results = trainer.evaluate()

            # Print results
            if results:
                click.echo("Evaluation results:")
                for key, value in results.items():
                    if isinstance(value, (int, float)):
                        click.echo(f"  {key}: {value:.4f}")
            else:
                click.echo("No evaluation results available")

    except Exception as e:
        click.echo(f"Error during evaluation: {e}", err=True)
        raise click.Abort()


@main.command()
def info() -> None:
    """Show information about TTSTune."""
    click.echo("TTSTune - Configuration-driven TTS model fine-tuning framework")
    click.echo("")
    click.echo("Supported models:")
    for model_type in ModelType:
        click.echo(f"  - {model_type.value}")
    click.echo("")
    click.echo("Dataset formats:")
    click.echo("  - wav_txt: Paired .wav and .txt files")
    click.echo("  - hf_dataset: Hugging Face datasets")
    click.echo("  - metadata_csv: CSV with audio_path, text columns")
    click.echo("  - metadata_json: JSON lines format")
    click.echo("")
    click.echo("Example usage:")
    click.echo("  1. Create config: ttstune create-config --model-type chatterbox")
    click.echo("  2. Edit config.yaml to match your setup")
    click.echo("  3. Validate config: ttstune validate-config --config config.yaml")
    click.echo("  4. Start training: ttstune train --config config.yaml")


if __name__ == "__main__":
    main()
