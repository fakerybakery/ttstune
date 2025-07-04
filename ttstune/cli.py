"""TTSTune CLI."""

import click


@click.group()
def main() -> None:
    """TTSTune CLI."""

@main.command()
def train() -> None:
    """Train a TTS model."""

if __name__ == "__main__":
    main()
