"""TTSTune CLI."""

import click


@click.group()  # type: ignore[misc]
def main() -> None:
    """TTSTune CLI."""

@main.command()  # type: ignore[misc]
def train() -> None:
    """Train a TTS model."""

if __name__ == "__main__":
    main()
