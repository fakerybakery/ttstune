# TTSTune Documentation

Welcome to TTSTune - a configuration-driven framework for fine-tuning Text-to-Speech (TTS) models.

## Table of Contents

- [Quick Start](quick-start.md)
- [Installation](installation.md)
- [Configuration Guide](configuration.md)
- [Model Support](models/README.md)
  - [Chatterbox](models/chatterbox.md)
- [Dataset Formats](datasets.md)
- [Training Guide](training.md)
- [CLI Reference](cli.md)
- [API Reference](api/README.md)
- [Examples](examples/README.md)
- [Troubleshooting](troubleshooting.md)
- [Contributing](../CONTRIBUTING.md)

## Overview

TTSTune provides a unified, configuration-driven interface for fine-tuning various TTS models. Instead of writing custom training scripts for each model, you simply define your training setup in a YAML configuration file.

### Key Features

- **Configuration-driven**: Define everything in YAML - no code changes needed
- **Multiple model support**: Chatterbox, F5-TTS, CSM 1B, Orpheus, StyleTTS 2 (planned)
- **Flexible dataset formats**: wav+txt pairs, Hugging Face datasets, CSV/JSON metadata
- **Built-in utilities**: Automatic checkpointing, logging, wandb integration
- **Modular architecture**: Easy to extend for new models and features

### Supported Models

| Model | Status | Components |
|-------|--------|------------|
| Chatterbox | ✅ Available | T3 (text-to-tokens), S3Gen (tokens-to-speech) |
| F5-TTS | 🚧 Planned | - |
| CSM 1B | 🚧 Planned | - |
| Orpheus | 🚧 Planned | - |
| StyleTTS 2 | 🚧 Planned | - |

### Quick Example

```bash
# 1. Create a configuration file
ttstune create-config --model-type chatterbox --output my_config.yaml

# 2. Edit the configuration to point to your dataset
# Edit my_config.yaml...

# 3. Validate the configuration
ttstune validate-config --config my_config.yaml

# 4. Start training
ttstune train --config my_config.yaml
```

## Getting Help

- Check the [troubleshooting guide](troubleshooting.md) for common issues
- Look at [examples](examples/README.md) for real-world usage
- Read the [API documentation](api/README.md) for programmatic usage
- Open an issue on GitHub for bug reports or feature requests
