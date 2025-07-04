# TTSTune

**Configuration-driven framework for fine-tuning Text-to-Speech (TTS) models.**

TTSTune provides a unified, YAML-based interface for fine-tuning various TTS models without writing custom training code. Simply define your training setup in a configuration file and let TTSTune handle the rest.

## 🚀 Quick Start

```bash
# 1. Install TTSTune
pip install ttstune

# 2. Create a configuration file
ttstune create-config --model-type chatterbox --output config.yaml

# 3. Edit config.yaml to point to your dataset
# 4. Start training
ttstune train --config config.yaml
```

## ✨ Features

- **Configuration-driven**: Define everything in YAML - no code changes needed
- **Multiple model support**: Chatterbox (✅), F5-TTS, CSM 1B, Orpheus, StyleTTS 2 (planned)
- **Flexible dataset formats**: wav+txt pairs, Hugging Face datasets, CSV/JSON metadata
- **Built-in utilities**: Automatic checkpointing, logging, wandb integration
- **Modular architecture**: Easy to extend for new models and features
- **Production-ready**: Memory optimization, multi-GPU support, mixed precision

## 🏗️ Architecture

TTSTune uses an abstract `TTSTuneTrainer` base class that specific model trainers inherit from. The framework handles:

- **Configuration management**: YAML-based config with validation
- **Dataset loading**: Unified interface for different data formats
- **Training orchestration**: Logging, checkpointing, evaluation
- **Multi-component training**: For models with multiple trainable parts
- **Utilities**: Device management, wandb integration, checkpoint management

## 📖 Documentation

Comprehensive documentation is available in the [docs/](docs/) folder:

- [Quick Start Guide](docs/quick-start.md) - Get up and running in 10 minutes
- [Quick Reference](docs/quick-reference.md) - Copy-paste configs for common scenarios
- [Configuration Guide](docs/configuration.md) - Complete config reference
- [Chatterbox Training Guide](docs/models/chatterbox-training-guide.md) - Comprehensive training guide
- [Model-Specific Guides](docs/models/) - Detailed guides for each model
- [Dataset Formats](docs/datasets.md) - Supported data formats
- [CLI Reference](docs/cli.md) - Command-line interface
- [Examples](examples/) - Real-world configuration examples
- [Config Templates](config/) - Ready-to-use configuration templates

## 🎯 Supported Models

| Model | Status | Components | Use Cases |
|-------|--------|------------|-----------|
| **Chatterbox** | ✅ Available | T3 + S3Gen | Voice cloning, adaptation |
| F5-TTS | 🚧 Planned | - | Fast, efficient TTS |
| CSM 1B | 🚧 Planned | - | Large-scale TTS |
| Orpheus | 🚧 Planned | - | High-quality synthesis |
| StyleTTS 2 | 🚧 Planned | - | Style transfer |

## 📊 Dataset Formats

TTSTune supports multiple dataset formats out of the box:

### 1. wav_txt (Paired Files)
```
dataset/
├── audio_001.wav
├── audio_001.txt
├── audio_002.wav
├── audio_002.txt
└── ...
```

### 2. Hugging Face Datasets
```yaml
dataset:
  dataset_type: hf_dataset
  dataset_name: mozilla-foundation/common_voice_11_0
  dataset_config_name: en
```

### 3. CSV Metadata
```yaml
dataset:
  dataset_type: metadata_csv
  dataset_path: ./metadata.csv  # audio_path,text columns
```

### 4. JSON Lines
```yaml
dataset:
  dataset_type: metadata_json
  dataset_path: ./metadata.jsonl
```

## ⚡ Example Configurations

### Basic Voice Cloning (Chatterbox)

```yaml
model:
  model_type: chatterbox
  base_model: ResembleAI/chatterbox
  freeze_components: [voice_encoder, s3gen]

dataset:
  dataset_path: ./my_voice_data
  dataset_type: wav_txt
  eval_split_size: 0.01

training:
  output_dir: ./outputs
  num_train_epochs: 10
  per_device_train_batch_size: 4
  learning_rate: 1e-4
  fp16: true

wandb:
  enabled: true
  project: my-voice-clone
```

### German Language Training (Yodas Dataset)

```yaml
model:
  model_type: chatterbox
  base_model: ResembleAI/chatterbox
  freeze_components: []

dataset:
  dataset_type: hf_dataset
  dataset_name: MrDragonFox/DE_Emilia_Yodas_680h
  text_column_name: text_scribe
  eval_split_size: 0.0002

training:
  output_dir: ./checkpoints/chatterbox_finetuned_yodas
  num_train_epochs: 1
  per_device_train_batch_size: 4
  gradient_accumulation_steps: 2
  learning_rate: 5e-5
  fp16: true
```

### Memory-Efficient Training

```yaml
training:
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 8
  gradient_checkpointing: true
  fp16: true
```

### Multi-GPU Training

```yaml
training:
  per_device_train_batch_size: 4  # Per GPU
  # Automatically uses all available GPUs
```

## 🛠️ CLI Commands

```bash
# Create example configuration
ttstune create-config --model-type chatterbox

# Validate configuration
ttstune validate-config --config config.yaml

# Start training
ttstune train --config config.yaml --verbose

# Evaluate model
ttstune evaluate --config config.yaml --checkpoint ./outputs/checkpoint-1000

# Get help
ttstune --help
```

## 🔧 Installation

### From PyPI
```bash
pip install ttstune
```

### From Source
```bash
git clone https://github.com/fakerybakery/ttstune.git
cd ttstune
pip install -e .
```

### Development Setup
```bash
pip install uv
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
pre-commit install
```

## 📝 Example Usage

### Programmatic API

```python
from ttstune import TTSTuneConfig, ChatterboxTrainer

# Load configuration
config = TTSTuneConfig.from_yaml("config.yaml")

# Create and run trainer
with ChatterboxTrainer(config) as trainer:
    results = trainer.train()
    eval_results = trainer.evaluate()
```

### Using Trained Models

```python
from chatterbox.tts import ChatterboxTTS

# Load your fine-tuned model
model = ChatterboxTTS.from_local("./outputs/final_model")

# Generate speech
audio = model.speak("Hello from my fine-tuned voice!")
```

## 🤝 Contributing

We welcome contributions! See our [Contributing Guide](docs/contributing.md) for details.

### Adding New Models

1. Create a new trainer class inheriting from `TTSTuneTrainer`
2. Implement the required abstract methods
3. Add model type to `config.py`
4. Update CLI to support the new model
5. Add documentation and examples

## 📄 License

This project is dual-licensed under the MIT and Apache 2.0 licenses. See the [LICENSE.MIT](LICENSE.MIT) and [LICENSE.APACHE](LICENSE.APACHE) files for details.

## 🙏 Acknowledgments

TTSTune is built on and wraps the work of many open-source projects:

* [chatterbox-finetuning (@stlohrey)](https://github.com/stlohrey/chatterbox-finetuning)
* [F5-TTS](https://github.com/SWivid/F5-TTS)
* [Transformers](https://github.com/huggingface/transformers)
* [Datasets](https://github.com/huggingface/datasets)

Without these projects, TTSTune would not be possible.
