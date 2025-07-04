# Quick Start Guide

This guide will get you up and running with TTSTune in 10 minutes.

## Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended)
- At least 8GB of GPU memory for most models

## Installation

```bash
# Install TTSTune
pip install ttstune

# Or install from source
git clone https://github.com/your-org/ttstune.git
cd ttstune
pip install -e .
```

## Prepare Your Dataset

TTSTune supports several dataset formats. The simplest is the `wav_txt` format:

```
dataset/
├── 1.wav
├── 1.txt
├── 2.wav
├── 2.txt
└── ...
```

Each `.wav` file should have a corresponding `.txt` file with the transcription.

### Example Dataset Structure

```
my_voice_dataset/
├── audio_001.wav    # "Hello, this is a test recording."
├── audio_001.txt
├── audio_002.wav    # "The weather is nice today."
├── audio_002.txt
└── ...
```

## Create Configuration

Create a configuration file for your training:

```bash
ttstune create-config --model-type chatterbox --output config.yaml
```

This creates a `config.yaml` file with example settings:

```yaml
model:
  model_type: chatterbox
  base_model: ResembleAI/chatterbox
  freeze_components:
    - voice_encoder
    - s3gen

dataset:
  dataset_path: ./dataset
  dataset_type: wav_txt
  eval_split_size: 0.01

training:
  output_dir: ./outputs
  num_train_epochs: 10
  per_device_train_batch_size: 4
  learning_rate: 1.0e-4
  save_steps: 1000
  eval_steps: 500
  early_stopping_patience: 5

wandb:
  enabled: true
  project: ttstune-training
  tags:
    - chatterbox
    - finetuning
```

## Edit Configuration

Update the configuration to match your setup:

1. **Dataset path**: Update `dataset.dataset_path` to point to your dataset directory
2. **Output directory**: Change `training.output_dir` to where you want outputs saved
3. **Batch size**: Adjust `per_device_train_batch_size` based on your GPU memory
4. **Learning rate**: Fine-tune the learning rate if needed
5. **Wandb**: Set up weights & biases logging (optional but recommended)

## Validate Configuration

Check that your configuration is valid:

```bash
ttstune validate-config --config config.yaml
```

This will verify:
- Configuration file syntax
- Dataset path exists
- Model parameters are valid
- Required dependencies are available

## Start Training

Begin training your model:

```bash
ttstune train --config config.yaml
```

### Training Output

You'll see output like:

```
Loading configuration from config.yaml
Model: chatterbox
Base model: ResembleAI/chatterbox
Dataset: wav_txt
Output directory: ./outputs
Using device: cuda
Found 1500 samples from wav_txt dataset
Filtered to 1485 valid samples (removed 15)
Training components: T3=True, S3Gen=False
Starting T3 training...
Training completed successfully!
Final Chatterbox model saved to ./outputs/final_model
```

## Monitor Training

### Using Weights & Biases

If you enabled wandb in your config, you can monitor training at [wandb.ai](https://wandb.ai).

### Local Logs

Training logs are saved to:
- `./outputs/training.log` - Detailed training logs
- `./outputs/t3/` - T3 component outputs
- `./outputs/s3gen/` - S3Gen component outputs (if enabled)
- `./outputs/final_model/` - Final combined model

## Test Your Model

After training, your model will be saved in the `final_model` directory. You can use it with the original Chatterbox code:

```python
from chatterbox.tts import ChatterboxTTS

# Load your fine-tuned model
model = ChatterboxTTS.from_local("./outputs/final_model")

# Generate speech
audio = model.speak("Hello! This is my fine-tuned voice.")
```

## Next Steps

- Read the [Configuration Guide](configuration.md) for detailed options
- Check out [Training Guide](training.md) for advanced techniques
- See [Examples](examples/README.md) for real-world use cases
- Learn about [Dataset Formats](datasets.md) for different data types

## Common Issues

### Out of Memory

Reduce batch size in your config:

```yaml
training:
  per_device_train_batch_size: 2  # Reduce from 4
  gradient_accumulation_steps: 2   # Compensate with accumulation
```

### Slow Training

Enable mixed precision:

```yaml
training:
  fp16: true  # or bf16: true for newer GPUs
```

### Dataset Not Found

Check your dataset path and structure:

```bash
ls -la your_dataset_path/
# Should show .wav and .txt files
```

For more help, see the [Troubleshooting Guide](troubleshooting.md).
