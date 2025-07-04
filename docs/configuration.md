# Configuration Guide

TTSTune uses YAML configuration files to define all aspects of model training. This guide covers all available configuration options.

## Configuration Structure

A TTSTune configuration file has four main sections:

```yaml
model:      # Model settings
  # ...
dataset:    # Dataset settings
  # ...
training:   # Training parameters
  # ...
wandb:      # Weights & Biases logging (optional)
  # ...
```

## Complete Example

Here's a comprehensive configuration example:

```yaml
# Global settings
seed: 42
device: null  # Auto-detect (cuda/mps/cpu)

# Model configuration
model:
  model_type: chatterbox
  base_model: ResembleAI/chatterbox
  local_model_dir: null
  cache_dir: null
  freeze_components:
    - voice_encoder
    - s3gen
  model_params: {}

# Dataset configuration
dataset:
  dataset_path: ./my_dataset
  dataset_type: wav_txt
  dataset_name: null
  dataset_config_name: null
  train_split_name: train
  eval_split_name: validation
  text_column_name: text
  audio_column_name: audio
  eval_split_size: 0.01
  preprocessing_num_workers: 4
  ignore_verifications: false

  # Audio processing
  max_audio_duration_s: 30.0
  min_audio_duration_s: 1.0
  sample_rate: 16000

  # Text processing
  max_text_length: 200
  min_text_length: 5

# Training configuration
training:
  output_dir: ./outputs
  num_train_epochs: 10
  per_device_train_batch_size: 4
  per_device_eval_batch_size: 8
  gradient_accumulation_steps: 1
  learning_rate: 1.0e-4
  weight_decay: 0.01
  warmup_steps: 0
  warmup_ratio: 0.1
  logging_steps: 10
  eval_steps: 500
  save_steps: 1000
  save_total_limit: 3
  eval_strategy: steps
  save_strategy: steps
  load_best_model_at_end: true
  metric_for_best_model: eval_loss
  greater_is_better: false
  early_stopping_patience: 5

  # Advanced options
  fp16: false
  bf16: false
  gradient_checkpointing: false
  dataloader_num_workers: 0
  dataloader_pin_memory: true
  remove_unused_columns: false
  optim: adamw_torch
  lr_scheduler_type: linear
  resume_from_checkpoint: null

  # Custom parameters
  custom_params: {}

# Weights & Biases configuration
wandb:
  enabled: true
  project: ttstune-training
  entity: your-wandb-team
  name: chatterbox-finetune-experiment
  tags:
    - chatterbox
    - finetuning
    - experiment
  notes: "Fine-tuning Chatterbox on custom voice dataset"
  config: {}
```

## Model Configuration

### Basic Settings

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_type` | str | **required** | Type of model to train (`chatterbox`, `f5_tts`, etc.) |
| `base_model` | str | **required** | Base model name or Hugging Face model ID |
| `local_model_dir` | str | `null` | Path to local model directory (overrides `base_model`) |
| `cache_dir` | str | `null` | Directory to cache downloaded models |

### Component Freezing

The `freeze_components` list controls which parts of the model to freeze during training:

**For Chatterbox:**
- `voice_encoder` - Freeze the voice encoder (speaker embedding model)
- `s3gen` - Freeze the S3Gen component (speech generation)
- `t3` - Freeze the T3 component (text-to-tokens)
- `speaker_encoder` - Freeze speaker encoder within S3Gen
- `s3_tokenizer` - Freeze S3 tokenizer

**Common combinations:**
```yaml
# Train only T3 (text-to-tokens)
freeze_components: [voice_encoder, s3gen]

# Train only S3Gen (tokens-to-speech)
freeze_components: [voice_encoder, t3]

# Train everything (not recommended for small datasets)
freeze_components: []
```

## Dataset Configuration

### Dataset Types

#### 1. wav_txt (Paired Files)
```yaml
dataset:
  dataset_type: wav_txt
  dataset_path: ./my_dataset  # Directory with .wav/.txt pairs
```

#### 2. hf_dataset (Hugging Face)
```yaml
dataset:
  dataset_type: hf_dataset
  dataset_name: mozilla-foundation/common_voice_11_0
  dataset_config_name: en
  text_column_name: sentence
  audio_column_name: audio
```

#### 3. metadata_csv (CSV File)
```yaml
dataset:
  dataset_type: metadata_csv
  dataset_path: ./metadata.csv  # CSV with audio_path,text columns
```

#### 4. metadata_json (JSON Lines)
```yaml
dataset:
  dataset_type: metadata_json
  dataset_path: ./metadata.jsonl  # One JSON object per line
```

### Audio Processing

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_audio_duration_s` | float | `null` | Maximum audio length in seconds |
| `min_audio_duration_s` | float | `null` | Minimum audio length in seconds |
| `sample_rate` | int | `null` | Target sample rate (auto-detect if null) |

### Text Processing

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_text_length` | int | `null` | Maximum text length in characters |
| `min_text_length` | int | `null` | Minimum text length in characters |

## Training Configuration

### Basic Training

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `output_dir` | str | **required** | Directory to save outputs |
| `num_train_epochs` | int | `3` | Number of training epochs |
| `per_device_train_batch_size` | int | `8` | Training batch size per device |
| `per_device_eval_batch_size` | int | `8` | Evaluation batch size per device |
| `learning_rate` | float | `5e-5` | Learning rate |
| `weight_decay` | float | `0.01` | Weight decay for regularization |

### Learning Rate Scheduling

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `warmup_steps` | int | `0` | Number of warmup steps |
| `warmup_ratio` | float | `0.0` | Warmup ratio (alternative to steps) |
| `lr_scheduler_type` | str | `linear` | LR scheduler (`linear`, `cosine`, etc.) |

### Evaluation and Checkpointing

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `eval_strategy` | str | `steps` | When to evaluate (`steps`, `epoch`, `no`) |
| `eval_steps` | int | `null` | Evaluate every N steps |
| `save_strategy` | str | `steps` | When to save (`steps`, `epoch`) |
| `save_steps` | int | `500` | Save every N steps |
| `save_total_limit` | int | `3` | Maximum checkpoints to keep |
| `load_best_model_at_end` | bool | `true` | Load best model after training |
| `metric_for_best_model` | str | `eval_loss` | Metric to determine best model |
| `early_stopping_patience` | int | `null` | Early stopping patience (null = disabled) |

### Mixed Precision Training

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `fp16` | bool | `false` | Enable FP16 mixed precision |
| `bf16` | bool | `false` | Enable BF16 mixed precision (RTX 30xx+) |
| `gradient_checkpointing` | bool | `false` | Enable gradient checkpointing (saves memory) |

### Data Loading

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dataloader_num_workers` | int | `0` | Number of data loading workers |
| `dataloader_pin_memory` | bool | `true` | Pin memory for faster GPU transfer |
| `preprocessing_num_workers` | int | `null` | Workers for dataset preprocessing |

## Wandb Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enabled` | bool | `false` | Enable Weights & Biases logging |
| `project` | str | `null` | W&B project name |
| `entity` | str | `null` | W&B team/username |
| `name` | str | `null` | Run name (auto-generated if null) |
| `tags` | list | `[]` | List of tags for the run |
| `notes` | str | `null` | Notes about the run |

## Performance Tuning

### Memory Optimization

For limited GPU memory:
```yaml
training:
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 8  # Effective batch size = 8
  gradient_checkpointing: true
  fp16: true
```

### Speed Optimization

For faster training:
```yaml
training:
  dataloader_num_workers: 4
  dataloader_pin_memory: true
  bf16: true  # If supported
dataset:
  preprocessing_num_workers: 8
```

### Quality Optimization

For best quality:
```yaml
training:
  learning_rate: 1e-5  # Lower learning rate
  num_train_epochs: 20  # More epochs
  eval_steps: 100      # More frequent evaluation
  early_stopping_patience: 10
  warmup_ratio: 0.1    # Gradual warmup
```

## Environment Variables

You can override config values with environment variables:

```bash
export TTSTUNE_OUTPUT_DIR=/path/to/outputs
export TTSTUNE_BATCH_SIZE=2
export TTSTUNE_LEARNING_RATE=1e-5

ttstune train --config config.yaml
```

## Validation

Always validate your configuration before training:

```bash
ttstune validate-config --config config.yaml
```

This checks:
- YAML syntax
- Required parameters
- Parameter value ranges
- Dataset path existence
- Model compatibility
