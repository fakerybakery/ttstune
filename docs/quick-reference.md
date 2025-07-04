# TTSTune Quick Reference

This guide provides quick copy-paste configurations for common training scenarios.

## Training Commands

```bash
# Basic training
ttstune train --config config.yaml

# Training with validation
ttstune validate-config --config config.yaml && ttstune train --config config.yaml

# Resume training
ttstune train --config config.yaml --resume-from-checkpoint ./outputs/checkpoint-1000

# Multi-GPU training
CUDA_VISIBLE_DEVICES=0,1,2,3 ttstune train --config config.yaml

# Training with verbose output
ttstune train --config config.yaml --verbose
```

## Common Configurations

### 1. Voice Cloning (Small Dataset)

```yaml
model:
  model_type: chatterbox
  base_model: ResembleAI/chatterbox
  freeze_components: [voice_encoder]

dataset:
  dataset_type: wav_txt
  dataset_path: ./my_voice_data
  eval_split_size: 0.1

training:
  output_dir: ./outputs/voice_clone
  num_train_epochs: 20
  per_device_train_batch_size: 8
  learning_rate: 1e-4
  fp16: true
```

### 2. Language Adaptation (Large Dataset)

```yaml
model:
  model_type: chatterbox
  base_model: ResembleAI/chatterbox
  freeze_components: []

dataset:
  dataset_type: hf_dataset
  dataset_name: MrDragonFox/DE_Emilia_Yodas_680h
  text_column_name: text_scribe
  eval_split_size: 0.001

training:
  output_dir: ./outputs/german_adaptation
  num_train_epochs: 1
  per_device_train_batch_size: 4
  gradient_accumulation_steps: 2
  learning_rate: 5e-5
  fp16: true
```

### 3. Memory-Efficient Training

```yaml
training:
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 16
  gradient_checkpointing: true
  fp16: true
  dataloader_num_workers: 4
```

### 4. Fast Prototyping

```yaml
dataset:
  eval_split_size: 0.1  # Larger eval set for quick feedback

training:
  num_train_epochs: 5
  eval_steps: 100       # Frequent evaluation
  save_steps: 200       # Frequent saving
  logging_steps: 10     # Verbose logging
  eval_on_start: true   # Baseline evaluation
```

## Dataset Formats

### Hugging Face Datasets

```yaml
dataset:
  dataset_type: hf_dataset
  dataset_name: mozilla-foundation/common_voice_11_0
  dataset_config_name: en
  text_column_name: sentence
  audio_column_name: audio
```

### Local wav+txt Files

```yaml
dataset:
  dataset_type: wav_txt
  dataset_path: ./data
  # Expects: 001.wav, 001.txt, 002.wav, 002.txt, ...
```

### CSV Metadata

```yaml
dataset:
  dataset_type: metadata_csv
  dataset_path: ./metadata.csv
  # CSV with columns: audio_path, text
```

### JSON Lines

```yaml
dataset:
  dataset_type: metadata_json
  dataset_path: ./metadata.jsonl
  # Each line: {"audio_path": "...", "text": "..."}
```

## Component Freezing Strategies

### Voice Cloning Only
```yaml
freeze_components: [voice_encoder]  # Train T3 and S3Gen
```

### Language Adaptation Only
```yaml
freeze_components: [voice_encoder, s3gen]  # Train T3 only
```

### Speaker Adaptation Only
```yaml
freeze_components: [t3]  # Train S3Gen and voice_encoder
```

### Full Fine-tuning
```yaml
freeze_components: []  # Train all components
```

## Performance Optimization

### For Large Datasets
```yaml
training:
  per_device_train_batch_size: 16
  gradient_accumulation_steps: 1
  dataloader_num_workers: 16
  dataloader_pin_memory: true
  fp16: true
```

### For Limited Memory
```yaml
training:
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 32
  gradient_checkpointing: true
  fp16: true
  dataloader_num_workers: 2
```

### For Multi-GPU
```yaml
training:
  per_device_train_batch_size: 8  # Per GPU
  # Automatically uses all available GPUs
```

## Monitoring and Logging

### Tensorboard Only
```yaml
training:
  report_to: [tensorboard]
  logging_steps: 10

# View logs: tensorboard --logdir ./outputs/logs
```

### Weights & Biases
```yaml
wandb:
  enabled: true
  project: my-tts-project
  name: experiment-1
  tags: [chatterbox, german]

training:
  report_to: [wandb]
```

### Both Tensorboard and Wandb
```yaml
wandb:
  enabled: true
  project: my-tts-project

training:
  report_to: [tensorboard, wandb]
```

## Advanced Features

### Component-Specific Learning Rates
```yaml
advanced:
  component_learning_rates:
    t3: 1e-4
    s3gen: 2e-4
    voice_encoder: 5e-5
```

### Progressive Training
```yaml
advanced:
  progressive_training: true
  # Trains components sequentially
```

### Data Augmentation
```yaml
advanced:
  data_augmentation:
    enabled: true
    speed_perturbation: [0.9, 1.0, 1.1]
    volume_perturbation: [0.8, 1.0, 1.2]
```

## Troubleshooting

### Out of Memory
```yaml
training:
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 8
  gradient_checkpointing: true
  fp16: true
```

### Training Too Slow
```yaml
training:
  dataloader_num_workers: 16
  dataloader_pin_memory: true
  fp16: true
  per_device_train_batch_size: 16  # Increase if memory allows
```

### Poor Quality Results
```yaml
training:
  learning_rate: 1e-5  # Lower learning rate
  num_train_epochs: 20  # More epochs
  warmup_steps: 1000    # Longer warmup

model:
  freeze_components: []  # Train all components
```

### Training Instability
```yaml
training:
  learning_rate: 5e-6
  gradient_clip_norm: 1.0
  warmup_steps: 2000
```

## Example Training Sessions

### German Voice Cloning
```bash
# 1. Use provided config
cp config/chatterbox_german_yodas.yaml my_config.yaml

# 2. Start training
ttstune train --config my_config.yaml

# 3. Monitor progress
tensorboard --logdir ./checkpoints/chatterbox_finetuned_yodas/logs
```

### Quick Voice Clone
```bash
# 1. Create basic config
ttstune create-config --model-type chatterbox --output quick_clone.yaml

# 2. Edit for your data
# dataset:
#   dataset_path: ./my_voice_data
#   dataset_type: wav_txt

# 3. Train
ttstune train --config quick_clone.yaml
```

### Multi-language Training
```bash
# 1. Use multilingual config
cp examples/chatterbox_multilingual.yaml multi_config.yaml

# 2. Edit dataset settings for your language
# 3. Train
ttstune train --config multi_config.yaml
```
