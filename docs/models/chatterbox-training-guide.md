# Chatterbox Training Guide

This guide provides comprehensive instructions for training Chatterbox models using TTSTune, including component explanations, training recipes, and best practices for different languages.

## Table of Contents

1. [Understanding Chatterbox Architecture](#understanding-chatterbox-architecture)
2. [Training Recipes](#training-recipes)
3. [Multilingual Training](#multilingual-training)
4. [Component-Specific Training](#component-specific-training)
5. [Troubleshooting](#troubleshooting)

## Understanding Chatterbox Architecture

Chatterbox is a multi-component TTS system consisting of two main parts:

### 1. T3 (Text-to-Tokens) Model
- **Purpose**: Converts text input into intermediate token representations
- **Training**: Learns text-to-speech alignment and phonetic representations
- **When to train**:
  - New languages with different phonetic systems
  - Domain-specific vocabulary (technical terms, names)
  - Accent adaptation

### 2. S3Gen (Speech Generation) Model
- **Purpose**: Converts tokens into mel-spectrograms and then to audio
- **Training**: Learns voice characteristics and audio generation
- **When to train**:
  - Voice cloning and adaptation
  - Audio quality improvements
  - Style transfer

### 3. Voice Encoder
- **Purpose**: Extracts speaker embeddings from reference audio
- **Training**: Usually frozen, but can be fine-tuned for new speaker types
- **When to train**:
  - Completely new speaker types
  - Cross-lingual speaker adaptation

## Training Recipes

### Recipe 1: Basic Voice Cloning (Single Speaker)

Perfect for cloning a specific voice with 1-10 hours of data.

```yaml
# config/voice_clone_basic.yaml
model:
  model_type: chatterbox
  base_model: ResembleAI/chatterbox
  freeze_components: [voice_encoder]  # Keep voice encoder frozen

dataset:
  dataset_type: wav_txt
  dataset_path: ./my_voice_data
  eval_split_size: 0.1

training:
  num_train_epochs: 20
  per_device_train_batch_size: 8
  learning_rate: 1e-4
  warmup_steps: 500
  fp16: true
```

**Expected results**: High-quality voice cloning with 5-10 hours of clean data.

### Recipe 2: Language Adaptation (German Example)

For adapting to a new language with large datasets.

```yaml
# config/chatterbox_german_yodas.yaml (see full file in config/)
model:
  model_type: chatterbox
  base_model: ResembleAI/chatterbox
  freeze_components: []  # Train all components

dataset:
  dataset_type: hf_dataset
  dataset_name: MrDragonFox/DE_Emilia_Yodas_680h
  text_column_name: text_scribe
  eval_split_size: 0.0002

training:
  num_train_epochs: 1  # Large dataset, fewer epochs needed
  per_device_train_batch_size: 4
  gradient_accumulation_steps: 2
  learning_rate: 5e-5
  fp16: true
```

**Expected results**: Native-quality German TTS with proper pronunciation.

### Recipe 3: Multi-Speaker Adaptation

For training on datasets with multiple speakers.

```yaml
# config/multi_speaker.yaml
model:
  model_type: chatterbox
  base_model: ResembleAI/chatterbox
  freeze_components: [t3]  # Keep text processing, adapt voice generation

dataset:
  dataset_type: hf_dataset
  dataset_name: mozilla-foundation/common_voice_11_0
  dataset_config_name: en
  eval_split_size: 0.01

training:
  num_train_epochs: 5
  per_device_train_batch_size: 16
  learning_rate: 2e-4
  warmup_steps: 1000

advanced:
  component_learning_rates:
    t3: 1e-5        # Lower LR for text model
    s3gen: 2e-4     # Higher LR for voice model
    voice_encoder: 5e-5  # Moderate LR for speaker adaptation
```

### Recipe 4: Domain-Specific Fine-tuning

For specific domains like audiobooks, news, or technical content.

```yaml
# config/domain_specific.yaml
model:
  model_type: chatterbox
  base_model: ResembleAI/chatterbox
  freeze_components: [voice_encoder, s3gen]  # Only adapt text processing

dataset:
  dataset_type: metadata_csv
  dataset_path: ./technical_docs.csv
  eval_split_size: 0.05

training:
  num_train_epochs: 10
  per_device_train_batch_size: 12
  learning_rate: 1e-4

advanced:
  component_learning_rates:
    t3: 1e-4  # Focus on text understanding
```

## Multilingual Training

### Supported Languages

Chatterbox works well with:
- **High-resource**: English, German, French, Spanish, Italian
- **Medium-resource**: Dutch, Portuguese, Russian, Japanese
- **Low-resource**: May require additional preprocessing

### Language-Specific Considerations

#### German (Yodas Dataset Example)
```bash
# 1. Create configuration
ttstune create-config --model-type chatterbox --output german_config.yaml

# 2. Edit configuration for German dataset
# Use the provided config/chatterbox_german_yodas.yaml as reference

# 3. Start training
ttstune train --config german_config.yaml
```

**German-specific settings**:
- Use `text_scribe` column for transcriptions
- Sample rate: 22050 Hz (standard for Yodas)
- Longer training due to complex phonetics

#### Japanese
```yaml
dataset:
  text_column_name: transcript
  # Japanese often needs special tokenization
  preprocessing:
    normalize_text: true
    remove_punctuation: false  # Keep for proper intonation
```

#### Low-Resource Languages
```yaml
training:
  num_train_epochs: 50  # More epochs for limited data
  learning_rate: 5e-5   # Lower learning rate
  warmup_steps: 200     # Longer warmup

model:
  freeze_components: [voice_encoder]  # Freeze more components
```

## Component-Specific Training

### When to Train Each Component

| Component | Train When | Freeze When |
|-----------|------------|-------------|
| **T3** | New language, domain adaptation | Voice cloning only |
| **S3Gen** | Voice cloning, quality improvement | Language adaptation only |
| **Voice Encoder** | New speaker types | Most cases |

### Progressive Training Strategy

For best results, train components sequentially:

```yaml
# Phase 1: T3 only
model:
  freeze_components: [s3gen, voice_encoder]
training:
  num_train_epochs: 5
  learning_rate: 1e-4

# Phase 2: S3Gen only
model:
  freeze_components: [t3, voice_encoder]
training:
  num_train_epochs: 10
  learning_rate: 2e-4

# Phase 3: Fine-tune all
model:
  freeze_components: []
training:
  num_train_epochs: 5
  learning_rate: 5e-5
```

## Training Commands

### Basic Training
```bash
# Train with German Yodas dataset
ttstune train --config config/chatterbox_german_yodas.yaml

# Monitor training with tensorboard
tensorboard --logdir ./checkpoints/chatterbox_finetuned_yodas/logs
```

### Advanced Training
```bash
# Multi-GPU training
CUDA_VISIBLE_DEVICES=0,1,2,3 ttstune train --config config.yaml

# Resume from checkpoint
ttstune train --config config.yaml --resume-from-checkpoint ./checkpoints/checkpoint-4000

# Evaluate specific checkpoint
ttstune evaluate --config config.yaml --checkpoint ./checkpoints/checkpoint-8000
```

## Troubleshooting

### Common Issues

#### 1. Out of Memory
```yaml
training:
  per_device_train_batch_size: 1  # Reduce batch size
  gradient_accumulation_steps: 8  # Maintain effective batch size
  gradient_checkpointing: true    # Enable gradient checkpointing
  fp16: true                      # Use mixed precision
```

#### 2. Poor Audio Quality
- Check sample rate matches dataset
- Increase `per_device_train_batch_size` if possible
- Try unfreezing more components
- Increase training epochs

#### 3. Training Instability
```yaml
training:
  learning_rate: 1e-5  # Lower learning rate
  warmup_steps: 1000   # Longer warmup
  gradient_clip_norm: 1.0  # Add gradient clipping
```

#### 4. Slow Training
```yaml
training:
  dataloader_num_workers: 16  # Increase workers
  dataloader_pin_memory: true # Enable pin memory
  fp16: true                  # Use mixed precision
```

### Monitoring Training

#### Key Metrics to Watch
- **Loss**: Should decrease steadily
- **Perplexity**: Lower is better for T3
- **MCD (Mel Cepstral Distortion)**: Lower is better for S3Gen
- **Audio samples**: Listen to generated samples

#### Tensorboard Logs
```bash
tensorboard --logdir ./checkpoints/*/logs
```

#### Wandb Integration
```yaml
wandb:
  enabled: true
  project: chatterbox-training
  tags: [german, voice-cloning]
```

## Best Practices

### Data Preparation
1. **Clean audio**: Remove background noise, normalize volume
2. **Accurate transcripts**: Ensure text matches audio exactly
3. **Consistent format**: Same sample rate, file format
4. **Balanced dataset**: Mix of sentence lengths and content

### Training Strategy
1. **Start simple**: Begin with frozen components
2. **Monitor closely**: Watch for overfitting
3. **Save frequently**: Use reasonable `save_steps`
4. **Evaluate regularly**: Check audio quality throughout training

### Resource Management
1. **Batch size**: Largest that fits in memory
2. **Gradient accumulation**: Maintain effective batch size
3. **Mixed precision**: Always use `fp16: true`
4. **Checkpointing**: Balance frequency vs. storage

## Example Training Session

Here's a complete example of training a German voice model:

```bash
# 1. Setup
mkdir -p ./checkpoints/german_training
cd ttstune-v2

# 2. Create configuration
cp config/chatterbox_german_yodas.yaml my_german_config.yaml

# 3. Validate configuration
ttstune validate-config --config my_german_config.yaml

# 4. Start training
ttstune train --config my_german_config.yaml --verbose

# 5. Monitor training
tensorboard --logdir ./checkpoints/chatterbox_finetuned_yodas/logs

# 6. Evaluate final model
ttstune evaluate --config my_german_config.yaml --checkpoint ./checkpoints/chatterbox_finetuned_yodas/final_model
```

This should give you a high-quality German TTS model ready for inference!
