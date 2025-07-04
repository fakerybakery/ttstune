# Chatterbox Model Guide

Chatterbox is a two-stage TTS model that consists of:
1. **T3**: Text-to-Token conversion (converts text to discrete speech tokens)
2. **S3Gen**: Speech generation (converts tokens to mel spectrograms and audio)

## Architecture Overview

```
Text → T3 → Speech Tokens → S3Gen → Mel Spectrogram → Audio
```

### Components

- **Voice Encoder (VE)**: Extracts speaker embeddings from reference audio
- **T3**: Transformer that converts text to discrete speech tokens
- **S3Gen**: Flow-based model that generates mel spectrograms from tokens
- **HiFiGAN**: Vocoder that converts mel spectrograms to audio

## Training Strategies

### 1. T3-Only Training (Recommended)

Train only the T3 component while freezing everything else:

```yaml
model:
  model_type: chatterbox
  base_model: ResembleAI/chatterbox
  freeze_components:
    - voice_encoder
    - s3gen
```

**When to use:**
- Voice cloning applications
- Adapting to new speakers
- Limited computational resources
- Most common use case

**Training time:** ~2-4 hours on single GPU

### 2. S3Gen-Only Training

Train only the S3Gen component:

```yaml
model:
  freeze_components:
    - voice_encoder
    - t3
```

**When to use:**
- Improving audio quality
- Adapting to different audio domains
- Advanced users only

**Training time:** ~6-12 hours on single GPU

### 3. Full Model Training

Train both T3 and S3Gen:

```yaml
model:
  freeze_components:
    - voice_encoder  # Always keep this frozen
```

**When to use:**
- Large, diverse datasets (>50 hours)
- Significant domain shifts
- Maximum customization

**Training time:** ~8-16 hours on single GPU

## Dataset Requirements

### Minimum Requirements

- **Audio quality**: 16kHz or higher, mono
- **Duration**: 30 minutes to 2 hours of speech
- **Speakers**: Single speaker (for voice cloning)
- **Text quality**: Clean transcriptions, proper punctuation

### Recommended Setup

- **Duration**: 2-10 hours of speech
- **Sample rate**: 22kHz or 44kHz
- **Format**: WAV files
- **Text**: Normalized, consistent style
- **Content**: Diverse sentences, various phonemes

### Audio Preprocessing

TTSTune automatically handles:
- Sample rate conversion
- Audio normalization
- Silence trimming (if configured)

## Configuration Examples

### Basic Voice Cloning

```yaml
model:
  model_type: chatterbox
  base_model: ResembleAI/chatterbox
  freeze_components: [voice_encoder, s3gen]

dataset:
  dataset_path: ./voice_data
  dataset_type: wav_txt
  max_audio_duration_s: 20.0
  min_audio_duration_s: 2.0

training:
  num_train_epochs: 10
  per_device_train_batch_size: 8
  learning_rate: 1e-4
  eval_steps: 200
  save_steps: 500
```

### High-Quality Training

```yaml
model:
  model_type: chatterbox
  base_model: ResembleAI/chatterbox
  freeze_components: [voice_encoder, s3gen]

dataset:
  dataset_path: ./voice_data
  dataset_type: wav_txt
  eval_split_size: 0.05  # Larger validation set

training:
  num_train_epochs: 20
  per_device_train_batch_size: 4
  learning_rate: 5e-5
  warmup_ratio: 0.1
  eval_steps: 100
  early_stopping_patience: 8
  save_steps: 200
```

### Memory-Efficient Training

```yaml
model:
  model_type: chatterbox
  base_model: ResembleAI/chatterbox
  freeze_components: [voice_encoder, s3gen]

training:
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 8
  gradient_checkpointing: true
  fp16: true
```

## Model-Specific Parameters

### T3 Component

Controlled via `training.custom_params`:

```yaml
training:
  custom_params:
    max_text_len: 256          # Maximum text tokens
    max_speech_len: 800        # Maximum speech tokens
    audio_prompt_duration_s: 3.0  # Reference audio length
```

### S3Gen Component

```yaml
training:
  custom_params:
    max_speech_token_len: 750    # Max speech tokens for S3Gen
    max_mel_len: 1500           # Max mel frames
    prompt_audio_duration_s: 3.0 # Prompt duration
```

## Performance Tips

### GPU Memory Usage

| Batch Size | GPU Memory | Training Speed |
|------------|------------|----------------|
| 1 | ~4GB | Slow |
| 2 | ~6GB | Medium |
| 4 | ~10GB | Fast |
| 8 | ~16GB | Very Fast |

### Training Speed Optimizations

1. **Use mixed precision:**
   ```yaml
   training:
     fp16: true  # or bf16 for newer GPUs
   ```

2. **Increase batch size:**
   ```yaml
   training:
     per_device_train_batch_size: 8
     dataloader_num_workers: 4
   ```

3. **Enable data loading optimizations:**
   ```yaml
   training:
     dataloader_pin_memory: true
   dataset:
     preprocessing_num_workers: 8
   ```

## Quality Optimization

### Text Preprocessing

Ensure your text data is:
- Consistently formatted
- Properly punctuated
- Free of special characters
- Normalized (consistent abbreviations, numbers)

### Audio Quality

- Use high-quality source audio (minimal background noise)
- Consistent recording conditions
- Proper gain levels (not clipped or too quiet)
- Single speaker per dataset

### Training Techniques

1. **Learning rate scheduling:**
   ```yaml
   training:
     learning_rate: 1e-4
     warmup_ratio: 0.1
     lr_scheduler_type: cosine
   ```

2. **Early stopping:**
   ```yaml
   training:
     early_stopping_patience: 5
     load_best_model_at_end: true
   ```

3. **Regular evaluation:**
   ```yaml
   training:
     eval_steps: 100
     metric_for_best_model: eval_loss
   ```

## Troubleshooting

### Common Issues

**Training loss not decreasing:**
- Reduce learning rate (try 5e-5 or 1e-5)
- Check data quality
- Ensure sufficient training data

**Out of memory errors:**
- Reduce batch size
- Enable gradient checkpointing
- Use gradient accumulation

**Poor audio quality:**
- Check source audio quality
- Increase training epochs
- Consider S3Gen training

**Slow convergence:**
- Increase learning rate
- Check data preprocessing
- Ensure model components aren't frozen incorrectly

### Validation Metrics

Monitor these metrics during training:

- `train_loss`: Should decrease steadily
- `eval_loss`: Should decrease and be close to train_loss
- `train_text_loss`: T3 text prediction loss
- `train_speech_loss`: T3 speech prediction loss

## Advanced Usage

### Multi-GPU Training

TTSTune automatically detects multiple GPUs:

```yaml
training:
  per_device_train_batch_size: 4  # Per GPU
  # Effective batch size = 4 * num_gpus
```

### Custom Model Paths

Load from local directory:

```yaml
model:
  local_model_dir: /path/to/chatterbox/model
  # This overrides base_model
```

### Resuming Training

```yaml
training:
  resume_from_checkpoint: ./outputs/checkpoint-1000
```

## Integration

After training, use your model with the original Chatterbox library:

```python
from chatterbox.tts import ChatterboxTTS

# Load your fine-tuned model
model = ChatterboxTTS.from_local("./outputs/final_model")

# Generate speech
audio = model.speak(
    text="Hello, this is my fine-tuned voice!",
    voice_id="custom"
)
```
