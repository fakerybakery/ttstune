"""Data utilities for dataset creation and collation."""

import logging
from pathlib import Path
from typing import Optional, Dict, Any, List, Union
import torch
from torch.utils.data import Dataset
import librosa
import numpy as np
import pandas as pd
import json
from datasets import load_dataset, Dataset as HFDataset
from ttstune.config import DatasetConfig, DatasetType

logger = logging.getLogger(__name__)


class BaseDataset(Dataset):
    """Base dataset class for TTS training."""

    def __init__(self, config: DatasetConfig, split: str = "train"):
        """Initialize base dataset.

        Args:
            config: Dataset configuration
            split: Dataset split (train/eval)
        """
        self.config = config
        self.split = split
        self.data: List[Dict[str, Any]] = []

        self._load_data()
        self._validate_data()

    def _load_data(self) -> None:
        """Load data based on dataset type."""
        if self.config.dataset_type == DatasetType.WAV_TXT:
            self._load_wav_txt_data()
        elif self.config.dataset_type == DatasetType.HF_DATASET:
            self._load_hf_data()
        elif self.config.dataset_type == DatasetType.METADATA_CSV:
            self._load_csv_data()
        elif self.config.dataset_type == DatasetType.METADATA_JSON:
            self._load_json_data()
        else:
            raise ValueError(f"Unsupported dataset type: {self.config.dataset_type}")

    def _load_wav_txt_data(self) -> None:
        """Load wav+txt paired files."""
        if not self.config.dataset_path:
            raise ValueError("dataset_path is required for wav_txt dataset type")

        dataset_path = Path(self.config.dataset_path)
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset path not found: {dataset_path}")

        # Find all wav files
        wav_files = list(dataset_path.rglob("*.wav"))

        for wav_file in wav_files:
            txt_file = wav_file.with_suffix(".txt")
            if txt_file.exists():
                try:
                    with open(txt_file, "r", encoding="utf-8") as f:
                        text = f.read().strip()

                    if text:  # Skip empty text files
                        self.data.append({"audio_path": str(wav_file), "text": text})
                except Exception as e:
                    logger.warning(f"Error reading {txt_file}: {e}")

        logger.info(f"Loaded {len(self.data)} samples from wav_txt dataset")

    def _load_hf_data(self) -> None:
        """Load Hugging Face dataset."""
        if not self.config.dataset_name:
            raise ValueError("dataset_name is required for hf_dataset type")

        dataset = load_dataset(
            self.config.dataset_name,
            self.config.dataset_config_name,
            verification_mode=(
                "no_checks" if self.config.ignore_verifications else "basic_checks"
            ),
        )

        # Get the appropriate split
        if self.split == "train":
            split_name = self.config.train_split_name
        else:
            split_name = self.config.eval_split_name or "validation"

        if split_name not in dataset:
            if self.split == "eval" and self.config.eval_split_size > 0:
                # Create eval split from train
                train_dataset = dataset[self.config.train_split_name]
                split_dataset = train_dataset.train_test_split(
                    test_size=self.config.eval_split_size, seed=42
                )
                hf_data = (
                    split_dataset["test"]
                    if self.split == "eval"
                    else split_dataset["train"]
                )
            else:
                raise ValueError(f"Split '{split_name}' not found in dataset")
        else:
            hf_data = dataset[split_name]

        # Convert to our format
        for item in hf_data:
            audio_data = item[self.config.audio_column_name]
            text = item[self.config.text_column_name]

            # Handle different audio formats
            if isinstance(audio_data, str):
                audio_path = audio_data
            elif isinstance(audio_data, dict) and "path" in audio_data:
                audio_path = audio_data["path"]
            else:
                # Audio array format - we'll handle this in __getitem__
                audio_path = None

            self.data.append(
                {
                    "audio_path": audio_path,
                    "audio_data": audio_data if audio_path is None else None,
                    "text": text,
                }
            )

        logger.info(f"Loaded {len(self.data)} samples from HF dataset")

    def _load_csv_data(self) -> None:
        """Load CSV metadata format."""
        if not self.config.dataset_path:
            raise ValueError("dataset_path is required for metadata_csv type")

        csv_path = Path(self.config.dataset_path)
        df = pd.read_csv(csv_path)

        # Expect columns: audio_path, text
        for _, row in df.iterrows():
            audio_path = row["audio_path"]
            text = row["text"]

            # Make path relative to CSV directory if not absolute
            if not Path(audio_path).is_absolute():
                audio_path = str(csv_path.parent / audio_path)

            self.data.append({"audio_path": audio_path, "text": text})

        logger.info(f"Loaded {len(self.data)} samples from CSV metadata")

    def _load_json_data(self) -> None:
        """Load JSON lines metadata format."""
        if not self.config.dataset_path:
            raise ValueError("dataset_path is required for metadata_json type")

        json_path = Path(self.config.dataset_path)

        with open(json_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    item = json.loads(line.strip())
                    audio_path = item["audio_path"]
                    text = item["text"]

                    # Make path relative to JSON directory if not absolute
                    if not Path(audio_path).is_absolute():
                        audio_path = str(json_path.parent / audio_path)

                    self.data.append({"audio_path": audio_path, "text": text})
                except Exception as e:
                    logger.warning(f"Error parsing JSON line: {e}")

        logger.info(f"Loaded {len(self.data)} samples from JSON metadata")

    def _validate_data(self) -> None:
        """Validate loaded data."""
        valid_data = []

        for item in self.data:
            # Validate text length
            text = item["text"]
            if self.config.min_text_length and len(text) < self.config.min_text_length:
                continue
            if self.config.max_text_length and len(text) > self.config.max_text_length:
                continue

            # Validate audio file exists (if path provided)
            if item.get("audio_path"):
                audio_path = Path(item["audio_path"])
                if not audio_path.exists():
                    logger.warning(f"Audio file not found: {audio_path}")
                    continue

            valid_data.append(item)

        logger.info(
            f"Filtered to {len(valid_data)} valid samples (removed {len(self.data) - len(valid_data)})"
        )
        self.data = valid_data

    def __len__(self) -> int:
        """Get dataset length."""
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get dataset item."""
        item = self.data[idx]

        # Load audio
        audio = self._load_audio(item)
        if audio is None:
            # Return a dummy item or raise an exception
            return None

        return {
            "audio": audio,
            "text": item["text"],
            "sample_rate": self.config.sample_rate or 16000,
        }

    def _load_audio(self, item: Dict[str, Any]) -> Optional[np.ndarray]:
        """Load audio from item."""
        try:
            if item.get("audio_path"):
                # Load from file
                audio, sr = librosa.load(
                    item["audio_path"], sr=self.config.sample_rate, mono=True
                )
            elif item.get("audio_data"):
                # From HF dataset format
                audio_data = item["audio_data"]
                if isinstance(audio_data, dict):
                    audio = np.array(audio_data["array"], dtype=np.float32)
                    original_sr = audio_data["sampling_rate"]

                    # Resample if needed
                    if (
                        self.config.sample_rate
                        and original_sr != self.config.sample_rate
                    ):
                        audio = librosa.resample(
                            audio,
                            orig_sr=original_sr,
                            target_sr=self.config.sample_rate,
                        )
                else:
                    audio = np.array(audio_data, dtype=np.float32)
            else:
                return None

            # Validate audio duration
            duration = len(audio) / (self.config.sample_rate or 16000)
            if (
                self.config.min_audio_duration_s
                and duration < self.config.min_audio_duration_s
            ):
                return None
            if (
                self.config.max_audio_duration_s
                and duration > self.config.max_audio_duration_s
            ):
                # Truncate
                max_samples = int(
                    self.config.max_audio_duration_s
                    * (self.config.sample_rate or 16000)
                )
                audio = audio[:max_samples]

            return audio.astype(np.float32)

        except Exception as e:
            logger.warning(f"Error loading audio: {e}")
            return None


class BaseDataCollator:
    """Base data collator for TTS training."""

    def __init__(self, pad_token_id: int = 0, max_length: Optional[int] = None):
        """Initialize data collator.

        Args:
            pad_token_id: Token ID for padding
            max_length: Maximum sequence length
        """
        self.pad_token_id = pad_token_id
        self.max_length = max_length

    def __call__(self, batch: List[Optional[Dict[str, Any]]]) -> Dict[str, Any]:
        """Collate batch of samples.

        Args:
            batch: List of samples

        Returns:
            Collated batch
        """
        # Filter out None samples
        valid_batch = [item for item in batch if item is not None]

        if not valid_batch:
            return {}

        # This is a base implementation - specific models should override
        return {"batch_size": len(valid_batch), "samples": valid_batch}


def create_dataset(config: DatasetConfig, split: str = "train") -> BaseDataset:
    """Create dataset based on configuration.

    Args:
        config: Dataset configuration
        split: Dataset split (train/eval)

    Returns:
        BaseDataset: Created dataset
    """
    return BaseDataset(config, split)


def create_data_collator(collator_type: str = "base", **kwargs) -> BaseDataCollator:
    """Create data collator.

    Args:
        collator_type: Type of collator to create
        **kwargs: Additional arguments for collator

    Returns:
        BaseDataCollator: Created data collator
    """
    if collator_type == "base":
        return BaseDataCollator(**kwargs)
    else:
        raise ValueError(f"Unknown collator type: {collator_type}")


def split_dataset(
    dataset: BaseDataset, eval_split_size: float = 0.1, seed: int = 42
) -> tuple[BaseDataset, BaseDataset]:
    """Split dataset into train and eval sets.

    Args:
        dataset: Dataset to split
        eval_split_size: Fraction of data for evaluation
        seed: Random seed for splitting

    Returns:
        Tuple of (train_dataset, eval_dataset)
    """
    import random

    random.seed(seed)

    indices = list(range(len(dataset)))
    random.shuffle(indices)

    split_idx = int(len(indices) * (1 - eval_split_size))
    train_indices = indices[:split_idx]
    eval_indices = indices[split_idx:]

    # Create new datasets with filtered data
    train_dataset = BaseDataset(dataset.config, "train")
    eval_dataset = BaseDataset(dataset.config, "eval")

    train_dataset.data = [dataset.data[i] for i in train_indices]
    eval_dataset.data = [dataset.data[i] for i in eval_indices]

    return train_dataset, eval_dataset
