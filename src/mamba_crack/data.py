from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset


@dataclass(frozen=True)
class Split:
    train: list[Path]
    val: list[Path]
    test: list[Path]


def split_images(root: Path, seed: int, train_fraction: float, val_fraction: float) -> Split:
    images = sorted((root / "images").glob("*"))
    images = [path for path in images if path.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    if not images:
        raise FileNotFoundError(f"No images found in {root / 'images'}")
    generator = random.Random(seed)
    generator.shuffle(images)
    train_end = int(len(images) * train_fraction)
    val_end = train_end + int(len(images) * val_fraction)
    return Split(images[:train_end], images[train_end:val_end], images[val_end:])


def compute_stats(paths: list[Path], image_size: int) -> tuple[list[float], list[float]]:
    channel_sum = np.zeros(3, dtype=np.float64)
    channel_squared_sum = np.zeros(3, dtype=np.float64)
    pixel_count = 0
    for path in paths:
        image = Image.open(path).convert("RGB").resize((image_size, image_size), Image.Resampling.BILINEAR)
        values = np.asarray(image, dtype=np.float32) / 255.0
        channel_sum += values.sum(axis=(0, 1))
        channel_squared_sum += (values**2).sum(axis=(0, 1))
        pixel_count += values.shape[0] * values.shape[1]
    mean = channel_sum / pixel_count
    variance = np.maximum(channel_squared_sum / pixel_count - mean**2, 1e-8)
    return mean.tolist(), np.sqrt(variance).tolist()


class CrackDataset(Dataset):
    def __init__(
        self,
        image_paths: list[Path],
        root: Path,
        image_size: int,
        mean: list[float],
        std: list[float],
        training: bool,
        rotation: bool,
        gain_min: float,
        gain_max: float,
    ) -> None:
        self.image_paths = image_paths
        self.root = root
        self.image_size = image_size
        self.mean = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).view(3, 1, 1)
        self.training = training
        self.rotation = rotation
        self.gain_min = gain_min
        self.gain_max = gain_max

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        image_path = self.image_paths[index]
        label_path = self.root / "labels" / f"{image_path.stem}.png"
        image = Image.open(image_path).convert("RGB").resize(
            (self.image_size, self.image_size), Image.Resampling.BILINEAR
        )
        label = Image.open(label_path).convert("L").resize(
            (self.image_size, self.image_size), Image.Resampling.NEAREST
        )
        if self.training and self.rotation:
            angle = random.choice((0, 90, 180, 270))
            image = image.rotate(angle, resample=Image.Resampling.BILINEAR)
            label = label.rotate(angle, resample=Image.Resampling.NEAREST)
            gain = random.uniform(self.gain_min, self.gain_max)
            image = Image.fromarray(np.clip(np.asarray(image, dtype=np.float32) * gain, 0, 255).astype(np.uint8))
        image_tensor = torch.from_numpy(np.asarray(image, dtype=np.float32).copy()).permute(2, 0, 1) / 255.0
        image_tensor = (image_tensor - self.mean) / self.std
        label_tensor = torch.from_numpy((np.asarray(label, dtype=np.float32).copy() > 127).astype(np.float32)).unsqueeze(0)
        return image_tensor, label_tensor, image_path.stem


def make_loaders(config: dict) -> tuple[dict[str, DataLoader], Split, tuple[list[float], list[float]]]:
    data_config = config["data"]
    root = Path(data_config["root"])
    split = split_images(root, config["seed"], data_config["train_fraction"], data_config["val_fraction"])
    if data_config.get("compute_stats", True):
        mean, std = compute_stats(split.train, data_config["image_size"])
    else:
        mean, std = data_config["mean"], data_config["std"]
    config["data"]["mean"], config["data"]["std"] = mean, std
    common = dict(
        root=root,
        image_size=data_config["image_size"],
        mean=mean,
        std=std,
        rotation=config["augmentation"]["rotation"],
        gain_min=config["augmentation"]["gain_min"],
        gain_max=config["augmentation"]["gain_max"],
    )
    datasets = {
        "train": CrackDataset(split.train, training=True, **common),
        "val": CrackDataset(split.val, training=False, **common),
        "test": CrackDataset(split.test, training=False, **common),
    }
    loaders = {
        name: DataLoader(
            dataset,
            batch_size=config["training"]["batch_size"],
            shuffle=name == "train",
            num_workers=data_config["num_workers"],
            pin_memory=True,
        )
        for name, dataset in datasets.items()
    }
    return loaders, split, (mean, std)