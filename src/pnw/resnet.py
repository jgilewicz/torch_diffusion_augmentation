from __future__ import annotations

import random

import torch
import torch.nn as nn
from omegaconf import DictConfig
from torchvision import models, transforms

from pnw.datasets import ImageFolderDataset


def image_size(dataset_name: str, sizes: DictConfig) -> int:
    return int(sizes.get(dataset_name, sizes.default))


def get_transforms(dataset_name: str, augmented: bool, cfg: DictConfig):
    size = image_size(dataset_name, cfg.image_sizes)
    steps = [transforms.Resize((size, size))]

    if augmented:
        aug = cfg.augmentation
        steps.extend(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(float(aug.rotation_degrees)),
                transforms.ColorJitter(
                    brightness=float(aug.color_jitter.brightness),
                    contrast=float(aug.color_jitter.contrast),
                    saturation=float(aug.color_jitter.saturation),
                    hue=float(aug.color_jitter.hue),
                ),
            ]
        )

    steps.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=list(cfg.normalization.mean),
                std=list(cfg.normalization.std),
            ),
        ]
    )
    return transforms.Compose(steps)


def create_resnet18(num_classes: int, device: torch.device) -> nn.Module:
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model.to(device)


def stratified_indices(
    dataset: ImageFolderDataset,
    *,
    seed: int,
    train_fraction: float,
    split: str,
) -> list[int]:
    rng = random.Random(seed)
    by_class: dict[int, list[int]] = {}
    for idx, label in enumerate(dataset.labels):
        by_class.setdefault(label, []).append(idx)

    selected = []
    for indices in by_class.values():
        shuffled = indices[:]
        rng.shuffle(shuffled)
        train_count = max(1, int(len(shuffled) * train_fraction))
        if split == "train":
            selected.extend(shuffled[:train_count])
        elif split == "eval":
            selected.extend(shuffled[train_count:])
        else:
            raise ValueError(f"Unsupported split: {split}")

    return sorted(selected)
