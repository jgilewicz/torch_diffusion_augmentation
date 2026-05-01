from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm import tqdm


RESNET_DATA_DIR = Path("resnet_data")
OUTPUT_DIR = Path("outputs/stage3")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATASETS = ["cifar10", "eurosat_rgb", "beans"]
AUGMENTATION_TYPES = ["original", "augmented", "generated"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 1e-3
RANDOM_SEED = 42
TRAIN_FRACTION = 0.8


class ImageFolderDataset(Dataset):
    def __init__(self, root_dir: Path, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.images: list[Path] = []
        self.labels: list[int] = []
        self.class_to_idx: dict[str, int] = {}

        class_dirs = sorted(path for path in self.root_dir.iterdir() if path.is_dir())
        for idx, class_dir in enumerate(class_dirs):
            self.class_to_idx[class_dir.name] = idx

        for class_dir in class_dirs:
            class_idx = self.class_to_idx[class_dir.name]
            for image_path in sorted(class_dir.iterdir()):
                if image_path.suffix.lower() in IMAGE_EXTENSIONS:
                    self.images.append(image_path)
                    self.labels.append(class_idx)

        if not self.images:
            raise ValueError(f"No images found in {self.root_dir}")

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        from PIL import Image

        image = Image.open(self.images[idx]).convert("RGB")
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def image_size(dataset_name: str) -> int:
    if dataset_name == "cifar10":
        return 32
    if dataset_name == "eurosat_rgb":
        return 64
    return 224


def get_transforms(dataset_name: str, augmented: bool):
    size = image_size(dataset_name)
    steps = [transforms.Resize((size, size))]

    if augmented:
        steps.extend(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                    hue=0.05,
                ),
            ]
        )

    steps.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
    return transforms.Compose(steps)


def create_model(num_classes: int) -> nn.Module:
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model.to(DEVICE)


def train_epoch(model, dataloader, criterion, optimizer) -> float:
    model.train()
    total_loss = 0.0

    for images, labels in tqdm(dataloader, desc="Training", leave=False):
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

    return total_loss / len(dataloader.dataset)


def stratified_train_indices(dataset: ImageFolderDataset) -> list[int]:
    """Return the fixed train split used for all augmentation variants."""
    rng = random.Random(RANDOM_SEED)
    by_class: dict[int, list[int]] = {}
    for idx, label in enumerate(dataset.labels):
        by_class.setdefault(label, []).append(idx)

    train_indices = []
    for indices in by_class.values():
        shuffled = indices[:]
        rng.shuffle(shuffled)
        train_count = max(1, int(len(shuffled) * TRAIN_FRACTION))
        train_indices.extend(shuffled[:train_count])

    return sorted(train_indices)


def train_variant(
    dataset_name: str,
    aug_type: str,
    train_data_path: Path,
    original_data_path: Path,
) -> None:
    use_augmented_transform = aug_type == "augmented"
    train_transform = get_transforms(dataset_name, augmented=use_augmented_transform)
    test_transform = get_transforms(dataset_name, augmented=False)

    original_dataset = ImageFolderDataset(original_data_path, transform=test_transform)
    train_base_dataset = ImageFolderDataset(train_data_path, transform=test_transform)
    if train_base_dataset.class_to_idx != original_dataset.class_to_idx:
        raise ValueError(
            f"{train_data_path} classes do not match {original_data_path} classes."
        )

    num_classes = len(original_dataset.class_to_idx)
    train_indices = stratified_train_indices(original_dataset)
    print(f"  training {dataset_name}/{aug_type} on {len(train_indices)} images")

    train_dataset = ImageFolderDataset(train_data_path, transform=train_transform)
    train_subset = torch.utils.data.Subset(train_dataset, train_indices)
    train_loader = DataLoader(
        train_subset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

    model = create_model(num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    for epoch in range(EPOCHS):
        train_loss = train_epoch(model, train_loader, criterion, optimizer)
        scheduler.step()

        wandb.log(
            {
                f"{aug_type}/epoch": epoch + 1,
                f"{aug_type}/train_loss": train_loss,
            }
        )

    model_path = OUTPUT_DIR / f"{dataset_name}_{aug_type}.pt"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_path)
    print(f"    saved {model_path}")


def main() -> None:
    seed_everything(RANDOM_SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for dataset_name in DATASETS:
        original_data_path = RESNET_DATA_DIR / dataset_name / "original"
        if not original_data_path.exists():
            print(f"Skipping {dataset_name}: missing {original_data_path}")
            continue

        wandb.init(
            project="PNW",
            name=f"stage3-train-{dataset_name}",
            group="stage3-resnet-training",
            tags=["stage3", "resnet", "train", dataset_name],
            config={
                "dataset": dataset_name,
                "augmentation_types": AUGMENTATION_TYPES,
                "batch_size": BATCH_SIZE,
                "epochs": EPOCHS,
                "learning_rate": LEARNING_RATE,
                "random_seed": RANDOM_SEED,
                "train_fraction": TRAIN_FRACTION,
            },
            reinit=True,
        )

        try:
            for aug_type in AUGMENTATION_TYPES:
                data_path = RESNET_DATA_DIR / dataset_name / aug_type
                if not data_path.exists():
                    print(f"Skipping {data_path}: missing directory")
                    continue

                train_variant(dataset_name, aug_type, data_path, original_data_path)
        finally:
            wandb.finish()


if __name__ == "__main__":
    main()
