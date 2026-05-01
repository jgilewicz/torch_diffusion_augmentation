from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import models, transforms
from tqdm import tqdm


RESNET_DATA_DIR = Path("resnet_data")
OUTPUT_DIR = Path("outputs/stage3")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATASETS = ["cifar10", "eurosat_rgb", "beans"]
AUGMENTATION_TYPES = ["original", "augmented", "generated"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

NUM_FOLDS = 5
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 1e-3
RANDOM_SEED = 42


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


def evaluate(model, dataloader, criterion) -> tuple[float, float, float]:
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Validation", leave=False):
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)
            preds = torch.argmax(outputs, dim=1)

            total_loss += loss.item() * images.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    loss = total_loss / len(dataloader.dataset)
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    return loss, accuracy, f1


def train_variant(dataset_name: str, aug_type: str, data_path: Path) -> None:
    use_augmented_transform = aug_type == "augmented"
    train_transform = get_transforms(dataset_name, augmented=use_augmented_transform)
    test_transform = get_transforms(dataset_name, augmented=False)

    base_dataset = ImageFolderDataset(data_path, transform=test_transform)
    num_classes = len(base_dataset.class_to_idx)
    fold_splits = list(
        KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=RANDOM_SEED).split(
            base_dataset
        )
    )

    for fold, (train_idx, test_idx) in enumerate(fold_splits):
        print(f"  {dataset_name}/{aug_type} fold {fold + 1}/{NUM_FOLDS}")

        train_dataset = ImageFolderDataset(data_path, transform=train_transform)
        test_dataset = ImageFolderDataset(data_path, transform=test_transform)
        train_subset = Subset(train_dataset, train_idx)
        test_subset = Subset(test_dataset, test_idx)

        train_loader = DataLoader(
            train_subset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=0,
        )
        test_loader = DataLoader(
            test_subset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=0,
        )

        model = create_model(num_classes)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

        best_f1 = -1.0
        model_path = OUTPUT_DIR / f"{dataset_name}_{aug_type}_fold{fold}.pt"

        for epoch in range(EPOCHS):
            train_loss = train_epoch(model, train_loader, criterion, optimizer)
            val_loss, val_acc, val_f1 = evaluate(model, test_loader, criterion)
            scheduler.step()

            wandb.log(
                {
                    "dataset": dataset_name,
                    "augmentation": aug_type,
                    "fold": fold,
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "val_f1": val_f1,
                }
            )

            if val_f1 > best_f1:
                best_f1 = val_f1
                model_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), model_path)

        print(f"    saved {model_path} (best f1={best_f1:.4f})")


def main() -> None:
    seed_everything(RANDOM_SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    wandb.init(project="PNW", name="stage3-resnet-training")

    for dataset_name in DATASETS:
        for aug_type in AUGMENTATION_TYPES:
            data_path = RESNET_DATA_DIR / dataset_name / aug_type
            if not data_path.exists():
                print(f"Skipping {data_path}: missing directory")
                continue

            train_variant(dataset_name, aug_type, data_path)

    wandb.finish()


if __name__ == "__main__":
    main()
