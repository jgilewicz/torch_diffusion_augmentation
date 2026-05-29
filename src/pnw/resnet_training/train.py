from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from pnw.common import path, seed_everything, torch_device
from pnw.datasets import ImageFolderDataset, class_count
from pnw.resnet import create_resnet18, get_transforms, stratified_indices


def train_epoch(model, dataloader, criterion, optimizer, device: torch.device) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(dataloader, desc="Training", leave=False):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)

        preds = torch.argmax(outputs, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / len(dataloader.dataset), correct / total


def train_variant(
    dataset_name: str,
    aug_type: str,
    train_data_path,
    original_data_path,
    cfg: DictConfig,
    device: torch.device,
) -> None:
    train_transform = get_transforms(dataset_name, False, cfg.resnet)
    test_transform = get_transforms(dataset_name, False, cfg.resnet)

    original_dataset = ImageFolderDataset(original_data_path, transform=test_transform)
    train_base_dataset = ImageFolderDataset(train_data_path, transform=test_transform)
    if train_base_dataset.class_to_idx != original_dataset.class_to_idx:
        raise ValueError(f"{train_data_path} classes do not match {original_data_path} classes.")

    train_indices = stratified_indices(
        train_base_dataset,
        seed=int(cfg.random_seed),
        train_fraction=float(cfg.train_fraction),
        split="train",
    )
    print(f"  training {dataset_name}/{aug_type} on {len(train_indices)} images")

    train_dataset = ImageFolderDataset(train_data_path, transform=train_transform)
    train_subset = torch.utils.data.Subset(train_dataset, train_indices)
    train_loader = DataLoader(
        train_subset,
        batch_size=int(cfg.batch_size),
        shuffle=True,
        num_workers=int(cfg.num_workers),
    )

    model = create_resnet18(class_count(original_dataset), device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=float(cfg.learning_rate))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(cfg.epochs))

    for epoch in range(int(cfg.epochs)):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        scheduler.step()
        wandb.log({
            f"{aug_type}/epoch": epoch + 1,
            f"{aug_type}/train_loss": train_loss,
            f"{aug_type}/train_acc": train_acc,
        })

    model_path = path(cfg.output_dir) / f"{dataset_name}_{aug_type}.pt"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_path)
    print(f"    saved {model_path}")


def run(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg, resolve=True))
    seed_everything(int(cfg.random_seed))
    device = torch_device()
    path(cfg.output_dir).mkdir(parents=True, exist_ok=True)

    for dataset_name in cfg.datasets:
        original_data_path = path(cfg.resnet_data_dir) / dataset_name / "original"
        if not original_data_path.exists():
            print(f"Skipping {dataset_name}: missing {original_data_path}")
            continue

        wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.get("entity"),
            name=f"resnet-train-{dataset_name}",
            group=cfg.wandb.group,
            tags=list(cfg.wandb.tags) + [dataset_name],
            config=OmegaConf.to_container(cfg, resolve=True),
            reinit=True,
        )

        try:
            for aug_type in cfg.augmentation_types:
                data_path = path(cfg.resnet_data_dir) / dataset_name / aug_type
                if not data_path.exists():
                    print(f"Skipping {data_path}: missing directory")
                    continue
                train_variant(dataset_name, aug_type, data_path, original_data_path, cfg, device)
        finally:
            wandb.finish()
