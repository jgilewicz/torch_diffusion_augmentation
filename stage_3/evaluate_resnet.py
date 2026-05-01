from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import wandb
from scipy import stats
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import models, transforms
from tqdm import tqdm


RESNET_DATA_DIR = Path("resnet_data")
MODELS_DIR = Path("outputs/stage3")
OUTPUT_DIR = Path("outputs/stage3_eval")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATASETS = ["cifar10", "eurosat_rgb", "beans"]
AUGMENTATION_TYPES = ["original", "augmented", "generated"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

NUM_FOLDS = 5
BATCH_SIZE = 32
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


def image_size(dataset_name: str) -> int:
    if dataset_name == "cifar10":
        return 32
    if dataset_name == "eurosat_rgb":
        return 64
    return 224


def get_transform(dataset_name: str):
    return transforms.Compose(
        [
            transforms.Resize((image_size(dataset_name), image_size(dataset_name))),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def create_model(num_classes: int) -> nn.Module:
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model.to(DEVICE)


def evaluate_model(model, dataloader) -> tuple[float, float]:
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating", leave=False):
            images = images.to(DEVICE)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    return accuracy, f1


def evaluate_fold(
    test_dataset_path: Path,
    model_path: Path,
    test_idx: np.ndarray,
    transform,
    num_classes: int,
) -> tuple[float, float]:
    dataset = ImageFolderDataset(test_dataset_path, transform=transform)
    test_subset = Subset(dataset, test_idx)
    test_loader = DataLoader(
        test_subset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    model = create_model(num_classes)
    state_dict = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    return evaluate_model(model, test_loader)


def stratified_eval_indices(dataset: ImageFolderDataset) -> list[int]:
    """Return held-out original samples not used by stage_3/train_resnet.py."""
    rng = random.Random(RANDOM_SEED)
    by_class: dict[int, list[int]] = {}
    for idx, label in enumerate(dataset.labels):
        by_class.setdefault(label, []).append(idx)

    eval_indices = []
    for indices in by_class.values():
        shuffled = indices[:]
        rng.shuffle(shuffled)
        train_count = max(1, int(len(shuffled) * TRAIN_FRACTION))
        eval_indices.extend(shuffled[train_count:])

    return sorted(eval_indices)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_results = {}

    for dataset_name in DATASETS:
        print(f"\n{'=' * 60}\nDataset: {dataset_name}\n{'=' * 60}")
        original_path = RESNET_DATA_DIR / dataset_name / "original"
        if not original_path.exists():
            print(f"Skipping {dataset_name}: missing {original_path}")
            continue

        transform = get_transform(dataset_name)
        original_dataset = ImageFolderDataset(original_path, transform=transform)
        num_classes = len(original_dataset.class_to_idx)
        eval_indices = stratified_eval_indices(original_dataset)
        if len(eval_indices) < NUM_FOLDS:
            raise ValueError(
                f"Not enough held-out samples for {NUM_FOLDS}-fold evaluation: "
                f"{len(eval_indices)} samples in {original_path}"
            )

        fold_splits = list(
            KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=RANDOM_SEED).split(
                eval_indices
            )
        )

        wandb.init(
            project="PNW",
            name=f"stage3-eval-{dataset_name}",
            group="stage3-resnet-evaluation",
            tags=["stage3", "resnet", "eval", dataset_name],
            config={
                "dataset": dataset_name,
                "augmentation_types": AUGMENTATION_TYPES,
                "num_folds": NUM_FOLDS,
                "batch_size": BATCH_SIZE,
                "random_seed": RANDOM_SEED,
                "train_fraction": TRAIN_FRACTION,
                "held_out_samples": len(eval_indices),
            },
            reinit=True,
        )

        per_fold_metrics = {aug: {"acc": [], "f1": []} for aug in AUGMENTATION_TYPES}

        try:
            for aug_type in AUGMENTATION_TYPES:
                data_path = RESNET_DATA_DIR / dataset_name / aug_type
                if not data_path.exists():
                    print(f"  skipping {aug_type}: missing {data_path}")
                    continue
                variant_dataset = ImageFolderDataset(data_path, transform=transform)
                if variant_dataset.class_to_idx != original_dataset.class_to_idx:
                    raise ValueError(f"{data_path} classes do not match {original_path}")

                print(f"\n  Evaluating {aug_type} on original held-out folds")
                model_path = MODELS_DIR / f"{dataset_name}_{aug_type}.pt"
                if not model_path.exists():
                    print(f"    missing {model_path}")
                    continue

                for fold, (_, fold_eval_positions) in enumerate(fold_splits):
                    test_idx = np.array(
                        [eval_indices[position] for position in fold_eval_positions]
                    )
                    acc, f1 = evaluate_fold(
                        original_path,
                        model_path,
                        test_idx,
                        transform,
                        num_classes,
                    )
                    per_fold_metrics[aug_type]["acc"].append(acc)
                    per_fold_metrics[aug_type]["f1"].append(f1)
                    wandb.log(
                        {
                            f"{aug_type}/fold_{fold}/acc": acc,
                            f"{aug_type}/fold_{fold}/f1": f1,
                        }
                    )
                    print(f"    fold {fold + 1}: acc={acc:.4f}, f1={f1:.4f}")

                if per_fold_metrics[aug_type]["acc"]:
                    avg_acc = float(np.mean(per_fold_metrics[aug_type]["acc"]))
                    std_acc = float(np.std(per_fold_metrics[aug_type]["acc"]))
                    avg_f1 = float(np.mean(per_fold_metrics[aug_type]["f1"]))
                    std_f1 = float(np.std(per_fold_metrics[aug_type]["f1"]))
                    print(
                        f"    summary: acc={avg_acc:.4f} +/- {std_acc:.4f}, "
                        f"f1={avg_f1:.4f} +/- {std_f1:.4f}"
                    )
                    wandb.log(
                        {
                            f"{aug_type}/avg_acc": avg_acc,
                            f"{aug_type}/std_acc": std_acc,
                            f"{aug_type}/avg_f1": avg_f1,
                            f"{aug_type}/std_f1": std_f1,
                        }
                    )

            variants = [
                aug for aug in AUGMENTATION_TYPES if per_fold_metrics[aug]["acc"]
            ]
            for i, aug_a in enumerate(variants):
                for aug_b in variants[i + 1 :]:
                    acc_a = per_fold_metrics[aug_a]["acc"]
                    acc_b = per_fold_metrics[aug_b]["acc"]
                    if len(acc_a) != len(acc_b) or len(acc_a) < 2:
                        continue

                    _, p_ttest = stats.ttest_rel(acc_a, acc_b)
                    _, p_wilcox = stats.wilcoxon(acc_a, acc_b)
                    wandb.log(
                        {
                            f"{aug_a}_vs_{aug_b}/ttest_p": float(p_ttest),
                            f"{aug_a}_vs_{aug_b}/wilcoxon_p": float(p_wilcox),
                        }
                    )

            all_results[dataset_name] = {"per_fold_metrics": per_fold_metrics}
        finally:
            wandb.finish()

    output_path = OUTPUT_DIR / "evaluation_results.json"
    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(all_results, output_file, indent=2)

    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
