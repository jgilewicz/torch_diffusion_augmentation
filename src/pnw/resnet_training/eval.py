from __future__ import annotations

import json

import numpy as np
import torch
import wandb
from omegaconf import DictConfig, OmegaConf
from scipy import stats
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from pnw.common import path, torch_device
from pnw.datasets import ImageFolderDataset, class_count
from pnw.resnet import create_resnet18, get_transforms, stratified_indices


def evaluate_model(model, dataloader, device: torch.device) -> tuple[float, float]:
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating", leave=False):
            images = images.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    return (
        accuracy_score(all_labels, all_preds),
        f1_score(all_labels, all_preds, average="weighted", zero_division=0),
    )


def evaluate_fold(
    test_dataset_path,
    model_path,
    test_idx: np.ndarray,
    transform,
    num_classes: int,
    cfg: DictConfig,
    device: torch.device,
) -> tuple[float, float]:
    dataset = ImageFolderDataset(test_dataset_path, transform=transform)
    test_subset = Subset(dataset, test_idx)
    test_loader = DataLoader(
        test_subset,
        batch_size=int(cfg.batch_size),
        shuffle=False,
        num_workers=int(cfg.num_workers),
    )

    model = create_resnet18(num_classes, device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    return evaluate_model(model, test_loader, device)


def _log_summary(aug_type: str, metrics: dict[str, list[float]]) -> None:
    avg_acc = float(np.mean(metrics["acc"]))
    std_acc = float(np.std(metrics["acc"]))
    avg_f1 = float(np.mean(metrics["f1"]))
    std_f1 = float(np.std(metrics["f1"]))
    print(f"    summary: acc={avg_acc:.4f} +/- {std_acc:.4f}, f1={avg_f1:.4f} +/- {std_f1:.4f}")
    wandb.log(
        {
            f"{aug_type}/avg_acc": avg_acc,
            f"{aug_type}/std_acc": std_acc,
            f"{aug_type}/avg_f1": avg_f1,
            f"{aug_type}/std_f1": std_f1,
        }
    )


def run(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg, resolve=True))
    output_dir = path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch_device()
    all_results = {}

    for dataset_name in cfg.datasets:
        print(f"\n{'=' * 60}\nDataset: {dataset_name}\n{'=' * 60}")
        original_path = path(cfg.resnet_data_dir) / dataset_name / "original"
        if not original_path.exists():
            print(f"Skipping {dataset_name}: missing {original_path}")
            continue

        transform = get_transforms(dataset_name, False, cfg.resnet)
        original_dataset = ImageFolderDataset(original_path, transform=transform)
        eval_indices = stratified_indices(
            original_dataset,
            seed=int(cfg.random_seed),
            train_fraction=float(cfg.train_fraction),
            split="eval",
        )
        if len(eval_indices) < int(cfg.num_folds):
            raise ValueError(
                f"Not enough held-out samples for {cfg.num_folds}-fold evaluation: "
                f"{len(eval_indices)} samples in {original_path}"
            )

        fold_splits = list(
            KFold(n_splits=int(cfg.num_folds), shuffle=True, random_state=int(cfg.random_seed)).split(eval_indices)
        )

        wandb.init(
            project=cfg.wandb.project,
            name=f"resnet-eval-{dataset_name}",
            group=cfg.wandb.group,
            tags=list(cfg.wandb.tags) + [dataset_name],
            config=OmegaConf.to_container(cfg, resolve=True),
            reinit=True,
        )

        per_fold_metrics = {aug: {"acc": [], "f1": []} for aug in cfg.augmentation_types}
        try:
            for aug_type in cfg.augmentation_types:
                data_path = path(cfg.resnet_data_dir) / dataset_name / aug_type
                if not data_path.exists():
                    print(f"  skipping {aug_type}: missing {data_path}")
                    continue
                variant_dataset = ImageFolderDataset(data_path, transform=transform)
                if variant_dataset.class_to_idx != original_dataset.class_to_idx:
                    raise ValueError(f"{data_path} classes do not match {original_path}")

                print(f"\n  Evaluating {aug_type} on original held-out folds")
                model_path = path(cfg.models_dir) / f"{dataset_name}_{aug_type}.pt"
                if not model_path.exists():
                    print(f"    missing {model_path}")
                    continue

                for fold, (_, fold_eval_positions) in enumerate(fold_splits):
                    test_idx = np.array([eval_indices[position] for position in fold_eval_positions])
                    acc, f1 = evaluate_fold(
                        original_path,
                        model_path,
                        test_idx,
                        transform,
                        class_count(original_dataset),
                        cfg,
                        device,
                    )
                    per_fold_metrics[aug_type]["acc"].append(acc)
                    per_fold_metrics[aug_type]["f1"].append(f1)
                    wandb.log({f"{aug_type}/fold_{fold}/acc": acc, f"{aug_type}/fold_{fold}/f1": f1})
                    print(f"    fold {fold + 1}: acc={acc:.4f}, f1={f1:.4f}")

                if per_fold_metrics[aug_type]["acc"]:
                    _log_summary(aug_type, per_fold_metrics[aug_type])

            variants = [aug for aug in cfg.augmentation_types if per_fold_metrics[aug]["acc"]]
            for index, aug_a in enumerate(variants):
                for aug_b in variants[index + 1 :]:
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

    output_path = output_dir / "evaluation_results.json"
    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(all_results, output_file, indent=2)
    print(f"\nResults saved to {output_path}")
