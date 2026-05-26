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


def _log_summary(aug_type: str, metrics: dict[str, list[float]]) -> tuple[float, float, float, float]:
    avg_acc = float(np.mean(metrics["acc"]))
    std_acc = float(np.std(metrics["acc"]))
    avg_f1 = float(np.mean(metrics["f1"]))
    std_f1 = float(np.std(metrics["f1"]))
    print(f"    summary: acc={avg_acc:.4f} +/- {std_acc:.4f}, f1={avg_f1:.4f} +/- {std_f1:.4f}")
    return avg_acc, std_acc, avg_f1, std_f1


def _select_paired_test(
    values_a: list[float],
    values_b: list[float],
    alpha: float = 0.05,
) -> dict[str, float | str | bool]:
    if len(values_a) != len(values_b):
        raise ValueError("Paired statistical tests require inputs of the same length.")
    if len(values_a) < 3:
        raise ValueError("Shapiro-Wilk requires at least 3 paired observations.")

    diffs = np.asarray(values_a, dtype=float) - np.asarray(values_b, dtype=float)
    _, shapiro_p = stats.shapiro(diffs)
    normality_not_rejected = bool(shapiro_p >= alpha)

    if normality_not_rejected:
        _, selected_p = stats.ttest_rel(values_a, values_b)
        selected_test = "ttest_rel"
    else:
        _, selected_p = stats.wilcoxon(values_a, values_b)
        selected_test = "wilcoxon"

    return {
        "shapiro_p": float(shapiro_p),
        "normality_not_rejected": normality_not_rejected,
        "selected_p": float(selected_p),
        "selected_test": selected_test,
    }


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

        fold_metrics_table = wandb.Table(columns=["Augmentation", "Fold", "Accuracy", "F1"])
        summary_metrics_table = wandb.Table(columns=["Augmentation", "Avg Accuracy", "Std Accuracy", "Avg F1", "Std F1"])
        stats_test_table = wandb.Table(columns=["Model A", "Model B", "Shapiro p", "Normality Not Rejected", "Selected Test", "Selected p"])

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
                    fold_metrics_table.add_data(aug_type, fold, acc, f1)
                    print(f"    fold {fold + 1}: acc={acc:.4f}, f1={f1:.4f}")

                if per_fold_metrics[aug_type]["acc"]:
                    avg_acc, std_acc, avg_f1, std_f1 = _log_summary(aug_type, per_fold_metrics[aug_type])
                    summary_metrics_table.add_data(aug_type, avg_acc, std_acc, avg_f1, std_f1)

            variants = [aug for aug in cfg.augmentation_types if per_fold_metrics[aug]["acc"]]
            for index, aug_a in enumerate(variants):
                for aug_b in variants[index + 1 :]:
                    acc_a = per_fold_metrics[aug_a]["acc"]
                    acc_b = per_fold_metrics[aug_b]["acc"]
                    if len(acc_a) != len(acc_b) or len(acc_a) < 3:
                        print(
                            f"  skipping significance test for {aug_a} vs {aug_b}: "
                            "Shapiro-Wilk requires at least 3 paired folds"
                        )
                        continue

                    test_result = _select_paired_test(acc_a, acc_b)
                    print(
                        f"  {aug_a} vs {aug_b}: Shapiro p={test_result['shapiro_p']:.4f}, "
                        f"selected={test_result['selected_test']}, p={test_result['selected_p']:.4f}"
                    )
                    stats_test_table.add_data(
                        aug_a,
                        aug_b,
                        test_result["shapiro_p"],
                        bool(test_result["normality_not_rejected"]),
                        test_result["selected_test"],
                        test_result["selected_p"],
                    )

            wandb.log(
                {
                    "Fold Metrics": fold_metrics_table,
                    "Summary Metrics": summary_metrics_table,
                    "Statistical Tests": stats_test_table,
                }
            )
            all_results[dataset_name] = {"per_fold_metrics": per_fold_metrics}
        finally:
            wandb.finish()

    output_path = output_dir / "evaluation_results.json"
    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(all_results, output_file, indent=2)
    print(f"\nResults saved to {output_path}")
