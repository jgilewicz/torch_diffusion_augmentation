from __future__ import annotations

import os
import subprocess
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from pnw.common import path


def hub_namespace() -> str:
    namespace = os.getenv("HF_ORG") or os.getenv("HF_USERNAME")
    if not namespace:
        raise RuntimeError("Set HF_USERNAME or HF_ORG in .env before running stage 2.")
    return namespace


def validate_inputs(cfg: DictConfig) -> None:
    train_script = path(cfg.train_script)
    if not train_script.exists():
        raise FileNotFoundError(
            f"{train_script} does not exist. Run `make setup` before stage 2."
        )

    missing = [
        path(dataset.dataset_dir) / "train" / "metadata.jsonl"
        for dataset in cfg.datasets
        if not (path(dataset.dataset_dir) / "train" / "metadata.jsonl").exists()
    ]
    if missing:
        missing_text = ", ".join(str(item) for item in missing)
        raise FileNotFoundError(
            f"Missing prepared datasets: {missing_text}. Run `make prepare-sana-datasets` first."
        )

    if not (os.getenv("WANDB_API_KEY") or os.getenv("WANDB_MODE") == "offline"):
        print("WANDB_API_KEY is not set; assuming you already ran `make login-wandb`.")

    if not os.getenv("HF_TOKEN"):
        print("HF_TOKEN is not set; assuming you already ran `make login-hf`.")


def train_lora(dataset: DictConfig, cfg: DictConfig, namespace: str) -> None:
    output_dir = path(cfg.output_root) / dataset.output_name
    output_dir.mkdir(parents=True, exist_ok=True)
    hub_model_id = f"{namespace}/{dataset.output_name}"

    cmd = [
        "accelerate",
        "launch",
        str(path(cfg.train_script)),
        f"--pretrained_model_name_or_path={cfg.model_name}",
        f"--dataset_name={path(dataset.dataset_dir) / 'train'}",
        "--image_column=image",
        "--caption_column=text",
        f"--output_dir={output_dir}",
        f"--mixed_precision={cfg.mixed_precision}",
        f"--resolution={cfg.resolution}",
        f"--train_batch_size={cfg.train_batch_size}",
        f"--gradient_accumulation_steps={cfg.gradient_accumulation_steps}",
        "--use_8bit_adam",
        f"--learning_rate={cfg.learning_rate}",
        f"--lr_scheduler={cfg.lr_scheduler}",
        f"--lr_warmup_steps={cfg.lr_warmup_steps}",
        f"--max_train_steps={cfg.max_train_steps}",
        f"--rank={cfg.lora_rank}",
        f"--lora_alpha={cfg.lora_alpha}",
        f"--validation_prompt={dataset.validation_prompt}",
        "--report_to=wandb",
        "--push_to_hub",
        f"--hub_model_id={hub_model_id}",
        f"--instance_prompt={cfg.instance_prompt}",
    ]

    env = os.environ.copy()
    env["WANDB_PROJECT"] = str(cfg.wandb.project)
    env["WANDB_RUN_GROUP"] = str(cfg.wandb.run_group)
    env["WANDB_NAME"] = str(dataset.output_name)

    print(f"Training {dataset.name} -> {output_dir}")
    print(f"Using dataset {path(dataset.dataset_dir) / 'train'}")
    print(f"Pushing to {hub_model_id}")
    subprocess.run(cmd, check=True, env=env)


def run(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg, resolve=True))
    validate_inputs(cfg)
    namespace = hub_namespace()
    for dataset in cfg.datasets:
        train_lora(dataset, cfg, namespace)

