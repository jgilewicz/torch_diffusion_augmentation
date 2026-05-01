#!/usr/bin/env python3
"""Fine-tune one SANA LoRA per prepared dataset and push results to HF Hub."""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path


MODEL_NAME = os.getenv(
    "SANA_MODEL_NAME", "Efficient-Large-Model/Sana_1600M_1024px_BF16_diffusers"
)
TRAIN_SCRIPT = Path("diffusers/examples/dreambooth/train_dreambooth_lora_sana.py")
OUTPUT_ROOT = Path("outputs/stage2")
WANDB_PROJECT = os.getenv("WANDB_PROJECT", "stage2-sana-lora")

RESOLUTION = int(os.getenv("SANA_RESOLUTION", "512"))
MAX_TRAIN_STEPS = int(os.getenv("SANA_MAX_TRAIN_STEPS", "300"))
LORA_RANK = int(os.getenv("SANA_LORA_RANK", "2"))
LORA_ALPHA = int(os.getenv("SANA_LORA_ALPHA", "2"))
LEARNING_RATE = os.getenv("SANA_LEARNING_RATE", "1e-4")
GRADIENT_ACCUMULATION_STEPS = int(os.getenv("SANA_GRADIENT_ACCUMULATION_STEPS", "1"))


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    dataset_dir: Path
    output_name: str
    validation_prompt: str


DATASETS = [
    DatasetConfig(
        name="cifar10",
        dataset_dir=Path("data/sana_cifar10"),
        output_name="SANA-LoRA-CIFAR10",
        validation_prompt="a low-resolution photo of an airplane, CIFAR-10 style",
    ),
    DatasetConfig(
        name="eurosat_rgb",
        dataset_dir=Path("data/sana_eurosat_rgb"),
        output_name="SANA-LoRA-EuroSAT-RGB",
        validation_prompt="a satellite image of AnnualCrop, EuroSAT style",
    ),
    DatasetConfig(
        name="beans",
        dataset_dir=Path("data/sana_beans"),
        output_name="SANA-LoRA-Beans",
        validation_prompt="a photo of a bean leaf with healthy",
    ),
]


def hub_namespace() -> str:
    namespace = os.getenv("HF_ORG") or os.getenv("HF_USERNAME")
    if not namespace:
        raise RuntimeError("Set HF_USERNAME or HF_ORG in .env before running stage 2.")
    return namespace


def validate_inputs() -> None:
    if not TRAIN_SCRIPT.exists():
        raise FileNotFoundError(
            f"{TRAIN_SCRIPT} does not exist. Run `just setup` before stage 2."
        )

    missing = [
        config.dataset_dir
        for config in DATASETS
        if not (config.dataset_dir / "train" / "metadata.jsonl").exists()
    ]
    if missing:
        missing_text = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(
            f"Missing prepared SANA datasets: {missing_text}. Run `just datasets` first."
        )

    if not (os.getenv("WANDB_API_KEY") or os.getenv("WANDB_MODE") == "offline"):
        print("WANDB_API_KEY is not set; assuming you already ran `just login-wandb`.")

    if not os.getenv("HF_TOKEN"):
        print("HF_TOKEN is not set; assuming you already ran `just login-hf`.")


def train_lora(config: DatasetConfig, namespace: str) -> None:
    output_dir = OUTPUT_ROOT / config.output_name
    hub_model_id = f"{namespace}/{config.output_name}"
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "accelerate",
        "launch",
        str(TRAIN_SCRIPT),
        f"--pretrained_model_name_or_path={MODEL_NAME}",
        f"--instance_data_dir={config.dataset_dir / 'train'}",
        f"--output_dir={output_dir}",
        "--mixed_precision=bf16",
        f"--resolution={RESOLUTION}",
        "--train_batch_size=1",
        f"--gradient_accumulation_steps={GRADIENT_ACCUMULATION_STEPS}",
        "--use_8bit_adam",
        f"--learning_rate={LEARNING_RATE}",
        "--lr_scheduler=constant",
        "--lr_warmup_steps=0",
        f"--max_train_steps={MAX_TRAIN_STEPS}",
        f"--rank={LORA_RANK}",
        f"--lora_alpha={LORA_ALPHA}",
        f"--validation_prompt={config.validation_prompt}",
        "--report_to=wandb",
        "--push_to_hub",
        f"--hub_model_id={hub_model_id}",
        "--instance_prompt=a photo",
    ]

    env = os.environ.copy()
    env["WANDB_PROJECT"] = WANDB_PROJECT
    env["WANDB_RUN_GROUP"] = "sana-lora-finetune"
    env["WANDB_NAME"] = config.output_name

    print(f"Training {config.name} -> {output_dir}")
    print(f"Pushing to {hub_model_id}")
    subprocess.run(cmd, check=True, env=env)


def main() -> None:
    validate_inputs()
    namespace = hub_namespace()

    for config in DATASETS:
        train_lora(config, namespace)


if __name__ == "__main__":
    main()
