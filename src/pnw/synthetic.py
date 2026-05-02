from __future__ import annotations

from pathlib import Path

import torch
from diffusers import DiffusionPipeline
from omegaconf import DictConfig
from tqdm import tqdm

from pnw.common import diffusion_device, path


def _resize_size(dataset_name: str, cfg: DictConfig) -> tuple[int, int] | None:
    if dataset_name not in cfg.image_sizes:
        return None
    size = int(cfg.image_sizes[dataset_name])
    return size, size


def generate_dataset_images(dataset_name: str, dataset_cfg: DictConfig, cfg: DictConfig, device: str) -> None:
    print(f"  Loading base model and LoRA from {dataset_cfg.lora_id}")
    pipe = DiffusionPipeline.from_pretrained(
        cfg.base_model,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    pipe.load_lora_weights(dataset_cfg.lora_id)

    dataset_output_dir = path(cfg.resnet_data_dir) / dataset_name / "generated"
    dataset_output_dir.mkdir(parents=True, exist_ok=True)
    resize_size = _resize_size(dataset_name, cfg)

    for class_name in dataset_cfg.classes:
        class_dir = dataset_output_dir / class_name
        class_dir.mkdir(exist_ok=True)
        prompt = dataset_cfg.prompt_template.format(class_name=class_name)
        print(f"    {class_name}...", end=" ", flush=True)

        for index in tqdm(range(int(dataset_cfg.images_per_class)), desc=class_name, leave=False):
            try:
                with torch.no_grad():
                    image = pipe(prompt).images[0]
                if resize_size:
                    image = image.resize(resize_size)
                image.save(class_dir / f"{index:05d}.png")
            except Exception as error:
                print(f"\n    Error generating image {index} for {class_name}: {error}")
                continue

        print("done")


def run(cfg: DictConfig) -> None:
    device = diffusion_device()
    print(f"Using device: {device}")

    for dataset_name, dataset_cfg in cfg.datasets.items():
        print(f"\nGenerating {dataset_name}...")
        try:
            generate_dataset_images(dataset_name, dataset_cfg, cfg, device)
        except Exception as error:
            print(f"  Error processing {dataset_name}: {error}")
            continue

    print(f"\nSynthetic images generated in {Path(cfg.resnet_data_dir)}")
