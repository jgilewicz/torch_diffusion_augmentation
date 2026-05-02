from __future__ import annotations

import os
import subprocess
from pathlib import Path

import torch
import wandb
from omegaconf import DictConfig, OmegaConf
from torch_fidelity import calculate_metrics

from pnw.common import path


def configure_environment(cfg: DictConfig) -> None:
    for key, value in cfg.environment.items():
        os.environ[str(key)] = str(value)


def train_lora(model_name: str, model_id: str, cfg: DictConfig) -> None:
    out = path(cfg.output_dir) / model_name
    out.mkdir(parents=True, exist_ok=True)
    script = cfg.train_scripts[model_name]

    cmd = [
        "accelerate",
        "launch",
        str(script),
        f"--pretrained_model_name_or_path={model_id}",
        f"--instance_data_dir={cfg.horse_dataset}",
        f"--instance_prompt={cfg.prompt}",
        f"--output_dir={out / 'lora'}",
        f"--rank={cfg.lora_rank}",
        f"--max_train_steps={cfg.train_steps}",
        f"--train_batch_size={cfg.train_batch_size}",
        f"--mixed_precision={cfg.bf16_mixed_precision if model_name in cfg.bf16_models else cfg.fp16_mixed_precision}",
        "--gradient_checkpointing",
        "--report_to=wandb",
    ]
    subprocess.run(cmd, check=True)


def generate_images(model_name: str, model_id: str, cfg: DictConfig) -> None:
    from diffusers import AutoPipelineForText2Image

    lora_path = path(cfg.output_dir) / model_name / "lora"
    gen_path = path(cfg.output_dir) / model_name / "generated"
    gen_path.mkdir(parents=True, exist_ok=True)

    dtype = torch.bfloat16 if model_name in cfg.bf16_models else torch.float16
    pipe = AutoPipelineForText2Image.from_pretrained(model_id, torch_dtype=dtype).to("cuda")
    if model_name == "sana":
        pipe.vae.to(torch.bfloat16)

    pipe.load_lora_weights(str(lora_path))
    for index in range(int(cfg.num_images)):
        image = pipe(cfg.prompt, num_inference_steps=int(cfg.num_inference_steps)).images[0]
        image.save(gen_path / f"{index:04d}.png")

    pipe = None
    torch.cuda.empty_cache()


def compute_metrics(model_name: str, cfg: DictConfig) -> dict[str, float | str]:
    gen_path = str(path(cfg.output_dir) / model_name / "generated")
    real_path = str(path(cfg.horse_dataset))
    metrics = calculate_metrics(
        input1=real_path,
        input2=gen_path,
        cuda=torch.cuda.is_available(),
        fid=True,
        isc=True,
        verbose=False,
    )
    fid = round(metrics["frechet_inception_distance"], 3)
    isc = round(metrics["inception_score_mean"], 3)
    wandb.log({"model": model_name, "fid": fid, "isc": isc})
    return {"model": model_name, "FID": fid, "IS": isc}


def run(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg, resolve=True))
    configure_environment(cfg)
    wandb.init(project=cfg.wandb.project, name=cfg.wandb.name)

    results = []
    for model_name, model_id in cfg.models.items():
        train_lora(model_name, model_id, cfg)
        generate_images(model_name, model_id, cfg)
        results.append(compute_metrics(model_name, cfg))

    table = wandb.Table(
        columns=["model", "FID", "IS"],
        data=[[item["model"], item["FID"], item["IS"]] for item in results],
    )
    wandb.log({"results": table})
    best = min(results, key=lambda item: item["FID"])
    wandb.summary["best_model"] = best["model"]
    wandb.summary["best_fid"] = best["FID"]
    wandb.finish()
