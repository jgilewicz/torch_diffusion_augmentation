import os

os.environ["HF_HOME"] = "/workspace/hf_cache"
os.environ["HF_HUB_CACHE"] = "/workspace/hf_cache"
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import subprocess
from pathlib import Path

import torch
import wandb
from torch_fidelity import calculate_metrics

MODELS = {
    "sd15": "stable-diffusion-v1-5/stable-diffusion-v1-5",
    "lumina2": "Alpha-VLLM/Lumina-Image-2.0",
    "sana": "Efficient-Large-Model/Sana_600M_512px_diffusers",
}

HORSE_DATASET = "./data/horses"
OUTPUT_DIR = "./stage1"
PROMPT = "a photo of a horse"
NUM_IMAGES = 100
LORA_RANK = 2
TRAIN_STEPS = 300


def train_lora(model_name, model_id):
    out = Path(OUTPUT_DIR) / model_name
    out.mkdir(parents=True, exist_ok=True)

    script = {
        "sd15": "diffusers/examples/dreambooth/train_dreambooth_lora.py",
        "lumina2": "diffusers/examples/dreambooth/train_dreambooth_lora_lumina2.py",
        "sana": "diffusers/examples/dreambooth/train_dreambooth_lora_sana.py",
    }[model_name]

    cmd = [
        "accelerate",
        "launch",
        script,
        f"--pretrained_model_name_or_path={model_id}",
        f"--instance_data_dir={HORSE_DATASET}",
        f"--instance_prompt={PROMPT}",
        f"--output_dir={out}/lora",
        f"--rank={LORA_RANK}",
        f"--max_train_steps={TRAIN_STEPS}",
        "--train_batch_size=1",
        f"--mixed_precision={'bf16' if model_name in ('sana', 'lumina2') else 'fp16'}",
        "--gradient_checkpointing",
        "--report_to=wandb",
    ]

    subprocess.run(cmd, check=True)


def generate_images(model_name, model_id):
    from diffusers import AutoPipelineForText2Image

    lora_path = Path(OUTPUT_DIR) / model_name / "lora"
    gen_path = Path(OUTPUT_DIR) / model_name / "generated"
    gen_path.mkdir(parents=True, exist_ok=True)

    dtype = torch.bfloat16 if model_name in ("sana", "lumina2") else torch.float16
    pipe = AutoPipelineForText2Image.from_pretrained(
        model_id, torch_dtype=dtype
    ).to("cuda")

    if model_name == "sana":
        pipe.vae.to(torch.bfloat16)

    pipe.load_lora_weights(str(lora_path))

    for i in range(NUM_IMAGES):
        image = pipe(PROMPT, num_inference_steps=25).images[0]
        image.save(gen_path / f"{i:04d}.png")

    pipe = None
    torch.cuda.empty_cache()


def compute_metrics(model_name):
    gen_path = str(Path(OUTPUT_DIR) / model_name / "generated")
    real_path = str(Path(HORSE_DATASET))

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


if __name__ == "__main__":
    wandb.init(project="stage1-model-selection")

    results = []
    for model_name, model_id in MODELS.items():
        train_lora(model_name, model_id)
        generate_images(model_name, model_id)
        results.append(compute_metrics(model_name))

    table = wandb.Table(
        columns=["model", "FID", "IS"],
        data=[[r["model"], r["FID"], r["IS"]] for r in results],
    )
    wandb.log({"results": table})

    best = min(results, key=lambda x: x["FID"])
    wandb.summary["best_model"] = best["model"]
    wandb.summary["best_fid"] = best["FID"]

    wandb.finish()
