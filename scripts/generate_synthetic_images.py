from __future__ import annotations

from pathlib import Path

import torch
from diffusers import DiffusionPipeline
from tqdm import tqdm


BASE_MODEL = "Efficient-Large-Model/Sana_1600M_1024px_BF16_diffusers"


def get_device() -> str:
    """Auto-detect best available device."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


RESNET_DATA_DIR = Path("resnet_data")

DATASETS_CONFIG = {
    "cifar10": {
        "lora_id": "W1ndrunn3rr/SANA-LoRA-CIFAR10",
        "classes": [
            "airplane",
            "automobile",
            "bird",
            "cat",
            "deer",
            "dog",
            "frog",
            "horse",
            "ship",
            "truck",
        ],
        "prompt_template": "a low-resolution photo of a {class_name}, CIFAR-10 style",
        "images_per_class": 20,
    },
    "eurosat_rgb": {
        "lora_id": "W1ndrunn3rr/SANA-LoRA-EuroSAT-RGB",
        "classes": [
            "AnnualCrop",
            "Forest",
            "HerbaceousVegetation",
            "Highway",
            "Industrial",
            "Pasture",
            "PermanentCrop",
            "Residential",
            "River",
            "SeaLake",
        ],
        "prompt_template": "a satellite image of {class_name}, EuroSAT style",
        "images_per_class": 20,
    },
    "beans": {
        "lora_id": "W1ndrunn3rr/SANA-LoRA-Beans",
        "classes": ["angular_leaf_spot", "bean_rust", "healthy"],
        "prompt_template": "a photo of a bean leaf with {class_name}",
        "images_per_class": 20,
    },
}


def generate_dataset_images(
    dataset_name: str,
    lora_id: str,
    classes: list[str],
    prompt_template: str,
    images_per_class: int,
    device: str,
) -> None:

    print(f"  Loading base model and LoRA from {lora_id}")
    pipe = DiffusionPipeline.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    pipe.load_lora_weights(lora_id)

    dataset_output_dir = RESNET_DATA_DIR / dataset_name / "generated"
    dataset_output_dir.mkdir(parents=True, exist_ok=True)

    for class_name in classes:
        class_dir = dataset_output_dir / class_name
        class_dir.mkdir(exist_ok=True)

        prompt = prompt_template.format(class_name=class_name)

        print(f"    {class_name}...", end=" ", flush=True)

        for i in tqdm(range(images_per_class), desc=f"{class_name}", leave=False):
            try:
                with torch.no_grad():
                    image = pipe(prompt).images[0]

                if dataset_name == "cifar10":
                    image = image.resize((32, 32))
                elif dataset_name == "eurosat_rgb":
                    image = image.resize((64, 64))
                elif dataset_name == "beans":
                    image = image.resize((224, 224))

                image.save(class_dir / f"{i:05d}.png")
            except Exception as e:
                print(f"\n    Error generating image {i} for {class_name}: {e}")
                continue

        print("✓")


def main() -> None:
    device = get_device()
    print(f"Using device: {device}")

    for dataset_name, config in DATASETS_CONFIG.items():
        print(f"\nGenerating {dataset_name}...")

        try:
            generate_dataset_images(
                dataset_name=dataset_name,
                lora_id=config["lora_id"],
                classes=config["classes"],
                prompt_template=config["prompt_template"],
                images_per_class=config["images_per_class"],
                device=device,
            )
        except Exception as e:
            print(f"  Error processing {dataset_name}: {e}")
            continue

    print(f"\nSynthetic images generated in {RESNET_DATA_DIR}")


if __name__ == "__main__":
    main()
