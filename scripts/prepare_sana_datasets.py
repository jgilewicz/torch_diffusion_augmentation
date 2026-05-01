#!/usr/bin/env python3
"""Create captioned ImageFolder datasets for SANA LoRA fine-tuning."""

from __future__ import annotations

import json
from pathlib import Path
from shutil import copy2, rmtree


DATA_DIR = Path("data")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

CAPTION_TEMPLATES = {
    "cifar10": "a low-resolution photo of {article} {class_name}, CIFAR-10 style",
    "eurosat_rgb": "a satellite image of {class_name}, EuroSAT style",
    "beans": "a photo of a bean leaf with {class_name}",
}


def caption_class_name(class_dir_name: str) -> str:
    return class_dir_name.replace("_", " ").replace("-", " ").strip()


def article_for(name: str) -> str:
    return "an" if name[:1].lower() in {"a", "e", "i", "o", "u"} else "a"


def prepare_dataset(dataset_name: str, template: str) -> None:
    source_dir = DATA_DIR / dataset_name
    output_train_dir = DATA_DIR / f"sana_{dataset_name}" / "train"

    if not source_dir.exists():
        print(f"Skipping {dataset_name}: {source_dir} does not exist")
        return

    if output_train_dir.parent.exists():
        rmtree(output_train_dir.parent)
    output_train_dir.mkdir(parents=True)

    metadata_path = output_train_dir / "metadata.jsonl"
    written = 0

    with metadata_path.open("w", encoding="utf-8") as metadata_file:
        for class_dir in sorted(path for path in source_dir.iterdir() if path.is_dir()):
            class_name = caption_class_name(class_dir.name)
            caption = template.format(
                article=article_for(class_name),
                class_name=class_name,
            )

            for index, image_path in enumerate(sorted(class_dir.iterdir()), start=1):
                if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue

                output_name = f"{class_dir.name}_{index:04d}{image_path.suffix.lower()}"
                copy2(image_path, output_train_dir / output_name)
                metadata_file.write(
                    json.dumps({"file_name": output_name, "text": caption}) + "\n"
                )
                written += 1

    print(f"Prepared {DATA_DIR / f'sana_{dataset_name}'} ({written} images)")


def main() -> None:
    for dataset_name, template in CAPTION_TEMPLATES.items():
        prepare_dataset(dataset_name, template)


if __name__ == "__main__":
    main()
