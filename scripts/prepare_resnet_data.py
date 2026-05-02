from __future__ import annotations

import random
from pathlib import Path
from shutil import copy2, rmtree


DATA_DIR = Path("data")
RESNET_DATA_DIR = Path("resnet_data")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

DATASETS = ["cifar10", "eurosat_rgb", "beans"]
AUGMENTED_SAMPLES_PER_CLASS = 20


def copy_dataset_original(
    dataset_name: str,
    source_dir: Path,
    target_dir: Path,
    max_per_class: int | None = None,
) -> int:
    target_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    if source_dir.exists():
        for class_dir in sorted(source_dir.iterdir()):
            if not class_dir.is_dir():
                continue

            class_target = target_dir / class_dir.name
            class_target.mkdir(exist_ok=True)

            class_count = 0
            for image_path in sorted(class_dir.iterdir()):
                if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue
                if max_per_class is not None and class_count >= max_per_class:
                    break

                copy2(image_path, class_target / image_path.name)
                count += 1
                class_count += 1

    return count


def setup_generated_folders(source_dir: Path, target_dir: Path) -> int:
    target_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for class_dir in sorted(source_dir.iterdir()):
        if class_dir.is_dir():
            class_target = target_dir / class_dir.name
            class_target.mkdir(exist_ok=True)
            count += 1

    return count


def prepare_resnet_data() -> None:

    for dataset_name in DATASETS:
        source_dir = DATA_DIR / dataset_name

        if not source_dir.exists():
            print(f"Skipping {dataset_name}: {source_dir} does not exist")
            continue

        dataset_resnet_dir = RESNET_DATA_DIR / dataset_name

        print(f"\nPreparing {dataset_name}...")

        original_dir = dataset_resnet_dir / "original"
        if original_dir.exists():
            rmtree(original_dir)
        original_count = copy_dataset_original(dataset_name, source_dir, original_dir)
        print(f"  Original: copied {original_count} images")

        augmented_dir = dataset_resnet_dir / "augmented"
        if augmented_dir.exists():
            rmtree(augmented_dir)
        augmented_count = copy_dataset_original(
            dataset_name,
            source_dir,
            augmented_dir,
            max_per_class=AUGMENTED_SAMPLES_PER_CLASS,
        )
        print(
            f"  Augmented: copied {augmented_count} images "
            f"({AUGMENTED_SAMPLES_PER_CLASS} per class, transforms applied during training)"
        )

        generated_dir = dataset_resnet_dir / "generated"
        if generated_dir.exists():
            rmtree(generated_dir)
        class_count = setup_generated_folders(source_dir, generated_dir)
        print(
            f"  Generated: created {class_count} class folders (empty, ready for generation)"
        )


def main() -> None:
    RESNET_DATA_DIR.mkdir(exist_ok=True)
    random.seed(42)
    prepare_resnet_data()
    print(f"\n Resnet data prepared in {RESNET_DATA_DIR}")
    print("   - original/  : Original images (no transforms)")
    print("   - augmented/ : Original images (transforms applied during training)")
    print("   - generated/ : Generated images via LoRA")


if __name__ == "__main__":
    main()
