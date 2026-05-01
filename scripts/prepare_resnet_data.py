from __future__ import annotations

import random
from pathlib import Path
from shutil import copy2, rmtree

from torchvision import transforms
from PIL import Image


DATA_DIR = Path("data")
RESNET_DATA_DIR = Path("resnet_data")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

DATASETS = ["cifar10", "eurosat_rgb", "beans"]

augmentation_transforms = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    ]
)


def copy_dataset_normal(dataset_name: str, source_dir: Path, target_dir: Path) -> int:
    target_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    if source_dir.exists():
        for class_dir in sorted(source_dir.iterdir()):
            if not class_dir.is_dir():
                continue

            class_target = target_dir / class_dir.name
            class_target.mkdir(exist_ok=True)

            for image_path in sorted(class_dir.iterdir()):
                if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue

                copy2(image_path, class_target / image_path.name)
                count += 1

    return count


def augment_dataset(
    source_dir: Path, target_dir: Path, images_per_class: int = 200
) -> int:
    target_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for class_dir in sorted(source_dir.iterdir()):
        if not class_dir.is_dir():
            continue

        class_target = target_dir / class_dir.name
        class_target.mkdir(exist_ok=True)

        original_count = 0
        for image_path in sorted(class_dir.iterdir()):
            if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            original_image = Image.open(image_path)
            original_image.save(class_target / image_path.name)
            original_count += 1
            count += 1

        augmented_needed = max(0, images_per_class - original_count)
        for i in range(augmented_needed):
            image_path = random.choice(list(class_dir.glob("*")))
            if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            image = Image.open(image_path)
            if image.mode != "RGB":
                image = image.convert("RGB")

            augmented = augmentation_transforms(image)
            output_name = f"aug_{i:05d}_{image_path.stem}.png"
            augmented.save(class_target / output_name)
            count += 1

    return count


def setup_diffused_folders(source_dir: Path, target_dir: Path) -> int:
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

        normal_dir = dataset_resnet_dir / "normal"
        if normal_dir.exists():
            rmtree(normal_dir)
        normal_count = copy_dataset_normal(dataset_name, source_dir, normal_dir)
        print(f"  Normal: copied {normal_count} images")

        augmented_dir = dataset_resnet_dir / "augmented"
        if augmented_dir.exists():
            rmtree(augmented_dir)
        augmented_count = augment_dataset(
            source_dir, augmented_dir, images_per_class=200
        )
        print(f"  Augmented: created {augmented_count} images")

        diffused_dir = dataset_resnet_dir / "diffused"
        if diffused_dir.exists():
            rmtree(diffused_dir)
        class_count = setup_diffused_folders(source_dir, diffused_dir)
        print(
            f"  Diffused: created {class_count} class folders (empty, ready for generation)"
        )


def main() -> None:
    RESNET_DATA_DIR.mkdir(exist_ok=True)
    random.seed(42)
    prepare_resnet_data()
    print(f"\nResnet data prepared in {RESNET_DATA_DIR}")


if __name__ == "__main__":
    main()
