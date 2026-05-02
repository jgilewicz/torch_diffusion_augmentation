from __future__ import annotations

import json
import random
from pathlib import Path
from shutil import copy2, rmtree

from datasets import ClassLabel, load_dataset
from omegaconf import DictConfig
from PIL import Image, ImageEnhance, ImageOps

from pnw.common import IMAGE_EXTENSIONS, clean_name, path


def _class_name(dataset, label_column: str, label) -> str:
    feature = dataset.features[label_column]
    if isinstance(feature, ClassLabel):
        return clean_name(feature.int2str(int(label)))
    return clean_name(str(label))


def _target_reached(counts: dict[str, int], num_classes: int, samples_per_class: int) -> bool:
    return len(counts) == num_classes and all(
        count >= samples_per_class for count in counts.values()
    )


def download_datasets(cfg: DictConfig) -> None:
    data_dir = path(cfg.data_dir)
    cache_dir = data_dir / cfg.cache_subdir
    samples_per_class = int(cfg.samples_per_class)
    data_dir.mkdir(exist_ok=True)

    try:
        for name, dataset_cfg in cfg.datasets.items():
            print(f"Preparing {name}")
            output_dir = data_dir / name
            if output_dir.exists():
                rmtree(output_dir)
            output_dir.mkdir(parents=True)

            splits = list(dataset_cfg.splits)
            first_split = load_dataset(
                dataset_cfg.repo_id,
                split=splits[0],
                cache_dir=str(cache_dir),
            )
            num_classes = first_split.features[dataset_cfg.label_column].num_classes
            counts: dict[str, int] = {}

            for split in splits:
                if _target_reached(counts, num_classes, samples_per_class):
                    break

                dataset = (
                    first_split
                    if split == splits[0]
                    else load_dataset(dataset_cfg.repo_id, split=split, cache_dir=str(cache_dir))
                )
                print(f"  {split}")

                for row in dataset:
                    label = _class_name(dataset, dataset_cfg.label_column, row[dataset_cfg.label_column])
                    if counts.get(label, 0) >= samples_per_class:
                        continue

                    class_dir = output_dir / label
                    class_dir.mkdir(exist_ok=True)
                    counts[label] = counts.get(label, 0) + 1
                    image = row[dataset_cfg.image_column].convert("RGB")
                    image.save(class_dir / f"{counts[label]:04d}.png")

                    if _target_reached(counts, num_classes, samples_per_class):
                        break

            short = {label: count for label, count in counts.items() if count < samples_per_class}
            if short:
                print(f"  saved fewer than {samples_per_class} for {len(short)} classes")
    finally:
        for artifact in cfg.artifact_subdirs:
            artifact_path = data_dir / artifact
            if artifact_path.exists():
                rmtree(artifact_path)


def _caption_class_name(class_dir_name: str) -> str:
    return class_dir_name.replace("_", " ").replace("-", " ").strip()


def _article_for(name: str) -> str:
    return "an" if name[:1].lower() in {"a", "e", "i", "o", "u"} else "a"


def prepare_sana_datasets(cfg: DictConfig) -> None:
    data_dir = path(cfg.data_dir)
    for dataset_name, template in cfg.caption_templates.items():
        source_dir = data_dir / dataset_name
        output_train_dir = data_dir / f"{cfg.output_prefix}{dataset_name}" / "train"

        if not source_dir.exists():
            print(f"Skipping {dataset_name}: {source_dir} does not exist")
            continue

        if output_train_dir.parent.exists():
            rmtree(output_train_dir.parent)
        output_train_dir.mkdir(parents=True)

        metadata_path = output_train_dir / "metadata.jsonl"
        written = 0
        with metadata_path.open("w", encoding="utf-8") as metadata_file:
            for class_dir in sorted(item for item in source_dir.iterdir() if item.is_dir()):
                class_name = _caption_class_name(class_dir.name)
                caption = template.format(
                    article=_article_for(class_name),
                    class_name=class_name,
                )

                for index, image_path in enumerate(sorted(class_dir.iterdir()), start=1):
                    if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                        continue

                    output_name = f"{class_dir.name}_{index:04d}{image_path.suffix.lower()}"
                    copy2(image_path, output_train_dir / output_name)
                    metadata_file.write(json.dumps({"file_name": output_name, "text": caption}) + "\n")
                    written += 1

        print(f"Prepared {output_train_dir.parent} ({written} images)")


def _copy_dataset_original(source_dir: Path, target_dir: Path, max_per_class: int | None = None) -> int:
    target_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    if not source_dir.exists():
        return count

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


def _augment_image(image: Image.Image, rng: random.Random) -> Image.Image:
    augmented = image.convert("RGB")
    if rng.random() < 0.5:
        augmented = ImageOps.mirror(augmented)

    augmented = augmented.rotate(rng.uniform(-15, 15), resample=Image.Resampling.BICUBIC)
    augmented = ImageEnhance.Brightness(augmented).enhance(rng.uniform(0.8, 1.2))
    augmented = ImageEnhance.Contrast(augmented).enhance(rng.uniform(0.8, 1.2))
    augmented = ImageEnhance.Color(augmented).enhance(rng.uniform(0.8, 1.2))
    return augmented


def _prepare_augmented_dataset(
    source_dir: Path,
    target_dir: Path,
    samples_per_class: int,
    include_original: bool,
    seed: int,
) -> tuple[int, int]:
    original_count = _copy_dataset_original(source_dir, target_dir) if include_original else 0
    augmented_count = 0
    rng = random.Random(seed)

    for class_dir in sorted(path for path in source_dir.iterdir() if path.is_dir()):
        source_images = [
            image_path
            for image_path in sorted(class_dir.iterdir())
            if image_path.suffix.lower() in IMAGE_EXTENSIONS
        ]
        if not source_images:
            continue

        class_target = target_dir / class_dir.name
        class_target.mkdir(parents=True, exist_ok=True)

        for index in range(samples_per_class):
            image_path = source_images[index % len(source_images)]
            with Image.open(image_path) as image:
                augmented = _augment_image(image, rng)
            augmented.save(class_target / f"augmented_{index:05d}.png")
            augmented_count += 1

    return original_count, augmented_count


def _setup_generated_folders(source_dir: Path, target_dir: Path) -> int:
    target_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for class_dir in sorted(source_dir.iterdir()):
        if class_dir.is_dir():
            (target_dir / class_dir.name).mkdir(exist_ok=True)
            count += 1
    return count


def _prepare_generated_dataset(source_dir: Path, target_dir: Path, include_original: bool) -> int:
    if include_original:
        return _copy_dataset_original(source_dir, target_dir)
    return _setup_generated_folders(source_dir, target_dir)


def prepare_resnet_data(cfg: DictConfig) -> None:
    data_dir = path(cfg.data_dir)
    resnet_data_dir = path(cfg.resnet_data_dir)
    resnet_data_dir.mkdir(exist_ok=True)

    for dataset_name in cfg.datasets:
        source_dir = data_dir / dataset_name
        if not source_dir.exists():
            print(f"Skipping {dataset_name}: {source_dir} does not exist")
            continue

        dataset_resnet_dir = resnet_data_dir / dataset_name
        print(f"\nPreparing {dataset_name}...")

        original_dir = dataset_resnet_dir / "original"
        if original_dir.exists():
            rmtree(original_dir)
        original_count = _copy_dataset_original(source_dir, original_dir)
        print(f"  Original: copied {original_count} images")

        augmented_dir = dataset_resnet_dir / "augmented"
        if augmented_dir.exists():
            rmtree(augmented_dir)
        augmented_original_count, augmented_count = _prepare_augmented_dataset(
            source_dir,
            augmented_dir,
            samples_per_class=int(cfg.augmented_samples_per_class),
            include_original=bool(cfg.augmented_includes_original),
            seed=int(cfg.random_seed),
        )
        print(
            f"  Augmented: copied {augmented_original_count} original images "
            f"and created {augmented_count} augmented images "
            f"({cfg.augmented_samples_per_class} per class)"
        )

        generated_dir = dataset_resnet_dir / "generated"
        if generated_dir.exists():
            rmtree(generated_dir)
        generated_count = _prepare_generated_dataset(
            source_dir,
            generated_dir,
            include_original=bool(cfg.generated_includes_original),
        )
        if cfg.generated_includes_original:
            print(
                f"  Generated: copied {generated_count} original images "
                "and prepared folders for synthetic images"
            )
        else:
            print(f"  Generated: created {generated_count} class folders")

    print(f"\nResNet data prepared in {resnet_data_dir}")
