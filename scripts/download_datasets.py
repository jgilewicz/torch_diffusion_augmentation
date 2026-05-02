from pathlib import Path
from shutil import rmtree

from datasets import ClassLabel, load_dataset


DATA_DIR = Path("data")
CACHE_DIR = DATA_DIR / ".hf_cache"
ARTIFACT_DIRS = [CACHE_DIR, DATA_DIR / "hf"]
SAMPLES_PER_CLASS = 100

DATASETS = {
    "cifar10": ("uoft-cs/cifar10", "img", "label", ["train"]),
    "eurosat_rgb": (
        "giswqs/EuroSAT_RGB",
        "image",
        "label",
        ["train"],
    ),
    "beans": ("beans", "image", "labels", ["train"]),
}


def clean_name(name: str) -> str:
    return "_".join(name.replace("/", "_").replace("\\", "_").split())


def class_name(dataset, label_column: str, label) -> str:
    feature = dataset.features[label_column]
    if isinstance(feature, ClassLabel):
        return clean_name(feature.int2str(int(label)))
    return clean_name(str(label))


def target_reached(counts: dict[str, int], num_classes: int) -> bool:
    return len(counts) == num_classes and all(
        count >= SAMPLES_PER_CLASS for count in counts.values()
    )


def save_dataset(
    name: str, repo_id: str, image_column: str, label_column: str, splits: list[str]
) -> None:
    output_dir = DATA_DIR / name
    if output_dir.exists():
        rmtree(output_dir)
    output_dir.mkdir(parents=True)

    counts: dict[str, int] = {}
    first_split = load_dataset(repo_id, split=splits[0], cache_dir=str(CACHE_DIR))
    num_classes = first_split.features[label_column].num_classes

    for split in splits:
        if target_reached(counts, num_classes):
            break

        dataset = (
            first_split
            if split == splits[0]
            else load_dataset(repo_id, split=split, cache_dir=str(CACHE_DIR))
        )
        print(f"  {split}")

        for row in dataset:
            label = class_name(dataset, label_column, row[label_column])
            if counts.get(label, 0) >= SAMPLES_PER_CLASS:
                continue

            class_dir = output_dir / label
            class_dir.mkdir(exist_ok=True)

            counts[label] = counts.get(label, 0) + 1
            image = row[image_column].convert("RGB")
            image.save(class_dir / f"{counts[label]:04d}.png")

            if target_reached(counts, num_classes):
                break

    short = {
        label: count for label, count in counts.items() if count < SAMPLES_PER_CLASS
    }
    if short:
        print(f"  saved fewer than {SAMPLES_PER_CLASS} for {len(short)} classes")


def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)

    try:
        for name, config in DATASETS.items():
            print(f"Preparing {name}")
            save_dataset(name, *config)
    finally:
        for path in ARTIFACT_DIRS:
            if path.exists():
                rmtree(path)


if __name__ == "__main__":
    main()
