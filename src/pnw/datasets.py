from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image
from torch.utils.data import Dataset

from pnw.common import IMAGE_EXTENSIONS


class ImageFolderDataset(Dataset):
    def __init__(self, root_dir: str | Path, transform: Any = None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.images: list[Path] = []
        self.labels: list[int] = []
        self.class_to_idx: dict[str, int] = {}

        class_dirs = sorted(path for path in self.root_dir.iterdir() if path.is_dir())
        for idx, class_dir in enumerate(class_dirs):
            self.class_to_idx[class_dir.name] = idx

        for class_dir in class_dirs:
            class_idx = self.class_to_idx[class_dir.name]
            for image_path in sorted(class_dir.iterdir()):
                if image_path.suffix.lower() in IMAGE_EXTENSIONS:
                    self.images.append(image_path)
                    self.labels.append(class_idx)

        if not self.images:
            raise ValueError(f"No images found in {self.root_dir}")

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        image = Image.open(self.images[idx]).convert("RGB")
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


def class_count(dataset: ImageFolderDataset) -> int:
    return len(dataset.class_to_idx)

