from __future__ import annotations

import random
from collections.abc import Iterable
import os
from pathlib import Path

import numpy as np
import torch


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def path(value: str | Path) -> Path:
    return Path(value).expanduser()


def load_env_file(env_path: str | Path = ".env") -> None:
    env_file = Path(env_path)
    if not env_file.exists():
        return

    for raw_line in env_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def torch_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def diffusion_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def clean_name(name: str) -> str:
    return "_".join(name.replace("/", "_").replace("\\", "_").split())


def require_existing(paths: Iterable[Path], hint: str) -> None:
    missing = [item for item in paths if not item.exists()]
    if missing:
        missing_text = ", ".join(str(item) for item in missing)
        raise FileNotFoundError(f"Missing required paths: {missing_text}. {hint}")
