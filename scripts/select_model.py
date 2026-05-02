from __future__ import annotations

import hydra
from omegaconf import DictConfig

from pnw.common import load_env_file
from pnw.model_selection.selection import run


load_env_file()


@hydra.main(version_base=None, config_path="../configs", config_name="model_selection")
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()
