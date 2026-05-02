from __future__ import annotations

import hydra
from omegaconf import DictConfig

from pnw.common import load_env_file
from pnw.resnet_training.train import run


load_env_file()


@hydra.main(version_base=None, config_path="../configs", config_name="resnet_train")
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()
