from __future__ import annotations

import hydra
from omegaconf import DictConfig

from pnw.common import load_env_file
from pnw.data_prep import prepare_resnet_data


load_env_file()


@hydra.main(version_base=None, config_path="../configs", config_name="prepare_resnet_data")
def main(cfg: DictConfig) -> None:
    prepare_resnet_data(cfg)


if __name__ == "__main__":
    main()
