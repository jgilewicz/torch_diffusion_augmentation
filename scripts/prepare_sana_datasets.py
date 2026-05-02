from __future__ import annotations

import hydra
from omegaconf import DictConfig

from pnw.common import load_env_file
from pnw.data_prep import prepare_sana_datasets


load_env_file()


@hydra.main(version_base=None, config_path="../configs", config_name="prepare_sana_datasets")
def main(cfg: DictConfig) -> None:
    prepare_sana_datasets(cfg)


if __name__ == "__main__":
    main()
