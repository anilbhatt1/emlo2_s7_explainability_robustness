import rootutils

root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from typing import Any, Dict, List, Optional, Tuple
import hydra
import lightning as L
import rootutils
import torch
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig
from src import utils

@hydra.main(version_base="1.3", config_path="../configs", config_name="test_target.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    print(f'cfg is {cfg}')

    dm = hydra.utils.instantiate(cfg.data)
    print('\n')
    print(f'dm is {dm}')
    print(f'dm with new parm is {dm(new_parm="new")}')
    print(f'dm with new parm is {dm(new_parm2="new2")}')

if __name__ == "__main__":
    main()
