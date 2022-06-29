from ml_collections import FrozenConfigDict
from .d4rl import get_d4lr_dataset_dataloader


_ds_registry = {
    "d4rl": get_d4lr_dataset_dataloader,
}


def get_dataset_dataloader(cfg: FrozenConfigDict):
    return _ds_registry[cfg.DATA.DS_NAME.split("/")[0]](cfg)
