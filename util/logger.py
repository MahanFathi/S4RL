from typing import Union, Mapping
from brax.training import types
import wandb
from flax.metrics import tensorboard
from ml_collections import FrozenConfigDict

from datetime import datetime
from pathlib import Path


LOG_NAME = None
LOG_PATH = None
TB_SUMMARY_WRITER = None


def get_log_name(cfg: FrozenConfigDict) -> str:
    global LOG_NAME
    if LOG_NAME:
        return LOG_NAME
    log_name = "{}_{}_{}".format(
        cfg.EXP_NAME,
        cfg.ENV.ENV_NAME,
        datetime.now().strftime("%Y.%m.%d_%H:%M:%S"),
     )
    return log_name


def get_logdir_path(cfg: FrozenConfigDict) -> Path:
    global LOG_PATH
    if LOG_PATH:
        return LOG_PATH
    log_name = get_log_name(cfg)
    log_path = Path("./logs").joinpath(log_name)
    print(log_path)
    log_path.mkdir(parents=True, exist_ok=True)
    LOG_PATH = log_path
    return log_path


def get_summary_writer(cfg: FrozenConfigDict) -> tensorboard.SummaryWriter:
    log_path = get_logdir_path(cfg)
    return tensorboard.SummaryWriter(str(log_path))


def init_wandb(cfg: FrozenConfigDict):
    wandb.init(
        project="S4RL",
        dir=get_logdir_path(cfg),
        name=get_log_name(cfg),
        config=cfg.to_dict(),
    )


def log_metrics(cfg: FrozenConfigDict, num_steps: int, metrics: Mapping[str, Union[int, float]]):
    global TB_SUMMARY_WRITER
    if not TB_SUMMARY_WRITER:
        TB_SUMMARY_WRITER = get_summary_writer(cfg)
        if cfg.WANDB:
            init_wandb(cfg)
    for key, value in metrics.items():
        TB_SUMMARY_WRITER.scalar(key, value, num_steps)
        metrics[key] = float(value)
    if cfg.WANDB:
        wandb.log(metrics, num_steps)


def save_params(params: types.Params, name: str, logdir: str = None):
    params_dir = logdir.joinpath("params")
    params_dir.mkdir(exist_ok=True)
    params_file = params_dir.joinpath("{}.flax".format(name))

    param_bytes = flax.serialization.to_bytes(params)

    with open(params_file, "wb") as f:
        f.write(param_bytes)
