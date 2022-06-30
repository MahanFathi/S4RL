import os
from typing import Union, Mapping

from absl import app
from absl import flags
from absl import logging

import wandb
import jax
from jax.config import config
import ml_collections
from ml_collections.config_flags import config_flags

from model import create_model
from util import logger


_CONFIG = config_flags.DEFINE_config_file('cfg', './config/defaults.py')


def main(argv):
    del argv

    # freeze the config
    cfg = ml_collections.FrozenConfigDict(_CONFIG.value)
    logging.info("configuration: \n{}".format(cfg))

    if cfg.DEBUG:
        logging.set_verbosity(logging.DEBUG)
        config.update("jax_debug_nans", True)
        if cfg.MOCK_TPU:
            os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=8'
            jax.devices()

    logging.info(logger.get_logdir_path(cfg))

    def progress_fn(num_steps: int, metrics: Mapping[str, Union[int, float]]):
        logger.log_metrics(cfg, num_steps, metrics)

    train(cfg, progress_fn)
    wandb.finish()


if __name__ == '__main__':
  app.run(main)
