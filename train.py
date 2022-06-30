from typing import Any, Callable, Optional, Tuple, Mapping
from functools import partial
import jax
from jax import numpy as jnp
import flax
import optax
from flax import linen as nn
from flax.training.train_state import TrainState
from torch.utils import data
from ml_collections import FrozenConfigDict

from dataset import get_dataset_dataloader
from model import create_model
from util import types
from util.util import set_seed


def map_nested_fn(fn):
    """Recursively apply `fn to the key-value pairs of a nested dict / pytree."""

    def map_fn(nested_dict):
        return {
            k: (map_fn(v) if hasattr(v, "keys") else fn(k, v))
            for k, v in nested_dict.items()
        }

    return map_fn


def create_train_state_and_model(
        cfg: FrozenConfigDict,
        key: types.PRNGKey,
        dataloader: data.DataLoader,
        total_steps=-1,
):

    sample_state, sample_action, _ = next(iter(dataloader))
    batch_size, seq_len, in_dim = sample_state.shape
    batch_size, seq_len, out_dim = sample_action.shape

    model = create_model(cfg, out_dim, seq_len)

    key_init, key_dropout = jax.random.split(key, num=2)
    params = model.init(
        {"params": key_init, "dropout": key_dropout},
        jnp.ones((batch_size, seq_len, in_dim)),
    )[
        "params"
    ].unfreeze()  # Note: Added immediate `unfreeze()` to play well w/ Optax. See below!

    lr = cfg.TRAIN.LR
    lr_schedule = cfg.TRAIN.LR_SCHEDULE

    # Implement LR Schedule (No change for first 30% of training, then decay w/ cubic polynomial to 0 for last 70%)
    if lr_schedule:
        lr = optax.polynomial_schedule(
            init_value=lr,
            end_value=0.0,
            power=3,
            transition_begin=int(0.3 * total_steps),
            transition_steps=int(0.7 * total_steps),
        )

    model_name = cfg.MODEL.MODEL_NAME

    # # S4 uses a Fixed LR = 1e-3 with NO weight decay for the S4 Matrices, higher LR elsewhere
    if "s4" in model_name or "dss" in model_name:
        # Note for Debugging... this is all undocumented and so weird. The following links are helpful...
        #
        #   > Flax "Recommended" interplay w/ Optax (this bridge needs ironing):
        #       https://github.com/google/flax/blob/main/docs/flip/1009-optimizer-api.md#multi-optimizer
        #
        #   > But... masking doesn't work like the above example suggests!
        #       Root Explanation: https://github.com/deepmind/optax/issues/159
        #       Fix: https://github.com/deepmind/optax/discussions/167
        #
        #   > Also... Flax FrozenDict doesn't play well with rest of Jax + Optax...
        #       https://github.com/deepmind/optax/issues/160#issuecomment-896460796
        #
        #   > Solution: Use Optax.multi_transform!
        s4_fn = map_nested_fn(
            lambda k, _: "s4"
            if k in ["B", "Ct", "D", "log_step", "W"]
            else ("none" if k in [] else "regular")
        )
        tx = optax.multi_transform(
            {
                "none": optax.sgd(learning_rate=0.0),
                "s4": optax.adam(learning_rate=1e-3),
                "regular": optax.adamw(learning_rate=lr, weight_decay=0.01),
            },
            s4_fn,
        )

    else:
        tx = optax.adamw(learning_rate=lr, weight_decay=0.01)

    return TrainState.create(apply_fn=model.apply, params=params, tx=tx), model


@partial(jax.jit, static_argnums=(1, ))
def train_step(
        train_state, model, key, states, actions, masks,
):
    def loss_fn(params):
        predictions, mod_vars = model.apply(
            {"params": params},
            states,
            rngs={"dropout": key},
            mutable=["intermediates"],
        )
        loss = jnp.mean(jnp.linalg.norm(predictions - actions) * masks)
        return loss

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(train_state.params)
    train_state = train_state.apply_gradients(grads=grads)
    return train_state, loss


def train_epoch(train_state, model, key, trainloader):
    # Store Metrics
    batch_losses = []
    for batch_idx, (states, actions, masks) in enumerate(trainloader):
        states = jnp.array(states)
        actions = jnp.array(actions)
        masks = jnp.array(masks)
        key, key_dropout = jax.random.split(key)
        train_state, loss = train_step(
            train_state,
            model,
            key_dropout,
            states,
            actions,
            masks,
        )
        batch_losses.append(loss)

    # Return average loss over batches
    return train_state, jnp.mean(jnp.array(batch_losses))


def train(
        cfg: FrozenConfigDict,
        progress_fn: Callable[[int, types.Metrics], None] = lambda *args: None,
):
    seed = cfg.SEED
    set_seed(seed) # torch and python
    key = jax.random.PRNGKey(seed)
    key, key_model, key_train = jax.random.split(key, 3)

    epochs = cfg.TRAIN.EPOCHS

    dataset, dataloader = get_dataset_dataloader(cfg)

    train_state, model = create_train_state_and_model(
        cfg, key_model, dataloader,
        total_steps=len(dataloader) * epochs)

    for epoch in range(epochs):
        print(f"[*] Starting Training Epoch {epoch + 1}...")
        key_train = jax.random.fold_in(key_train, epoch)
        state, train_loss = train_epoch(
            train_state, model, key_train, dataloader)

        metrics = {
            "train_loss": train_loss,
        }
        progress_fn(len(dataloader) * (epoch + 1), metrics)
