from functools import partial
import jax
import jax.numpy as jnp
from flax import linen as nn


class SequenceBlock(nn.Module):
    layer: nn.Module
    seq_len: int
    dropout: float
    d_model: int
    decode: bool = False

    def setup(self):
        self.seq = self.layer(seq_len=self.seq_len, decode=self.decode)
        self.norm = nn.LayerNorm()
        self.out = nn.Dense(self.d_model)
        self.drop = nn.Dropout(self.dropout, broadcast_dims=[0])

    def __call__(self, x, training):
        x2 = self.seq(x)
        drop = partial(self.drop, deterministic=not training)
        z = drop(self.out(drop(nn.gelu(x2))))
        return self.norm(z + x)


class StackedModel(nn.Module):
    layer: nn.Module
    d_output: int
    d_model: int
    seq_len: int
    n_layers: int
    dropout: float = 0.2
    classification: bool = False
    decode: bool = False

    def setup(self):
        self.encoder = nn.Dense(self.d_model)
        self.decoder = nn.Dense(self.d_output)
        self.layers = [
            SequenceBlock(
                layer=self.layer,
                d_model=self.d_model,
                dropout=self.dropout,
                seq_len=self.seq_len,
                decode=self.decode,
            )
            for _ in range(self.n_layers)
        ]

    def __call__(self, x, training=True):
        x = self.encoder(x)
        for layer in self.layers:
            x = layer(x, training)
        if self.classification:
            x = jnp.mean(x, axis=0)
        x = self.decoder(x)
        # return nn.log_softmax(x, axis=-1)
        return x # for continuous prediction


BatchStackedModel = nn.vmap(
    StackedModel,
    in_axes=0,
    out_axes=0,
    variable_axes={"params": None, "dropout": None, "cache": 0, "prime": None},
    split_rngs={"params": False, "dropout": True},
)
