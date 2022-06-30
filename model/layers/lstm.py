import jax
from flax import linen as nn

class LSTM(nn.Module):
    d_model: int
    seq_len: int

    def setup(self):
        LSTM = nn.scan(
            nn.OptimizedLSTMCell,
            in_axes=0,
            out_axes=0,
            variable_broadcast="params",
            split_rngs={"params": False},
        )
        dummy_rng = jax.random.PRNGKey(0)
        self.init_h = nn.OptimizedLSTMCell.initialize_carry(
            dummy_rng, (), self.d_model
        )
        self.LSTM = LSTM(name="lstm_cell")

    def __call__(self, xs):
        return self.LSTM(self.init_h, xs)[1]


def LSTMInit(unused):
    return LSTM
