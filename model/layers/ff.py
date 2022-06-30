from flax import linen as nn

class FF(nn.Module):
    d_model: int
    seq_len: int

    def setup(self):
        self.dense = nn.Dense(self.d_model)

    def __call__(self, x):
        """x - L x N"""
        return nn.relu(self.dense(x))


def FFInit(unused):
    return FF
