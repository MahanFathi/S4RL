from functools import partial
import jax
import jax.numpy as jnp
from flax import linen as nn
from jax.nn.initializers import lecun_normal, uniform
from model.util import *


class SSM(nn.Module):
    A: jnp.DeviceArray  # HiPPO
    N: int
    l_max: int
    decode: bool = False

    def setup(self):
        # SSM parameters
        self.B = self.param("B", lecun_normal(), (self.N, 1))
        self.C = self.param("C", lecun_normal(), (1, self.N))
        self.D = self.param("D", nn.initializers.ones, (1,))

        # Step parameter
        self.log_step = self.param("log_step", log_step_initializer(), (1,))

        step = jnp.exp(self.log_step)
        self.ssm = discretize(self.A, self.B, self.C, step=step)
        self.K = K_conv(*self.ssm, self.l_max)

        # RNN cache for long sequences
        self.x_k_1 = self.variable("cache", "cache_x_k", jnp.zeros, (self.N,))

    def __call__(self, u):
        if not self.decode:
            # CNN Mode
            return non_circular_convolution(u, self.K) + self.D * u
        else:
            # RNN Mode
            x_k, y_s = scan_SSM(*self.ssm, u[:, jnp.newaxis], self.x_k_1.value)
            if self.is_mutable_collection("cache"):
                self.x_k_1.value = x_k
            return y_s.reshape(-1).real + self.D * u


def SSMInit(N):
    return partial(clone_layer(SSM), A=make_HiPPO(N), N=N)
