from functools import partial
import jax
import jax.numpy as jnp
from flax import linen as nn
from jax.nn.initializers import lecun_normal, uniform
from model.util import *


class DSS(nn.Module):
    Lambda: jnp.DeviceArray
    N: int
    l_max: int
    decode: bool = False

    def setup(self):
        # Learned Parameters
        self.W = self.param("W", lecun_normal(), (1, self.N, 2))
        self.W = self.W[..., 0] + 1j * self.W[..., 1]
        self.D = self.param("D", nn.initializers.ones, (1,))
        self.step = jnp.exp(
            self.param("log_step", log_step_initializer(), (1,))
        )
        if not self.decode:
            self.K = dss_kernel(self.W, self.Lambda, self.l_max, self.step)
        else:
            # FLAX code to ensure that we only compute discrete once during decoding.
            def init_discrete():
                return dss_ssm(self.W, self.Lambda, self.l_max, self.step)
            ssm_var = self.variable("prime", "ssm", init_discrete)
            if self.is_mutable_collection("prime"):
                ssm_var.value = init_discrete()
            self.ssm = ssm_var.value

            # RNN Cache
            self.x_k_1 = self.variable(
                "cache", "cache_x_k", jnp.zeros, (self.N,), jnp.complex64
            )

    def __call__(self, u):
        if not self.decode:
            return non_circular_convolution(u, self.K) + self.D * u
        else:
            x_k, y_s = scan_SSM(*self.ssm, u[:, jnp.newaxis], self.x_k_1.value)
            if self.is_mutable_collection("cache"):
                self.x_k_1.value = x_k
            return y_s.reshape(-1).real + self.D * u


def DSSInit(N):
    _, Lambda, _, _, _ = make_NPLR_HiPPO(2 * N)
    Lambda = Lambda[jnp.nonzero(Lambda.imag > 0, size=N)]
    return partial(clone_layer(DSS), N=N, Lambda=Lambda)
