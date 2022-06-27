from functools import partial
import jax
import jax.numpy as jnp
from flax import linen as nn
from jax.nn.initializers import lecun_normal, uniform
from model.util import *


class S4(nn.Module):
    A: jnp.DeviceArray
    Vc: jnp.DeviceArray
    p: jnp.DeviceArray
    q: jnp.DeviceArray
    Lambda: jnp.DeviceArray
    N: int
    l_max: int
    decode: bool = False

    def setup(self):
        # Learned Parameters (Ct is complex!)
        self.Ct = self.param("Ct", lecun_normal(), (1, self.N, 2))
        self.Ct = self.Ct[..., 0] + 1j * self.Ct[..., 1]
        self.B = self.Vc @ self.param("B", lecun_normal(), (self.N, 1))
        self.D = self.param("D", uniform(), (1,))
        self.step = jnp.exp(self.param("log_step", log_step_initializer(), (1,)))

        if not self.decode:
            # CNN mode, compute kernel.
            K_gen = K_gen_DPLR(
                self.Lambda,
                self.p,
                self.q,
                self.B,
                self.Ct,
                self.step[0],
                unmat=self.l_max > 1000,
            )
            self.K = conv_from_gen(K_gen, self.l_max)

        else:
            # RNN mode, discretize

            # Flax trick to cache discrete form during decoding.
            def init_discrete():
                return discrete_DPLR(
                    self.Lambda,
                    self.p,
                    self.q,
                    self.B,
                    self.Ct,
                    self.step[0],
                    self.l_max,
                )

            ssm_var = self.variable("prime", "ssm", init_discrete)
            if self.is_mutable_collection("prime"):
                ssm_var.value = init_discrete()
            self.ssm = ssm_var.value

            # RNN Cache
            self.x_k_1 = self.variable(
                "cache", "cache_x_k", jnp.zeros, (self.N,), jnp.complex64
            )

    def __call__(self, u):
        # This is identical to SSM Layer
        if not self.decode:
            # CNN Mode
            return non_circular_convolution(u, self.K) + self.D * u
        else:
            # RNN Mode
            x_k, y_s = scan_SSM(*self.ssm, u[:, jnp.newaxis], self.x_k_1.value)
            if self.is_mutable_collection("cache"):
                self.x_k_1.value = x_k
            return y_s.reshape(-1).real + self.D * u


def S4Init(N):
    _, Lambda, p, q, V = make_NPLR_HiPPO(N)
    Vc = V.conj().T
    p = Vc @ p
    q = Vc @ q.conj()
    A = jnp.diag(Lambda) - p[:, jnp.newaxis] @ q[:, jnp.newaxis].conj().T
    return partial(clone_layer(S4), N=N, A=A, Lambda=Lambda, p=p, q=q, Vc=Vc)
