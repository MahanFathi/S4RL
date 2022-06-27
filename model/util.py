from functools import partial
import jax
import jax.numpy as jnp
from flax import linen as nn
from jax.numpy.linalg import eig, inv, matrix_power
from jax.scipy.signal import convolve


def discretize(A, B, C, step):
    I = jnp.eye(A.shape[0])
    BL = inv(I - (step / 2.0) * A)
    Ab = BL @ (I + (step / 2.0) * A)
    Bb = (BL * step) @ B
    return Ab, Bb, C


def scan_SSM(Ab, Bb, Cb, u, x0):
    def step(x_k_1, u_k):
        x_k = Ab @ x_k_1 + Bb @ u_k
        y_k = Cb @ x_k
        return x_k, y_k

    return jax.lax.scan(step, x0, u)


def K_conv(Ab, Bb, Cb, L):
    return jnp.array(
        [(Cb @ matrix_power(Ab, l) @ Bb).reshape() for l in range(L)]
    )


def non_circular_convolution(u, K, nofft=False):
    if nofft:
        return convolve(u, K, mode="full")[: u.shape[0]]
    else:
        assert K.shape[0] == u.shape[0]
        ud = jnp.fft.rfft(jnp.pad(u, (0, K.shape[0])))
        Kd = jnp.fft.rfft(jnp.pad(K, (0, u.shape[0])))
        out = ud * Kd
        return jnp.fft.irfft(out)[: u.shape[0]]


def discrete_DPLR(Lambda, p, q, B, Ct, step, L):
    N = Lambda.shape[0]
    A = jnp.diag(Lambda) - p[:, jnp.newaxis] @ q[:, jnp.newaxis].conj().T
    I = jnp.eye(N)

    # Forward Euler
    A0 = (2.0 / step) * I + A

    # Backward Euler
    D = jnp.diag(1.0 / ((2.0 / step) - Lambda))
    qc = q.conj().T.reshape(1, -1)
    p2 = p.reshape(-1, 1)
    A1 = D - (D @ p2 * (1.0 / (1 + (qc @ D @ p2))) * qc @ D)

    # A bar and B bar
    Ab = A1 @ A0
    Bb = 2 * A1 @ B

    # Recover Cbar from Ct
    Cb = Ct @ inv(I - matrix_power(Ab, L)).conj()
    return Ab, Bb, Cb.conj()


@partial(jnp.vectorize, signature="(c),(),(c)->()")
def cauchy_dot(v, omega, lambd):
    return (v / (omega - lambd)).sum()


def K_gen_DPLR(Lambda, p, q, B, Ct, step, unmat=False):
    aterm = (Ct.conj().ravel(), q.conj().ravel())
    bterm = (B.ravel(), p.ravel())

    def gen(o):
        g = (2.0 / step) * ((1.0 - o) / (1.0 + o))
        c = 2.0 / (1.0 + o)

        def k(a):
            # Checkpoint this calculation for memory efficiency.
            if unmat:
                return jax.remat(cauchy_dot)(a, g, Lambda)
            else:
                return cauchy_dot(a, g, Lambda)

        k00 = k(aterm[0] * bterm[0])
        k01 = k(aterm[0] * bterm[1])
        k10 = k(aterm[1] * bterm[0])
        k11 = k(aterm[1] * bterm[1])
        return c * (k00 - k01 * (1.0 / (1.0 + k11)) * k10)

    return gen


def conv_from_gen(gen, L):
    # Evaluate at roots of unity
    # Generating function is (-)z-transform, so we evaluate at (-)root
    Omega_L = jnp.exp((-2j * jnp.pi) * (jnp.arange(L) / L))
    atRoots = jax.vmap(gen)(Omega_L)
    # Inverse FFT
    out = jnp.fft.ifft(atRoots, L).reshape(L)
    return out.real



def make_HiPPO(N):
    def v(n, k):
        if n > k:
            return jnp.sqrt(2 * n + 1) * jnp.sqrt(2 * k + 1)
        elif n == k:
            return n + 1
        else:
            return 0

    # Do it slow so we don't mess it up :)
    mat = [[v(n, k) for k in range(1, N + 1)] for n in range(1, N + 1)]
    return -jnp.array(mat)


def make_NPLR_HiPPO(N):
    # Make -HiPPO
    nhippo = make_HiPPO(N)

    # Add in a rank 1 term. Makes it Normal.
    p = 0.5 * jnp.sqrt(2 * jnp.arange(1, N + 1) + 1.0)
    q = 2 * p
    S = nhippo + p[:, jnp.newaxis] * q[jnp.newaxis, :]

    # Diagonalize to S to V \Lambda V^*
    Lambda, V = jax.jit(eig, backend="cpu")(S)
    # Lambda, V = eig(jax.device_put(S, device=jax.devices("cpu")[0]))
    return nhippo, Lambda, p, q, V


def log_step_initializer(dt_min=0.001, dt_max=0.1):
    def init(key, shape):
        return jax.random.uniform(key, shape) * (
            jnp.log(dt_max) - jnp.log(dt_min)
        ) + jnp.log(dt_min)

    return init


def clone_layer(layer):
    return nn.vmap(
        layer,
        in_axes=1,
        out_axes=1,
        variable_axes={"params": 1, "cache": 1, "prime": 1},
        split_rngs={"params": True},
    )
