import jax
from jax import numpy as jnp
from typing import Any, Mapping

Params = Any
PRNGKey = jnp.ndarray
Metrics = Mapping[str, jnp.ndarray]
