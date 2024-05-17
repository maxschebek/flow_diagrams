
import jax
import jax.numpy as jnp

def key_chain(
    seed: int | jnp.ndarray | jax.random.PRNGKeyArray,
) :
    """returns an iterator that automatically splits jax.random.PRNGKeys"""

    if isinstance(seed, int) or seed.ndim == 0:
        key = jax.random.PRNGKey(seed)
    else:
        key, _ = jax.random.split(seed)
    while True:
        new, key = jax.random.split(key)
        yield new