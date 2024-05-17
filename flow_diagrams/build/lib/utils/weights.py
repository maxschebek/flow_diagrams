# Library code
from jax import numpy as jnp
import numpy as np
import equinox as eqx
from utils import key_chain
import jax

from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions

class NormalFlow(eqx.Module):
  matrix: jax.Array

  def get(self, mean, std, key):
    chain = key_chain(key)
    return tfd.normal.Normal(loc=mean * jnp.ones(shape=self.matrix.shape),
                                scale= std * jnp.ones(shape=self.matrix.shape)).sample(seed=next(chain))

def is_symmetric(x):
    return isinstance(x, NormalFlow)

def maybe_symmetric(x, mean, std, key):
    chain = key_chain(key)
    if is_symmetric(x):
        return x.get(mean, std, next(chain))
    else:
        return x  # leave everything else unchanged

def resolve_symmetric(model, mean, std, key):
    chain = key_chain(key)
    return jax.tree_util.tree_map(lambda x :maybe_symmetric(x, mean=mean, std=std, key=next(chain)), model, is_leaf=is_symmetric)



is_linear = lambda x: isinstance(x, eqx.nn.Linear)

get_weights = lambda m: [x.weight
                         for x in jax.tree_util.tree_leaves(m, is_leaf=is_linear)
                         if is_linear(x)]
get_biases = lambda m: [x.bias
                         for x in jax.tree_util.tree_leaves(m, is_leaf=is_linear)
                         if is_linear(x)]