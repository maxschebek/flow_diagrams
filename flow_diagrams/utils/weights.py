# Library code
from jax import numpy as jnp
import numpy as np
import equinox as eqx
import jax
from flow_diagrams.utils.jax import key_chain
from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions

is_linear = lambda x: isinstance(x, eqx.nn.Linear)

get_weights = lambda m: [x.weight
                         for x in jax.tree_util.tree_leaves(m, is_leaf=is_linear)
                         if is_linear(x)]
get_biases = lambda m: [x.bias
                         for x in jax.tree_util.tree_leaves(m, is_leaf=is_linear)
                         if is_linear(x)]

def uniform_init(weight: jax.Array, lim, key: jax.random.PRNGKey) -> jax.Array:
  return jax.random.uniform(key, shape=weight.shape, minval=-lim, maxval=lim)

def normal_init(weight: jax.Array, lim, key: jax.random.PRNGKey) -> jax.Array:
  return jax.random.normal(key, shape=weight.shape) * lim

def zero_init(weight: jax.Array, lim, key: jax.random.PRNGKey) -> jax.Array:
  return jnp.zeros(shape=weight.shape)

def init_weights(model, lim, init_fn, key):

  weights = get_weights(model)
  biases = get_biases(model)
  new_weights = [init_fn(weight,lim, subkey)
                 for weight, subkey in zip(weights, jax.random.split(key, len(weights)))]
  new_biases = [init_fn(bias,lim, subkey)
                 for bias, subkey in zip(biases, jax.random.split(key, len(biases)))]
  new_model = eqx.tree_at(get_weights, model, new_weights)
  new_model = eqx.tree_at(get_biases, new_model, new_biases)
  return new_model
