import jax.numpy  as jnp
import jax.random as jrandom
from flow_diagrams.utils.jax import key_chain

def grid_conditional_variables(t_min, t_max, p_min, p_max, n_t, n_p):
    """Generates a grid of conditional variables. """

    temperatures = jnp.linspace(t_min, t_max, num=n_t)
    pressures = jnp.linspace(p_min, p_max, num=n_p)

    conditional_variables = jnp.array(jnp.meshgrid(temperatures, pressures)).T.reshape(-1, 2)

    return conditional_variables


def random_conditional_variables(t_min, t_max, p_min, p_max, n_points, key):
    """Draws n random conditional variables"""

    chain = key_chain(key)

    temperatures = jrandom.uniform(next(chain),(n_points,1),minval=t_min,maxval=t_max)
    pressures = jrandom.uniform(next(chain),(n_points,1),minval=p_min,maxval=p_max)

    conditional_variables = jnp.hstack((temperatures, pressures))
    return conditional_variables

def convert_from_reduced_t(epsilon, kb):
    """Transform temperature from reduced LJ units to real units"""
    return  epsilon / kb

def convert_from_reduced_p(epsilon, sigma):
    """Transform pressure from reduced LJ units to real units"""
    return  epsilon / sigma ** 3