import jax.numpy as jnp
from jax import Array

def wrap_to_unit_cube(pos, lower, upper):
    width = upper - lower
    return jnp.mod(pos - lower, width) + lower

def wrap_to_box(pos, box):
    return pos % box

def circular(x: Array,
             lower: float,
             upper: float,
             num_frequencies: int) -> Array:
  """Maps angles to points on the unit circle.
   Taken from https://github.com/google-deepmind/flows_for_atomic_solids.

  The mapping is such that the interval [lower, upper] is mapped to a full
  circle starting and ending at (1, 0). For num_frequencies > 1, the mapping
  also includes higher frequencies which are multiples of 2 pi/(lower-upper)
  so that [lower, upper] wraps around the unit circle multiple times.

  Args:
    x: array of shape [..., D].
    lower: lower limit, angles equal to this will be mapped to (1, 0).
    upper: upper limit, angles equal to this will be mapped to (1, 0).
    num_frequencies: number of frequencies to consider in the embedding.

  Returns:
    An array of shape [..., 2*num_frequencies*D].
  """
  base_frequency = 2. * jnp.pi / (upper - lower)
  frequencies = base_frequency * jnp.arange(1, num_frequencies+1)
  angles = frequencies * (x[..., None] - lower)
  # Reshape from [..., D, num_frequencies] to [..., D*num_frequencies].
  angles = angles.reshape(x.shape[:-1] + (-1,))
  cos = jnp.cos(angles)
  sin = jnp.sin(angles)
  return jnp.concatenate([cos, sin], axis=-1)


def remove_disp_of_first_atom(displacements):

    disp_at_1 = displacements[0,:]

    return displacements - disp_at_1


def transform_rel_coords_to_abs_coords(relative_coordinates: Array, side_length: Array):
    """Transforms relative coordinates inside the unit cube to absolute coordinates given a 3d box_vector."""
    assert relative_coordinates.shape[-1] == 3
    assert side_length.shape == (3,)
    return relative_coordinates * side_length

def transform_abs_coords_to_rel_coords(absolute_coordinates: Array, side_length: Array):
    """Transforms relative coordinates inside the unit cube to absolute coordinates given a 3d box_vector."""
    assert absolute_coordinates.shape[-1] == 3
    assert side_length.shape == (3,)
    return absolute_coordinates / side_length
