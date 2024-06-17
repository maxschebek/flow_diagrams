from typing import Callable, Tuple, TextIO, Dict, Any, Optional
import jax.numpy as jnp
from jax import ops
from jax.tree_util import tree_map
from jax import vmap
import haiku as hk
from jax.scipy.special import erfc  # error function
from jax_md import space, smap, partition, util
from functools import wraps, partial
from ml_collections import ConfigDict
from jax_md.energy import lennard_jones

maybe_downcast = util.maybe_downcast
PyTree = Any
Box = space.Box
DisplacementFn = space.DisplacementFn
DisplacementOrMetricFn = space.DisplacementOrMetricFn
f32 = util.f32
f64 = util.f64
Array = util.Array

NeighborFn = partition.NeighborFn
NeighborList = partition.NeighborList
NeighborListFormat = partition.NeighborListFormat


def fd_multiplicative_isotropic_cutoff(
    fn: Callable[..., Array],
    r_onset: float,
    r_cutoff: float,
    smooth: bool = False,
    shift: bool = False,
) -> Callable[..., Array]:
    """Takes an isotropic function and constructs a truncated function.

    Given a function `f:R -> R`, we construct a new function `f':R -> R` such
    that `f'(r) = f(r)` for `r < r_onset`, `f'(r) = 0` for `r > r_cutoff`, and
    `f(r)` is :math:`C^1` everywhere. To do this, we follow the approach outlined
    in HOOMD Blue  [#hoomd]_ (thanks to Carl Goodrich for the pointer). We
    construct a function `S(r)` such that `S(r) = 1` for `r < r_onset`,
    `S(r) = 0` for `r > r_cutoff`, and `S(r)` is :math:`C^1`. Then
    `f'(r) = S(r)f(r)`.

    Args:
      fn: A function that takes an ndarray of distances of shape `[n, m]` as well
        as varargs.
      r_onset: A float specifying the distance marking the onset of deformation.
      r_cutoff: A float specifying the cutoff distance.

    Returns:
      A new function with the same signature as fn, with the properties outlined
      above.

    .. rubric:: References
    .. [#hoomd] HOOMD Blue documentation. Accessed on 05/31/2019.
        https://hoomd-blue.readthedocs.io/en/stable/module-md-pair.html#hoomd.md.pair.pair
    """

    r_c = r_cutoff ** f32(2)
    r_o = r_onset ** f32(2)

    def smooth_fn(dr):
        r = dr ** f32(2)

        inner = jnp.where(
            dr < r_cutoff,
            (r_c - r) ** 2 * (r_c + 2 * r - 3 * r_o) / (r_c - r_o) ** 3,
            0,
        )

        return jnp.where(dr < r_onset, 1, inner)

    def smooth_fn_openmm(dr):
        x = (dr - r_onset) / (r_cutoff - r_onset)

        inner = jnp.where(
            dr < r_cutoff,
            1.0 - 6 * jnp.power(x, 5) + 15 * jnp.power(x, 4) - 10 * jnp.power(x, 3),
            0,
        )

        return jnp.where(dr < r_onset, 1, inner)

    def cut_fn(dr):
        return jnp.where(dr < r_cutoff, 1, 0)

    @wraps(fn)
    def cutoff_fn(dr, *args, **kwargs):
        if smooth:
            return smooth_fn_openmm(dr) * fn(dr, *args, **kwargs)
        elif shift:
            return cut_fn(dr) * (
                fn(dr, *args, **kwargs) - fn(r_cutoff, *args, **kwargs)
            )
        else:
            return cut_fn(dr) * fn(dr, *args, **kwargs)

    return cutoff_fn


def fd_lennard_jones_neighbor_list(
    displacement_or_metric: DisplacementOrMetricFn,
    box_size: Box,
    species: Optional[Array] = None,
    sigma: Array = 1.0,
    epsilon: Array = 1.0,
    alpha: Array = 2.0,
    r_onset: float = 2.0,
    r_cutoff: float = 2.5,
    dr_threshold: float = 0.5,
    per_particle: bool = False,
    fractional_coordinates: bool = False,
    shift: bool = False,
    smooth: bool = False,
    soft_core: bool = False,
    lambda_lj: float = 1,
    format: partition.NeighborListFormat = partition.OrderedSparse,
    **neighbor_kwargs
) -> Tuple[NeighborFn, Callable[[Array, NeighborList], Array]]:
    """Convenience wrapper to compute :ref:`Lennard-Jones <lj-pot>` using a neighbor list."""
    sigma = maybe_downcast(sigma)
    epsilon = maybe_downcast(epsilon)
    r_onset = maybe_downcast(r_onset) * jnp.max(sigma)
    r_cutoff = maybe_downcast(r_cutoff) * jnp.max(sigma)
    dr_threshold = maybe_downcast(dr_threshold)

    neighbor_fn = partition.neighbor_list(
        displacement_or_metric,
        box_size,
        r_cutoff,
        dr_threshold,
        fractional_coordinates=fractional_coordinates,
        format=format,
        **neighbor_kwargs
    )
    if soft_core:
        func = lambda dr, sigma, epsilon,: soft_lennard_jones(
            dr=dr, sigma=sigma, epsilon=epsilon, lambda_lj=lambda_lj
        )
    else:
        func = lennard_jones

    energy_fn = smap.pair_neighbor_list(
        fd_multiplicative_isotropic_cutoff(func, r_onset, r_cutoff, smooth, shift),
        space.canonicalize_displacement_or_metric(displacement_or_metric),
        ignore_unused_parameters=True,
        species=species,
        sigma=sigma,
        epsilon=epsilon,
        reduce_axis=(1,) if per_particle else None,
    )

    return neighbor_fn, energy_fn


def fd_lennard_jones_pair(
    displacement_or_metric: DisplacementOrMetricFn,
    species: Optional[Array] = None,
    sigma: Array = 1.0,
    epsilon: Array = 1.0,
    soft_core: bool = False,
    lambda_lj: float = 1,
    r_onset: Array = 2.0,
    r_cutoff: Array = 2.5,
    shift: bool = False,
    smooth: bool = False,
    per_particle: bool = False,
) -> Callable[[Array], Array]:
    """Convenience wrapper to compute :ref:`Lennard-Jones energy <lj-pot>` over a system."""
    sigma = maybe_downcast(sigma)
    epsilon = maybe_downcast(epsilon)
    r_onset = maybe_downcast(r_onset) * jnp.max(sigma)
    r_cutoff = maybe_downcast(r_cutoff) * jnp.max(sigma)

    if soft_core:
        func = lambda dr, sigma, epsilon,: soft_lennard_jones(
            dr=dr, sigma=sigma, epsilon=epsilon, lambda_lj=lambda_lj
        )
    else:
        func = lennard_jones

    return smap.pair(
        fd_multiplicative_isotropic_cutoff(func, r_onset, r_cutoff, smooth, shift),
        space.canonicalize_displacement_or_metric(displacement_or_metric),
        ignore_unused_parameters=True,
        species=species,
        sigma=sigma,
        epsilon=epsilon,
        reduce_axis=(1,) if per_particle else None,
    )


def soft_lennard_jones(
    dr: Array,
    sigma: Array = 1,
    epsilon: Array = 1,
    lambda_lj: Array = 1,
    **unused_kwargs
):
    r6 = (dr / sigma) ** 6
    r6 += 0.5 * (1.0 - lambda_lj) ** 2

    r6inv = 1.0 / r6
    energy = r6inv * (r6inv - 1.0)
    energy *= 4.0 * lambda_lj * epsilon
    return energy
