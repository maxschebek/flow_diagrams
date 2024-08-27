import numpy as np
import jax
import jax.numpy as jnp
# from openmm import unit

kB = 0.00831446261815324 # in(unit.kilojoule_per_mole / unit.kelvin)


def running_average(data: np.ndarray, tau: int = 0):
    """Running average"""
    if tau == 0:
        tau = len(data)
    av = np.zeros_like(data)
    for n in range(len(data)):
        if n <= tau:
            av[n] = np.mean(data[: n + 1])
        else:
            av[n] = av[n - 1] + (data[n] - av[n - 1]) / tau
    return av


def effective_sample_size(log_weights):
    """Kish effective sample size; log weights don't have to be normalized"""
    return jnp.exp(
        2 * jax.scipy.special.logsumexp(log_weights)
        - jax.scipy.special.logsumexp(2 * log_weights)
    )


"""An algorithm is sample efficient if it can get the most out of every sample."""


def sampling_efficiency(log_weights):
    """Kish effective sample size / sample size; log weights don't have to be normalized"""
    return effective_sample_size(log_weights) / len(log_weights)


def log_weights_given_latent(
    pos_prior,
    scale_prior,
    prior_energy,
    temp_and_pressure_target,
    temp_and_pressure_flow,
    reference_box,
    n_particles,
    pressure_prior,
    temp_prior,
    target_energy_fn,
    flow,
):
    """Computes the weights for one sample.

    params:
    --------------------------
    logw: unnormalized weights"""

    prior_vol = jnp.prod(scale_prior * reference_box)

    target_temp = temp_and_pressure_target[0]
    target_press = temp_and_pressure_target[1]

    flow_temp = temp_and_pressure_flow[0]
    flow_press = temp_and_pressure_flow[1]

    new_pos, new_scale, ldj = flow.forward(
        pos=pos_prior, scale=scale_prior, temp=flow_temp, press=flow_press
    )
    new_box = new_scale * reference_box
    new_vol = jnp.prod(new_box)

    target_energy = target_energy_fn(new_pos, new_scale)

    ldj_initial = n_particles * (jnp.log(1.0 / prior_vol))
    ldj_final = n_particles * (jnp.log(new_vol))

    ldj += ldj_final + ldj_initial
    logw = (
        -(target_energy + new_vol * target_press) / (kB * target_temp)
        + ldj
        + (prior_energy + prior_vol * pressure_prior) / (kB * temp_prior)
    )

    return logw


def normalize_weights(logw_unnormalized):
    """Normalizes the weights.

    params:
    --------------------------
    logw: unnormalized weights"""
    return logw_unnormalized - jax.scipy.special.logsumexp(logw_unnormalized, axis=0)


def delta_f_to_prior(logw):
    """Computes the free energy difference from the flow to the prior.

    params:
    --------------------------
    logw: unnormalized weights

    """
    return jnp.log(len(logw)) - jax.scipy.special.logsumexp(logw, axis=0)
