from jax import numpy as jnp
from openmm import unit
import jax
import equinox as eqx
import numpy as np
from flow_diagrams.utils.jax import key_chain

kB = unit.MOLAR_GAS_CONSTANT_R.value_in_unit(unit.kilojoule_per_mole / unit.kelvin)


def sample_loss(
    pos_prior,
    scale_prior,
    prior_energy,
    prior_pressure,
    prior_temp,
    target_temp,
    target_press,
    reference_box,
    flow,
    target_energy_fn,
):

    prior_vol = jnp.prod( scale_prior * reference_box)

    new_pos, new_scale, ldj = flow.forward(
        pos_prior, scale_prior, target_temp, target_press
    )

    new_box = new_scale * reference_box
    new_vol = jnp.prod(new_box)

    target_energy = target_energy_fn(new_pos, new_scale)

    ldj_initial = flow.n_particles * (jnp.log(1.0 / prior_vol))
    ldj_final = flow.n_particles * (jnp.log(new_vol))

    ldj += ldj_final + ldj_initial

    losses = (
        (target_energy + new_vol * target_press) / (kB * target_temp)
        - ldj
        - (prior_energy + prior_vol * prior_pressure) / (kB * prior_temp)
    )

    return losses


def batch_loss_temp_press_individual(
    prior_temp,
    batch_pos,
    batch_scale,
    batch_ene,
    temps_and_pressures,
    flow,
    prior_pressure,
    reference_box,
    target_energy_fn,
    key
):
    chain = key_chain(key)
    
    batch_size = batch_pos.shape[0]
    
    # ids = jax.random.choice(next(chain),
    #     jnp.arange(temps_and_pressures.shape[0]), shape=(batch_size,), replace=False
    # )

    t_min = temps_and_pressures[0,0]
    p_min = temps_and_pressures[0,1]
    
    t_max = temps_and_pressures[-1,0]
    p_max = temps_and_pressures[-1,1]

    t_rand = jax.random.uniform(next(chain),shape=(batch_size,),minval=t_min,maxval = t_max)
    p_rand = jax.random.uniform(next(chain),shape=(batch_size,),minval=p_min,maxval = p_max)


    losses = jax.vmap(
        lambda pos, ene, scal, temp, press: sample_loss(
            pos_prior=pos,
            prior_energy=ene,
            scale_prior=scal,
            flow=flow,
            target_temp=temp,
            prior_temp=prior_temp,
            prior_pressure=prior_pressure,
            reference_box=reference_box,
            target_press=press,
            target_energy_fn=target_energy_fn,
        )
    )(
        batch_pos,
        batch_ene,
        batch_scale,
        t_rand,
        p_rand,
    )
    return jnp.mean(losses, axis=0)


@eqx.filter_jit
def batch_loss_temp_press_vmap(
    batch_pos,
    batch_scale,
    batch_ene,
    temps_and_pressures,
    prior_pressure,
    prior_temp,
    target_energy_fn,reference_box,
    flow,
):
    losses = jax.vmap(
        lambda par: batch_loss(
            batch_pos=batch_pos,
            batch_energies=batch_ene,
            batch_scales=batch_scale,
            temp_and_pressure=par,
            flow=flow,
            prior_pressure=prior_pressure,
            prior_temp=prior_temp,
            reference_box=reference_box,
            target_energy_fn=target_energy_fn,
        )
    )(temps_and_pressures)
    return jnp.mean(losses, axis=0)

@eqx.filter_jit
def batch_loss(
    batch_pos,
    batch_energies,
    batch_scales,
    temp_and_pressure,
    flow,
    prior_temp,
    prior_pressure,
    reference_box,
    target_energy_fn,
):
    temp = temp_and_pressure[0]
    press = temp_and_pressure[1]

    losses = jax.vmap(
        lambda pos, ene, scal: sample_loss(
            pos_prior=pos,
            prior_energy=ene,
            scale_prior=scal,
            flow=flow,
            target_temp=temp,
            target_press=press,
            prior_temp=prior_temp,
            reference_box=reference_box,
            prior_pressure=prior_pressure,
            target_energy_fn=target_energy_fn,
        )
    )(batch_pos, batch_energies, batch_scales)
    return jnp.mean(losses, axis=0)


@eqx.filter_jit
def make_step(
    batch_pos,
    batch_ene,
    batch_scale,
    flow,
    temps_and_pressures,
    optimized_state,
    prior_pressure,
    prior_temp,
    target_energy_fn,
    reference_box,
    optim,
    key
):
    chain = key_chain(key)
    loss_train, grads = eqx.filter_value_and_grad(
        lambda flow: batch_loss_temp_press_individual(
            flow=flow,
            batch_pos=batch_pos,
            batch_scale=batch_scale,
            batch_ene=batch_ene,
            temps_and_pressures=temps_and_pressures,
            prior_pressure=prior_pressure,
            prior_temp=prior_temp,
            target_energy_fn=target_energy_fn,
            reference_box=reference_box,
            key=next(chain)
        )
    )(flow)

    updates, optimized_state = optim.update(grads, optimized_state)
    flow = eqx.apply_updates(flow, updates)
    return loss_train, flow, optimized_state
