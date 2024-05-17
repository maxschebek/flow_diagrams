from jax import numpy as jnp
import numpy as np
import equinox as eqx
from flow_diagrams.utils.weights import normal_init, init_weights
from flow_diagrams.utils.jax import key_chain
from flow_diagrams.models.coupling_layers import (
    ConditionalCouplingLayerNPT,
    CouplingLayerNVT,
    ConditionalCircularShiftLayerNPT,
    CircularShiftLayerNVT,
)


class ConditionalCouplingFlowNPT(eqx.Module):
    """Coupling flow for the NPT ensemble conditioned on temperature and pressure."""

    layers: list
    n_particles: int
    t_max: np.float32
    p_max: np.float32
    net_shape: eqx.Module

    def __init__(
        self,
        n_layers: int,
        n_blocks: int,
        n_bins: int,
        n_heads: int,
        t_max: np.float32,
        p_max: np.float32,
        lower: np.float32,
        upper: np.float32,
        n_freqs: int,
        dim_embedd: int,
        use_layer_norm: bool,
        num_hidden: int,
        dim_hidden: int,
        num_hidden_shape: int,
        dim_hidden_shape: int,
        n_particles: int,
        use_circular_shift: bool,
        init_identity: bool,
        key,
        init_width: float = 5e-3,
    ):
        """

        params:
        --------------------------
        - `n_layers`: Number of repetitions of all 6 possible splittings.
        - `n_blocks`: Number of transformer blocks
        - `n_bins`:  Number of bins for neural spline
        - `n_heads`:  Number of heads in multi-head attention
        - `t_max`: Maximum temperature used for conditioning
        - `p_max`: Maximum pressure used for conditioning
        - `lower`: Lower limit of the spline interval
        - `upper`: Upper limit of the spline interval
        - `n_freqs`: Number of frequencies for circular encoding
        - `dim_embedd`: Embedding dimension
        - `use_layer_norm`: `True` if layer normalisation is to be applied
        - `num_hidden`: Number of hidden layers in transformer  MLP
        - `dim_hidden`: Number of nodes per hidden layer in transformer MLP
        - `num_hidden_shape`: Number of hidden layers in  shape MLP
        - `dim_hidden_shape`: Number of nodes per hidden layer in shape MLP
        - `n_particles`: Number of particles
        - `use_circular_shift`: `True` if circular shift layers are to be used
        - `init_identity`: `True` if flow transformation is to be initialized close to the identity
        - `key`: A `jax.random.PRNGKey` used for random parameter initialisation.
        - `init_width`: Width of the parameter distribution for neural-network initialisation
        """

        chain = key_chain(key)

        self.layers = []
        self.p_max = p_max
        self.t_max = t_max
        self.n_particles = n_particles

        self.net_shape = eqx.nn.MLP(
            in_size=2, out_size=2, width_size=64, depth=6, key=next(chain)
        )

        # normalize weights
        if init_identity:
            self.net_shape = init_weights(
                self.net_shape, init_width, normal_init, next(chain)
            )

        # generate a bunch of coupling blocks
        list_dim = [[0], [1], [2], [1, 2], [0, 1], [0, 2]]

        for _ in range(n_layers):
            for i in list_dim:
                fixed_idxs = np.array(i, dtype=np.int32)
                changed_idxs = np.setdiff1d(np.arange(3, dtype=np.int32), fixed_idxs)
                if use_circular_shift:
                    self.layers.append(
                        ConditionalCircularShiftLayerNPT(
                            lower=lower,
                            changed_idxs=changed_idxs,
                            key=next(chain),
                            upper=upper,
                        )
                    )

                self.layers.append(
                    ConditionalCouplingLayerNPT(
                        fixed_idxs=fixed_idxs,
                        changed_idxs=changed_idxs,
                        num_hidden=num_hidden,
                        dim_hidden=dim_hidden,
                        n_particles=n_particles,
                        use_layer_norm=use_layer_norm,
                        n_freqs=n_freqs,
                        n_bins=n_bins,
                        lower=lower,
                        upper=upper,
                        n_heads=n_heads,
                        n_blocks=n_blocks,
                        dim_embed=dim_embedd,
                        init_identity=init_identity,
                        key=next(chain),
                    )
                )

    def forward(self, pos, scale, press, temp):
        """Forward pass.

         params:
        --------------------------
        - `pos`: Fractional coordinates in [0,1]
        - `scale`: Box scaling parameter
        - `press`: Pressure
        - `temp`: Temperature

        """

        if len(self.layers) == 0:
            return pos, scale, jnp.array(0.0)

        total_log_det_jacobian = 0

        # update volume with scaling parameter
        temp_scaled = temp / self.t_max
        press_scaled = press / self.p_max

        conditional_vars = jnp.array([temp_scaled, press_scaled])

        shape_params = self.net_shape(conditional_vars)

        alpha = shape_params[0]
        beta = shape_params[1]

        new_scale = (1.0 + alpha) * scale + beta

        ldj_scale = jnp.log(jnp.abs(1.0 + alpha))

        total_log_det_jacobian += ldj_scale

        for layer in self.layers:
            pos, log_det_jacobian = layer.forward(
                pos=pos, scale=new_scale, temp=temp_scaled, press=press_scaled
            )

            total_log_det_jacobian += log_det_jacobian

        return pos, new_scale.squeeze(), total_log_det_jacobian.squeeze()
