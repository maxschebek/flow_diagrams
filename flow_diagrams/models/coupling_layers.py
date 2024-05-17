import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from flow_diagrams.utils.jax import key_chain
import equinox as eqx
from flow_diagrams.models.transformer import TransformerBlock
from flow_diagrams.utils.weights import (
    uniform_init,
    zero_init,
    init_weights,
    normal_init
)
from flow_diagrams.models.spline import RationalQuadraticSpline
from flow_diagrams.utils.lattice import circular, wrap_to_unit_cube


class ConditionalCouplingLayerNPT(eqx.Module):
    """Coupling layer for the NPT ensemble conditioned on temperature and pressure.
    """
    
    net_embedd: eqx.Module
    net_affine: eqx.Module
    transformer_block: eqx.Module

    lower: np.float32
    upper: np.float32
    fixed_idxs: np.ndarray
    changed_idxs: np.ndarray
    n_particles: int
    n_freqs: int
    n_fixed: int
    n_changed: int
    dim_embedd: int
    n_hidden: int
    n_bins: int
    n_blocks: int

    def __init__(
        self,
        fixed_idxs: np.ndarray,
        num_hidden: int,
        dim_hidden: int,
        changed_idxs: np.ndarray,
        n_particles: int,
        n_bins: int,
        lower: np.float32,
        upper: np.float32,
        n_heads: int,
        n_freqs: int,
        dim_embed: int,
        use_layer_norm:bool,
        n_blocks: int,
        key,
        init_identity: bool,
        init_width: float=5e-3,
    ):
        """
        params:
        --------------------------
        - `fixed_idxs`: Indices of fixed dimensions
        - `num_hidden`: Number of hidden layers in  MLP
        - `dim_hidden`: Number of nodes per hidden layer in  MLP
        - `changed_idxs`: Indices of changed dimensions
        - `n_particles`: Number of particles
        - `n_bins`:  Number of bins for neural spline
        - `lower`: Lower limit of the spline interval
        - `upper`: Upper limit of the spline interval
        - `n_heads`:  Number of heads in multi-head attention
        - `n_freqs`: Number of frequencies for circular encoding
        - `dim_embedd`: Embedding dimension
        - `use_layer_norm`: `True` if layer normalisation is to be applied
        - `n_blocks`: Number of transformer blocks
        - `key`: A `jax.random.PRNGKey` used for random parameter initialisation.
        - `init_identity`: `True` if flow transformation is to be initialized close to the identity
        - `init_width`: Width of the parameter distribution for neural-network initialisation
        """

        chain = key_chain(key)

        self.lower = lower
        self.upper = upper
        self.fixed_idxs = fixed_idxs
        self.changed_idxs = changed_idxs
        self.n_freqs = n_freqs
        self.n_changed = changed_idxs.shape[0]
        self.n_fixed = fixed_idxs.shape[0]
        self.dim_embedd = dim_embed
        self.n_hidden = num_hidden
        self.n_bins = n_bins
        self.n_blocks = n_blocks

        self.n_particles = n_particles

        # input dimension: circular embedding + scaling + temp + press
        self.net_embedd = eqx.nn.MLP(
            in_size=2 * self.n_freqs * self.n_fixed + 3,
            out_size=self.dim_embedd,
            width_size=dim_hidden,
            depth=num_hidden,
            key=next(chain),
        )

        self.transformer_block = eqx.nn.Sequential(
            [
                TransformerBlock(
                    n_heads=n_heads,
                    dim_embedd=self.dim_embedd,
                    dim_hidden=dim_hidden,
                    num_hidden=self.n_hidden,
                    use_layer_norm=use_layer_norm,
                    n_particles=n_particles,
                    use_key_bias=True,
                    use_query_bias=False,
                    use_output_bias=True,
                    key=next(chain),
                )
                for _ in range(n_blocks)
            ]
        )

        self.net_affine = eqx.nn.MLP(
            in_size=self.dim_embedd,
            out_size=self.n_changed * (3 * n_bins + 1),
            width_size=dim_hidden,
            depth=num_hidden,
            key=next(chain),
        )
        if init_identity:
           self.net_affine = init_weights(self.net_affine ,init_width, normal_init, next(chain))

    def forward(self, pos, scale, press, temp, inverse: bool = False):
        """Forward pass.

         params:
        --------------------------
        - `pos`: Fractional coordinates in [0,1]
        - `press`: Pressure
        - `temp`: Temperature

        """
        assert pos.shape == (self.n_particles, 3)
        # the set of vars we dont change but condition on
        fixed = pos[:, self.fixed_idxs]

        # the set of vars we change
        changed = pos[1:, self.changed_idxs]
        features = circular(
            fixed, self.lower, self.upper, self.n_freqs
        )

        # fixed-size vector h
        h = jax.vmap(self.net_embedd)(
            jnp.hstack(
                (
                    features,
                    jnp.ones((features.shape[0], 1)) * scale,
                    jnp.ones((features.shape[0], 1)) * temp,
                    jnp.ones((features.shape[0], 1)) * press,
                )
            )
        )

        # Transformer update
        h = self.transformer_block(h)

        # Spline
        params = jax.vmap(self.net_affine)(h[1:])
        params = params.reshape(
            (self.n_particles - 1, self.n_changed, (3 * self.n_bins + 1))
        )

        rqs = RationalQuadraticSpline(
            params=params,
            range_min=self.lower,
            range_max=self.upper,
            boundary_slopes="circular",
        )
        if not inverse:
            changed, log_det_jacobian = rqs.forward_and_log_det(changed)
        else:
            changed, log_det_jacobian = rqs.inverse_and_log_det(changed)

        new_pos = jnp.zeros_like(pos)
        new_pos = new_pos.at[0].set(pos[0])
        new_pos = new_pos.at[1:, self.fixed_idxs].set(fixed[1:])
        new_pos = new_pos.at[1:, self.changed_idxs].set(
            changed.reshape(self.n_particles - 1, self.n_changed)
        )

        return new_pos, log_det_jacobian.sum()



class ConditionalCircularShiftLayerNPT(eqx.Module):
    """Conditional circular shift layer"""
    changed_idxs: np.ndarray
    lower: np.float32
    upper: np.float32
    mlp: eqx.Module
    n_changed: int

    def __init__(self, lower, upper, changed_idxs, key):
        chain = key_chain(key)

        self.changed_idxs = changed_idxs
        self.n_changed = changed_idxs.shape[0]

        self.lower = lower
        self.upper = upper
        self.mlp = eqx.nn.MLP(
            in_size=2, out_size=self.n_changed, width_size=4, depth=1, key=next(chain)
        )
        """
        params:
        --------------------------
        - `lower`: Lower limit of the spline interval
        - `upper`: Upper limit of the spline interval
        - `changed_idxs`: Indices of changed dimensions
        - `key`: A `jax.random.PRNGKey` used for random parameter initialisation.
        """

    def forward(self, pos, scale, temp, press, inverse: bool = False):
        """Forward pass.

         params:
        --------------------------
        - `pos`: Fractional coordinates in [0,1]
        - `temp`: Temperature
        - `press`: Pressure

        """
        shift = self.mlp(jnp.array([temp, press]))

        sign = jnp.power(-1, inverse)

        new_pos = jnp.copy(pos)
        new_pos = new_pos.at[:, self.changed_idxs].set(
            wrap_to_unit_cube(
                pos[:, self.changed_idxs] - sign * shift, self.lower, self.upper
            )
        )
        return new_pos, 0.0
