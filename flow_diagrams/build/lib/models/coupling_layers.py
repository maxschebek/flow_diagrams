import jax
import jax.numpy as jnp
import numpy as np
from flows_for_phase_diagrams.utils.jax import key_chain
import flows
import equinox as eqx

from ..utils.weights import get_weights, NormalFlow, resolve_symmetric, get_biases, get_weights
from spline import RationalQuadraticSpline
from ..utils.lattice import circular, wrap_to_unit_cube

class TransformerBlock(eqx.Module):
    transformer: eqx.Module
    mlp: eqx.Module

    def __init__(
        self,
        n_heads,
        dim_embedd,
        dim_hidden,
        n_hidden,
        key,
        use_key_bias: bool = True,
        use_query_bias: bool = False,
        use_output_bias: bool = True,
    ):
        chain = key_chain(key)

        self.transformer = eqx.nn.MultiheadAttention(
            n_heads,
            dim_embedd,
            use_key_bias=use_key_bias,
            use_query_bias=use_query_bias,
            use_output_bias=use_output_bias,
            key=next(chain),
        )

        self.mlp = eqx.nn.MLP(
            in_size=dim_embedd,
            out_size=dim_embedd,
            width_size=dim_hidden,
            depth=n_hidden,
            key=next(chain),
        )

    def __call__(self, x, key=None):
        x = x + self.transformer(x, x, x)
        x = x + jax.vmap(self.mlp)(x)
        return x


class CouplingLayer(eqx.Module):
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
        n_hidden: int,
        dim_hidden: int,
        changed_idxs: np.ndarray,
        n_particles: int,
        n_bins: int,
        lower: np.float32,
        upper: np.float32,
        n_heads: int,
        n_freqs: int,
        dim_embed: int,
        n_blocks: int,
        key,
        init_identity: bool,
    ):
        chain = key_chain(key)

        self.lower = lower
        self.upper = upper
        self.fixed_idxs = fixed_idxs
        self.changed_idxs = changed_idxs
        self.n_freqs = n_freqs
        self.n_changed = changed_idxs.shape[0]
        self.n_fixed = fixed_idxs.shape[0]
        self.dim_embedd = dim_embed
        self.n_hidden = n_hidden
        self.n_bins = n_bins
        self.n_blocks = n_blocks

        self.n_particles = n_particles

        # input dimension: circular embedding + scaling + temp + press
        self.net_embedd = eqx.nn.MLP(
            in_size=2 * self.n_freqs * self.n_fixed + 3,
            out_size=self.dim_embedd,
            width_size=dim_hidden,
            depth=n_hidden,
            key=next(chain),
        )

        self.transformer_block = eqx.nn.Sequential(
            [
                TransformerBlock(
                    n_heads=n_heads,
                    dim_embedd=self.dim_embedd,
                    dim_hidden=dim_hidden,
                    n_hidden=self.n_hidden,
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
            depth=n_hidden,
            key=next(chain),
        )
        if init_identity:
            # normalize weights
            self.net_affine = eqx.tree_at(
                get_weights, self.net_affine, replace_fn=NormalFlow
            )

            # # normalize biases
            self.net_affine = eqx.tree_at(
                get_biases, self.net_affine, replace_fn=NormalFlow
            )

            self.net_affine = resolve_symmetric(self.net_affine, mean=0, std=5e-3)

    def forward(self, pos, scale, press, temp, inverse: bool = False):
        """Forward coupling."""
        assert pos.shape == (self.n_particles, 3)
        # the set of vars we dont change but condition on
        fixed = pos[1:, self.fixed_idxs]

        # the set of vars we change
        changed = pos[1:, self.changed_idxs]
        features = circular(
            fixed[:, self.fixed_idxs], self.lower, self.upper, self.n_freqs
        )

        # fixed-size vector h
        # print(features.shape, self.net_embedd)
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
        params = jax.vmap(self.net_affine)(h)
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

        pos = pos.at[1:, self.fixed_idxs].set(fixed)
        pos = pos.at[1:, self.changed_idxs].set(
            changed.reshape(self.n_particles - 1, self.n_changed)
        )

        return pos, log_det_jacobian.sum()


class CircularShiftLayer(eqx.Module):
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

    def forward(self, pos, scale, temp, press, inverse: bool = False):
        """Forward coupling."""

        shift = self.mlp(jnp.array([temp, press]))

        sign = jnp.power(-1, inverse)

        new_pos = jnp.copy(pos)
        new_pos = new_pos.at[:, self.changed_idxs].set(
            wrap_to_unit_cube(
                pos[:, self.changed_idxs] - sign * shift, self.lower, self.upper
            )
        )
        return new_pos, 0.0
