from flow_diagrams.utils.jax import key_chain
import equinox as eqx
import jax
class TransformerBlock(eqx.Module):
    """Transformer block consisting of multi-head attention and MLP"""
    transformer: eqx.Module
    mlp: eqx.Module
    norm_transformer: eqx.Module
    norm_mlp: eqx.Module

    def __init__(
        self,
        n_heads,
        dim_embedd,
        dim_hidden,
        num_hidden,
        n_particles,
        key,
        use_layer_norm,
        use_key_bias: bool = True,
        use_query_bias: bool = False,
        use_output_bias: bool = True,
    ):
        """

        params:
        --------------------------

        - `n_heads`:  Number of heads in multi-head attention
        - `n_freqs`: Number of frequencies for circular encoding
        - `dim_embedd`: Embedding dimension
        - `use_layer_norm`: `True` if layer normalisation is to be applied
        - `num_hidden`: Number of hidden layers in  MLP
        - `dim_hidden`: Number of nodes per hidden layer in  MLP
        - `n_particles`: Number of particles
        - `key`: A `jax.random.PRNGKey` used for random parameter initialisation.
        - `use_query_bias`: `True` if query bias is to be used
        - `use_key_bias`: `True` if query bias is to be used
        - `use_output_bias`: `True` if output bias is to be used

                """
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
            depth=num_hidden,
            key=next(chain),
        )

        if use_layer_norm:
            self.norm_mlp = eqx.nn.LayerNorm(shape=(n_particles, dim_embedd))
            self.norm_transformer = eqx.nn.LayerNorm(shape=(n_particles, dim_embedd))
        else:
            self.norm_mlp = lambda x: x
            self.norm_transformer = lambda x: x

    def __call__(self, x, key=None):
        x = self.norm_transformer(x)
        x = x + self.transformer(x, x, x)
        x = self.norm_mlp(x)
        x = x + jax.vmap(self.mlp)(x)
        return x
