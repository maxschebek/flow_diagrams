from jax import numpy as jnp
import numpy as np
import equinox as eqx
from ..utils.jax import key_chain
from ..utils.weights import get_weights, get_biases, NormalFlow, resolve_symmetric

class CouplingFlow(eqx.Module):

    layers: list
    n_particles: int
    rescale: np.float32
    net_shape: eqx.Module


    def __init__(self, n_layers: int, n_blocks: int,
                n_bins: int, n_heads: int,
                lower: np.float32,
                upper: np.float32,
                n_freqs: int, dim_embedd: int,
                num_hidden: int, dim_hidden: int,
                n_particles: int, rescale: np.float32,
                use_circular_shift: bool,
                init_identity: bool,
                key):

        chain = key_chain(key)

        self.layers = []
        self.n_particles = n_particles
        self.rescale = rescale

        self.net_shape = eqx.nn.MLP(in_size=2,out_size=2,width_size=16,depth=2,key=next(chain))

        if init_identity:
            # normalize weights
            self.net_shape = eqx.tree_at(get_weights, self.net_shape, replace_fn=NormalFlow)

            # # normalize biases
            self.net_shape = eqx.tree_at(get_biases, self.net_shape, replace_fn=NormalFlow)

            self.net_shape = resolve_symmetric(self.net_shape, mean=0, std=5e-3)



         # generate a bunch of coupling blocks
        list_dim = [[0],[1],[2],[1,2],[0,1],[0,2]]

        for _ in range(n_layers):
            for i in list_dim:
                fixed_idxs =  np.array(i,dtype=np.int32)
                changed_idxs = np.setdiff1d(np.arange(3,dtype=np.int32), fixed_idxs)
                if use_circular_shift:
                    self.layers.append(
                        CircularShiftLayer(lower=lower,
                                        changed_idxs = changed_idxs,
                                        key=next(chain),
                                        upper=upper)
                    )

                self.layers.append(
                    CouplingLayer(
                        fixed_idxs = fixed_idxs,
                        changed_idxs = changed_idxs,
                        n_hidden=num_hidden,
                        dim_hidden=dim_hidden,
                        n_particles=n_particles,
                        n_freqs=n_freqs,
                        n_bins=n_bins,
                        lower=lower,
                        upper=upper,
                        n_heads=n_heads,
                        n_blocks=n_blocks,
                        dim_embed=dim_embedd,
                        init_identity=init_identity,
                        key=next(chain)
                    ) )

    def forward(self, pos, scale, press, temp):
        """Forward sequential coupling flow"""

        if len(self.layers) == 0:
            return pos, jnp.zeros(1)
        
        total_log_det_jacobian = 0

        # update volume with scaling parameter
        conditional_vars = jnp.array([temp,press])
        
        shape_params = self.net_shape(conditional_vars)
        
        alpha = shape_params[0] 
        beta = shape_params[1] 

        new_scale = (1. + alpha) * scale + beta

        ldj_scale = jnp.log(1. + alpha)

        total_log_det_jacobian += ldj_scale

        for layer in self.layers:
            
            pos, log_det_jacobian = layer.forward(pos, new_scale, press, temp)
            
            total_log_det_jacobian += log_det_jacobian
        
        pos_rescaled = pos / self.rescale
        return pos_rescaled, new_scale.squeeze(), total_log_det_jacobian.squeeze()

    def inverse(self, pos, scale, press, temp):
        """Inverse sequential coupling flow"""

        if len(self.layers) == 0:
            return pos, jnp.zeros(pos.shape[0])

        total_log_det_jacobian = 0
        for block in reversed(self.layers):
            pos,  log_det_jacobian = block.forward(pos, scale,press, temp, inverse=True)
            total_log_det_jacobian += log_det_jacobian

        # update volume with scaling parameter
        conditional_vars = jnp.array([temp,press])
        
        shape_params = self.net_shape(conditional_vars)
        
        alpha = shape_params[0] 
        beta = shape_params[1] 


        new_scale = (scale - beta) / (1. + alpha) 

        ldj_scale = -jnp.log(1. + alpha)

        total_log_det_jacobian += ldj_scale

        pos_rescaled = pos / self.rescale
        return pos_rescaled, new_scale.squeeze(), total_log_det_jacobian.squeeze()