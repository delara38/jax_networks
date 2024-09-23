import jax
import jax.numpy as jnp
import jax.random as jrandom
from jax import vmap, jit, grad
from typing import Any, Tuple 
from networks.utils import relu_apply, _init_linear_layer_weights, build_mlp



class MLP:
    def __init__(self, in_dim, out_dim, layers,  key, scale=1e-4):
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.layers = layers
        

        self.params = build_mlp(in_dim, out_dim, layers, scale, key)

    batch_apply = jit(vmap(relu_apply, in_axes=(None, 0)))
    bb_apply = jit(vmap(vmap(relu_apply, in_axes=(None,0)), in_axes=(None, 0)))
    bbb_apply = jit(vmap(vmap(vmap(relu_apply, in_axes=(None, 0)), in_axes=(None, 0)), in_axes=(None, 0)))
    
    
    def __call__(self, x):
        if jnp.ndim(x) == 1:
            return relu_apply(self.params, x)
        elif jnp.dim(x)==2:
            return self.batch_apply(self.params, x)
        elif jnp.dim(x) ==3:
            return self.bb_apply(self.params, x)
        elif jnp.dim(x)== 4:
            return self.bbb_apply(self.params, x)
        else:
            raise NotImplementedError("Passed argument has dimension higher than 3")
        