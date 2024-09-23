import jax
import jax.numpy as jnp
import jax.random as jrandom
from jax import vmap, jit, grad
from typing import Any, Tuple 


def _init_linear_layer_weights(in_dim, out_dim, scale, key):
    w_key, b_key = jrandom.split(key)
    weights = jrandom.normal(w_key, (out_dim, in_dim))
    bias = jrandom.normal(b_key, (out_dim, ))
    return weights, bias 


def build_mlp(in_dim, out_dim, layers, scale, key):
    
    first_key, key = jrandom.split(key)
    weights = [_init_linear_layer_weights(in_dim, layers[0], scale, first_key)]

    for l in range(1, len(layers)):
        l_key, key = jrandom.split(key)
        weights.append(_init_linear_layer_weights(layers[l-1], layers[l], scale, l_key))

    weights.append(_init_linear_layer_weights(layers[-1], out_dim), scale, key)
    return weights

@jit
def relu_apply(params, x):
    for i in range(len(params)-1):
        w,b = params[i]
        x = jax.nn.relu(jnp.dot(w,x)+b)
    x = jnp.dot(params[-1][0], x ) + params[-1][1]
    return x