import jax
import jax.numpy as jnp
import jax.random as jrandom
from jax import vmap, jit, grad
from typing import Any, Tuple 
from networks.utils import relu_apply, _init_linear_layer_weights, build_mlp


def gaussian_apply(params, x):
    x = relu_apply(params[:-2], x)

    mu = jnp.dot(params[-2][0], x) + params[-2][1]

    zus = jnp.concatenate([x, mu], axis=-1)

    sigma = jnp.dot(params[-1][0], zus) + params[-1][0]
    sigma = jnp.where(sigma < 0, jnp.exp(sigma), sigma+1)

    return mu, sigma

def gaussian_sample(key, mu, sigma):
    return mu + sigma*jnp.normal(key, sigma.shape())

def gaussian_logprob(mu, sigma, x):
    pdf = 1/(sigma*jnp.sqrt(2*jnp.pi))*jnp.exp( -1 * jnp.square((x-mu)/sigma) )

    return jnp.log(pdf)



batch_eval_lp = vmap(gaussian_logprob, in_axes=(None, None, 0))
bb_eval_lp = vmap(vmap(gaussian_logprob, in_axes=(None, None, 0)), in_axes=(None, None, 0))
bbb_eval_lp = vmap(vmap(vmap(gaussian_logprob, in_axes=(None, None, 0)), in_axes=(None, None, 0)), in_axes=(None, None, 0))

batch_lp = vmap(gaussian_logprob, in_axes=(0,0,0))
bb_lp = vmap(vmap(gaussian_logprob, in_axes=(0,0,0)), in_axes=(0,0,0))
bbb_lp = vmap(vmap(vmap(gaussian_logprob, in_axes=(0,0,0)), in_axes=(0,0,0)), in_axes=(0,0,0))

class GaussianMLP:
    def __init__(self, in_dim, layers,  key, scale=1e-4):
        self.in_dim = in_dim
        self.layers = layers
        

        trunk_key,mu_key, sigma_key = jrandom.split(key)
        self.params = build_mlp(in_dim, layers[-1], layers[:-1], scale, trunk_key)

        self.mu_layer = _init_linear_layer_weights(layers[-1], 1, scale, mu_key)
        self.sigma_layer = _init_linear_layer_weights(layers[-1]+1, 1, scale=mu_key)

    
    

    batch_apply = jit(vmap(gaussian_apply, in_axes=(None, 0)))
    bb_apply = jit(vmap(vmap(gaussian_apply, in_axes=(None,0)), in_axes=(None, 0)))
    bbb_apply = jit(vmap(vmap(vmap(gaussian_apply, in_axes=(None, 0)), in_axes=(None, 0)), in_axes=(None, 0)))
    
    
    def __call__(self, x):
        if jnp.ndim(x) == 1:
            return gaussian_apply(self.params, x)
        elif jnp.dim(x)==2:
            return self.batch_apply(self.params, x)
        elif jnp.dim(x) ==3:
            return self.bb_apply(self.params, x)
        elif jnp.dim(x)== 4:
            return self.bbb_apply(self.params, x)
        else:
            raise NotImplementedError("Passed argument has dimension higher than 3")
    
    def eval_log_prob(self, x, y):
        mus, sigmas = self.__call__(x)

        if jnp.ndim(x) == 1:
            return gaussian_logprob(mus, sigmas, y)
        elif jnp.dim(x)==2:
            return self.batch_eval_lp(mus, sigmas, y)
        elif jnp.dim(x) ==3:
            return self.bb_eval_lp(mus, sigmas, y)
        elif jnp.dim(x)== 4:
            return self.bbb_eval_lp(mus, sigmas, y)
        else:
            raise NotImplementedError("Passed argument has dimension higher than 3")

    def log_prob(self, x, y):
        mus, sigmas = self.__call__(x)

        if jnp.ndim(x) == 1:
            return gaussian_logprob(mus, sigmas, y)
        elif jnp.dim(x)==2:
            return self.batch_lp(mus, sigmas, y)
        elif jnp.dim(x) ==3:
            return self.bb_lp(mus, sigmas, y)
        elif jnp.dim(x)== 4:
            return self.bbb_lp(mus, sigmas, y)
        else:
            raise NotImplementedError("Passed argument has dimension higher than 3")
