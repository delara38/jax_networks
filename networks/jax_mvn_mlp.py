import jax
import jax.numpy as jnp
import jax.random as jrandom
from jax import vmap, jit, grad
from typing import Any, Tuple 
from networks.utils import relu_apply, _init_linear_layer_weights, build_mlp


def mvn_apply(params, x):
    x = relu_apply(params[:-2], x)

    mu = jnp.dot(params[-2][0], x) + params[-2][1]

    zus = jnp.concatenate([x, mu], axis=-1)

    sigma = jnp.dot(params[-1][0], zus) + params[-1][0]
    sigma = jnp.where(sigma < 0, jnp.exp(sigma), sigma+1)
    sigma = jnp.diagflat(sigma)


    return mu, sigma


def mvn_logprob(x, mu, cov):

    lp = jnp.power(jnp.pi, -1*jnp.shape(mu).shape[-1]/2)*jnp.power(jnp.linalg.det(cov), -0.5)*jnp.exp( -0.5* jnp.dot(x - mu, jnp.dot(jnp.linalg.inverse(cov), x-mu))  )
    return lp


def mvn_sample(mu, cov,key):
    return mu + jnp.dot(jnp.linalg.cholesky(cov), jrandom.normal(key, jnp.shape(mu)))




batch_eval_lp = jit(vmap(mvn_logprob, in_axes=(None, None, 0)))
bb_eval_lp = jit(vmap(vmap(mvn_logprob, in_axes=(None, None, 0)), in_axes=(None, None, 0)))
bbb_eval_lp = jit(vmap(vmap(vmap(mvn_logprob, in_axes=(None, None, 0)), in_axes=(None, None, 0)), in_axes=(None, None, 0)))

batch_lp = jit(vmap(mvn_logprob, in_axes=(0,0,0)))
bb_lp = jit(vmap(vmap(mvn_logprob, in_axes=(0,0,0)), in_axes=(0,0,0)))
bbb_lp = jit(vmap(vmap(vmap(mvn_logprob, in_axes=(0,0,0)), in_axes=(0,0,0)), in_axes=(0,0,0)))



class MultivariateNormalMLP:
    def __init__(self, in_dim, layers,  key, scale=1e-4):
        self.in_dim = in_dim
        self.layers = layers
        

        trunk_key,mu_key, sigma_key = jrandom.split(key)
        self.params = build_mlp(in_dim, layers[-1], layers[:-1], scale, trunk_key)

        self.mu_layer = _init_linear_layer_weights(layers[-1], 1, scale, mu_key)
        self.sigma_layer = _init_linear_layer_weights(layers[-1]+1, 1, scale=mu_key)

    
    

    batch_apply = jit(vmap(mvn_apply, in_axes=(None, 0)))
    bb_apply = jit(vmap(vmap(mvn_apply, in_axes=(None,0)), in_axes=(None, 0)))
    bbb_apply = jit(vmap(vmap(vmap(mvn_apply, in_axes=(None, 0)), in_axes=(None, 0)), in_axes=(None, 0)))
    
    
    def __call__(self, x):
        if jnp.ndim(x) == 1:
            return mvn_apply(self.params, x)
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
            return mvn_logprob(mus, sigmas, y)
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
            return mvn_logprob(mus, sigmas, y)
        elif jnp.dim(x)==2:
            return self.batch_lp(mus, sigmas, y)
        elif jnp.dim(x) ==3:
            return self.bb_lp(mus, sigmas, y)
        elif jnp.dim(x)== 4:
            return self.bbb_lp(mus, sigmas, y)
        else:
            raise NotImplementedError("Passed argument has dimension higher than 3")
