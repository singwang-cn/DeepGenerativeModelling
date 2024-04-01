from typing import Tuple
import jax
import jax.numpy as jnp

from diffusion.diffusion_utils import flatten_len
#from diffusion_utils import flatten_len

def split_mean_var(z):
  mean, log_var = jnp.split(z, 2, axis=-1)
  log_var = jnp.clip(log_var, -30.0, 20.0)
  std = jnp.exp(0.5 * log_var)
  return mean, std

def gaussian_reparam(rng, mean, std):
  return mean + std * jax.random.normal(rng, std.shape, dtype=jnp.float32)

def gaussian_kl(mu, sigmasq, coef=1.0e-03):
  """KL divergence from a diagonal Gaussian to the standard Gaussian."""
  return coef * jnp.mean(-0.5 * jnp.sum(1. + jnp.log(sigmasq) - mu**2. - sigmasq, axis=(1, 2, 3)), axis=0)

'''
def get_mean_cov(shape_input: Tuple):
    len_input = flatten_len(shape_input)
    mean = jnp.zeros(len_input, dtype=jnp.float32)
    cov = jnp.eye(N = len_input, dtype=jnp.float32)
    return mean, cov

def gaussian_noise_generator(rng, mean, cov):
    noise = jax.random.multivariate_normal(rng, mean, cov, dtype=jnp.float32, method='cholesky')
    return noise
'''

