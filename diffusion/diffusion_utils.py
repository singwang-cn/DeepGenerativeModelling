from typing import Tuple
from functools import partial
import jax
import jax.numpy as jnp

import diffusion

def norm_neg_one2one(x):
  return x * 2. - 1.

def norm_zero2one(x):
  return (x + 1.) * 0.5

def norm_maxmin(x):
  return (x - x.min()) / (x.max() - x.min() + 1e-20)

def norm_mean2zero(x):
  return x - x.mean()

def flatten_len(shape_tuple):
    len = 1
    for dim in shape_tuple:
        len *= dim
    return len

def img_to_onedsample(img):
    return jnp.array(norm_neg_one2one(img.reshape(-1)))

def get_beta_linear(num_timesteps, beta_1 = 1e-4, beta_t = 2e-2):
  return jnp.linspace(beta_1, beta_t, num_timesteps, dtype=jnp.float32)

def get_beta_cosine(num_timesteps, offset_s = 8e-3, pow = 2.):
  betas = jnp.arange(num_timesteps + 1) / num_timesteps + offset_s
  betas = betas / (1. + offset_s) * jnp.pi / 2.
  betas = jnp.power(jnp.cos(betas), pow) / betas[0]
  betas = 1 - betas[1:] / betas[:-1]
  betas = jnp.clip(betas, 0, 0.999)
  return betas

def get_alpha(betas, cum_prod=False):
    alphas = 1. - betas
    if cum_prod:
        return jnp.cumprod(alphas)
    else:
        return alphas

def get_timestep_samples(t_key, batch_size, timesteps, offset=0):
    return jax.random.randint(t_key, (batch_size,), offset, timesteps-1, dtype=jnp.int32)

def onedsample_to_img(x, shape_img: Tuple):
  xt = x.reshape(shape_img)
  xt = norm_zero2one(xt)
  return xt

def get_noisy_sample(x, noise, alphas_cum_prod):
  x_part = jnp.sqrt(alphas_cum_prod) * x
  noise_part = jnp.sqrt(1. - alphas_cum_prod) * noise
  return x_part + noise_part

def get_norm_noise_batch(rng, dummy, sample_size):
  _, H, W, C = dummy.shape
  noise_batch = jax.random.normal(rng, jnp.array([sample_size, H, W, C]), dtype=jax.numpy.float32)

  return noise_batch
