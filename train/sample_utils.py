import jax
from jax import numpy as jnp
from functools import partial

from diffusion.diffusion_utils import get_norm_noise_batch
from diffusion.diffusion_process import gaussian_reverse_process, ddim_reverse_process, normalized_gaussian_reverse_process


"""-------------------------Sampling---------------------------"""
def sampling_ddpm(logger, rng, model, params, dummy, alphas, alphas_cum_prod, betas, timesteps, sample_size=32, show_step=None):
  rng, xt_key = jax.random.split(rng)
  
  xt = get_norm_noise_batch(xt_key, dummy, sample_size)
  #xt = norm_neg_one2one(norm_maxmin(xt))
  #xt = jnp.clip(xt, -1., 1.)

  comean = 1. / jnp.sqrt(alphas)
  coeps = betas / jnp.sqrt(1. - alphas_cum_prod)
  ### simple variance
  #covar = jnp.sqrt(betas)
  ### beta_tilde variance
  covar = (1 - alphas_cum_prod[:-1]) / (1 - alphas_cum_prod[1:]) * betas[1:]
  covar = jnp.append(jnp.array([0]), covar, axis=0) ### at the last step, do not add any noise
  covar = jnp.sqrt(covar)
  '''
  xt = Image.open("/home/wangzq/workspace/diffusion_jax/out/step999.png")
  xt = onp.array(xt) / 255.
  xt = jnp.expand_dims(xt, 0)
  xt = norm_neg_one2one(xt)

  for i, img in enumerate(xt):
    img = norm_zero2one(img)
    img = Image.fromarray(onp.array(img * 255.).astype(onp.uint8))
    img.save(f'{"/home/wangzq/workspace/diffusion_jax/out"}/sample{i}_stepT.png')
  '''
  @jax.jit
  def recurrent_fn(xt, t, rng):
    rng, noise_key = jax.random.split(rng)
    z_rngs = jax.random.split(noise_key, sample_size)
    #if t == 0: z *= 0.  ### at the last step, do not add any noise
    t_vector = jnp.broadcast_to(t, sample_size)
    comean_vector = jnp.broadcast_to(comean[t], sample_size)
    coeps_vector = jnp.broadcast_to(coeps[t], sample_size)
    covar_vector = jnp.broadcast_to(covar[t], sample_size)
    eps = model.apply(params, xt, t_vector, train=False)
    xt = jax.vmap(gaussian_reverse_process)(xt, z_rngs, eps, comean_vector, coeps_vector, covar_vector)
    return xt, eps, rng
  
  for t in reversed(range(200, timesteps)):
    xt, eps, rng = recurrent_fn(xt, t, rng)
    
    if show_step is not None:
      if t % show_step == 0:
        logger.info(f"sampling step {t}\n" \
                    f"beta = {betas[t]} alpha = {alphas[t]} alphas_cum_prod = {alphas_cum_prod[t]}\n" \
                    f"epsilon {eps.max()} {eps.min()} {eps.mean()}\n" \
                    f"x(t-1) {xt.max()} {xt.min()} {xt.mean()}")
    
    ''' 
        print(f"sampling step {t}")
        print('beta =', betas[t], 'alpha =', alphas[t], 'alphas_cum_prod =', alphas_cum_prod[t])
        print("epsilon: ", eps.max(), eps.min(), eps.mean())
        print("x(t-1): ", xt.max(), xt.min(), xt.mean())
      
      
        for i, img in enumerate(xt):
          img = norm_zero2one(img)
          img = Image.fromarray(onp.array(img * 255.).astype(onp.uint8))
          img.save(f'{"/home/wangzq/workspace/diffusion_jax/out"}/sample{i}_step{t}.png')
        '''
  return xt

def sampling_ddim(logger, rng, model, params, dummy, alphas, timesteps, samplesteps, eta=0., sample_size=32, show_step=None):
  rng, xt_key = jax.random.split(rng)
  xt = get_norm_noise_batch(xt_key, dummy, sample_size)
  dt = int(timesteps / samplesteps)
  alphas = jnp.concatenate([jnp.array([1]), alphas])

  @jax.jit
  def recurrent_fn(xt, tau, rng):
    sigma = eta * jnp.sqrt((1 - alphas[tau - dt]) / (1 - alphas[tau])) \
          * jnp.sqrt((1 - alphas[tau] / alphas[tau - dt]))
    co1st = jnp.sqrt(alphas[tau - dt]) / jnp.sqrt(alphas[tau])
    co2nd = jnp.sqrt(alphas[tau - dt]) * jnp.sqrt(1 - alphas[tau]) / jnp.sqrt(alphas[tau])
    co3rd = jnp.sqrt(1 - alphas[tau - dt] - sigma**2)
    rng, noise_key = jax.random.split(rng)
    z_rngs = jax.random.split(noise_key, sample_size)
    tau_vector = jnp.broadcast_to(tau-1, sample_size)
    co1st_vector = jnp.broadcast_to(co1st, sample_size)
    co2nd_vector = jnp.broadcast_to(co2nd, sample_size)
    co3rd_vector = jnp.broadcast_to(co3rd, sample_size)
    covar_vector = jnp.broadcast_to(sigma, sample_size)
    eps = model.apply(params, xt, tau_vector, train=False)
    xt = jax.vmap(ddim_reverse_process)(xt, z_rngs, eps, co1st_vector, co2nd_vector, co3rd_vector, covar_vector)

    return xt, eps, rng

  for tau in range(timesteps, 200, -dt):
    xt, eps, rng = recurrent_fn(xt, tau, rng)

    if show_step is not None:
      if tau % show_step == 0:
        logger.info(f"sampling step {tau}\n" \
                    f"epsilon {eps.max()} {eps.min()} {eps.mean()}\n" \
                    f"x(t-1) {xt.max()} {xt.min()} {xt.mean()}")
        '''
        print(f"sampling step {tau}")
        print("sigma:", sigma)
        print("epsilon: ", eps.max(), eps.min(), eps.mean())
        print("x(t-1): ", xt.max(), xt.min(), xt.mean())
        '''

  return xt

def sampling_xpre(logger, rng, model, params, dummy, alphas, alphas_cum_prod, betas, timesteps, sample_size=32, show_step=None):
  xt = get_norm_noise_batch(rng, dummy, sample_size)
  for t in reversed(range(timesteps)):
    t_vector = jnp.broadcast_to(t, sample_size)
    xt = model.apply(params, xt, t_vector, train=False)

  return xt


def sampling_vae(rng, model, params, dummy, sample_size):
  z = get_norm_noise_batch(rng, dummy, sample_size)
  x = model.apply(params, z)

  return x

def sampling_sub(rng, model_squeue, state_squeue, dummy, sample_size):
  x = get_norm_noise_batch(rng, dummy, sample_size)
  for idx in reversed(range(len(model_squeue))):
    x = model_squeue[f'{idx:04d}'].apply(state_squeue[f'{idx:04d}'].params, x)

  return x
