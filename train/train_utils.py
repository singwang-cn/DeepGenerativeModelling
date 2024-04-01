import jax
from jax import numpy as jnp
from jax.experimental.gda_serialization.serialization import GlobalAsyncCheckpointManager
from flax import jax_utils
from flax.training import train_state, checkpoints
from flax.training import lr_schedule
import optax
from functools import partial

import os
import time
import datetime
import pickle
import logging
from PIL import Image
import numpy as onp

from diffusion.diffusion_utils import get_timestep_samples, norm_zero2one, norm_maxmin, norm_neg_one2one, get_norm_noise_batch
from diffusion.diffusion_process import gaussian_diffusion_process, gaussian_reverse_process, normalized_gaussian_diffusion_process, normalized_gaussian_reverse_process, ddim_reverse_process
from diffusion.distribution_utils import split_mean_var, gaussian_kl, gaussian_reparam


"""----------------------Cost functions----------------------"""
@jax.jit
def mse(logits_batch, labels_batch):
  # Define the squared loss for a single pair (x,y)
  def squared_error(logits, labels):
    logits = jnp.ravel(logits)
    labels = jnp.ravel(labels)
    return jnp.inner(logits-labels, logits-labels) / len(logits)
  # Vectorize the previous to compute the average of the loss on all samples.
  return jnp.mean(jax.vmap(squared_error)(logits_batch, labels_batch), axis=0)

def elbo(mu_z, sigma_z, logits, x):
  return jnp.mean(optax.l2_loss(predictions=jnp.ravel(logits), targets=jnp.ravel(x))) + gaussian_kl(mu_z, sigma_z)


"""--------------------------Training--------------------------"""
def create_train_state(rng, dummy, model, config, learning_rate_fn):
  """Creates initial `TrainState`."""
  if config.model_type == "vae" or config.model_type == "resnet":
    params = model.init(rng, dummy)
  elif config.model_type == "unet":
    params = model.init(rng, dummy, jnp.ones(dummy.shape[0]))
  ### chose the mehtod for optimizer
  if config.optimizer == "adamw":
    tx = optax.adamw(learning_rate_fn, weight_decay=config.weight_decay)
  elif config.optimizer == "sgd":
    tx = optax.sgd(learning_rate_fn, momentum=config.momentum)
  stats = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)
  return stats

def create_learning_rate_fn(config, steps_per_epoch):
  """Creates learning rate schedule."""
  '''
  warmup_fn = optax.linear_schedule(
      init_value=0., end_value=base_learning_rate,
      transition_steps=config.warmup_epochs * steps_per_epoch)
  cosine_epochs = max(config.num_epochs - config.warmup_epochs, 1)
  cosine_fn = optax.cosine_decay_schedule(
      init_value=base_learning_rate,
      decay_steps=cosine_epochs * steps_per_epoch)
  schedule_fn = optax.join_schedules(
      schedules=[warmup_fn, cosine_fn],
      boundaries=[config.warmup_epochs * steps_per_epoch])
  '''
  schedule_fn = lr_schedule.create_cosine_learning_rate_schedule(base_learning_rate=config.learning_rate,
                                                                steps_per_epoch=steps_per_epoch,
                                                                halfcos_epochs=config.num_epochs,
                                                                warmup_length=config.warmup_length)
  schedule_fn = lr_schedule.create_constant_learning_rate_schedule(base_learning_rate=config.learning_rate,
                                                                  steps_per_epoch=steps_per_epoch,
                                                                  warmup_length=config.warmup_length)
  return schedule_fn

@partial(jax.jit, static_argnums=(0, 5))
def single_accelaretor_update_eps(model, state, img, noise_rngs, drop_rng, dp_rate, alphas_cum_prod, t):
  def loss_fn(params):
    """The input here should be consistent with the variables that need to be differentiated"""
    logits = model.apply(params, noisy_imgs, t, dp_rate=dp_rate, train=True, rngs={'dropout': drop_rng})
    loss = jnp.mean(optax.l2_loss(predictions=jnp.ravel(logits), targets=jnp.ravel(noises)))
    return loss
  noises, noisy_imgs = jax.vmap(gaussian_diffusion_process)(img, noise_rngs, alphas_cum_prod[t])
  #noises, noisy_imgs = jax.vmap(normalized_gaussian_diffusion_process)(img, noise_rngs, alphas_cum_prod[t])
  grad_fn = jax.value_and_grad(loss_fn)
  loss, grads = grad_fn(state.params)
  state = state.apply_gradients(grads=grads)
  
  return state, loss

@partial(jax.jit, static_argnums=(0, 5))
def single_accelaretor_update_xpre(model, state, img, noise_rngs, drop_rng, dp_rate, alphas_cum_prod, t):
  def loss_fn(params):
    """The input here should be consistent with the variables that need to be differentiated"""
    logits = model.apply(params, noisy_imgs, t, dp_rate=dp_rate, train=True, rngs={'dropout': drop_rng})
    loss = jnp.mean(optax.l2_loss(predictions=jnp.ravel(logits), targets=jnp.ravel(noisy_imgs_pre)))
    return loss
  _, noisy_imgs = jax.vmap(gaussian_diffusion_process)(img, noise_rngs[0], alphas_cum_prod[t])
  _, noisy_imgs_pre = jax.vmap(gaussian_diffusion_process)(img, noise_rngs[1], alphas_cum_prod[t-1])
  grad_fn = jax.value_and_grad(loss_fn)
  loss, grads = grad_fn(state.params)
  state = state.apply_gradients(grads=grads)
  
  return state, loss

@partial(jax.pmap, axis_name='num_devices', static_broadcasted_argnums=(0, 5))
def data_parallel_update(model, state, img, noise_rngs, drop_rng, dp_rate, alphas_cum_prod, t):
  def loss_fn(params):
    """The input here should be consistent with the variables that need to be differentiated"""
    logits = model.apply(params, noisy_imgs, t, dp_rate=dp_rate, train=True, rngs={'dropout': drop_rng})
    loss = jnp.mean(optax.l2_loss(predictions=jnp.ravel(logits), targets=jnp.ravel(noises)))
    return loss
  noises, noisy_imgs = jax.vmap(gaussian_diffusion_process)(img, noise_rngs, alphas_cum_prod[t])
  grad_fn = jax.value_and_grad(loss_fn)
  loss, grads = grad_fn(state.params)
  grads = jax.lax.pmean(grads, axis_name='num_devices')
  loss = jax.lax.pmean(loss, axis_name='num_devices')
  state = state.apply_gradients(grads=grads)
  
  return state, loss

@partial(jax.jit, static_argnums=(1, 2))
def single_accelaretor_update_vae(rng, encoder, decoder, state_en, state_de, img):
  def loss_fn(params_en, params_de):
    """The input here should be consistent with the variables that need to be differentiated"""
    latent_params = encoder.apply(params_en, img)
    mean, std = split_mean_var(latent_params)
    z = gaussian_reparam(rng, mean, std)
    logits = decoder.apply(params_de, z)
    loss = elbo(mean, std, logits, img)
    return loss
  
  grad_fn = jax.value_and_grad(loss_fn, [0, 1])
  loss, grads = grad_fn(state_en.params, state_de.params)
  state_en = state_en.apply_gradients(grads=grads[0])
  state_de = state_de.apply_gradients(grads=grads[1])

  return state_en, state_de, loss


def train_step_unet(rng, cfg, model, state, img_batch, num_acce, alphas_cum_prod):
  """Train for a single step."""
  start_time = time.time()
  loss = 0.
  dp_rate = cfg.dropout_rate
  B = cfg.batch_size
  timesteps = cfg.num_timesteps
  large_batch_size, H, W, C = img_batch.shape
  t_rng, noise_rng, drop_rng = jax.random.split(rng, 3)

  ### For single accelaretor ###
  if num_acce == 1:
    if cfg.logit_type == 'epsilon':
      noise_rngs = jax.random.split(noise_rng, large_batch_size)
      t = get_timestep_samples(t_rng, large_batch_size, timesteps)
      state, loss = single_accelaretor_update_eps(model, state, img_batch, noise_rngs, drop_rng, dp_rate, alphas_cum_prod, t)
    elif cfg.logit_type == 'x_pre':
      noise_rngs = jax.random.split(noise_rng, large_batch_size*2)
      noise_rngs = noise_rngs.reshape(2, -1, 2)
      #t = get_timestep_samples(t_rng, large_batch_size, timesteps, offset=1)
      t = jnp.broadcast_to(199, large_batch_size)
      state, loss = single_accelaretor_update_xpre(model, state, img_batch, noise_rngs, drop_rng, dp_rate, alphas_cum_prod, t)

    

  ### For multiple accelaretors ###
  elif num_acce > 1:
    img_batch = img_batch.reshape((num_acce, B, H, W, C))
    noise_rngs = jax.random.split(noise_rng, large_batch_size).reshape((num_acce, B, -1))
    drop_rngs = jax.random.split(drop_rng, num_acce)
    t = get_timestep_samples(t_rng, large_batch_size, timesteps).reshape((num_acce, B))
    alphas_cum_prod = jnp.broadcast_to(alphas_cum_prod, (num_acce, len(alphas_cum_prod)))
    state = jax_utils.replicate(state)
    state, loss = data_parallel_update(model, state, img_batch, noise_rngs, drop_rngs, dp_rate, alphas_cum_prod, t)
    state = jax_utils.unreplicate(state)
    loss = jax_utils.unreplicate(loss)

  metrics = {
    'loss': loss,
    'step_time': time.time() - start_time,
  }

  return state, metrics

def train_step_vae(rng, cfg, model, state, img_batch, num_acce, _):
  start_time = time.time()
  loss = 0.
  state_en, state_de, loss = single_accelaretor_update_vae(rng, model['encoder'], model['decoder'], 
                                                           state['encoder'], state['decoder'], img_batch)
  state['encoder'] = state_en
  state['decoder'] = state_de
  metrics = {
    'loss': loss,
    'step_time': time.time() - start_time,
  }

  return state, metrics
  

def train_epoch(logger, cfg, rng, model, state, learning_rate_fn, train_loader, num_acce, alphas_cum_prod):
  """Train for a single epoch."""
  start_time = time.time()
  batch_loss = []
  if cfg.model_type == "vae":
    train_fn = train_step_vae
  elif cfg.model_type == "unet":
    train_fn = train_step_unet
  for img_batch in train_loader:
    #if img_batch.shape[0] < cfg.batch_size * num_acce: continue
    rng, step_rng = jax.random.split(rng)
    state, step_metrics = train_fn(step_rng, cfg, model, state, img_batch, num_acce, alphas_cum_prod)
    batch_loss.append(step_metrics['loss'])
    if cfg.model_type == "vae":
      step_metrics['step'] = state[list(state.keys())[0]].step
    elif cfg.model_type == "unet":
      step_metrics['step'] = state.step
    log_message(logger, step_metrics, "train_step")
    '''
    print(f"Train Step: {state.step}",
          f"loss@train: {step_metrics['loss']:.8f}",
          f"time: {step_metrics['step_time']:.2f} sec")
    '''

  mean_loss = onp.array(batch_loss).reshape(-1).mean()
  epoch_time = time.time() - start_time
  if cfg.model_type == "vae":
    metrics = {
      'learning_rate': learning_rate_fn(state[list(state.keys())[0]].step),
      'loss': mean_loss,
      'time': epoch_time,
    }
  elif cfg.model_type == "unet":
    metrics = {
      'learning_rate': learning_rate_fn(state.step),
      'loss': mean_loss,
      'time': epoch_time,
    }
  return state, metrics


"""-------------------------Validation-------------------------"""
@partial(jax.jit, static_argnums=(1,))
def eval_step(rng, model, state, img_batch, timesteps, alphas_cum_prod):
  t_rng, noise_rng = jax.random.split(rng)
  noise_rngs = jax.random.split(noise_rng, img_batch.shape[0])
  t = get_timestep_samples(t_rng, img_batch.shape[0], timesteps)

  noises, noisy_imgs = jax.vmap(gaussian_diffusion_process)(img_batch, noise_rngs, alphas_cum_prod[t])
  logits = model.apply(state.params, noisy_imgs, t, train=False)
  loss = jnp.mean(optax.l2_loss(predictions=jnp.ravel(logits), targets=jnp.ravel(noises)))

  return loss

def eval_step_vae(rng, model, state, img, _, __):
  encoder = model['encoder']
  decoder = model['decoder']
  @jax.jit
  def loss_fn(params_en, params_de):
    """The input here should be consistent with the variables that need to be differentiated"""
    latent_params = encoder.apply(params_en, img)
    mean, std = split_mean_var(latent_params)
    z = gaussian_reparam(rng, mean, std)
    logits = decoder.apply(params_de, z)
    loss = elbo(mean, std, logits, img)
    return loss
  loss = loss_fn(state['encoder'].params, state['decoder'].params)

  return loss

def eval_epoch(rng, cfg, model, state, valid_loader, timesteps, alphas_cum_prod):
  batch_loss = []
  if cfg.model_type == "vae":
    eval_fn = eval_step_vae
  elif cfg.model_type == "unet":
    eval_fn = eval_step
  for img_batch in valid_loader:
    if img_batch.shape[0] < cfg.batch_size: continue
    rng, step_rng = jax.random.split(rng)
    loss_per_train = eval_fn(step_rng, model, state, img_batch, timesteps, alphas_cum_prod)
    batch_loss.append(loss_per_train)
  mean_loss = onp.array(batch_loss).reshape(-1).mean()
  metrics = {
      'loss': mean_loss,
  }
  return metrics


"""------------------Saving & loading models-------------------"""
def save_model(state, ckpt_dir: str, step=0, prefix='checkpoint_'):
  ckpt_file =  checkpoints.save_checkpoint(
    ckpt_dir=ckpt_dir,
    target=state,
    step=step,
    prefix=prefix,
    keep=5,
    overwrite=True)
  return f"The state of the model has been saved as [{ckpt_file}]."

def load_model(ckpt_dir: str, dummy_state):
  return checkpoints.restore_checkpoint(ckpt_dir=ckpt_dir, target=dummy_state)


"""----------------------System utilities-----------------------"""
def save_sample_images(cfg, images, save_path, file_name='sample'):
  #images = norm_neg_one2one(norm_maxmin(images))
  if cfg.model_type == "vae":
    images = jnp.clip(images, 0., 1.)
  elif cfg.model_type == "unet" or cfg.model_type == "unfold":
    images = jnp.clip(images, -1., 1.)
    images = norm_zero2one(images)
  for i, img in enumerate(images):
    img = Image.fromarray(onp.array(img * 255.).astype(onp.uint8))
    img.save(f'{save_path}/{file_name}{i}.png')

def get_logger(log_file="./log"):
  formatter = logging.Formatter(fmt='%(asctime)s|%(name)s|%(levelname)s|%(message)s', datefmt="%Y%m%d-%H:%M:%S")

  console_handler = logging.StreamHandler()
  console_handler.setLevel(logging.INFO)
  console_handler.setFormatter(formatter)

  file_handler = logging.FileHandler(log_file, mode='w')
  file_handler.setLevel(logging.INFO)
  file_handler.setFormatter(formatter)

  logger = logging.getLogger("dpm")
  logger.setLevel(logging.DEBUG)

  logger.addHandler(console_handler)
  logger.addHandler(file_handler)

  return logger

def log_message(logger, messages, contents=None):
  if contents == None:
    msg = messages
  elif contents == "train_step":
    msg = f"Step {messages['step']} " \
          f"loss@train {messages['loss']:.8f} " \
          f"time {messages['step_time']:.2f} sec"
  elif contents == "train_epoch":
    msg = f"Epoch {messages['epoch']} " \
          f"loss@train {messages['loss']:.8f} " \
          f"lr {messages['learning_rate']:.6e} " \
          f"time {messages['time']:.2f} sec"
  elif contents == "valid_epoch":
    msg = f"Epoch {messages['epoch']} loss@valid {messages['loss']:.8f}"

  logger.info(msg=msg)





