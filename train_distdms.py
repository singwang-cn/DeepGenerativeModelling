import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] ='true'
os.environ['XLA_PYTHON_CLIENT_ALLOCATOR']='platform'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import time

### for debugging nan
#from jax.config import config
#config.update("jax_debug_nans", True)
#config.update("jax_enable_x64", True) ### Force to set global double precision
import numpy as onp
import jax
from jax import numpy as jnp
import optax
from functools import partial

from diffusion.diffusion_utils import get_beta_cosine, get_beta_linear, get_alpha, get_timestep_samples

from model.models import ResNet, Unet

from data.datasets import DatasetFromRGBIamges
from data.loader import NumpyLoader
from data.data_utils import Normalize, NormalizeAndResize, get_dataset_path

from train.train_utils import create_train_state, create_learning_rate_fn
#from train.train_utils import train_epoch, eval_epoch
from train.sample_utils import sampling_ddpm, sampling_ddim, sampling_sub
from train.train_utils import save_model, load_model, save_sample_images
from train.train_utils import get_logger, log_message
from diffusion.diffusion_process import gaussian_diffusion_process

from absl import app
from absl import flags
import wandb

FLAGS = flags.FLAGS

### data
flags.DEFINE_enum('dataset', default='cifar10', enum_values=['cifar10', 'cifar100', 'imagenet', 'celebahq256', 'lsun_bedroom', 'lsun_church'], help='Dataset for training')
flags.DEFINE_integer('batch_size', default=512, help='Batch size for training')
flags.DEFINE_integer('sample_size', default=32, help='Batch size for sampling')
flags.DEFINE_list('input_size', default=[32, 32], help='Resolution of input')
flags.DEFINE_integer('num_channels', default=3, help='Number of channels of input')
### training
flags.DEFINE_integer('num_epochs', default=500, help='Number of epochs to train for')
flags.DEFINE_bool('do_val', default=False, help='Whether to do validation during training')
flags.DEFINE_integer('val_epochs', default=50, help='Interval in epochs for validation')
flags.DEFINE_bool('save_ckpts', default=False, help='Whether to save checkpoints during training')
flags.DEFINE_float('dropout_rate', default=0.1, help='Dropout rate for the network')
flags.DEFINE_enum('optimizer', default='adamw', enum_values=['adamw', 'sgd'], help='Optimizer to apply')
flags.DEFINE_enum('scheduler', default='constant', enum_values=['constant', 'cosine'], help='Scheduler to apply')
flags.DEFINE_float('learning_rate', default=2e-4, help='Initial learning rate')
flags.DEFINE_float('weight_decay', default=1e-4, help='Weight decay coefficient for regularization')
flags.DEFINE_float('momentum', default=0.9, help='Momentum factor for SGD optimizer')
flags.DEFINE_float('warmup_length', default=0, help='Number of epochs for linear warmup')
flags.DEFINE_bool('wandb_on', default=False, help='Whether to use wandb')
### diffusion
flags.DEFINE_enum('noise_schedule_type', default='linear', enum_values=['cosine', 'linear'], help='Schedule type for noise')
flags.DEFINE_enum('logit_type', default='x_pre', enum_values=['x_pre', 'epsilon'], help='Target type for prediction')
flags.DEFINE_integer('num_timesteps', default=1000, help='Number of timesteps')
flags.DEFINE_integer('skip_timesteps', default=0, help='Number of timesteps to skip')
flags.DEFINE_float('beta_1', default=1e-4, help='Beta 1')
flags.DEFINE_float('beta_t', default=0.02, help='Beta T')
### model
flags.DEFINE_enum('model_type', default='unet', enum_values=['vae', 'unet', 'resnet'], help='Model to train')
flags.DEFINE_list('feature_size', default=[64, 128, 256, 512], help='Number of feature maps')
flags.DEFINE_integer('num_heads_att', default=1, help='Number of heads for attention layers')
flags.DEFINE_integer('num_models', default=2, help='Number of models')

exp_num = 2
logger = get_logger(f"./log{exp_num}")

@partial(jax.jit, static_argnums=(0,))
def single_accelaretor_update_xpre(model, state, img, noise_rngs, alphas_cum_prod_t, alphas_cum_prod_dt):
  def loss_fn(params):
    logits = model.apply(params, imgs_dt)
    loss = jnp.mean(optax.l2_loss(predictions=jnp.ravel(logits), targets=jnp.ravel(imgs_t)))
    return loss
  _, imgs_t = jax.vmap(gaussian_diffusion_process)(img, noise_rngs[0], alphas_cum_prod_t)
  _, imgs_dt = jax.vmap(gaussian_diffusion_process)(img, noise_rngs[1], alphas_cum_prod_dt)
  grad_fn = jax.value_and_grad(loss_fn)
  loss, grads = grad_fn(state.params)
  state = state.apply_gradients(grads=grads)
  
  return state, loss

@partial(jax.jit, static_argnums=(0,))
def single_accelaretor_update_epsilon(model, state, img, noise_rngs, alphas_cum_prod_t, alphas_cum_prod_dt):
  def loss_fn(params):
    logits = model.apply(params, imgs_dt)
    loss = jnp.mean(optax.l2_loss(predictions=jnp.ravel(logits), targets=jnp.ravel(noise)))
    return loss
  noise, imgs_dt = jax.vmap(gaussian_diffusion_process)(img, noise_rngs[1], alphas_cum_prod_dt)
  grad_fn = jax.value_and_grad(loss_fn)
  loss, grads = grad_fn(state.params)
  state = state.apply_gradients(grads=grads)
  
  return state, loss

@partial(jax.jit, static_argnums=(0, 8))
def single_accelaretor_update_unet_xpre(model, state, img, noise_rngs, alphas_cum_prod_t, alphas_cum_prod_dt, dt, drop_rng, dp_rate):
  def loss_fn(params):
    logits = model.apply(params, imgs_dt, dt, dp_rate=dp_rate, train=True, rngs={'dropout': drop_rng})
    loss = jnp.mean(optax.l2_loss(predictions=jnp.ravel(logits), targets=jnp.ravel(imgs_t)))
    return loss
  _, imgs_t = jax.vmap(gaussian_diffusion_process)(img, noise_rngs[0], alphas_cum_prod_t)
  _, imgs_dt = jax.vmap(gaussian_diffusion_process)(img, noise_rngs[1], alphas_cum_prod_dt)
  grad_fn = jax.value_and_grad(loss_fn)
  loss, grads = grad_fn(state.params)
  state = state.apply_gradients(grads=grads)
  
  return state, loss

@partial(jax.jit, static_argnums=(0, 8))
def single_accelaretor_update_unet_epsilon(model, state, img, noise_rngs, alphas_cum_prod_t, alphas_cum_prod_dt, dt, drop_rng, dp_rate):
  def loss_fn(params):
    logits = model.apply(params, imgs_dt, dt, dp_rate=dp_rate, train=True, rngs={'dropout': drop_rng})
    loss = jnp.mean(optax.l2_loss(predictions=jnp.ravel(logits), targets=jnp.ravel(noise)))
    return loss
  noise, imgs_dt = jax.vmap(gaussian_diffusion_process)(img, noise_rngs[1], alphas_cum_prod_dt)
  grad_fn = jax.value_and_grad(loss_fn)
  loss, grads = grad_fn(state.params)
  state = state.apply_gradients(grads=grads)
  
  return state, loss

def train_step(rng, cfg, model, state, img_batch, num_acce, alphas_cum_prod, sub_timesteps, offset_timesteps):
  """Train for a single step."""
  start_time = time.time()
  loss = 0.
  B = cfg.batch_size
  large_batch_size, H, W, C = img_batch.shape
  ### For single accelaretor ###
  if cfg.model_type == 'unet': rng, drop_rng = jax.random.split(rng)
  noise_rngs = jax.random.split(rng, large_batch_size * 2)
  noise_rngs = noise_rngs.reshape(2, -1, 2)
  dt = jnp.broadcast_to(offset_timesteps+sub_timesteps, large_batch_size)
  alphas_cum_prod_t = jnp.broadcast_to(alphas_cum_prod[offset_timesteps], large_batch_size)
  alphas_cum_prod_dt = jnp.broadcast_to(alphas_cum_prod[offset_timesteps+sub_timesteps], large_batch_size)
  if cfg.model_type == 'unet':
    if cfg.logit_type == 'x_pre': update_fn = single_accelaretor_update_unet_xpre
    elif cfg.logit_type == 'epsilon': update_fn = single_accelaretor_update_unet_epsilon
    state, loss = update_fn(model, state, img_batch, noise_rngs, alphas_cum_prod_t, alphas_cum_prod_dt, dt, drop_rng, cfg.dropout_rate)
  else:
    if cfg.logit_type == 'x_pre': update_fn = single_accelaretor_update_xpre
    elif cfg.logit_type == 'epsilon': update_fn = single_accelaretor_update_epsilon
    state, loss = update_fn(model, state, img_batch, noise_rngs, alphas_cum_prod_t, alphas_cum_prod_dt)

  metrics = {
    'loss': loss,
    'step_time': time.time() - start_time,
  }

  return state, metrics

def train_step_2models(rng, cfg, model, state, img_batch, num_acce, alphas_cum_prod, sub_timesteps, offset_timesteps):
  """Train for a single step."""
  start_time = time.time()
  loss = 0.
  B = cfg.batch_size
  large_batch_size, H, W, C = img_batch.shape
  t_rng, noise_rng, drop_rng = jax.random.split(rng, 3)
  ### For single accelaretor ###
  noise_rngs = jax.random.split(noise_rng, large_batch_size * 2)
  noise_rngs = noise_rngs.reshape(2, -1, 2)
  t = jax.random.randint(t_rng, (large_batch_size,), offset_timesteps, offset_timesteps+sub_timesteps, dtype=jnp.int32)
  alphas_cum_prod_t = alphas_cum_prod[t-1]
  alphas_cum_prod_dt = alphas_cum_prod[t]
  if cfg.model_type == 'unet':
    if cfg.logit_type == 'x_pre': update_fn = single_accelaretor_update_unet_xpre
    elif cfg.logit_type == 'epsilon': update_fn = single_accelaretor_update_unet_epsilon
    state, loss = update_fn(model, state, img_batch, noise_rngs, alphas_cum_prod_t, alphas_cum_prod_dt, t, drop_rng, cfg.dropout_rate)
  else:
    if cfg.logit_type == 'x_pre': update_fn = single_accelaretor_update_xpre
    elif cfg.logit_type == 'epsilon': update_fn = single_accelaretor_update_epsilon
    state, loss = update_fn(model, state, img_batch, noise_rngs, alphas_cum_prod_t, alphas_cum_prod_dt)

  metrics = {
    'loss': loss,
    'step_time': time.time() - start_time,
  }

  return state, metrics

def train_epoch(logger, cfg, rng, model, state, learning_rate_fn, train_loader, num_acce, alphas_cum_prod, sub_timesteps, offset_timesteps):
  """Train for a single epoch."""
  start_time = time.time()
  batch_loss = []
  train_fn = train_step
  for img_batch in train_loader:
    #if img_batch.shape[0] < cfg.batch_size * num_acce: continue
    rng, step_rng = jax.random.split(rng)
    state, step_metrics = train_fn(step_rng, cfg, model, state, img_batch, num_acce, alphas_cum_prod, sub_timesteps, offset_timesteps)
    batch_loss.append(step_metrics['loss'])

    step_metrics['step'] = state.step
    log_message(logger, step_metrics, "train_step")

  mean_loss = onp.array(batch_loss).reshape(-1).mean()
  epoch_time = time.time() - start_time

  metrics = {
      'learning_rate': learning_rate_fn(state.step),
      'loss': mean_loss,
      'time': epoch_time,
  }
  return state, metrics

def run(argv):
  if FLAGS.wandb_on: wandb.init(
    project="distributed_ldm",
    config=FLAGS,
    job_type="diffusion_train",
    name=f"{FLAGS.model_type}_{FLAGS.feature_size}_{FLAGS.num_heads_att}_{FLAGS.dataset}_{FLAGS.input_size}_{FLAGS.noise_schedule_type}_b{FLAGS.batch_size}_{time.strftime('%Y%m%d_%H%M%S')}"
  )

  if FLAGS.wandb_on: cfg = wandb.config
  else: cfg = FLAGS

  ### data variables
  train_data_path, valid_data_path = get_dataset_path(cfg)
  img_out_path = f"/home/wangzq/workspace/diffusion_jax/out/exp{exp_num}"
  ckpt_path = f"/home/wangzq/workspace/diffusion_jax/check_points/exp{exp_num}"
  num_acce = jax.local_device_count() ### set to 1 if want to run on single accelaretor

  if cfg.noise_schedule_type == "linear":
    betas = get_beta_linear(cfg.num_timesteps, cfg.beta_1, cfg.beta_t)
  elif cfg.noise_schedule_type == "cosine":
    betas = get_beta_cosine(cfg.num_timesteps, offset_s = 8e-3, pow = 2.)
  alphas = get_alpha(betas)
  alphas_cum_prod = get_alpha(betas, cum_prod=True)

  train_ds = DatasetFromRGBIamges(train_data_path, transform=NormalizeAndResize(cfg.input_size))
  trainloader = NumpyLoader(dataset=train_ds, batch_size=cfg.batch_size * num_acce, shuffle=True)
  if cfg.do_val:
    valid_ds = DatasetFromRGBIamges(valid_data_path, transform=NormalizeAndResize(cfg.input_size))
    validloader = NumpyLoader(dataset=valid_ds, batch_size=cfg.batch_size * num_acce, shuffle=True)

  feature_size = tuple(cfg.feature_size) ### tuple for jit
  model_squeue = {}
  state_squeue = {}
  rng = jax.random.PRNGKey(0)
  rng, init_rng = jax.random.split(rng)
  init_rng = jax.random.split(init_rng, cfg.num_models)
  dummy = jnp.ones([1, cfg.input_size[0], cfg.input_size[1], cfg.num_channels])
  learning_rate_fn = create_learning_rate_fn(cfg, len(train_ds)//(cfg.batch_size * num_acce))
  for idx in range(cfg.num_models):
    #model_squeue[f'{idx:04d}'] = Unet(idx, feature_size, cfg.num_heads_att)
    model_squeue[f'{idx:04d}'] = Unet(feature_size, cfg.num_heads_att)
    state_squeue[f'{idx:04d}'] = create_train_state(init_rng[idx], dummy, model_squeue[f'{idx:04d}'], cfg, learning_rate_fn)
  del init_rng
  #state = load_model(ckpt_path, state)
  
  
  log_message(logger, f"{FLAGS.model_type}_{FLAGS.feature_size}_{FLAGS.num_heads_att}\n\
              {FLAGS.dataset}_{FLAGS.input_size}_b{FLAGS.batch_size}\n\
              {time.strftime('%Y%m%d_%H%M%S')}")
  log_message(logger, "Model Initialized.")
  log_metrics = {}

  #for idx in range(1):
  #for idx in range(19, cfg.num_models):
  for idx in range(cfg.num_models):
    model = model_squeue[f'{idx:04d}']
    state = state_squeue[f'{idx:04d}']
    sub_timesteps = cfg.num_timesteps // cfg.num_models
    offset_timesteps = sub_timesteps * idx

    for epoch in range(1, cfg.num_epochs + 1):
      log_message(logger, f"Epoch {epoch} started.")
      rng, epoch_rng = jax.random.split(rng)
      # Run an optimization step over a training batch
      state, train_metrics = train_epoch(logger, cfg, epoch_rng, model, state, learning_rate_fn, trainloader, num_acce, alphas_cum_prod, sub_timesteps, offset_timesteps)
      train_metrics['epoch'] = epoch
      log_message(logger, train_metrics, "train_epoch")
      log_metrics['Train Loss'] = train_metrics['loss']
      log_metrics['Epoch Time'] = train_metrics['time']
      log_metrics['Learning Rate'] = train_metrics['learning_rate']
      '''
      if cfg.do_val and epoch % cfg.val_epochs == 0:
        # Evaluate on the test set after each training epoch
        valid_metrics = eval_epoch(rng, cfg, model, state, validloader, cfg.num_timesteps, alphas_cum_prod)
        valid_metrics['epoch'] = epoch
        log_message(logger, valid_metrics, "valid_epoch")
        log_metrics['Validation Loss'] = valid_metrics['loss']
      '''
      if FLAGS.wandb_on: wandb.log(log_metrics, step=idx*cfg.num_epochs+epoch)
    if cfg.save_ckpts: log_message(logger, save_model(state=state, ckpt_dir=ckpt_path, step=epoch, prefix=f'ckp_{idx}_'))
    model_squeue[f'{idx:04d}'] = model
    state_squeue[f'{idx:04d}'] = state
  
  rng, sample_rng = jax.random.split(rng)
  log_message(logger, "Start sampling")
  samples = sampling_sub(rng, model_squeue, state_squeue, dummy, cfg.sample_size)
  save_sample_images(cfg, samples, img_out_path)
  log_message(logger, "Sampling finished")
  
if __name__ == "__main__":
  app.run(run)

