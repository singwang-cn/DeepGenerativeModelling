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
from functools import partial

from diffusion.diffusion_utils import get_beta_cosine, get_beta_linear, get_alpha

from model.models import ResNet

from data.datasets import DatasetFromRGBIamges
from data.loader import NumpyLoader
from data.data_utils import Normalize, NormalizeAndResize, get_dataset_path

from train.train_utils import create_train_state, create_learning_rate_fn
#from train.train_utils import train_epoch, eval_epoch
from train.sample_utils import sampling_ddpm, sampling_ddim, sampling_sub
from train.train_utils import save_model, load_model, save_sample_images
from train.train_utils import get_logger, log_message
from diffusion.diffusion_process import gaussian_diffusion_process, ddim_reverse_process
from diffusion.diffusion_utils import get_norm_noise_batch

from absl import app
from absl import flags
import wandb

FLAGS = flags.FLAGS

### data
flags.DEFINE_enum('dataset', default='cifar10', enum_values=['cifar10', 'cifar100', 'imagenet', 'celebahq256', 'lsun_bedroom', 'lsun_church'], help='Dataset for training')
flags.DEFINE_integer('batch_size', default=1024, help='Batch size for training')
flags.DEFINE_integer('sample_size', default=32, help='Batch size for sampling')
flags.DEFINE_list('input_size', default=[32, 32], help='Resolution of input')
flags.DEFINE_integer('num_channels', default=3, help='Number of channels of input')
### training
flags.DEFINE_integer('num_epochs', default=50, help='Number of epochs to train for')
flags.DEFINE_bool('do_val', default=False, help='Whether to do validation during training')
flags.DEFINE_integer('val_epochs', default=50, help='Interval in epochs for validation')
flags.DEFINE_bool('save_ckpts', default=False, help='Whether to save checkpoints during training')
flags.DEFINE_float('dropout_rate', default=0.1, help='Dropout rate for the network')
flags.DEFINE_enum('optimizer', default='adamw', enum_values=['adamw', 'sgd'], help='Optimizer to apply')
flags.DEFINE_enum('scheduler', default='constant', enum_values=['constant', 'cosine'], help='Scheduler to apply')
flags.DEFINE_float('learning_rate', default=2e-3, help='Initial learning rate')
flags.DEFINE_float('weight_decay', default=1e-4, help='Weight decay coefficient for regularization')
flags.DEFINE_float('momentum', default=0.9, help='Momentum factor for SGD optimizer')
flags.DEFINE_float('warmup_length', default=0, help='Number of epochs for linear warmup')
flags.DEFINE_bool('wandb_on', default=False, help='Whether to use wandb')
### diffusion
flags.DEFINE_enum('noise_schedule_type', default='linear', enum_values=['cosine', 'linear'], help='Schedule type for noise')
flags.DEFINE_enum('logit_type', default='epsilon', enum_values=['x_pre', 'epsilon'], help='Target type for prediction')
flags.DEFINE_integer('num_timesteps', default=1000, help='Number of timesteps')
flags.DEFINE_integer('skip_timesteps', default=0, help='Number of timesteps to skip')
flags.DEFINE_float('beta_1', default=1e-4, help='Beta 1')
flags.DEFINE_float('beta_t', default=0.02, help='Beta T')
### model
flags.DEFINE_enum('model_type', default='resnet', enum_values=['vae', 'unet', 'resnet'], help='Model to train')
flags.DEFINE_list('feature_size', default=[128], help='Number of feature maps')
flags.DEFINE_integer('num_heads_att', default=1, help='Number of heads for attention layers')
flags.DEFINE_integer('num_models', default=20, help='Number of models')


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
  img_out_path = "/home/wangzq/workspace/diffusion_jax/out/exp6"
  ckpt_path = "/home/wangzq/workspace/diffusion_jax/check_points/exp6"
  num_acce = jax.local_device_count() ### set to 1 if want to run on single accelaretor

  if cfg.noise_schedule_type == "linear":
    betas = get_beta_linear(cfg.num_timesteps, cfg.beta_1, cfg.beta_t)
  elif cfg.noise_schedule_type == "cosine":
    betas = get_beta_cosine(cfg.num_timesteps, offset_s = 8e-3, pow = 2.)
  #alphas = get_alpha(betas)
  alphas_cum_prod = get_alpha(betas, cum_prod=True)

  feature_size = tuple(cfg.feature_size) ### tuple for jit
  model_squeue = {}
  state_squeue = {}
  rng = jax.random.PRNGKey(0)
  dummy = jnp.ones([1, cfg.input_size[0], cfg.input_size[1], cfg.num_channels])
  for idx in range(cfg.num_models):
    model_squeue[f'{idx:04d}'] = ResNet(idx, feature_size, cfg.num_heads_att)
    state_squeue[f'{idx:04d}'] = load_model(f'{ckpt_path}/ckp_{idx}_100', None)
  print(len(state_squeue), 'models have been loaded.')

  xt = get_norm_noise_batch(rng, dummy, cfg.sample_size)
  eta = 0
  for idx in reversed(range(cfg.num_models)):
    model = model_squeue[f'{idx:04d}']
    state = state_squeue[f'{idx:04d}']
    sub_timesteps = cfg.num_timesteps // cfg.num_models
    tau = sub_timesteps * idx
    t = tau + sub_timesteps

    eps = model.apply(state['params'], xt)
    if tau == 0:
      sigma = 0.
    else:
      sigma = eta * jnp.sqrt((1 - alphas_cum_prod[tau]) / (1 - alphas_cum_prod[t - 1])) \
            * jnp.sqrt((1 - alphas_cum_prod[t - 1] / alphas_cum_prod[tau]))
    co1st = jnp.sqrt(alphas_cum_prod[tau] / alphas_cum_prod[t - 1])
    coeps = jnp.sqrt(1 - alphas_cum_prod[t - 1])
    co2nd = jnp.sqrt(1 - alphas_cum_prod[tau] - sigma**2)
    rng, noise_key = jax.random.split(rng)
    z_rngs = jax.random.split(noise_key, cfg.sample_size)
    co1st_vector = jnp.broadcast_to(co1st, cfg.sample_size)
    co2nd_vector = jnp.broadcast_to(co2nd, cfg.sample_size)
    coeps_vector = jnp.broadcast_to(coeps, cfg.sample_size)
    covar_vector = jnp.broadcast_to(sigma, cfg.sample_size)
    xt = jax.vmap(ddim_reverse_process)(xt, z_rngs, eps, co1st_vector, co2nd_vector, coeps_vector, covar_vector)
    print('eps', eps.max(), eps.min(), eps.mean(), eps.var())
    print('xt', xt.max(), xt.min(), xt.mean(), xt.var())
  save_sample_images(cfg, xt, img_out_path)
  
if __name__ == "__main__":
  app.run(run)

