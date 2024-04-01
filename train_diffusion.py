import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] ='true'
os.environ['XLA_PYTHON_CLIENT_ALLOCATOR']='platform'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import time

### for debugging nan
#from jax.config import config
#config.update("jax_debug_nans", True)
#config.update("jax_enable_x64", True) ### Force to set global double precision
import jax
from jax import numpy as jnp

from diffusion.diffusion_utils import get_beta_cosine, get_beta_linear, get_alpha

from model.models import Unet

from data.datasets import DatasetFromRGBIamges
from data.loader import NumpyLoader
from data.data_utils import Normalize, NormalizeAndResize, get_dataset_path

from train.train_utils import create_train_state, create_learning_rate_fn
from train.train_utils import train_epoch, eval_epoch
from train.sample_utils import sampling_ddpm, sampling_ddim, sampling_xpre
from train.train_utils import save_model, load_model, save_sample_images
from train.train_utils import get_logger, log_message

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
flags.DEFINE_integer('num_epochs', default=1000, help='Number of epochs to train for')
flags.DEFINE_bool('do_val', default=False, help='Whether to do validation during training')
flags.DEFINE_integer('val_epochs', default=100, help='Interval in epochs for validation')
flags.DEFINE_bool('save_ckpts', default=False, help='Whether to save checkpoints during training')
flags.DEFINE_float('dropout_rate', default=0.1, help='Dropout rate for the network')
flags.DEFINE_enum('optimizer', default='adamw', enum_values=['adamw', 'sgd'], help='Optimizer to apply')
flags.DEFINE_enum('scheduler', default='constant', enum_values=['constant', 'cosine'], help='Scheduler to apply')
flags.DEFINE_float('learning_rate', default=2e-4, help='Initial learning rate')
flags.DEFINE_float('weight_decay', default=1e-4, help='Weight decay coefficient for regularization')
flags.DEFINE_float('momentum', default=0.9, help='Momentum factor for SGD optimizer')
flags.DEFINE_float('warmup_length', default=100, help='Number of epochs for linear warmup')
flags.DEFINE_bool('wandb_on', default=False, help='Whether to use wandb')
### diffusion
flags.DEFINE_enum('noise_schedule_type', default='linear', enum_values=['cosine', 'linear'], help='Schedule type for noise')
flags.DEFINE_enum('logit_type', default='x_pre', enum_values=['x_pre', 'epsilon'], help='Target type for prediction')
flags.DEFINE_integer('num_timesteps', default=1000, help='Number of timesteps')
flags.DEFINE_integer('skip_timesteps', default=0, help='Number of timesteps to skip')
flags.DEFINE_float('beta_1', default=1e-4, help='Beta 1')
flags.DEFINE_float('beta_t', default=0.02, help='Beta T')
### model
flags.DEFINE_enum('model_type', default='unet', enum_values=['vae', 'unet'], help='Model to train')
flags.DEFINE_list('feature_size', default=[64, 128, 256, 512], help='Number of feature maps')
flags.DEFINE_integer('num_heads_att', default=8, help='Number of heads for attention layers')

exp_num = 8
logger = get_logger(f"./log{exp_num}")

def run(argv):
  if FLAGS.wandb_on: wandb.init(
    project="distributed_ldm",
    config=FLAGS,
    job_type="diffusion_train",
    name=f"{FLAGS.model_type}_{FLAGS.feature_size}_{FLAGS.num_heads_att}_{FLAGS.dataset}_{FLAGS.input_size}_b{FLAGS.batch_size}_{time.strftime('%Y%m%d_%H%M%S')}"
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

  ###model variables
  feature_size = tuple(cfg.feature_size) ### tuple for jit
  model = Unet(feature_size, cfg.num_heads_att)

  train_ds = DatasetFromRGBIamges(train_data_path, transform=NormalizeAndResize(cfg.input_size))
  trainloader = NumpyLoader(dataset=train_ds, batch_size=cfg.batch_size * num_acce, shuffle=True)
  if cfg.do_val:
    valid_ds = DatasetFromRGBIamges(valid_data_path, transform=NormalizeAndResize(cfg.input_size))
    validloader = NumpyLoader(dataset=valid_ds, batch_size=cfg.batch_size * num_acce, shuffle=True)

  rng = jax.random.PRNGKey(0)
  rng, init_rng = jax.random.split(rng)
  dummy = jnp.ones([1, cfg.input_size[0], cfg.input_size[1], cfg.num_channels])

  learning_rate_fn = create_learning_rate_fn(cfg, len(train_ds)//(cfg.batch_size * num_acce))
  state = create_train_state(init_rng, dummy, model, cfg, learning_rate_fn)
  del init_rng
  #state = load_model(ckpt_path, state)

  log_message(logger, f"{FLAGS.model_type}_{FLAGS.feature_size}_{FLAGS.num_heads_att}\n\
              {FLAGS.dataset}_{FLAGS.input_size}_b{FLAGS.batch_size}\n\
              {time.strftime('%Y%m%d_%H%M%S')}")
  log_message(logger, "Model Initialized.")
  log_metrics = {}

  for epoch in range(1, cfg.num_epochs + 1):
    log_message(logger, f"Epoch {epoch} started.")
    rng, epoch_rng = jax.random.split(rng)
    # Run an optimization step over a training batch
    state, train_metrics = train_epoch(logger, cfg, epoch_rng, model, state, learning_rate_fn, trainloader, num_acce, alphas_cum_prod)
    train_metrics['epoch'] = epoch
    log_message(logger, train_metrics, "train_epoch")
    log_metrics['Train Loss'] = train_metrics['loss']
    log_metrics['Epoch Time'] = train_metrics['time']
    log_metrics['Learning Rate'] = train_metrics['learning_rate']
    
    if cfg.save_ckpts and epoch % cfg.val_epochs == 0:
      log_message(logger, save_model(state=state, ckpt_dir=ckpt_path, step=epoch))

    if cfg.do_val and epoch % cfg.val_epochs == 0:
      # Evaluate on the test set after each training epoch
      valid_metrics = eval_epoch(rng, cfg, model, state, validloader, cfg.num_timesteps, alphas_cum_prod)
      valid_metrics['epoch'] = epoch
      log_message(logger, valid_metrics, "valid_epoch")
      log_metrics['Validation Loss'] = valid_metrics['loss']
    
    if FLAGS.wandb_on: wandb.log(log_metrics, step=epoch)

  rng, sample_rng = jax.random.split(rng)
  log_message(logger, "Start sampling")
  if cfg.logit_type == 'epsilon':
    samples = sampling_ddpm(logger, sample_rng, model, state.params, dummy, alphas, alphas_cum_prod, betas, cfg.num_timesteps-cfg.skip_timesteps, cfg.sample_size, show_step=100)
  elif cfg.logit_type == 'x_pre':
    samples = sampling_xpre(logger, sample_rng, model, state.params, dummy, alphas, alphas_cum_prod, betas, cfg.num_timesteps-cfg.skip_timesteps, cfg.sample_size, show_step=100)
  save_sample_images(cfg, samples, img_out_path)
  log_message(logger, "Sampling completed")

if __name__ == "__main__":
  app.run(run)

