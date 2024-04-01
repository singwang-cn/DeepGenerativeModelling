import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] ='false'
os.environ['XLA_PYTHON_CLIENT_ALLOCATOR']='platform'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import argparse

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
from data.data_utils import Normalize, NormalizeAndResize

from train.train_utils import create_train_state, create_learning_rate_fn
from train.train_utils import train_epoch, eval_epoch
from train.sample_utils import sampling_from_noise
from train.train_utils import save_model, load_model, save_sample_images
from train.train_utils import get_logger, log_message


def run(logger, args):
  cfg = args

  ### data variables
  #train_data_path = "/home/wangzq/workspace/dataset/cifar10_imgs/train/automobile"
  #valid_data_path = "/home/wangzq/workspace/dataset/cifar10_imgs/val/automobile"
  train_data_path = "/home/wangzq/workspace/dataset/cifar10_imgs/train/"
  valid_data_path = "/home/wangzq/workspace/dataset/cifar10_imgs/val"
  img_out_path = "/home/wangzq/workspace/diffusion_jax/out/exp3"
  ckpt_path = "/home/wangzq/workspace/diffusion_jax/check_points/exp3"
  num_acce = jax.local_device_count() ### set to 1 if want to run on single accelaretor
  batch_size = 256
  sample_size = 16
  input_size = (64, 64)
  num_channels = 3

  ### diffuison variables
  num_timesteps = 1000
  clip_timesteps = 0
  beta_1, beta_t = 1e-4, 0.02
  if cfg.noise_schedule == "linear":
    betas = get_beta_linear(num_timesteps, beta_1, beta_t)
  elif cfg.noise_schedule == "cosine":
    betas = get_beta_cosine(num_timesteps, offset_s = 8e-3, pow = 2.)
  alphas = get_alpha(betas)
  alphas_cum_prod = get_alpha(betas, cum_prod=True)

  ### train variables
  '''
  num_epochs = 1000
  val_epochs = 100
  optimizer = "adamw"
  learning_rate = 2e-5
  weight_decay = 1e-4
  '''

  ###model variables
  feature_size = (64, 128, 256, 512) ### tuple for jit
  model = Unet(feature_size)

  train_ds = DatasetFromRGBIamges(train_data_path, transform=NormalizeAndResize(input_size))
  trainloader = NumpyLoader(dataset=train_ds, batch_size=batch_size * num_acce, shuffle=True)
  valid_ds = DatasetFromRGBIamges(valid_data_path, transform=NormalizeAndResize(input_size))
  validloader = NumpyLoader(dataset=valid_ds, batch_size=batch_size * num_acce, shuffle=True)

  rng = jax.random.PRNGKey(0)
  rng, init_rng = jax.random.split(rng)
  dummy = jnp.ones([1, input_size[0], input_size[1], num_channels])

  learning_rate_fn = create_learning_rate_fn(cfg, cfg.learning_rate, len(train_ds)//(batch_size * num_acce))
  state = create_train_state(init_rng, dummy, model, cfg, learning_rate_fn)
  del init_rng
  #state = load_model(ckpt_path, state)

  log_message(logger, "Model Initialized.")

  for epoch in range(1, cfg.num_epochs + 1):
    log_message(logger, f"Epoch {epoch} started.")
    rng, epoch_rng = jax.random.split(rng)
    # Run an optimization step over a training batch
    state, train_metrics = train_epoch(logger, epoch_rng, model, state, cfg.dropout_rate, learning_rate_fn, trainloader, num_acce, batch_size, num_timesteps, alphas_cum_prod)
    train_metrics['epoch'] = epoch
    log_message(logger, train_metrics, "train_epoch")
    '''
    print(f"Epoch: {epoch}",
          f"loss@train: {train_metrics['loss']:.8f}",
          f"lr: {train_metrics['learning_rate']:.6e}",
          f"time: {train_metrics['time']:.2f} sec")
    '''
    
    if epoch % cfg.val_epochs == 0:
      #log_message(logger, save_model(state=state, ckpt_dir=ckpt_path, step=epoch))
      # Evaluate on the test set after each training epoch
      valid_metrics = eval_epoch(rng, model, state, validloader, batch_size, num_timesteps, alphas_cum_prod)
      valid_metrics['epoch'] = epoch
      log_message(logger, valid_metrics, "valid_epoch")
      #print('Epoch: %d, loss@valid: %.8f' % (epoch, valid_metrics['loss']))


  rng, sample_rng = jax.random.split(rng)
  #print("Start sampling")
  log_message(logger, "Start sampling")
  samples = sampling_from_noise(logger, sample_rng, model, state.params, dummy, alphas, alphas_cum_prod, betas, num_timesteps-clip_timesteps, sample_size, show_step=1)
  save_sample_images(samples, img_out_path)
  #print("Sampling finished")
  log_message(logger, "Sampling finished")

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--num_epochs', type=int, default=1000)
  parser.add_argument('--val_epochs', type=int, default=500)
  parser.add_argument('--dropout_rate', type=float, default=0.0)
  parser.add_argument('--optimizer', type=str, choices=['adamw', 'sgd'], default='adamw')
  parser.add_argument('--learning_rate', type=float, default=8e-5)
  parser.add_argument('--weight_decay', type=float, default=1e-4)
  parser.add_argument('--momentum', type=float, default=0.9)
  parser.add_argument('--warmup_length', type=float, default=50)
  parser.add_argument('--noise_schedule', type=str, choices=['cosine', 'linear'], default='linear')
  args = parser.parse_args()

  logger = get_logger()

  run(logger, args)

