import jax
from jax import numpy as jnp

from diffusion.diffusion_utils import get_beta_cosine, get_beta_linear, get_alpha, norm_neg_one2one
from train.train_utils import create_train_state, create_learning_rate_fn, save_sample_images, load_model
from train.sample_utils import sampling_ddpm, sampling_ddim
from model.models import Unet
from train.train_utils import get_logger, log_message

from absl import app
from absl import flags

FLAGS = flags.FLAGS
flags.DEFINE_enum('model_type', default='unet', enum_values=['vae', 'unet'], help='Model to train')


img_out_path = "/home/wangzq/workspace/diffusion_jax/out/test"
#img_out_path = "/home/wangzq/workspace/diffusion_jax/out/test_cifar10_ddpm"
#img_out_path = "/home/wangzq/workspace/diffusion_jax/out/test_cifar10_2step_ddim100"
#ckpt_path = "/home/wangzq/workspace/diffusion_jax/pretrained_model/checkpoint_norm_noise_celebahq"
#ckpt_path = "/home/wangzq/workspace/diffusion_jax/pretrained_model/checkpoint_norm_noise_cifar10_all"
#ckpt_path = "/home/wangzq/workspace/diffusion_jax/pretrained_model/ckpt_cifar10_automobile_64_cosine_1000ep"
#ckpt_path = "/home/wangzq/workspace/diffusion_jax/pretrained_model/ckpt_cifar10_all_64_linear_1000ep"
#ckpt_path = "/home/wangzq/workspace/diffusion_jax/pretrained_model/ckpt_celebahq_64_linear_1000ep"
#ckpt_path = "/home/wangzq/workspace/diffusion_jax/pretrained_model/ckpt_celebahq_128_linear_1200ep"
#ckpt_path = "/home/wangzq/workspace/diffusion_jax/check_points/exp3/checkpoint_1000"
#ckpt_path = "/home/wangzq/workspace/diffusion_jax/check_points/checkpoint_800"
#ckpt_path = "/home/wangzq/workspace/diffusion_jax/pretrained_model/diffusion/unet_[64, 128, 256, 512]_8 _cifar10_[32, 32]_b512 _20230506_184655/checkpoint"
#ckpt_path2 = "/home/wangzq/workspace/diffusion_jax/pretrained_model/diffusion/unetx2_[64, 128, 256, 512]_8 _cifar10_[32, 32]_b512_split200_20230527_030716/ckpt_step2"
ckpt_path1 = "/home/wangzq/workspace/diffusion_jax/check_points/exp3/checkpoint_1000"
ckpt_path2 = "/home/wangzq/workspace/diffusion_jax/check_points/exp6/checkpoint_1000"
logger = get_logger("./log")

def run(argv):
  #num_acce = jax.local_device_count() ### set to 1 if want to run on single accelaretor
  cfg = FLAGS

  batch_size = 32
  sample_size = 32
  input_size = (32, 32)
  num_channels = 3

  ### diffuison variables
  num_timesteps = 1000
  num_samplesteps = 100
  skip_timesteps = 0
  beta_1, beta_t = 1e-4, 0.02
  #betas = get_beta_cosine(num_timesteps, offset_s = 8e-3, pow = 2.)
  betas = get_beta_linear(num_timesteps, beta_1, beta_t)
  alphas = get_alpha(betas)
  alphas_cum_prod = get_alpha(betas, cum_prod=True)

  ###model variables
  feature_size = (64, 128, 256, 512)  ### tuple for jit
  model = Unet(feature_size, 8)

  rng = jax.random.PRNGKey(0)
  #rng, init_rng = jax.random.split(rng)

  dummy = jnp.ones([1, input_size[0], input_size[1], num_channels])
  state1 = load_model(ckpt_path1, None)
  state2 = load_model(ckpt_path2, None)


  print("Start sampling")
  for ite in range(sample_size//batch_size):
    rng, sample_rng = jax.random.split(rng)
    samples = sampling_ddpm(logger, sample_rng, model, state1['params'], dummy, alphas, alphas_cum_prod, betas, num_timesteps-skip_timesteps, batch_size, show_step=None)
    #samples = sampling_ddim(logger, sample_rng, model, state1['params'], dummy, alphas_cum_prod, num_timesteps-skip_timesteps, num_samplesteps, eta=0.0, sample_size=batch_size, show_step=None)
    save_sample_images(cfg, samples, img_out_path, f'sample{ite}_eps_')
    print(samples.max(), samples.min())
    '''
    for t in reversed(range(199, 201)):
      rng, step_rng = jax.random.split(rng)
      if t == 1: sig = 0
      else: sig = jnp.sqrt(betas[t])
      t_vector = jnp.broadcast_to(t, batch_size)
      samples = model.apply(state2['params'], samples, t_vector, train=False) + sig * jax.random.normal(step_rng, samples.shape, dtype=jax.numpy.float32)
    '''
    t_vector = jnp.broadcast_to(199, batch_size)
    samples = model.apply(state2['params'], samples, t_vector, train=False)

    save_sample_images(cfg, norm_neg_one2one(samples), img_out_path, f'sample{ite}_pre_')
    print(samples.max(), samples.min())

  print("Sampling finished")

if __name__ == "__main__":
  app.run(run)