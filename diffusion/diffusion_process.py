import jax
from jax import numpy as jnp

from diffusion.diffusion_utils import img_to_onedsample, get_noisy_sample, onedsample_to_img, norm_neg_one2one, norm_maxmin, norm_mean2zero
#from diffusion.distribution_utils import get_mean_cov, gaussian_noise_generator

#from diffusion_utils import img_to_onedsample, get_noisy_sample, onedsample_to_img, norm_neg_one2one, norm_maxmin
#from distribution_utils import get_mean_cov, gaussian_noise_generator

@jax.jit
def gaussian_diffusion_process(img, noise_key, alphas_cum_prod):
  noise_t = jax.random.normal(noise_key, img.shape, dtype=jnp.float32)
  xt = norm_neg_one2one(img)
  xt = get_noisy_sample(xt, noise_t, alphas_cum_prod)
  
  return noise_t, xt

@jax.jit
def gaussian_reverse_process(xt, z_rng, eps, comean, coeps, covar):
  z = jax.random.normal(z_rng, xt.shape, dtype=jnp.float32)
  x = comean * (xt - coeps * eps) + covar * z

  return x

@jax.jit
def normalized_gaussian_diffusion_process(img, noise_key, alphas_cum_prod):
  noise_t = jax.random.normal(noise_key, img.shape, dtype=jnp.float32)
  noise_t = norm_neg_one2one(norm_maxmin(noise_t))
  xt = norm_neg_one2one(img)
  xt = get_noisy_sample(xt, noise_t, alphas_cum_prod)
  xt = norm_neg_one2one(norm_maxmin(xt))
  
  return noise_t, xt

@jax.jit
def normalized_gaussian_reverse_process(xt, z_rng, eps, comean, coeps, covar):
  z = jax.random.normal(z_rng, xt.shape, dtype=jnp.float32)
  x = comean * (norm_neg_one2one(norm_maxmin(xt)) \
    - coeps * eps) \
    + covar * norm_neg_one2one(norm_maxmin(z)) ### coefficient of z also can be (1− ̄αt−1)/(1− ̄α)t*β
  '''
  x = comean * (xt - coeps * eps) + covar * norm_neg_one2one(norm_maxmin(z))
  '''
  return norm_neg_one2one(norm_maxmin(x))
  #return jnp.clip(x, a_min=-1, a_max=1)
  #return x

@jax.jit
def ddim_reverse_process(xt, z_rng, eps, co1st, co2nd, co3rd, covar):
  z = jax.random.normal(z_rng, xt.shape, dtype=jnp.float32)
  x = co1st * xt - co2nd * eps + co3rd * eps + covar * z

  return x




if __name__ == "__main__":
    from jax import vmap
    from jax import numpy as jnp
    from PIL import Image
    import numpy as onp
    from diffusion_utils import get_beta_cosine, get_alpha, get_timestep_samples, norm_zero2one
    '''
    imgs = []
    for i in range(10):
        img = Image.open(f"/home/wangzq/workspace/dataset/cifar10_imgs/train/automobile/{i+28:04d}.png")
        img = onp.array(img) /255.
        imgs.append(img)
    imgs = onp.array(imgs)
    '''
    

    '''
    t = get_timestep_samples(t_key, imgs.shape[0], timesteps)
    print(t)
    
    _, imgs = vmap(gaussian_diffusion_process)(imgs, noise_keys, alphas_cum_prod[t])
    for i, img in enumerate(imgs):
        img = Image.fromarray(onp.array(img).astype(onp.uint8))
        img.save(f'out/sample{i}.png')
    '''
    
    x = Image.open("/home/wangzq/workspace/dataset/celebahq_imgs/data256x256/00006.jpg")
    #x = Image.open("/home/wangzq/workspace/dataset/cifar10_imgs/train/automobile/0028.png")
    #x = jax.image.resize(onp.array(x), (128, 128, 3), method="bilinear")
    x = onp.array(x) / 255.

    print(x.reshape(-1).max(), x.reshape(-1).min(), x.reshape(-1).mean())

    timesteps = 1000
    beta_1 = 1e-4
    beta_t = 0.02
    rng = jax.random.PRNGKey(0)
    t_key, noise_key = jax.random.split(rng)
    
    betas = get_beta_cosine(timesteps)
    alphas_cum_prod = get_alpha(betas, cum_prod=True)

    for timestep, alpha_prod, in zip(range(timesteps), alphas_cum_prod):
        noise_key, noise_key_current = jax.random.split(noise_key)
        _, img = gaussian_diffusion_process(x, noise_key_current, alpha_prod)
        print(timestep, img.max(), img.min(), img.mean())
        if (timestep+1) % 50 == 0:
            img = Image.fromarray(onp.array(norm_zero2one(img)*255.).astype(onp.uint8))
            img.save(f'out/step{timestep}.png')
            print(f'out/step{timestep}.png saved')
    



    

