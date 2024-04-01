from pyexpat import features
from typing import List
import jax
import jax.numpy as jnp
from flax import linen as nn

"""--------------------Auxiliary Functions---------------------"""
def interpolate_2d(x):
  B, H, W, C = x.shape
  new_shape = (B, 2 * H, 2 * W, C)
  return jax.image.resize(x, new_shape, method="bilinear")

def get_sinusoidal_time_embedding(timesteps, embedding_dim: int):
  half_dim = embedding_dim // 2
  ## compute w
  emb = jnp.log(10000) / (half_dim - 1)
  emb = jnp.exp(jnp.arange(half_dim) * -emb)
  ## compute w*t
  emb = timesteps[:, None] @ emb[None, :]
  ## compute embedding
  emb = jnp.concatenate((jnp.sin(emb)[:,:,None], jnp.cos(emb)[:,:,None]), axis=2, dtype=jnp.float32)
  emb = emb.reshape(len(timesteps), 2 * half_dim)
  if embedding_dim % 2 == 1:
    jnp.pad(emb, [[0, 0], [0, 1]])
  return emb

"""-------------------Components of Networks-------------------"""
class DownCovBlock(nn.Module):
  features: int

  @nn.compact
  def __call__(self, x, bottle_neck=False):
    x = nn.Conv(features=self.features, kernel_size=(3, 3))(x)
    x = nn.GroupNorm(num_groups=32, epsilon=1e-06)(x)
    x = nn.swish(x)
    x = nn.Conv(features=self.features, kernel_size=(3, 3))(x)
    x = nn.swish(x)
    if not bottle_neck:
      x_crop = x.copy()
      x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
      return x, x_crop
    return x

class UpCovBlock(nn.Module):
    features: int

    @nn.compact
    def __call__(self, x, x_crop):
        x = interpolate_2d(x)
        x = nn.GroupNorm(num_groups=32, epsilon=1e-06)(x)
        x = nn.ConvTranspose(features=self.features, kernel_size=(2, 2))(x)
        x = jnp.concatenate((x, x_crop), axis=-1)
        x = nn.Conv(features=self.features, kernel_size=(3, 3))(x)
        x = nn.swish(x)
        x = nn.Conv(features=self.features, kernel_size=(3, 3))(x)
        x = nn.swish(x)

        return x

class SinusoidalPosEmb(nn.Module):

  @nn.compact
  def __call__(self, timesteps, embedding_dim):
    emb = get_sinusoidal_time_embedding(timesteps, embedding_dim)
    emb = nn.Dense(features=embedding_dim * 4)(emb)
    emb = nn.swish(emb)
    emb = nn.Dense(features=embedding_dim * 4)(emb)
    return emb

class AttentionBlock(nn.Module):

  @nn.compact
  def __call__(self, x):
    B, H, W, C = x.shape
    h = nn.GroupNorm(num_groups=32, epsilon=1e-06)(x)
    h = h.reshape(B, H*W, C)
    h = nn.SelfAttention(1)(x)
    h = h.reshape(B, H, W, C)
    return h + x

class MultiHeadAttentionBlock(nn.Module):
  num_heads: int

  @nn.compact
  def __call__(self, x):
    B, H, W, C = x.shape
    h = nn.GroupNorm(num_groups=32, epsilon=1e-06)(x)
    h = nn.SelfAttention(self.num_heads)(h)
    h = h.reshape(B, H, W, C)
    return h + x
  
class SimpleResdualBlock(nn.Module):
  feature: int

  @nn.compact
  def __call__(self, x,):
    h = nn.GroupNorm(num_groups=32, epsilon=1e-06)(x)
    h = nn.swish(h)
    h = nn.Conv(features=self.feature, kernel_size=(3, 3))(h)
    h = nn.GroupNorm(num_groups=32, epsilon=1e-06)(x)
    h = nn.swish(h)
    h = nn.Conv(features=self.feature, kernel_size=(3, 3))(h)
    if h.shape[-1] != x.shape[-1]:
      x = nn.Conv(features=self.feature, kernel_size=(1, 1))(x)
    return h + x

class ResidualBlock(nn.Module):
  features: int
  dp_rate: float

  @nn.compact
  def __call__(self, x, temb, train):
    h = nn.GroupNorm(num_groups=32, epsilon=1e-06)(x)
    h = nn.swish(h)
    h = nn.Conv(features=self.features, kernel_size=(3, 3))(h)
    h += nn.Dense(features=self.features)(temb)[:, None, None, :]
    h = nn.GroupNorm(num_groups=32, epsilon=1e-06)(h)
    h = nn.swish(h)
    h = nn.Dropout(rate=self.dp_rate, deterministic=not train)(h)
    h = nn.Conv(features=self.features, kernel_size=(3, 3))(h)

    if h.shape[-1] != x.shape[-1]:
      x = nn.Conv(features=self.features, kernel_size=(1, 1))(x)
    return h + x

class DownSample(nn.Module):
  
  @nn.compact
  def __call__(self, x):
    _, _, _, C = x.shape
    x = nn.Conv(features=C, kernel_size=(3, 3))(x)
    x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2), padding='SAME')

    return x

class UpSample(nn.Module):

  @nn.compact
  def __call__(self, x):
    _, _, _, C = x.shape
    x = nn.ConvTranspose(features=C, kernel_size=(3, 3), strides=(2, 2))(x)
    return x

class DownSampling(nn.Module):
  features_list: List
  num_heads: int

  @nn.compact
  def __call__(self, x, temb, dp_rate, train):
    x_crops = []
    for features in self.features_list:
      x = ResidualBlock(features, dp_rate)(x, temb, train)
      x = MultiHeadAttentionBlock(self.num_heads)(x)
      x_crops.append(x)
      x = DownSample()(x)
    return x, x_crops

class UpSampling(nn.Module):
  features_list: List
  num_heads: int

  @nn.compact
  def __call__(self, x, x_crops, temb, dp_rate, train):
    for features in self.features_list:
      x = UpSample()(x)
      x = jnp.concatenate((x, x_crops.pop()), axis=-1)
      x = ResidualBlock(features, dp_rate)(x, temb, train)
      x = MultiHeadAttentionBlock(self.num_heads)(x)
    return x

class MiddleBottleNeck(nn.Module): 
  num_heads: int

  @nn.compact
  def __call__(self, x, temb, dp_rate, train):
    _, _, _, C = x.shape
    x = ResidualBlock(C, dp_rate)(x, temb, train)
    x = MultiHeadAttentionBlock(self.num_heads)(x)
    x = ResidualBlock(C, dp_rate)(x, temb, train)

    return x

  

if __name__ == "__main__":
  '''
    jcov = DownSample([64, 128, 256, 512])
    params = jcov.init(jax.random.PRNGKey(0), jnp.ones([1, 32, 32, 3]))['params']
    x = jax.random.normal(jax.random.PRNGKey(0), (1, 32, 32, 3))
    xt, crops= jcov.apply({'params': params}, x)
    print(xt.shape)
    for c in crops:
        print(c.shape)
    crops.reverse()
    for c in crops:
        print(c.shape)
    upcov = UpSample([512, 256, 128, 64])
    params = upcov.init(jax.random.PRNGKey(0), jnp.ones([1, 2, 2, 1024]), crops)['params']
    x = jax.random.normal(jax.random.PRNGKey(0), (1, 2, 2, 1024))
    xi = upcov.apply({'params': params}, x, crops)
    print(xi.shape)
  '''

  """test for ResidualBlock"""
  '''
  testRes = ResidualBlock(64)
  params = testRes.init(jax.random.PRNGKey(0), jnp.ones([1, 32, 32, 64]), jnp.array([500]))['params']
  x = jax.random.normal(jax.random.PRNGKey(1), (1, 32, 32, 64))
  xt = testRes.apply({'params': params}, x, jnp.array([500]))
  print(xt.shape)
  '''
  """test for AttentionBlock"""
  '''
  testAtt = AttentionBlock()
  params = testAtt.init(jax.random.PRNGKey(0), jnp.ones([1, 32, 32, 64]))['params']
  x = jax.random.normal(jax.random.PRNGKey(1), (1, 32, 32, 64))
  xt = testAtt.apply({'params': params}, x)
  print(xt.shape)
  '''
  timeemb = SinusoidalPosEmb()
  params = timeemb.init(jax.random.PRNGKey(0), jnp.array([500]), 64)['params']
  temb = timeemb.apply({'params': params}, jnp.array([500]), 64)
  block = DownSampling([64, 128, 256, 512])
  params = block.init(jax.random.PRNGKey(0), jnp.ones([1, 32, 32, 64]), [], temb)['params']
  x = jax.random.normal(jax.random.PRNGKey(1), (1, 32, 32, 64))
  xt, xts = block.apply({'params': params}, x, [], temb)
  for x in xts:
    print(x.shape)
  block2 = MiddleBottleNeck()
  params = block2.init(jax.random.PRNGKey(0), xt, temb)['params']
  xt = block2.apply({'params': params}, xt, temb)
  block3 = UpSampling([512, 256, 128, 64])
  params = block3.init(jax.random.PRNGKey(3), xt, xts, temb)['params']
  xt = block3.apply({'params': params}, xt, xts, temb)


  print(xt.shape)





    