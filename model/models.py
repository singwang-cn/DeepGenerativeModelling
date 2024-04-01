from typing import List, Tuple, Union
import jax
import jax.numpy as jnp
from flax import linen as nn

from model.model_utils import DownSampling, UpSampling, SinusoidalPosEmb, MiddleBottleNeck
from model.model_utils import SimpleResdualBlock, AttentionBlock, MultiHeadAttentionBlock, DownSample, UpSample
#from model_utils import DownSampling, UpSampling, SinusoidalPosEmb, MiddleBottleNeck, SimpleResdualBlock
#from model_utils import SimpleResdualBlock, AttentionBlock, DownSample, UpSample

class Unet(nn.Module):
  features: List
  num_heads: int

  @nn.compact
  def __call__(self, x, timesteps, dp_rate=0.0, train=False):
    temb = SinusoidalPosEmb()(timesteps, self.features[0])

    x = nn.Conv(features=self.features[0], kernel_size=(3, 3))(x)
    x, x_crops = DownSampling(self.features, self.num_heads)(x, temb, dp_rate, train)
    x = MiddleBottleNeck(self.num_heads)(x, temb, dp_rate, train)
    reversed_features_list = reversed(self.features)
    x = UpSampling(reversed_features_list, self.num_heads)(x, x_crops, temb, dp_rate, train)

    x = nn.GroupNorm(num_groups=32, epsilon=1e-06)(x)
    x = nn.swish(x)
    x = nn.Conv(features=3, kernel_size=(3, 3))(x)

    return x
  
class Encoder(nn.Module):
  features: List
  out_feature: int

  @nn.compact
  def __call__(self, x):
    x = nn.Conv(features=self.features[0], kernel_size=(3, 3))(x)
    for feature in self.features:
      x = SimpleResdualBlock(feature)(x)
      x = DownSample()(x)
    x = SimpleResdualBlock(self.features[-1])(x)
    x = AttentionBlock()(x)
    x = SimpleResdualBlock(self.features[-1])(x)
    x = nn.GroupNorm(num_groups=32, epsilon=1e-06)(x)
    x = nn.swish(x)
    x = nn.Conv(features=2*self.out_feature, kernel_size=(3, 3))(x)

    return x
  
class Decoder(nn.Module):
  features: List
  img_feature: int

  @nn.compact
  def __call__(self, x):
    x = nn.Conv(features=self.features[0], kernel_size=(3, 3))(x)
    x = SimpleResdualBlock(self.features[0])(x)
    x = AttentionBlock()(x)
    x = SimpleResdualBlock(self.features[0])(x)
    for feature in self.features:
      x = SimpleResdualBlock(feature)(x)
      x = UpSample()(x)
    
    x = nn.GroupNorm(num_groups=32, epsilon=1e-06)(x)
    x = nn.swish(x)
    x = nn.Conv(features=self.img_feature, kernel_size=(3, 3))(x)

    return x

class ResNet(nn.Module):
  index: int
  features: List
  num_heads: int

  @nn.compact
  def __call__(self, x):
    h = nn.Conv(features=self.features[0], kernel_size=(3, 3))(x)
    for feature in self.features:
      h = SimpleResdualBlock(feature)(h)
      h = MultiHeadAttentionBlock(self.num_heads)(h)
      h = SimpleResdualBlock(feature)(h)
    h = nn.Conv(features=3, kernel_size=(3, 3))(h)
    '''
    ### x_tau = Axt -  Beps + Ceps + 0
    B, H, W, C = h.shape
    h = h.reshape(B, H*W*C)
    h = nn.Dense(features=H*W*C)(h)
    h = nn.swish(h)
    x = x.reshape(B, H*W*C)
    x = nn.Dense(features=H*W*C)(x-h)
    x = x.reshape(B, H, W, C)
    '''
    return x



if __name__ == "__main__":
    '''
    jcov = Encoder([64, 128, 256], 512)
    params = jcov.init(jax.random.PRNGKey(0), jnp.ones([1, 32, 32, 3]))
    x = jax.random.normal(jax.random.PRNGKey(0), (1, 32, 32, 3))
    #print(x.reshape(-1).max(), x.reshape(-1).min())
    xt = jcov.apply(params, x)
    #print(xt.reshape(-1).max(), xt.reshape(-1).min())
    print(xt.shape)
    '''
    x = jnp.array([[[1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6]],[[1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6]],[[1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6]],[[1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6]]])
    z = gaussian_reparam(jax.random.PRNGKey(0), x)
    print(jnp.sum(z))
    
    
