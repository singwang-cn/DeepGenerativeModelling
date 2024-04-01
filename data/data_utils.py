import numpy as onp
import jax
from jax import numpy as jnp

def get_dataset_path(cfg):
  if cfg.dataset == 'cifar10':
    tr_data_path = "/home/wangzq/workspace/dataset/cifar10_imgs/train/"
    va_data_path = "/home/wangzq/workspace/dataset/cifar10_imgs/val"
  elif cfg.dataset == 'cifar100':
    tr_data_path = "/home/wangzq/workspace/dataset/cifar100_imgs/train"
    va_data_path = "/home/wangzq/workspace/dataset/cifar100_imgs/val"
  elif cfg.dataset == 'imagenet':
    tr_data_path = "/mnt/nfs/datasets/ILSVRC2012/train/"
    va_data_path = "/mnt/nfs/datasets/ILSVRC2012/val/"
  elif cfg.dataset == 'celebahq256':
    tr_data_path = "/home/wangzq/workspace/dataset/celebahq_imgs/data256x256"
    va_data_path = None
  elif cfg.dataset == 'lsun_bedroom':
    tr_data_path = "/home/wangzq/workspace/dataset/lsun/imgs/bedroom/train"
    va_data_path = "/home/wangzq/workspace/dataset/lsun/imgs/bedroom/val"
  elif cfg.dataset == 'lsun_church':
    tr_data_path = "/home/wangzq/workspace/dataset/lsun/imgs/church/train"
    va_data_path = "/home/wangzq/workspace/dataset/lsun/imgs/church/val"
  return tr_data_path, va_data_path

class FlattenAndCast(object):
  def __call__(self, pic):
    return onp.ravel(onp.array(pic, dtype=jnp.float32))

class Normalize(object):
  def __call__(self, pic):
    return onp.array(pic / 255., dtype=jnp.float32)

class NormalizeAndResize(object):
  def __init__(self, shape) -> None:
    self.shape = shape
  def __call__(self, pic):
    _, _, C = pic.shape
    pic = onp.array(pic / 255., dtype=jnp.float32)
    pic = jax.image.resize(pic, (self.shape[0], self.shape[1], C), method="bilinear")
    return pic
