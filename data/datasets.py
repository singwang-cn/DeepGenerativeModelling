import os
from PIL import Image
import numpy as onp
from torch.utils.data import Dataset

class DatasetFromRGBIamges(Dataset):
  def __init__(self, root, transform=None, lim_len=30000):
    self.img_dir = root
    self.imgs = []
    self.transform = transform
    for dirpath, _, filenames in os.walk(self.img_dir):
      for filename in filenames:
        if filename.split('.')[-1] in ['jpg', 'png', 'JPEG', 'webp']:
          self.imgs.append(os.path.join(dirpath, filename))
          if len(self.imgs) >= lim_len: return

  def __len__(self):
    return len(self.imgs)

  def __getitem__(self, idx):
    img_path = self.imgs[idx]
    image = Image.open(img_path)
    image = onp.array(image)
    if self.transform:
      image = self.transform(image)
    return image
        

if __name__ == "__main__":
  myds = DatasetFromRGBIamges("/home/wangzq/workspace/dataset/lsun/imgs/bedroom/train")
  print(len(myds))
  print(myds[0].shape)

