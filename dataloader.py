from patlib import Path

import numpy as np
import math
import tensorflow.keras.preprocessing.image as image
import tensorflow.keras.utils as utils

def random_crop(img, size=(256, 256)):
  H, W = img.shape[0], img.shape[1]
  if isinstance(size, int):
    if size > min(H, W):
      size = min(H, W)
    dy, dx = size, size
  elif len(size) == 2:
    dy, dx = size
    if dy > H:
      dy = H
    if dx > W:
      dx = W
  else:
    raise ValueError('The size must be an integer or a tuple (or list) of 2 integers: {}'.format(size))
  y = np.random.randint(0, H - dy + 1)
  x = np.random.randint(0, W - dx + 1)
  return img[y:(y + dy), x:(x + dx), :]

def center_crop(img, size=(256, 256)):
  H, W = img.shape[0], img.shape[1]
  if isinstance(size, int):
    if size > min(H, W):
      size = min(H, W)
    dy, dx = size, size
  elif len(size) == 2:
    dy, dx = size
    if dy > H:
      dy = H
    if dx > W:
      dx = W
  else:
    raise ValueError('The size must be an integer or a tuple (or list) of 2 integers: {}'.format(size))
  y = int((H - dy) / 2)
  x = int((W - dx) / 2)
  return img[y:(y + dy), x:(x + dx), :]

class ImageFolderLoader(utils.Sequence):
  def __init__(self, root, n_per_epoch=1000, batch_size=8, image_shape=None, crop=None, crop_size=256):

    # root directory
    self.root = root
    root = Path(root)
    if not root.exists():
      raise ValueError('The root directory is not exist: {}'.format(root))

    # training settings
    self.n_per_epoch = n_per_epoch
    self.batch_size = batch_size
    if batch_size > n_per_epoch:
      raise ValueError('The batch is greater than the total images per epoch: {} > {}'.format(batch_size, n_per_epoch))

    # image transfomer: path -> np.array
    if image_shape is not None and len(image_shape) != 2:
      raise ValueError('The image_shape should be None (to use original shape of the image) or 2 dimensional tuple: {}'.format(image_shape))
    if not (crop is None or isinstance(crop_size, int) or len(crop_size) == 1 or len(crop_size == 2)):
      raise ValueError('The dimension of the croped image must be 1 (rectangle), 2: {}'.format(crop_size))
    transform = lambda x: image.load_img(x, target_size=image_shape)
    transform = lambda x: image.img_to_array(transform(x))
    if crop is None:
      pass
    elif crop == 'random':
      transform = lambda x: random_crop(transform(x), size=crop_size)
    elif crop == 'center':
      transform = lambda x: center_crop(transform(x), size=crop_size)
    else:
      raise ValueError('Unsuppored crop option: {}'.format(crop))
    self.transform = transform
    
    # find images
    self.images = list(root.glob('**/*.*'))
    self.n_images = len(self.images)

    # order of images
    self.indices = []
    while len(self.indices) < n_per_epoch:
      self.indices.extend(np.random.permutation(self.n_images))
    self.cur_indices = self.indices[:n_per_epoch]
    self.indices = self.indices[n_per_epoch:]

  def __len__(self):
    return math.ceil(self.n_per_epoch / self.batch_size)

  def __getitem__(self, idx):
    batch_x = self.cur_indices[idx * self.batch_size:(idx + 1) * self.batch_size]
    batch_x = [self.transform(x) for x in batch_x]
    return np.array(batch_x), np.ones(self.batch_size, dtype=np.float32)

  def on_epoch_end(self):
    if len(self.indices) < self.n_per_epoch:
      while len(self.indices) < self.n_per_epoch:
        self.indices.extend(np.random.permutation(self.n_images))
    self.cur_indices = self.indices[:self.n_per_epoch]
    self.indices = self.indices[self.n_per_epoch:]
