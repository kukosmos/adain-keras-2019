from pathlib import Path

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

class ContentStyleLoader(utils.Sequence):
  def __init__(self, content_root=None, content_image_shape=None, content_crop=None, content_crop_size=256, style_root=None, style_image_shape=None, style_crop=None, style_crop_size=256, n_per_epoch=1000, batch_size=8):

    # training settings
    self.n_per_epoch = n_per_epoch
    self.batch_size = batch_size
    if batch_size > n_per_epoch:
      raise ValueError('The batch is greater than the total images per epoch: {} > {}'.format(batch_size, n_per_epoch))

    # content images
    self.content_root = content_root
    content_root = Path(content_root)
    if not content_root.exists():
      raise ValueError('The content root directory is not exist: {}'.format(content_root))
    self.content_images = list(content_root.glob('**/*.*'))
    self.n_content = len(self.content_images)

    # order of content images
    self.content_indices = []
    while len(self.content_indices) < n_per_epoch:
      self.content_indices.extend(np.random.permutation(self.n_content))
    self.cur_content_indices = self.content_indices[:n_per_epoch]
    self.content_indices = self.content_indices[n_per_epoch:]

    # content image transfomer: path -> np.array
    if content_image_shape is not None and len(content_image_shape) != 2:
      raise ValueError('The content_image_shape should be None (to use original shape of the image) or 2 dimensional tuple: {}'.format(content_image_shape))
    if not (content_crop is None or isinstance(content_crop_size, int) or len(content_crop_size) == 1 or len(content_crop_size == 2)):
      raise ValueError('The dimension of the cropped content image must be 1 (rectangle), 2: {}'.format(content_crop_size))
    self.content_transform = [
      lambda x: image.load_img(x, target_size=content_image_shape),
      lambda x: image.img_to_array(x)
    ]
    if content_crop is None:
      pass
    elif content_crop == 'random':
      self.content_transform.append(lambda x: random_crop(x, size=content_crop_size))
    elif content_crop == 'center':
      self.content_transform.append(lambda x: center_crop(x, size=content_crop_size))
    else:
      raise ValueError('Unsuppored crop option: {}'.format(content_crop))

    # content images
    self.style_root = style_root
    style_root = Path(style_root)
    if not style_root.exists():
      raise ValueError('The style root directory is not exist: {}'.format(style_root))
    self.style_images = list(style_root.glob('**/*.*'))
    self.n_style = len(self.style_images)

    # order of style images
    self.style_indices = []
    while len(self.style_indices) < n_per_epoch:
      self.style_indices.extend(np.random.permutation(self.n_style))
    self.cur_style_indices = self.style_indices[:n_per_epoch]
    self.style_indices = self.style_indices[n_per_epoch:]

    # style image transfomer: path -> np.array
    if style_image_shape is not None and len(style_image_shape) != 2:
      raise ValueError('The style_image_shape should be None (to use original shape of the image) or 2 dimensional tuple: {}'.format(style_image_shape))
    if not (style_crop is None or isinstance(style_crop_size, int) or len(style_crop_size) == 1 or len(style_crop_size == 2)):
      raise ValueError('The dimension of the cropped style image must be 1 (rectangle), 2: {}'.format(style_crop_size))
    self.style_transform = [
      lambda x: image.load_img(x, target_size=style_image_shape),
      lambda x: image.img_to_array(x)
    ]
    if style_crop is None:
      pass
    elif style_crop == 'random':
      self.style_transform.append(lambda x: random_crop(x, size=style_crop_size))
    elif style_crop == 'center':
      self.style_transform.append(lambda x: center_crop(x, size=style_crop_size))
    else:
      raise ValueError('Unsuppored crop option: {}'.format(style_crop))

  def __len__(self):
    return math.ceil(self.n_per_epoch / self.batch_size)

  def __getitem__(self, idx):
    # get content
    idx_content = self.cur_content_indices[idx * self.batch_size:(idx + 1) * self.batch_size]
    batch_content = []
    for i in idx_content:
      content = self.content_images[i]
      for fn in self.content_transform:
        content = fn(content)
      batch_content.append(content)

    # get style
    idx_style = self.cur_style_indices[idx * self.batch_size:(idx + 1) * self.batch_size]
    batch_style = []
    for i in idx_style:
      style = self.style_images[i]
      for fn in self.style_transform:
        style = fn(style)
      batch_style.append(style)

    return [np.array(batch_content), np.array(batch_style)], np.ones(self.batch_size, dtype=np.float32)

  def on_epoch_end(self):
    # order of content images
    self.content_indices = []
    while len(self.content_indices) < self.n_per_epoch:
      self.content_indices.extend(np.random.permutation(self.n_content))
    self.cur_content_indices = self.content_indices[:self.n_per_epoch]
    self.content_indices = self.content_indices[self.n_per_epoch:]

    # order of style images
    self.style_indices = []
    while len(self.style_indices) < self.n_per_epoch:
      self.style_indices.extend(np.random.permutation(self.n_style))
    self.cur_style_indices = self.style_indices[:self.n_per_epoch]
    self.style_indices = self.style_indices[self.n_per_epoch:]

def load_image(filepath, image_shape=None, crop=None, crop_size=None):
  filepath = Path(filepath)
  if not filepath.exists() or filepath.is_dir():
    raise ValueError('The file is not exists: {}'.format(filepath))
  if image_shape is not None and len(image_shape) != 2:
    raise ValueError('The image_shape should be None (to use original shape of the image) or 2 dimensional tuple: {}'.format(image_shape))
  img = image.load_img(filepath, target_size=image_shape)
  img = image.img_to_array(img)
  if not (crop is None or isinstance(crop_size, int) or len(crop_size) == 1 or len(crop_size == 2)):
    raise ValueError('The dimension of the cropped image must be 1 (rectangle), 2: {}'.format(crop_size))
  if crop is None:
    pass
  elif crop == 'random':
    img = random_crop(img, size=crop_size)
  elif crop == 'center':
    img = center_crop(img, size=crop_size)
  else:
    raise ValueError('Unsupported crop option: {}'.format(crop))
  return img
