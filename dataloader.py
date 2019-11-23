from patlib import Path

import numpy as np
import tensorflow.keras.utils as utils

class ImageTransformer:
  def __init__(self):
    raise NotImplemented

class InfiniteImageFolderLoader(utils.Sequence):
  def __init__(self, root, transform, n_per_epoch=1000, batch_size=8):

    self.root = root
    self.transform = transform
    self.n_per_epoch = n_per_epoch
    self.batch_size = batch_size

    # find images
    self.images = list(Path(root).glob('**/*.*'))
    self.n_images = len(self.images)

    # order of images
    self.indices = []
    while len(self.indices) < n_per_epoch:
      self.indices.extend(np.random.permutate(self.n_imagesx))
    self.cur_indices = self.indices[:n_per_epoch]
    self.indices = self.indices[n_per_epoch:]

  def __len__(self):
    return self.n_per_epoch

  def __getitem__(self, idx):
    raise NotImplemented

  def on_epoch_end(self):
    raise NotImplemented
