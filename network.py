from tensorflow.keras.applications.vgg19 import VGG19

import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import tensorflow.keras.models as models

class Encoder(keras.Model):
  def __init__(self, name='encoder', **kwargs):
    super(Encoder, self).__init__(name=name, **kwargs)

  def call(self):
    raise NotImplemented

class Decoder(keras.Model):
  def __init__(self, name='decoder', **kwargs):
    super(Decoder, self).__init__(name=name, **kwargs)
    self.decoder = models.Sequential([
      layers.Conv2D,
      layers.Upsample(size=2, interpolation='nearest'),
      layers.Conv2D,
      layers.Conv2D,
      layers.Conv2D,
      layers.Conv2D,
      layers.Upsample(size=2, interpolation='nearest'),
      layers.Conv2D,
      layers.Conv2D,
      layers.Upsample(size=2, interpolation='nearest'),
      layers.Conv2D,
      layers.Conv2D
    ])

  def call(self):
    raise NotImplemented

class AdaIN(layers.Layer):
  def __init__(self, name='adain', **kwargs):
    super(AdaIN, self).__init__(name=name, **kwargs)

  def call(self):
    raise NotImplemented

class Stylizer(keras.Model):
  def __init__(self, name='stylizer', **kwargs):
    super(Stylizer, self).__init__(name=name, **kwargs)

  def call(self):
    raise NotImplemented
