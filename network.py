from tensorflow.keras.applications.vgg19 import VGG19

import tensorflow.keras as keras
import tensorflow.keras.layers as layers

class Encoder(keras.Model):
  def __init__(self, name='encoder', encoder_layers=['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1'], input_shape=(None, None, 3), **kwargs):
    assert len(encoder_layers) > 0, 'No "encoder_layers" is provided.'
    
    super(Encoder, self).__init__(name=name, **kwargs)
    vgg = VGG19(input_tensor=layers.Input(shape=input_shape))
    output_layers = [vgg.get_layer(layer_name).output for layer_name in encoder_layers]
    self.encoder = keras.Model(inputs=vgg.input, outputs=output_layers)

  def call(self, x):
    return self.encoder(x)

class Decoder(keras.Model):
  def __init__(self, name='decoder', **kwargs):
    super(Decoder, self).__init__(name=name, **kwargs)

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
