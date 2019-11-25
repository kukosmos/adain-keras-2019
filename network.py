from tensorflow.keras.applications.vgg19 import VGG19

import tensorflow.keras.backend as K
import tensorflow.keras.layers as layers
import tensorflow.keras.models as models

class ReflectionPad(layers.Layer):
  def __init__(self, padding, name='reflection', *args, **kwargs):
    super(ReflectionPad, self).__init__(name=name, **kwargs)
    self.pad_left, self.pad_right, self.pad_top, self.pad_bottom = padding

  def compute_output_shape(self, input_shape):
    return (input_shape[0], input_shape[1] + self.pad_left + self.pad_right, input_shape[2], self.pad_top, self.pad_bottom, input_shape[3])

  def call(self, x):
    x = K.concatenate([K.reverse(x, 1)[:, (-1 - self.pad_left):-1, :, :], x, K.reverse(x, 1)[:, 1:(1 + self.pad_right), :, :]], axis=1)
    x = K.concatenate([K.reverse(x, 2)[:, :, (-1 - self.pad_top):-1, :], x, K.reverse(x, 2)[:, :, 1:(1 + self.pad_bottom), :]], axis=2)
    return x

class AdaIN(layers.Layer):
  def __init__(self, name='adain', alpha=1.0, **kwargs):
    super(AdaIN, self).__init__(name=name, **kwargs)
    self.alpha = alpha

  def compute_output_shape(self, input_shape):
    return input_shape[0]

  def call(self, x):
    content_features, style_features = x
    content_mean = K.mean(content_features, axis=[1, 2], keepdim=True)
    content_var = K.variance(content_features, axis=[1, 2], keepdim=True)
    style_mean = K.mean(style_features, axis=[1, 2], keepdim=True)
    style_var = K.variance(style_features, axis=[1, 2], keepdim=True)
    normalized_content_features = K.batch_normalization(content_features, content_mean, content_var, style_mean, K.sqrt(style_var), epsilon=1e-5)
    return self.alpha * normalized_content_features + (1 - self.alpha) * content_features

class Encoder(models.Model):
  def __init__(self, encoder_layers=['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1'], input_shape=(None, None, 3), pretrained=True, name='encoder', **kwargs):
    assert len(encoder_layers) > 0, 'No "encoder_layers" is provided.'
    
    vgg = VGG19(input_tensor=layers.Input(shape=input_shape), weights=('imagenet' if pretrained else None), include_top=False)
    output_layers = [vgg.get_layer(layer_name).output for layer_name in encoder_layers]
    super(Encoder, self).__init__(inputs=vgg.input, outputs=output_layers, name=name, **kwargs)

class Decoder(models.Model):
  def __init__(self, name='decoder', **kwargs):
    super(Decoder, self).__init__(name=name, **kwargs)
    self.decoder = models.Sequential([
      ReflectionPad((1, 1, 1, 1)),
      layers.Conv2D(256, (3, 3)),
      layers.UpSampling2D(size=2, interpolation='nearest'),
      ReflectionPad((1, 1, 1, 1)),
      layers.Conv2D(256, (3, 3)),
      ReflectionPad((1, 1, 1, 1)),
      layers.Conv2D(256, (3, 3)),
      ReflectionPad((1, 1, 1, 1)),
      layers.Conv2D(256, (3, 3)),
      ReflectionPad((1, 1, 1, 1)),
      layers.Conv2D(128, (3, 3)),
      layers.UpSampling2D(size=2, interpolation='nearest'),
      ReflectionPad((1, 1, 1, 1)),
      layers.Conv2D(128, (3, 3)),
      ReflectionPad((1, 1, 1, 1)),
      layers.Conv2D(64, (3, 3)),
      layers.UpSampling2D(size=2, interpolation='nearest'),
      ReflectionPad((1, 1, 1, 1)),
      layers.Conv2D(64, (3, 3)),
      ReflectionPad((1, 1, 1, 1)),
      layers.Conv2D(3, (3, 3))
    ])

  def call(self, x):
    return self.decoder(x)
    
class Stylizer(models.Model):
  def __init__(self, alpha=1.0, pretrained=True, input_shape=(None, None, 3), name='stylizer', **kwargs):
    super(Stylizer, self).__init__(name=name, **kwargs)
    self.encoder = Encoder(input_shape=input_shape, pretrained=pretrained)
    if pretrained:
      for l in self.encoder.layers:
        l.trainable = False
    self.adain = AdaIN(alpha=alpha)
    self.decoder = Decoder()

  def call(self, x):
    contents, styles = x
    content_features = self.encoder(contents)
    style_features = self.encoder(styles)
    normalized_features = self.adain([content_features, style_features])
    return self.decoder(normalized_features)

  def set_alpha(self, alpha=1.0):
    self.adain.alpha = alpha
