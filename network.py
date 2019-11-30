from tensorflow.keras.applications.vgg19 import VGG19

import tensorflow.keras.backend as K
import tensorflow.keras.layers as layers
import tensorflow.keras.models as models

class ReflectionPad(layers.Layer):
  def __init__(self, padding, name='reflection', *args, **kwargs):
    super(ReflectionPad, self).__init__(name=name, **kwargs)
    self.pad_left, self.pad_right, self.pad_top, self.pad_bottom = padding

  def compute_output_shape(self, input_shape):
    return (input_shape[0], input_shape[1] + self.pad_left + self.pad_right, input_shape[2] + self.pad_top + self.pad_bottom, input_shape[3])

  def call(self, x):
    x = K.concatenate([K.reverse(x, 1)[:, (-1 - self.pad_left):-1, :, :], x, K.reverse(x, 1)[:, 1:(1 + self.pad_right), :, :]], axis=1)
    x = K.concatenate([K.reverse(x, 2)[:, :, (-1 - self.pad_top):-1, :], x, K.reverse(x, 2)[:, :, 1:(1 + self.pad_bottom), :]], axis=2)
    return x

class AdaIN(layers.Layer):
  def __init__(self, name='adain', alpha=1.0, epsilon=1e-5, **kwargs):
    super(AdaIN, self).__init__(name=name, **kwargs)
    self.alpha = alpha
    self.epsilon = epsilon

  def compute_output_shape(self, input_shape):
    return input_shape[0]

  def call(self, x):
    content_features, style_features = x
    content_mean = K.mean(content_features, axis=[1, 2], keepdims=True)
    content_std = K.std(content_features, axis=[1, 2], keepdims=True)
    style_mean = K.mean(style_features, axis=[1, 2], keepdims=True)
    style_std = K.std(style_features, axis=[1, 2], keepdims=True)
    normalized_content_features = (content_features - content_mean) / (content_std + self.epsilon) *style_std + style_mean
    return self.alpha * normalized_content_features + (1 - self.alpha) * content_features

class Encoder(models.Model):
  def __init__(self, encoder_layers=['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1'], input_shape=(None, None, 3), pretrained=True, name='encoder', **kwargs):
    assert len(encoder_layers) > 0, 'No "encoder_layers" is provided.'
    
    vgg = VGG19(input_tensor=layers.Input(shape=input_shape), weights=('imagenet' if pretrained else None), include_top=False)
    output_layers = [vgg.get_layer(layer_name).output for layer_name in encoder_layers]
    super(Encoder, self).__init__(inputs=vgg.input, outputs=output_layers, name=name, **kwargs)

class Decoder(models.Model):
  def __init__(self, input_shape=(None, None, 512), name='decoder', **kwargs):
    input_layer = layers.Input(shape=input_shape)
    out = ReflectionPad((1, 1, 1, 1), name='block4_reflection1')(input_layer)
    out = layers.Conv2D(256, (3, 3), activation='relu', name='block4_conv1')(out)
    out = layers.UpSampling2D(size=2, interpolation='nearest', name='block3_upsample')(out)
    out = ReflectionPad((1, 1, 1, 1), name='block3_reflection4')(out)
    out = layers.Conv2D(256, (3, 3), activation='relu', name='block3_conv4')(out)
    out = ReflectionPad((1, 1, 1, 1), name='block3_reflection3')(out)
    out = layers.Conv2D(256, (3, 3), activation='relu', name='block3_conv3')(out)
    out = ReflectionPad((1, 1, 1, 1), name='block3_reflection2')(out)
    out = layers.Conv2D(256, (3, 3), activation='relu', name='block3_conv2')(out)
    out = ReflectionPad((1, 1, 1, 1), name='block3_reflection1')(out)
    out = layers.Conv2D(128, (3, 3), activation='relu', name='block3_conv1')(out)
    out = layers.UpSampling2D(size=2, interpolation='nearest', name='block2_upsample')(out)
    out = ReflectionPad((1, 1, 1, 1), name='block2_reflection2')(out)
    out = layers.Conv2D(128, (3, 3), activation='relu', name='block2_conv2')(out)
    out = ReflectionPad((1, 1, 1, 1), name='block2_reflection1')(out)
    out = layers.Conv2D(64, (3, 3), activation='relu', name='block2_conv1')(out)
    out = layers.UpSampling2D(size=2, interpolation='nearest', name='block1_upsample')(out)
    out = ReflectionPad((1, 1, 1, 1), name='block1_reflection2')(out)
    out = layers.Conv2D(64, (3, 3), activation='relu', name='block1_conv2')(out)
    out = ReflectionPad((1, 1, 1, 1), name='block1_reflection1')(out)
    out = layers.Conv2D(3, (3, 3), name='block1_conv1')(out)
    super(Decoder, self).__init__(inputs=input_layer, outputs=out, name=name, **kwargs)
