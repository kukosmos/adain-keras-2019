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
  
  def get_config(self):
    config = super(ReflectionPad, self).get_config().copy()
    config.update({
      'padding': (self.pad_left, self.pad_right, self.pad_top, self.pad_bottom)
    })
    return config

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
    content_std = K.sqrt(K.var(content_features, axis=[1, 2], keepdims=True) + self.epsilon)
    style_mean = K.mean(style_features, axis=[1, 2], keepdims=True)
    style_std = K.sqrt(K.var(style_features, axis=[1, 2], keepdims=True) + self.epsilon)
    normalized_content_features = (content_features - content_mean) / (content_std + self.epsilon) * style_std + style_mean
    return self.alpha * normalized_content_features + (1 - self.alpha) * content_features
  
  def get_config(self):
    config = super(AdaIN, self).get_config().copy()
    config.update({
      'alpha': self.alpha,
      'epsilon': self.epsilon
    })
    return config

def Encoder(input_tensor=None, input_shape=(256, 256, 3), pretrained=True, name='encoder', **kwargs):
  if input_tensor is None:
    input_tensor = layers.Input(shape=input_shape)
  
  vgg = VGG19(input_tensor=input_tensor, weights='imagenet' if pretrained else None, include_top=False)
  output_layers = [vgg.get_layer(layer_name) for layer_name in ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1']]
  return models.Model(inputs=[input_tensor], outputs=[layer.output for layer in output_layers], name=name, **kwargs)

def Decoder(input_tensor=None, input_shape=(32, 32, 512), name='decoder', **kwargs):
  if input_tensor is None:
    input_tensor = layers.Input(shape=input_shape)

  x = ReflectionPad((1, 1, 1, 1), name='block4_reflection1')(input_tensor)
  x = layers.Conv2D(256, (3, 3), activation='relu', name='block4_conv1')(x)

  x = layers.UpSampling2D(size=2, interpolation='nearest', name='block3_upsample')(x)
  x = ReflectionPad((1, 1, 1, 1), name='block3_reflection4')(x)
  x = layers.Conv2D(256, (3, 3), activation='relu', name='block3_conv4')(x)
  x = ReflectionPad((1, 1, 1, 1), name='block3_reflection3')(x)
  x = layers.Conv2D(256, (3, 3), activation='relu', name='block3_conv3')(x)
  x = ReflectionPad((1, 1, 1, 1), name='block3_reflection2')(x)
  x = layers.Conv2D(256, (3, 3), activation='relu', name='block3_conv2')(x)
  x = ReflectionPad((1, 1, 1, 1), name='block3_reflection1')(x)
  x = layers.Conv2D(128, (3, 3), activation='relu', name='block3_conv1')(x)
  
  x = layers.UpSampling2D(size=2, interpolation='nearest', name='block2_upsample')(x)
  x = ReflectionPad((1, 1, 1, 1), name='block2_reflection2')(x)
  x = layers.Conv2D(128, (3, 3), activation='relu', name='block2_conv2')(x)
  x = ReflectionPad((1, 1, 1, 1), name='block2_reflection1')(x)
  x = layers.Conv2D(64, (3, 3), activation='relu', name='block2_conv1')(x)
  
  x = layers.UpSampling2D(size=2, interpolation='nearest', name='block1_upsample')(x)
  x = ReflectionPad((1, 1, 1, 1), name='block1_reflection2')(x)
  x = layers.Conv2D(64, (3, 3), activation='relu', name='block1_conv2')(x)
  x = ReflectionPad((1, 1, 1, 1), name='block1_reflection1')(x)
  x = layers.Conv2D(3, (3, 3), name='block1_conv1')(x)

  return models.Model(inputs=[input_tensor], outputs=[x], name=name, **kwargs)
  