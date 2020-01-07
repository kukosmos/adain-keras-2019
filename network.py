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

class Encoder(models.Model):
  def __init__(self, name='encoder', **kwargs):
    
    super(Encoder, self).__init__(name=name, **kwargs)
    
    self.block1_conv1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')
    self.block1_conv2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')
    self.block1_pool = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')

    self.block2_conv1 = layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')
    self.block2_conv2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')
    self.block2_pool = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')

    self.block3_conv1 = layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')
    self.block3_conv2 = layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')
    self.block3_conv3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')
    self.block3_conv4 = layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv4')
    self.block3_pool = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')

    self.block4_conv1 = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')

  def load_vgg(self, vgg=None):
    if vgg is None:
      raise ValueError('The variable "vgg" should not be None.')

    if not self.built:
      self.build(input_shape=vgg.input_shape)

    for self_layer, vgg_layer in zip(self.layers, vgg.layers[1:]):
      self_layer.set_weights(vgg_layer.get_weights())

  def call(self, x):
    out1 = self.block1_conv1(x)
    out2 = self.block1_conv2(out1)
    out2 = self.block1_pool(out2)
    
    out2 = self.block2_conv1(out2)
    out3 = self.block2_conv2(out2)
    out3 = self.block2_pool(out3)

    out3 = self.block3_conv1(out3)
    out4 = self.block3_conv2(out3)
    out4 = self.block3_conv3(out4)
    out4 = self.block3_conv4(out4)
    out4 = self.block3_pool(out4)

    out4 = self.block4_conv1(out4)

    return out1, out2, out3, out4
    
  def get_config(self):
    return super(Encoder, self).get_config().copy()

class Decoder(models.Model):
  def __init__(self, name='decoder', **kwargs):

    super(Decoder, self).__init__(name=name, **kwargs)

    self.block4_reflection1 = ReflectionPad((1, 1, 1, 1), name='block4_reflection1')
    self.block4_conv1 = layers.Conv2D(256, (3, 3), activation='relu', name='block4_conv1')

    self.block3_upsample = layers.UpSampling2D(size=2, interpolation='nearest', name='block3_upsample')
    self.block3_reflection4 = ReflectionPad((1, 1, 1, 1), name='block3_reflection4')
    self.block3_conv4 = layers.Conv2D(256, (3, 3), activation='relu', name='block3_conv4')
    self.block3_reflection3 = ReflectionPad((1, 1, 1, 1), name='block3_reflection3')
    self.block3_conv3 = layers.Conv2D(256, (3, 3), activation='relu', name='block3_conv3')
    self.block3_reflection2 = ReflectionPad((1, 1, 1, 1), name='block3_reflection2')
    self.block3_conv2 = layers.Conv2D(256, (3, 3), activation='relu', name='block3_conv2')
    self.block3_reflection1 = ReflectionPad((1, 1, 1, 1), name='block3_reflection1')
    self.block3_conv1 = layers.Conv2D(128, (3, 3), activation='relu', name='block3_conv1')
    
    self.block2_upsample = layers.UpSampling2D(size=2, interpolation='nearest', name='block2_upsample')
    self.block2_reflection2 = ReflectionPad((1, 1, 1, 1), name='block2_reflection2')
    self.block2_conv2 = layers.Conv2D(128, (3, 3), activation='relu', name='block2_conv2')
    self.block2_reflection1 = ReflectionPad((1, 1, 1, 1), name='block2_reflection1')
    self.block2_conv1 = layers.Conv2D(64, (3, 3), activation='relu', name='block2_conv1')
    
    self.block1_upsample = layers.UpSampling2D(size=2, interpolation='nearest', name='block1_upsample')
    self.block1_reflection2 = ReflectionPad((1, 1, 1, 1), name='block1_reflection2')
    self.block1_conv2 = layers.Conv2D(64, (3, 3), activation='relu', name='block1_conv2')
    self.block1_reflection1 = ReflectionPad((1, 1, 1, 1), name='block1_reflection1')
    self.block1_conv1 = layers.Conv2D(3, (3, 3), name='block1_conv1')
  
  def call(self, x):
    out = self.block4_reflection1(x)
    out = self.block4_conv1(out)
    
    out = self.block3_upsample(out)
    out = self.block3_reflection4(out)
    out = self.block3_conv4(out)
    out = self.block3_reflection3(out)
    out = self.block3_conv3(out)
    out = self.block3_reflection2(out)
    out = self.block3_conv2(out)
    out = self.block3_reflection1(out)
    out = self.block3_conv1(out)
    
    out = self.block2_upsample(out)
    out = self.block2_reflection2(out)
    out = self.block2_conv2(out)
    out = self.block2_reflection1(out)
    out = self.block2_conv1(out)
    
    out = self.block1_upsample(out)
    out = self.block1_reflection2(out)
    out = self.block1_conv2(out)
    out = self.block1_reflection1(out)
    out = self.block1_conv1(out)
    
    return out

  def get_config(self):
    return super(Decoder, self).get_config().copy()
