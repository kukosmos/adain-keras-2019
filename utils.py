import tensorflow.keras.backend as K


def learning_rate_decay(lr, decay, iteration):
  return lr / (1.0 + decay * iteration)


class calc_loss():
  def __init__(self, style_weight, content_weight, eps = 1e-5):
    self.style_weight = style_weight
    self.content_weight = content_weight
    self.eps = eps

  def content_loss(self, x):
    return K.mean(K.square(x[0] - x[1]))

  def style_loss(self, x):
    style = x[0]
    g = x[1]
    length = x[2]
    loss = 0.0

    for i in range(length):
      mean_s = K.mean(style[i], axis=[1,2])
      std_s = K.sqrt(K.var(style[i], axis=[1,2]) + self.eps)

      mean_g = K.mean(g[i], axis=[1,2])
      std_g = K.sqrt(K.var(g[i], axis=[1,2]) + self.eps)

      loss += K.mean(K.square(mean_s - mean_g)) + K.mean(K.square(std_s - std_g))
    return loss

  def sum_loss(self, x):
    return self.content_weight*x[0] + self.style_weight*x[1]
