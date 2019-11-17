def learning_rate_decay(lr, decay, iteration):
  return lr / (1.0 + decay * iteration)