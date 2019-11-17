assert __name__ == '__main__', 'This file cannot be imported.'

import tensorflow as tf

flags = tf.compat.v1.flags

# necessary arguments
flags.DEFINE_string('content_dir', None, 'Directory with content images', short_name='cd')
flags.DEFINE_string('style_dir', None, 'Directory with style images', short_name='sd')
flags.mark_flag_as_required('content_dir')
flags.mark_flag_as_required('style_dir')

# optional arguments for training
flags.DEFINE_string('save_dir', './experiments', 'Directory to save trained models, default=./experiments')
flags.DEFINE_string('log-dir', './logs', 'Directory to save logs, default=./logs')
flags.DEFINE_integer('log_image_every', 100, 'Period for loging generated images, non-positive for disabling, default=100')
flags.DEFINE_integer('save_interval', 10000, 'Period for saving model, default=10000')
flags.DEFINE_integer('n_threads', 2, 'Number of threads used for dataloader, default=2')

# hyper-parameters
flags.DEFINE_float('learning_rate', 1e-4, 'Learning rate, default=1e-4')
flags.DEFINE_float('learning_rate_decay', 5e-5, 'Learning rate decay, default=5e-5')
flags.DEFINE_integer('max_iter', 160000, 'Maximun number of iteration, default=160000')
flags.DEFINE_integer('batch_size', 8, 'Size of the batch, default=8')
flags.DEFINE_float('style_weight', 10.0, 'Weight of style loss, default=10.0')
flags.DEFINE_float('content_weight', 1.0, 'Weight of content loss, default=1.0')

FLAGS = flags.FLAGS

import os
import tensorflow.keras.backend as K

from tensorflow.keras import optimizers
from tensorflow.keras.models import Model, clone_model
from tf.keras.layers import Lambda

from dataloader import * #not yet
from network import AdaIN, Encoder, Decoder
from pathlib import Path
from PIL import Image, ImageFile
from tensorboardX import SummaryWriter
from tqdm import tqdm
from utils import learning_rate_decay


def content_loss(self, x, y):
  return K.mean(K.square(x - y))

def style_loss(style, g, eps=1e-5):
  loss = 0
  for i in range(4):
    mean_s = K.mean(style[i], axis=[1,2])
    std_s = K.sqrt(K.var(style[i], axis=[1,2]) + eps)

    mean_g = K.mean(g[i], axis=[1,2])
    std_g = K.sqrt(K.var(g[i], axis=[1,2]) + eps)

    loss += K.mean(K.square(mean_s - mean_g)) + K.mean(K.square(std_s - std_g))
  return loss

def sum_loss(x, y, style_weight = 1, content_weight = 0):
  return content_weight * x + style_weight * y

def main(self, argv):


  # for handling errors
  Image.MAX_IMAGE_PIXELS = None
  ImageFile.LOAD_TRUNCATED_IMAGES = True

  # directory trained models
  save_dir = Path(FLAGS.save_dir)
  save_dir.mkdir(exist_ok=True, parents=True)
  # directory for logs
  log_dir = Path(FLAGS.log_dir)
  log_dir.mkdir(exist_ok=True, parents=True)

  # content dataset
  
  # style dataset

  #model
  encoder = Encoder()
  content_input, content_output = clone_model(encoder)
  style_input, style_output = clone_model(encoder)
  
  adain = AdaIN()([content_output, style_output])
  decoder_output = Decoder()(adain)
  f_g = clone_model(encoder)(decoder_output)# ??

  loss_content = Lambda(self.content_loss)(adain, f_g)
  loss_style = Lambda(self.style_loss)(style_input, decoder_output)
  loss = Lambda(self.sum_loss(style_weight=FLAGS.style_weight, content_weight=FLAGS.content_weight))(loss_content, loss_style)


  model = Model(inputs=[content_input, style_input], outputs=[loss])
  adam = optimizers.Adam(learning_rate=FLAGS.learning_rate)
  model.compile(optimizer=adam, loss=lambda x, loss : loss)

  # log writer
  writer = SummaryWriter(log_dir=str(log_dir))

  # for maximum iteration
  for i in tqdm(range(FLAGS.max_iter)):
    # adjust learning rate
    lr = learning_rate_decay(FLAGS.learning_rate, FLAGS.learning_rate_decay, i)
    K.set_value(model.optimizer.lr, lr)

    # get images
    content_images = [] # not yet
    style_images = [] # not yet

    hist = model.fit([content_images, style_images], content_images, batch_size=FLAGS.batch_size)


    generate_model = Model(inputs=[content_input, style_input], outputs=decoder_output)
    g = generate_model.predict([content_images, style_images])


    writer.add_scalar('Loss/Loss', hist.history['loss'], i + 1)
    if FLAGS.log_image_every > 0 and ((i + 1) % FLAGS.log_image_every == 0 or i == 0 or (i + 1) == FLAGS.max_iter):
      writer.add_image('Image/Content', content_images[0], i + 1)
      writer.add_image('Image/Style', style_images[0], i + 1)
      writer.add_image('Image/Generated', g[0], i + 1)

    # save model
    if (i + 1) % FLAGS.save_interval == 0 or (i + 1) == FLAGS.max_iter:
      model.save(os.path.join(save_dir, 'iter_{}.pth'.format(i + 1)))



  writer.close()


if __name__ == '__main__':
    tf.compat.v1.app.run(main)