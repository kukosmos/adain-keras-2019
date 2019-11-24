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
from tensorflow.keras.layers import Lambda
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from dataloader import crop_generator
from network import AdaIN, Encoder, Decoder
from pathlib import Path
from PIL import Image, ImageFile
from tensorboardX import SummaryWriter
from tqdm import tqdm
from utils import learning_rate_decay, calc_loss

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


  image_generator = ImageDataGenerator(rescale=1./255)

  # content dataset
  content_generator = image_generator.flow_from_directory(FLAGS.content_dir, target_size = (512, 512), batch_size=FLAGS.batch_size, shuffle=True)
  content_iter = crop_generator(content_generator, 256)

  # style dataset
  style_generator = image_generator.flow_from_directory(FLAGS.style_dir, target_size = (512, 512), batch_size=FLAGS.batch_size, shuffle=True)
  style_iter = crop_generator(style_generator, 256)


  #model
  encoder = Encoder()
  content_encoder = clone_model(encoder)
  style_encoder = clone_model(encoder)
  
  adain = AdaIN()([content_encoder.outputs[-1], style_encoder.outputs[-1]])
  decoder_output = Decoder()(adain)
  f_g = clone_model(encoder)(decoder_output)

  loss_function = calc_loss(FLAGS.style_weight, FLAGS.content_weight)
  loss_content = Lambda(loss_function.content_loss)([adain, f_g[-1]])
  loss_style = Lambda(loss_function.style_loss)([style_encoder.outputs, f_g, len(encoder.layers)])
  loss_total = Lambda(loss_function.sum_loss)([loss_content, loss_style])


  model_train = Model(inputs=[content_encoder.inputs, style_encoder.inputs], outputs=[loss_total])
  adam = optimizers.Adam(learning_rate=FLAGS.learning_rate)
  model_train.compile(optimizer=adam, loss=lambda _, loss : loss)

  model_generate = Model(inputs=[content_encoder.inputs, style_encoder.inputs], outputs=decoder_output)
   
  # log writer
  writer = SummaryWriter(log_dir=str(log_dir))

  # for maximum iteration
  for i in tqdm(range(FLAGS.max_iter)):
    # adjust learning rate
    lr = learning_rate_decay(FLAGS.learning_rate, FLAGS.learning_rate_decay, i)
    K.set_value(model_train.optimizer.lr, lr)

    # get images
    content_images = next(content_iter)
    style_images = next(style_iter)
    hist = model_train.train_on_batch([content_images, style_images], content_images)

    g = model_generate.predict([content_images, style_images])


    writer.add_scalar('Loss/Loss', hist.history['loss'], i + 1)
    if FLAGS.log_image_every > 0 and ((i + 1) % FLAGS.log_image_every == 0 or i == 0 or (i + 1) == FLAGS.max_iter):
      writer.add_image('Image/Content', content_images[0], i + 1)
      writer.add_image('Image/Style', style_images[0], i + 1)
      writer.add_image('Image/Generated', g[0], i + 1)

    # save model
    if (i + 1) % FLAGS.save_interval == 0 or (i + 1) == FLAGS.max_iter:
      model_train.save(os.path.join(save_dir, 'iter_{}.pth'.format(i + 1)))


  writer.close()


if __name__ == '__main__':
    tf.compat.v1.app.run(main)