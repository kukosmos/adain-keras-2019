from absl import app
from absl import flags
from absl import logging
from pathlib import Path
from PIL import Image
from PIL import ImageFile
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Lambda
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K
import tensorflow.keras.losses as losses
import tensorflow.keras.optimizers as optimizers

from dataloader import ContentStyleLoader
from network import AdaIN
from network import Encoder
from network import Decoder
from utils import rm_tree

# dataset options
flags.DEFINE_string('content_dir', default=None, help='Directory with content images', short_name='cd')
flags.mark_flag_as_required('content_dir')
flags.DEFINE_string('style_dir', default=None, help='Directory with style images', short_name='sd')
flags.mark_flag_as_required('style_dir')
flags.DEFINE_integer('image_size', default=512, help='Size of images to load')
flags.DEFINE_integer('crop_size', default=256, help='Size of images after random crop')
flags.DEFINE_integer('batch_size', default=8, help='Size of the batch')
flags.DEFINE_integer('dataset_size', default=1000, help='Size of the dataset per epoch')
flags.DEFINE_integer('workers', default=1, help='Number of threads for input preprocessing')

# hyper-parameters
flags.DEFINE_float('content_weight', default=1.0, help='Weight of content loss')
flags.DEFINE_float('style_weight', default=10.0, help='Weight of style loss')
flags.DEFINE_integer('epochs', default=160, help='Total epochs')
flags.DEFINE_float('learning_rate', default=1e-4, help='Learning rate')
flags.DEFINE_float('learning_rate_decay', default=5e-5, help='Learning rate decay')

# logging
flags.DEFINE_string('save_dir', default='./experiments', help='Directory to save trained models')
flags.DEFINE_string('tensorboard', default='./logs', help='Directory to save tensorboard logs')
flags.DEFINE_bool('save_best_only', default=False, help='Option to save one best model')
flags.DEFINE_integer('save_every', default=10, help='Number of epochs between checkpoints')

FLAGS = flags.FLAGS

def calculate_style_loss(x):
  style_features, generated_features = x
  loss = [
    losses.mean_squared_error(K.mean(style_feature, keepdims=True), K.mean(generated_feature, keepdims=True))
    + losses.mean_squared_error(K.var(style_feature, keepdims=True), K.var(generated_feature, keepdims=True))
    for style_feature, generated_feature in zip(style_features, generated_features)
  ]
  return K.sum(loss)

def calculate_content_loss(x):
  content_features, generated_features = x
  return losses.mean_squared_error(content_features[-1], generated_features[-1])

def run():

  # create directories
  save_dir = Path(FLAGS.save_dir)
  if save_dir.exists():
    logging.warning('"save_dir={}" already exist. The files may be overwrited.'.format(FLAGS.save_dir))
  save_dir.mkdir(exist_ok=True)
  log_dir = Path(FLAGS.tensorboard)
  if log_dir.exists():
    logging.warning('"tensorboard={}" already exist. The directory and contents will be removed.'.format(FLAGS.tensorboard))
    rm_tree(log_dir)
  log_dir.mkdir(exist_ok=True)

  # to handle errors while loading images
  Image.MAX_IMAGE_PIXELS = None
  ImageFile.LOAD_TRUNCATED_IMAGES = True

  # image generator
  dataset = ContentStyleLoader(
    content_root=FLAGS.content_dir,
    content_image_shape=(FLAGS.image_size, FLAGS.image_size),
    content_crop='random',
    content_crop_size=FLAGS.crop_size,
    style_root=FLAGS.style_dir,
    style_image_shape=(FLAGS.image_size, FLAGS.image_size),
    style_crop='random',
    style_crop_size=FLAGS.crop_size,
    n_per_epoch=FLAGS.dataset_size,
    batch_size=FLAGS.batch_size
  )

  # create model
  encoder = Encoder(input_shape=(FLAGS.crop_size, FLAGS.crop_size, 3), pretrained=True)
  for l in encoder.layers:  # freeze the model
    l.trainable = False
  adain = AdaIN(alpha=1.0)
  decoder = Decoder(input_shape=encoder.output_shape[-1][1:])

  # place holders for inputs
  content_input = Input(shape=(FLAGS.crop_size, FLAGS.crop_size, 3), name='content_input')
  style_input = Input(shape=(FLAGS.crop_size, FLAGS.crop_size, 3), name='style_input')

  # forwarding
  content_features = encoder(content_input)
  style_features = encoder(style_input)
  normalized_features = adain([content_features[-1], style_features[-1]])
  generated = decoder(normalized_features)

  # loss calculation
  generated_features = encoder(generated)
  content_loss = Lambda(calculate_content_loss)([content_features, generated_features])
  style_loss = Lambda(calculate_style_loss)([style_features, generated_features])
  loss = Lambda(lambda x: FLAGS.content_weight * x[0] + FLAGS.style_weight * x[1])([content_loss, style_loss])

  # trainer
  trainer = Model(inputs=[content_input, style_input], outputs=[loss])
  optim = optimizers.Adam(learning_rate=FLAGS.learning_rate)
  trainer.compile(optimizer=optim, loss=lambda _, y_pred: y_pred)

  # callbacks
  callbacks = [
    # learning rate scheduler
    LearningRateScheduler(lambda epoch, _: FLAGS.learning_rate / (1.0 + FLAGS.learning_rate_decay * FLAGS.dataset_size * epoch)),
    # Tensor Board
    TensorBoard(str(log_dir), write_graph=False, update_freq='batch'),
    # save model
    ModelCheckpoint(str(save_dir / 'epoch-{epoch:d}.h5'), save_best_only=FLAGS.save_best_only, period=FLAGS.save_every)
  ]

  # train
  trainer.fit_generator(dataset, epochs=FLAGS.epochs, workers=FLAGS.workers, callbacks=callbacks)

def main(argv):
  del argv
  run()

if __name__ == '__main__':
  app.run(main)
