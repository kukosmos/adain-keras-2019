from absl import app
from absl import flags
from absl import logging
from pathlib import Path
from PIL import Image
from PIL import ImageFile
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Lambda
from tensorflow.keras.models import Model
import numpy as np
import tensorflow.keras.backend as K
import tensorflow.keras.optimizers as optimizers

from dataloader import ContentStyleLoader
from network import AdaIN
from network import Encoder
from network import Decoder
from utils import mse_loss
from utils import rm_path

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
flags.DEFINE_integer('save_every', default=None, help='Number of batches between checkpoints, default=per_epoch')

FLAGS = flags.FLAGS

def calculate_style_loss(x, epsilon=1e-5):
  y_trues, y_preds = x
  loss = [
    mse_loss(K.mean(y_true, axis=(1, 2)), K.mean(y_pred, axis=(1, 2)))
    + mse_loss(K.sqrt(K.var(y_true, axis=(1, 2)) + epsilon), K.sqrt(K.var(y_pred, axis=(1, 2)) + epsilon))
    for y_true, y_pred in zip(y_trues, y_preds)
  ]
  return K.sum(loss)

def calculate_content_loss(x):
  y_trues, y_preds = x
  return mse_loss(y_trues[-1], y_preds[-1])

class SubmodelCheckpoint(Callback):
  def __init__(self, filepath, submodel_name, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', save_freq='epoch', **kwargs):
    super(SubmodelCheckpoint, self).__init__()
    self.filepath = filepath
    self.submodel_name = submodel_name
    self.monitor = monitor
    self.verbose = verbose
    self.save_best_only = save_best_only
    self.save_weights_only = save_weights_only
    if mode == 'min':
      self.monitor_op = np.less
      self.best = np.Inf
    elif mode == 'max':
      self.monitor_op = np.greater
      self.best = -np.Inf
    else: # mode == 'auto'
      if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
        self.monitor_op = np.greater
        self.best = -np.Inf
      else: # monitor loss
        self.monitor_op = np.less
        self.best = np.Inf
    self.save_freq = save_freq
    if 'period' in kwargs:
      self.period = kwargs['period']
    else:
      self.period = 1
    self.epochs_since_last_save = 0
    self.samples_since_last_save = 0
    self.current_epoch = None
  
  def on_batch_end(self, batch, logs=None):
    logs = logs or {}
    if isinstance(self.save_freq, int):
      self.samples_since_last_save += 1
      if self.samples_since_last_save >= self.save_freq:
        self.save_model(name_dict={'batch': batch}, logs=logs)
        self.samples_since_last_save = 0

  def on_epoch_start(self, epoch, logs=None):
    self.current_epoch = epoch

  def on_epoch_end(self, epoch, logs=None):
    logs = logs or {}
    if self.save_freq == 'epoch':
      self.epochs_since_last_save += 1
      if self.epochs_since_last_save >= self.period:
        self.save_model(name_dict={'epoch': epoch}, logs=logs)
        self.epochs_since_last_save = 0

  def save_model(self, name_dict=None, logs=None):
    name_dict = name_dict or {}
    if 'epoch' not in name_dict:
      name_dict['epoch'] = self.current_epoch
    name_dict['epoch'] += 1
    logs = logs or {}
    filepath = self.filepath.format(**name_dict, **logs)
    submodel = self.model.get_layer(self.submodel_name)
    if self.save_best_only:
      current = logs.get(self.monitor)
      if current is None:
        logging.warning('Can save best model only with {} available, skipping.'.format(self.monitor))
      else:
        if self.monitor_op(current, self.best):
          if self.verbose > 0:
            print('\nEpoch {:5d}: {} improved from {:.5f} to {:.5f}, save model to {}'.format(self.current_epoch + 1, self.monitor, self.best, current, filepath))
          if self.save_weights_only:
            submodel.save_weights(filepath, overwrite=True)
          else:
            submodel.save(filepath, overwrite=True)
    else:
      if self.verbose > 0:
        print('\nEpoch {:5d}: save mode to {}'.format(self.current_epoch + 1, filepath))
      if self.save_weights_only:
        submodel.save_weights(filepath, overwrite=True)
      else:
        submodel.save(filepath, overwrite=True)

def run():

  # create directories
  save_dir = Path(FLAGS.save_dir)
  if save_dir.exists():
    logging.warning('"save_dir={}" already exist. The files may be overwrited.'.format(FLAGS.save_dir))
  save_dir.mkdir(exist_ok=True)
  log_dir = Path(FLAGS.tensorboard)
  if log_dir.exists():
    logging.warning('"tensorboard={}" already exist. The directory and contents will be removed.'.format(FLAGS.tensorboard))
    rm_path(log_dir)
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
  encoder = Encoder(input_shape=(FLAGS.crop_size, FLAGS.crop_size, 3), pretrained=True, name='encoder')
  for l in encoder.layers:  # freeze the model
    l.trainable = False
  adain = AdaIN(alpha=1.0, name='adain')
  decoder = Decoder(input_shape=encoder.output_shape[-1][1:], name='decoder')

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
  content_loss = Lambda(calculate_content_loss, name='content_loss')([content_features, generated_features])
  style_loss = Lambda(calculate_style_loss, name='style_loss')([style_features, generated_features])
  loss = Lambda(lambda x: FLAGS.content_weight * x[0] + FLAGS.style_weight * x[1], name='loss')([content_loss, style_loss])

  # trainer
  trainer = Model(inputs=[content_input, style_input], outputs=[loss])
  optim = optimizers.Adam(learning_rate=FLAGS.learning_rate)
  trainer.compile(optimizer=optim, loss=lambda _, y_pred: y_pred)
  trainer.summary()

  # callbacks
  callbacks = [
    # learning rate scheduler
    LearningRateScheduler(lambda epoch, _: FLAGS.learning_rate / (1.0 + FLAGS.learning_rate_decay * FLAGS.dataset_size * epoch)),
    # Tensor Board
    TensorBoard(str(log_dir), write_graph=False, update_freq='batch'),
    # save model
    SubmodelCheckpoint(str(save_dir / 'decoder.epoch-{epoch:d}.h5'), submodel_name='decoder', save_best_only=FLAGS.save_best_only, save_freq=FLAGS.save_every if FLAGS.save_every else 'epoch')
  ]

  # train
  trainer.fit_generator(dataset, epochs=FLAGS.epochs, workers=FLAGS.workers, callbacks=callbacks)

def main(argv):
  del argv
  run()

if __name__ == '__main__':
  app.run(main)
