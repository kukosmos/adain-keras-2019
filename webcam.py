from absl import app
from absl import flags
from pathlib import Path
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
import numpy as np

from dataloader import load_image
from network import AdaIN
from network import Decoder
from network import Encoder

# required options
flags.DEFINE_string('style', default=None, help='Path to a style image')
flags.DEFINE_string('decoder', default=None, help='Path to the weights of the decoder')
flags.mark_flags_as_required(['style', 'decoder'])

# model option
flags.DEFINE_float('alpha', default=1.0, help='The amount of stylization', lower_bound=0.0, upper_bound=1.0)

# style image options
flags.DEFINE_integer('style_size', default=256, help='The size of style image', lower_bound=1)
flags.DEFINE_bool('preserve_color', default=False, help='Option for preserving the color of content images')

FLAGS = flags.FLAGS

def run():

  # create model
  if not Path(FLAGS.decoder).exists():
    raise ValueError('The decoder model is not found: {}'.format(FLAGS.decoder))
  encoder = Encoder(pretrained=True)
  content_feature_input = Input(shape=encoder.output_shape[-1][1:])
  style_feature_input = Input(shape=encoder.output_shape[-1][1:])
  adain = AdaIN(alpha=FLAGS.alpha)
  adain = Model(inputs=[content_feature_input, style_feature_input], outputs=[adain([content_feature_input, style_feature_input])])
  decoder = Decoder(input_shape=encoder.output_shape[-1][1:])
  decoder.load_weights(FLAGS.decoder)
  
  # load and encode style image
  style_path = Path(FLAGS.style)
  if not style_path.exists():
    raise ValueError('The style image is not exist: {}'.format(style_path))
  style = np.expand_dims(load_image(style_path, image_shape=(FLAGS.style_size, FLAGS.style_size)), axis=0)
  style_feature = encoder.predict(style)[-1]

  # TODO

def main(argv):
  del argv
  run()

if __name__ == '__main__':
  app.run(main)