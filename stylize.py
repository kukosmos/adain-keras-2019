from absl import app
from absl import flags
from absl import logging
from pathlib import Path
from tensorflow.keras.layers import Input
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import array_to_img
from tqdm import tqdm
import numpy as np

from dataloader import load_image
from network import AdaIN
from network import Decoder
from network import Encoder
from utils import rm_path

# required options
flags.DEFINE_list('contents', default=None, help='Paths to content images or directories that contain content images')
flags.DEFINE_string('style', default=None, help='Path to a style image')
flags.DEFINE_string('decoder', default=None, help='Path to the weights of the decoder')
flags.mark_flags_as_required(['content', 'style', 'decoder'])

# model option
flags.DEFINE_float('alpha', default=1.0, help='The amount of stylization', lower_bound=0.0, upper_bound=1.0)

# save options
flags.DEFINE_string('ext', default='jpg', help='Extension name of generated images')
flags.DEFINE_string('output', default='./generated', help='Directory for saving generated images')
flags.DEFINE_bool('preserve_color', default=False, help='Option for preserving the color of content images')

FLAGS = flags.FLAGS

def run():
  
  # load data
  contents = []
  for c in FLAGS.contents:
    p = Path(c)
    if not p.exists():
      raise ValueError('The content image or directory is not exist: {}'.format(p))
    if p.is_dir():
      for f in p.glob('**/*.*'):
        contents.append((f, load_image(f)))
    else:
      contents.append((p, load_image(p)))
  style_path = Path(FLAGS.style)
  if not style_path.exists():
    raise ValueError('The style image is not exist: {}'.format(style_path))
  style = np.expand_dims(load_image(style_path), axis=0)

  # input place holder
  content_input = Input(shape=(None, None, 3))
  style_input = Input(shape=(None, None, 3))

  # create model
  if not Path(FLAGS.decoder).exists():
    raise ValueError('The decoder model is not found: {}'.format(FLAGS.decoder))
  encoder = Encoder(pretrained=True)
  adain = AdaIN(alpha=FLAGS.alpha)
  decoder = Decoder(input_shape=encoder.output_shape[-1][1:])
  decoder.load_weights(FLAGS.decoder)
  
  content_feature = encoder(content_input)
  style_feature = encoder(style_input)
  normalized_feature = adain([content_feature, style_feature])
  generated = decoder(normalized_feature)
  
  stylizer = Model(inputs=[content_input, style_input], outputs=[generated])

  # output directory
  output_dir = Path(FLAGS.output) / style_path.stem
  if output_dir.exists():
    logging.warning('The folder will be deleted: {}'.format(output_dir))
    rm_path(output_dir)
  output_dir.mkdir(exist_ok=True, parents=True)
  
  for content_path, content in tqdm(contents):
    
    # generate image
    content = np.expand_dims(content, axis=0)
    generated = stylizer.predict([content, style])

    # save image
    img_path = output_dir / '{}.{}'.format(content_path.stem, FLAGS.ext)
    img = array_to_img(generated[0])
    img.save(img_path)

def main(argv):
  del argv
  run()

if __name__ == '__main__':
  app.run(main)
