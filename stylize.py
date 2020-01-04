from absl import app
from absl import flags
from absl import logging
from pathlib import Path
from tensorflow.keras.layers import Input
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
flags.mark_flags_as_required(['contents', 'style', 'decoder'])

# model option
flags.DEFINE_float('alpha', default=1.0, help='The amount of stylization', lower_bound=0.0, upper_bound=1.0)

# style image option
flags.DEFINE_integer('style_size', default=256, help='The size of style image', lower_bound=1)

# save options
flags.DEFINE_string('ext', default='jpg', help='Extension name of generated images')
flags.DEFINE_string('output', default='./generated', help='Directory for saving generated images')

FLAGS = flags.FLAGS

def run():
  
  # paths to data
  content_paths = []
  for c in FLAGS.contents:
    p = Path(c)
    if not p.exists():
      raise ValueError('The content image or directory is not exist: {}'.format(p))
    if p.is_dir():
      for f in p.glob('**/*.*'):
        content_paths.append(f)
    else:
      content_paths.append(p)
  style_path = Path(FLAGS.style)
  if not style_path.exists():
    raise ValueError('The style image is not exist: {}'.format(style_path))

  # output directory
  output_dir = Path(FLAGS.output) / style_path.stem
  if output_dir.exists():
    logging.warning('The folder will be deleted: {}'.format(output_dir))
    rm_path(output_dir)
  output_dir.mkdir(exist_ok=True, parents=True)

  # create model
  if not Path(FLAGS.decoder).exists():
    raise ValueError('The decoder model is not found: {}'.format(FLAGS.decoder))
  encoder = Encoder(input_shape=(None, None, 3), pretrained=True)
  content_feature_input = Input(shape=encoder.output_shape[-1][1:])
  style_feature_input = Input(shape=encoder.output_shape[-1][1:])
  adain = AdaIN(alpha=FLAGS.alpha)
  adain = Model(inputs=[content_feature_input, style_feature_input], outputs=[adain([content_feature_input, style_feature_input])])
  decoder = Decoder(input_shape=encoder.output_shape[-1][1:])
  decoder.load_weights(FLAGS.decoder)
  
  # load and encode style image
  style = np.expand_dims(load_image(style_path, image_shape=(FLAGS.style_size, FLAGS.style_size)), axis=0)
  style_feature = encoder.predict(style)[-1]

  for content_path in tqdm(content_paths):
    
    # load and encode content image
    content = load_image(content_path)
    content = np.expand_dims(content, axis=0)
    content_feature = encoder.predict(content)[-1]

    # normalize the feature
    normalized_feature = adain.predict([content_feature, style_feature])

    # generate image
    generated = decoder.predict(normalized_feature)
    
    # save image
    img_path = output_dir / '{}.{}'.format(content_path.stem, FLAGS.ext)
    img = array_to_img(generated[0])
    img.save(img_path)

def main(argv):
  del argv
  run()

if __name__ == '__main__':
  app.run(main)
