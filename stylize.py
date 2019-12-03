from absl import app
from absl import flags
from pathlib import Path
from tensorflow.keras.models import load_model

from dataloader import load_image
from network import AdaIN
from network import Decoder
from network import Encoder

# required options
flags.DEFINE_string('decoder', default=None, help='Path to decoder')
flags.DEFINE_list('contents', default=None, help='Path to content images')
flags.DEFINE_string('style', default=None, help='Path to style image')
flags.mark_flags_as_required(['decoder', 'content', 'style'])

FLAGS = flags.FLAGS

def run():
  print(FLAGS)

def main(argv):
  del argv
  run()

if __name__ == '__main__':
  app.run(main)
