from absl import flags
from absl import app

# required option
flags.DEFINE_string('decoder', default=None, help='Path to the weights of the decoder')
flags.DEFINE_string('style', default=None, help='Path to the style image')
flags.mark_flags_as_required(['decoder'])

# model option
flags.DEFINE_float('alpha', default=1.0, help='The amount of stylization', lower_bound=0.0, upper_bound=1.0)

FLAGS = flags.FLAGS

def run():
  print(FLAGS)

def main(argv):
  del argv
  run()

if __name__ == '__main__':
  app.run(main)
