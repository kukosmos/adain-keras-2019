from absl import app
from absl import flags
from datetime import datetime
from pathlib import Path
# from tensorflow.keras.layers import Input
# from tensorflow.keras.models import Model
import cv2
import numpy as np

# from dataloader import load_image
# from network import AdaIN
# from network import Decoder
# from network import Encoder

# required options
flags.DEFINE_string('style', default=None, help='Path to a style image')
flags.DEFINE_string('decoder', default=None, help='Path to the weights of the decoder')
# flags.mark_flags_as_required(['style', 'decoder'])

# model option
flags.DEFINE_float('alpha', default=1.0, help='The amount of stylization', lower_bound=0.0, upper_bound=1.0)

# webcam options
flags.DEFINE_integer('width', default=None, help='Width of webcam')
flags.DEFINE_integer('height', default=None, help='Height of webcam')

# style image options
flags.DEFINE_integer('style_size', default=256, help='The size of style image', lower_bound=1)

FLAGS = flags.FLAGS

def run():

  # # create model
  # if not Path(FLAGS.decoder).exists():
  #   raise ValueError('The decoder model is not found: {}'.format(FLAGS.decoder))
  # encoder = Encoder(pretrained=True)
  # content_feature_input = Input(shape=encoder.output_shape[-1][1:])
  # style_feature_input = Input(shape=encoder.output_shape[-1][1:])
  # adain = AdaIN(alpha=FLAGS.alpha)
  # adain = Model(inputs=[content_feature_input, style_feature_input], outputs=[adain([content_feature_input, style_feature_input])])
  # decoder = Decoder(input_shape=encoder.output_shape[-1][1:])
  # decoder.load_weights(FLAGS.decoder)
  
  # # load and encode style image
  # style_path = Path(FLAGS.style)
  # if not style_path.exists():
  #   raise ValueError('The style image is not exist: {}'.format(style_path))
  # style = np.expand_dims(load_image(style_path, image_shape=(FLAGS.style_size, FLAGS.style_size)), axis=0)
  # style_feature = encoder.predict(style)[-1]

  # create webcam stream
  cap = cv2.VideoCapture(0)
  if FLAGS.width and FLAGS.height:
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FLAGS.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FLAGS.height)

  window_name = 'AdaIN Keras'
  cv2.namedWindow(window_name)
  # cv2.createTrackbar('alpha', window_name, int(adain.alpha * 100), 100, lambda x: adain.layers[0].set_alpha(float(x) / 100))
  cv2.createTrackbar('alpha', window_name, 100, 100, lambda x: x)

  start_time = datetime.now()
  prev_time = start_time
  total_frame = 0

  while cv2.getWindowProperty(window_name, 0) >= 0:
    ret, frame = cap.read()

    if not ret:
      raise OSError('Failed to get a frame from webcam.')
    
    frame = np.expand_dims(frame, axis=0)
    # frame_feature = encoder.predict(frame)[-1]
    # normalized_feature = adain.predict([frame_feature, style_feature])
    # generated = decoder.predict(normalized_feature)
    generated = frame

    cv2.imshow(window_name, generated[0])
    key = cv2.waitKey(10)
    if key & 0xFF == ord('q'):
      break

    total_frame += 1
    
  end_time = datetime.now()
  cap.release()
  cv2.destroyAllWindows()

  elapsed = (end_time - start_time).total_seconds()
  print('Time elapsed: {:.0f}'.format(elapsed))
  print('FPS: {:.4f}'.format(elapsed / total_frame))

def main(argv):
  del argv
  run()

if __name__ == '__main__':
  app.run(main)