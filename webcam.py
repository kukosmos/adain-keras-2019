from absl import app
from absl import flags
from datetime import datetime
from pathlib import Path
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
import cv2
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

# webcam options
flags.DEFINE_integer('width', default=None, help='Width of webcam')
flags.DEFINE_integer('height', default=None, help='Height of webcam')

# style image options
flags.DEFINE_integer('style_size', default=256, help='The size of style image', lower_bound=1)

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

  # create webcam stream
  cap = cv2.VideoCapture(0)
  if FLAGS.width and FLAGS.height:
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FLAGS.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FLAGS.height)

  frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
  frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

  # create window
  window_name = 'AdaIN Keras'
  cv2.namedWindow(window_name)
  cv2.createTrackbar('alpha', window_name, int(adain.alpha * 100), 100, lambda x: adain.layers[0].set_alpha(float(x) / 100))

  # information tracker
  display_info = False

  txt_font = cv2.FONT_HERSHEY_SIMPLEX
  txt_scale = 0.3
  txt_color = (255, 255, 150)
  txt_tickness = 1
  txt_left_margin = 10
  txt_top_margin = 10

  start_time = datetime.now()
  prev_time = start_time
  total_frame = 0
  fps_txt = 'fps: {:.2f}'

  # while window is open
  while cv2.getWindowProperty(window_name, 0) >= 0:
    ret, frame = cap.read()

    if not ret:
      raise OSError('Failed to get a frame from webcam.')
    
    frame = np.expand_dims(frame, axis=0)
    frame_feature = encoder.predict(frame)[-1]
    normalized_feature = adain.predict([frame_feature, style_feature])
    generated = decoder.predict(normalized_feature)[0]

    cur_time = datetime.now()
    frame_elapsed = (cur_time - prev_time).total_seconds()
    fps = 1 / frame_elapsed
    if display_info:
      txt = fps_txt.format(fps)
      txt_size = cv2.getTextSize(txt, txt_font, txt_scale, txt_tickness)
      top_margin = txt_top_margin
      cv2.putText(generated, txt, (frame_width - txt_size[0][0] - txt_left_margin, txt_size[0][1] + top_margin), txt_font, txt_scale, txt_color, thickness=txt_tickness)

    cv2.imshow(window_name, generated)
    key = cv2.waitKey(10)
    if key & 0xFF == ord('q'):
      break
    if key & 0xFF == ord('f'):
      display_info = not display_info

    total_frame += 1
    prev_time = cur_time
    
  end_time = datetime.now()
  cap.release()
  cv2.destroyAllWindows()

  elapsed = (end_time - start_time).total_seconds()
  print('Time elapsed: {:.0f}'.format(elapsed))
  print('FPS: {:.4f}'.format(total_frame /  elapsed))

def main(argv):
  del argv
  run()

if __name__ == '__main__':
  app.run(main)