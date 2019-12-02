from pathlib import Path
import tensorflow.keras.backend as K

def rm_dir(p):
  p = Path(p)
  for child in p.glob('*'):
    if child.is_file():
      child.unlink()
    else:
      rm_dir(child)
  p.rmdir()

def mse_loss(y_true, y_pred):
  return K.mean(K.square(y_true - y_pred))
