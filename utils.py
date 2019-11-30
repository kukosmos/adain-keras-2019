from pathlib import Path

def rm_tree(p):
  p = Path(p)
  for child in p.glob('*'):
    if child.is_file():
      child.unlink()
    else:
      rm_tree(child)
  p.rmdir()
