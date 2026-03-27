import json
import os
import jax
from google3.pyglib import gfile


def save_to_sponge(name, data):
  outdir = os.getenv('TEST_UNDECLARED_OUTPUTS_DIR')
  if not outdir:
    raise ValueError(
        'Failed to save to sponge: TEST_UNDECLARED_OUTPUTS_DIR is not set.'
    )
  with gfile.GFile(os.path.join(outdir, name), 'a') as f:
    f.write(str(data) + '\n')


def print_table(header, rows, *, col_width_extra=None):
  if col_width_extra is None:
    col_width_extra = {}
  sz = len(header)
  col_width = [(len(str(h)) + 3 + col_width_extra.get(h, 0)) for h in header]
  start_separator = '╒' + '╤'.join('═' * w for w in col_width) + '╕'
  middle_separator = '╞' + '╪'.join('═' * w for w in col_width) + '╡'
  end_separator = '╘' + '╧'.join('═' * w for w in col_width) + '╛'
  fmt = '│' + '│'.join('{{:<{}}}'.format(w) for w in col_width) + '│'
  print(start_separator)
  print(fmt.format(*header))
  for row in rows:
    assert len(row) == sz
    print(middle_separator)
    print(fmt.format(*row))
  print(end_separator)