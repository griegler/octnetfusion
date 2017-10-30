import numpy as np
import collections
import json
import os
from glob import glob

Calibration = collections.namedtuple('Calibration', 'P, K, R, T, C')

def load_values(out_root, metric, set):
  metric_paths = glob(os.path.join(out_root, 'metrics_e*_s%02d.json' % (set)))
  metric_paths.sort()
  vals = []
  for metric_path in metric_paths:
    with open(metric_path, 'r') as f:
      data = json.load(f)
      vals.append( data[metric] )
  return np.array(vals, dtype=np.float32)

def write_metric_json(out_path, sads, rmses, ious, ta_paths, timings=None):
  def write_list(f, name, vals, fmt, last_list):
    f.write('  "%s": [\n' % name)
    for vidx, val in enumerate(vals):
      f.write('    ')
      f.write(fmt % val)
      if vidx < len(vals) - 1:
        f.write(',\n')
      else:
        f.write('\n')
    if last_list:
      f.write('  ]\n')
    else:
      f.write('  ],\n')
  with open(out_path, 'w') as f:
    f.write('{\n')
    write_list(f, 'sads', sads, '%f', False)
    write_list(f, 'rmses', rmses, '%f', False)
    write_list(f, 'ious', ious, '%f', False)
    if timings is not None:
      write_list(f, 'timings', timings, '%f', False)
    write_list(f, 'ta_paths', ta_paths, '"%s"', True)
    f.write('}')
