#!/usr/bin/env python2

import numpy as np
import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
import os
from glob import glob
import random
import argparse
import time
import multiprocessing
import collections

import cPickle as pickle
import gzip

from modelnet_fusion_common import *

# TODO: make sure pyrender, pyfusion and oc are in your PYTHONPATH
import pyrender
import pyfusion
import oc
pyoctnet = oc.pyoctnet

random.seed(42)
np.random.seed(42)

sel_vx_resolutions = [64,128,256]
settings = [
  Setting(view_setting_idx=1, encoding='occ', vx_resolutions=sel_vx_resolutions, noise=(0.02, 'n02')),

  Setting(view_setting_idx=1, encoding='tsdf_hist', vx_resolutions=sel_vx_resolutions, noise=(0.01, 'n01')),

  # GT
  Setting(view_setting_idx=0, encoding='tsdf', vx_resolutions=sel_vx_resolutions, noise=(0,'')),
]


# TODO: set Modelnet40 database
data_root = 'PATH/TO/MODELNET40/SHAPES'
models = get_models_list()
n_train_paths = 200
n_val_paths = 5
n_test_paths = 20
n_threads = 8

out_root = './preprocessed/'

if not os.path.isdir(out_root):
  os.makedirs(out_root)


# collect model paths
off_paths = {
  'train': get_off_paths(data_root, models, 'train', 0, n_train_paths),
  'val': get_off_paths(data_root, models, 'train', n_train_paths, n_train_paths+n_val_paths),
  'test': get_off_paths(data_root, models, 'test', 0, n_test_paths)
}
for k in off_paths:
  print('%s contains %d paths' % (k, len(off_paths[k])))

# create camera params
calibs = get_calibs()



def worker(off_path_idx, off_path, train_test, category, n_native_threads):
  print('create %d/%d - %s/%s/%s' % (off_path_idx+1, len(off_paths[train_test]), train_test, category, off_path))
  tic = time.time()

  # load/create depthmaps
  tic_sub = time.time()
  dm_path = os.path.join(out_root, 'data_%s_%08d_%s.pkl' % (train_test, off_path_idx, category))
  if os.path.exists(dm_path):
    with gzip.open(dm_path, 'rb') as f:
      data = pickle.load(f)
      depthmaps, Ks, Rs, Ts = data['depthmaps'], data['Ks'], data['Rs'], data['Ts']
  else:
    # read mesh
    tic_ssub = time.time()
    verts, faces = load_mesh(off_path)
    print('  loading mesh took %f[s]' % (time.time() - tic_ssub))

    tic_ssub = time.time()
    depthmaps, Ks,Rs,Ts = render_depthmaps(verts, faces, calibs)
    print('  rendering depthmaps took %f[s]' % (time.time() - tic_ssub))
    with gzip.open(dm_path, 'wb', 4) as f:
      pickle.dump({'depthmaps':depthmaps, 'Ks':Ks, 'Rs':Rs, 'Ts':Ts}, f)
  depthmaps_a, Ks_a, Rs_a, Ts_a = depthmaps, Ks, Rs, Ts
  print('  loading depthmaps took %f[s]' % (time.time() - tic_sub))

  for setting in settings:
    cluster = get_view_setting(setting.view_setting_idx, len(calibs))
    depthmaps, Ks,Rs,Ts = get_cluster_data(cluster, depthmaps_a,Ks_a,Rs_a,Ts_a)
    depthmaps = add_depth_noise(depthmaps, setting.noise[0], off_path_idx)

    views = pyfusion.PyViews(depthmaps, Ks,Rs,Ts)

    # convert gt to octree structure
    for vx_res in setting.vx_resolutions:
      out_prefix = os.path.join(out_root, '%s_%08d_%s_%s_s%d%s_r%d' % (train_test, off_path_idx, category, setting.encoding, setting.view_setting_idx, setting.noise[1], vx_res))

      truncation = get_truncation_value(vx_res)
      vx_size = float(1) / vx_res

      # create grid with set encoding
      tic_sub = time.time()
      if setting.encoding == 'mask':
        projmask = pyfusion.projmask_gpu(views, vx_res,vx_res,vx_res, vx_size, False)
        grid = pyoctnet.Octree.create_from_dense(projmask, n_threads=n_native_threads)
        print(projmask.sum(), grid.to_cdhw().sum())
        if projmask.sum() != grid.to_cdhw().sum():
          raise Exception('oc mask does not match dense mask')

      elif setting.encoding == 'occ':
        tsdf = pyfusion.occupancy_gpu(views, vx_res,vx_res,vx_res, vx_size, truncation, False)
        grid = pyoctnet.Octree.create_from_dense(tsdf, n_threads=n_native_threads)
        if np.any(np.sign(tsdf) != np.sign(grid.to_cdhw())):
          print(np.sum(np.sign(tsdf) != np.sign(grid.to_cdhw())))
          raise Exception('oc tsdf does not match dense tsdf')

      elif setting.encoding == 'tsdf':
        tsdf = pyfusion.tsdf_gpu(views, vx_res,vx_res,vx_res, vx_size, truncation, False)
        grid = pyoctnet.Octree.create_from_dense(tsdf, n_threads=n_native_threads)
        if np.any(np.sign(tsdf) != np.sign(grid.to_cdhw())):
          print(np.sum(np.sign(tsdf) != np.sign(grid.to_cdhw())))
          raise Exception('oc tsdf does not match dense tsdf')

      elif setting.encoding == 'tsdf_hist':
        n_bins = 10
        bin_width = (2 * truncation) / (n_bins - 2)
        bins = np.linspace(-truncation-bin_width/2, truncation+bin_width/2, num=n_bins, endpoint=True, dtype=np.float32)
        tsdf_hist = pyfusion.tsdf_hist_gpu(views, vx_res,vx_res,vx_res, vx_size, truncation, False, bins)
        grid = pyoctnet.Octree.create_from_dense(tsdf_hist, n_threads=n_native_threads)
        if np.any(np.sign(tsdf_hist) != np.sign(grid.to_cdhw())):
          print(np.sum(np.sign(tsdf_hist) != np.sign(grid.to_cdhw())))
          raise Exception('oc tsdf does not match dense tsdf')

      else:
        raise Exception('unknown encoding %s' % setting.encoding)
      print('  fusion for vx_res=%d took %f[s]' % (vx_res, time.time() - tic_sub))


      # write grid
      tic_sub = time.time()
      grid.write_bin('%s.oc' % (out_prefix))
      print('  octree sparsity: %d/%d=%f' % (grid.n_leafs(), vx_res**3, float(grid.n_leafs()) / float(vx_res**3)))
      print('  writing octree for vx_res=%d took %f[s]' % (vx_res, time.time() - tic_sub))

  print('took %f[s]' % (time.time() - tic))


for train_test in off_paths:
  for off_path_idx, (off_path, category) in enumerate(off_paths[train_test]):
    worker(off_path_idx, off_path, train_test, category, n_threads)


