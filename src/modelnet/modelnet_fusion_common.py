import numpy as np
from glob import glob
import os
import random
import time
import sys
import h5py

sys.path.append('../')
sys.path.append('../../')
sys.path.append('../../../')
from completion_common import *
import pyrender
import pyfusion
import oc
pyoctnet = oc.pyoctnet


cam_intr = np.array([351, 351, 320, 240], dtype=np.float64)
img_size = np.array([480, 640], dtype=np.int32)

Setting = collections.namedtuple('Setting', ['view_setting_idx', 'encoding', 'vx_resolutions', 'noise'])


def faces_to_triangles(faces):
  new_faces = []
  for f in faces:
    if f[0] == 3:
      new_faces.append([f[1], f[2], f[3]])
    elif f[0] == 4:
      new_faces.append([f[1], f[2], f[3]])
      new_faces.append([f[3], f[4], f[1]])
    else:
      raise Exception('unknown face count %d', f[0])
  return new_faces

def read_off(path):
  f = open(path, 'r')
  lines = f.readlines()
  f.close()

  # parse header
  if lines[0].strip().lower() != 'off':
    raise Exception('Header error')

  splits = lines[1].strip().split(' ')
  n_verts = int(splits[0])
  n_faces = int(splits[1])

  # parse content
  line_nb = 2
  verts = []
  for idx in xrange(line_nb, line_nb + n_verts):
    verts.append([float(v) for v in lines[idx].strip().split(' ')])
    line_nb += 1
  faces = []
  for idx in xrange(line_nb, line_nb + n_faces):
    faces.append([int(v) for v in lines[idx].strip().split(' ')])

  faces = faces_to_triangles(faces)
  return np.array(verts, dtype=np.float32), np.array(faces, dtype=np.int32)

def scale_vertices(verts, faces, scale=1, pad=0):
  verts_ind = np.unique(faces.ravel())
  used_verts = verts[verts_ind]
  min_verts = used_verts.min(axis=0)
  max_verts = used_verts.max(axis=0)
  dist = np.abs(min_verts - max_verts)
  max_dist = np.max(dist)
  verts -= (0.5 * (max_verts + min_verts))
  verts /= max_dist
  verts *= scale * (1-pad)
  return verts



def get_models_list():
  return ['airplane', 'bed', 'car', 'desk', 'chair', 'guitar', 'piano', 'person', 'toilet', 'piano']

def get_off_paths(data_root, models, train_test, from_idx, to_idx):
  off_paths = []
  for model in models:
    paths = glob(os.path.join(data_root, model, train_test, '*.off'))
    paths.sort()
    for pidx in range(from_idx, min(to_idx, len(paths))):
      off_paths.append((paths[pidx], model))
  off_paths.sort(key=lambda tup: tup[0])
  return off_paths

def load_mesh(model_path):
  verts, faces = read_off(model_path)
  verts = scale_vertices(verts, faces, scale=3, pad=0.05)
  verts = verts.T.astype(np.float64).copy()
  faces = faces.T.astype(np.float64).copy()
  return verts, faces

def render_depthmaps(verts, faces, calibs):
  # get camera cluster
  cluster = get_view_setting(0, len(calibs))

  # # visualize cluster
  # fig = plt.figure()
  # ax = fig.add_subplot(111, projection='3d')
  # plot_camera_cluster(ax, calibs=get_calibs(), size=0.25, cluster=cluster)
  # axis_equal_3d(ax)
  # plt.show()

  # render depthmaps
  depthmaps = []
  Ks = []
  Rs = []
  Ts = []
  for vidx in cluster:
    K = calibs[vidx].K
    R = calibs[vidx].R
    T = calibs[vidx].T

    render_verts = R.dot(verts) + T.reshape((3,1))

    # render depth/mask
    depth, mask, img = pyrender.render(render_verts, faces, cam_intr, img_size, linewidth=0, colors=None)

    depth[mask != 1] = 10

    Ks.append(K)
    Rs.append(R)
    Ts.append(T)
    depthmaps.append(depth)

  Ks = np.array(Ks, dtype=np.float32)
  Rs = np.array(Rs, dtype=np.float32)
  Ts = np.array(Ts, dtype=np.float32)
  depthmaps = np.array(depthmaps, dtype=np.float32)

  return depthmaps, Ks, Rs*3, Ts

def add_depth_noise(depthmaps, noise_sigma, seed):
  # add noise
  if noise_sigma > 0:
    random.seed(seed)
    np.random.seed(seed)
    sigma = noise_sigma
    print('  add noise with sigma=%f' % sigma)
    noise = np.random.normal(0,1, size=depthmaps.shape).astype(np.float32)
    depthmaps = depthmaps + noise * sigma * depthmaps
  return depthmaps

def get_cluster_data(cluster, depthmaps, Ks, Rs, Ts):
  return depthmaps[cluster,...], Ks[cluster,...], Rs[cluster,...], Ts[cluster,...]


def get_truncation_value(vx_res):
  truncation = 0.025 * 256.0 / float(vx_res)
  return truncation


def create_icosahedron(divisions=0):
  # create dicrete orientations with icosahedron
  theta = 26.56505117707799 * np.pi / 180.0
  stheta = np.sin(theta)
  ctheta = np.cos(theta)

  vertices = []
  vertices.append((0.0, 0.0, -1.0))

  phi = np.pi / 5.0
  for i in xrange(1, 6):
    vertices.append((ctheta * np.cos(phi), ctheta * np.sin(phi), -stheta))
    phi += 2 * np.pi / 5.0

  phi = 0.0
  for i in xrange(6, 11):
    vertices.append((ctheta * np.cos(phi), ctheta * np.sin(phi), stheta))
    phi += 2 * np.pi / 5.0

  vertices.append((0.0, 0.0, 1.0))

  faces = []
  faces.append((vertices[0], vertices[2], vertices[1]))
  faces.append((vertices[0], vertices[3], vertices[2]))
  faces.append((vertices[0], vertices[4], vertices[3]))
  faces.append((vertices[0], vertices[5], vertices[4]))
  faces.append((vertices[0], vertices[1], vertices[5]))
  faces.append((vertices[1], vertices[2], vertices[7]))
  faces.append((vertices[2], vertices[3], vertices[8]))
  faces.append((vertices[3], vertices[4], vertices[9]))
  faces.append((vertices[4], vertices[5], vertices[10]))
  faces.append((vertices[5], vertices[1], vertices[6]))
  faces.append((vertices[1], vertices[7], vertices[6]))
  faces.append((vertices[2], vertices[8], vertices[7]))
  faces.append((vertices[3], vertices[9], vertices[8]))
  faces.append((vertices[4], vertices[10], vertices[9]))
  faces.append((vertices[5], vertices[6], vertices[10]))
  faces.append((vertices[6], vertices[7], vertices[11]))
  faces.append((vertices[7], vertices[8], vertices[11]))
  faces.append((vertices[8], vertices[9], vertices[11]))
  faces.append((vertices[9], vertices[10], vertices[11]))
  faces.append((vertices[10], vertices[6], vertices[11]))

  for division in xrange(divisions):
    faces_new = []
    for face in faces:
      v1, v2, v3 = face
      v4 = (v1 + v2); v4 /= np.linalg.norm(v4)
      v5 = (v2 + v3); v5 /= np.linalg.norm(v5)
      v6 = (v3 + v1); v6 /= np.linalg.norm(v6)
      v4 = (v4[0], v4[1], v4[2])
      v5 = (v5[0], v5[1], v5[2])
      v6 = (v6[0], v6[1], v6[2])
      vertices.append(v4)
      vertices.append(v5)
      vertices.append(v6)
      faces_new.append((v1, v4, v6))
      faces_new.append((v4, v2, v5))
      faces_new.append((v6, v5, v3))
      faces_new.append((v6, v4, v5))
    faces = faces_new

  vertices = np.array(vertices, dtype=np.float32)
  faces = np.array(faces, dtype=np.float32)
  return vertices, faces

def rotation_matrices_from_icosahedron(divisions, use_faces=True, vec_z=(0,0,1)):
  def skew_matrix(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]], dtype=np.float32)

  ih_verts, ih_faces = create_icosahedron(divisions)
  vecs_t = []
  if use_faces:
    for face in ih_faces:
      vec_t = np.mean(face, axis=0)
      vec_t /= np.linalg.norm(vec_t)
      vecs_t.append(vec_t)
  else:
    vecs_t = [ih_verts[idx] for idx in xrange(ih_verts.shape[0])]

  Rs = []
  vec_z = np.array(vec_z, dtype=np.float32)
  for vec_t in vecs_t:
    v = np.cross(vec_z, vec_t)
    if np.linalg.norm(v) < 1e-6:
      raise Exception('error in v == 0')
    s = np.linalg.norm(v)
    c = np.dot(vec_z, vec_t)
    vx = skew_matrix(v)
    R = np.eye(3) + vx + np.dot(vx, vx) * (1 - c) / s**2

    vec_z_on_t = R.dot(vec_z)
    vec_z_on_t /= np.linalg.norm(vec_z_on_t)
    if np.linalg.norm(vec_t - vec_z_on_t) > 1e-6:
      raise Exception('some error with R')

    R = rot_z(np.pi/2).dot(R)
    Rs.append(R.astype(np.float32))
  return np.array(Rs, dtype=np.float32)

def rot_z(z):
  return np.array([[np.cos(z),-np.sin(z),0], [np.sin(z),np.cos(z),0], [0,0,1]], dtype=np.float32)

def get_calibs():
  calibs = {}
  K = np.array([[cam_intr[0],0,cam_intr[2]], [0,cam_intr[1],cam_intr[3]], [0,0,1]], dtype=np.float32)
  vidx = 0
  T = np.array([0,0,3], dtype=np.float32)
  Rs = rotation_matrices_from_icosahedron(1, use_faces=True)
  for R in Rs:
    R = rot_z(np.pi/2).dot(R)
    # T = -R.dot(C)
    C = -R.T.dot(T.reshape((3,1)))
    calibs[vidx] = Calibration(P=None, K=K, R=R, T=T, C=C)
    vidx += 1

  return calibs

def get_view_setting(setting, n):
  if setting == 0: 
    cluster = range(n)
  elif setting == 1: 
    cluster = np.linspace(0, n-1, num=4, dtype=np.int32)
  elif setting == 2:
    cluster = np.linspace(0, n-1, num=1, dtype=np.int32)
  elif setting == 3:
    cluster = np.linspace(0, n-1, num=2, dtype=np.int32)
  elif setting == 4:
    cluster = np.linspace(0, n-1, num=6, dtype=np.int32)

  elif setting == 5:
    cluster = np.linspace(0, n-1, num=8, dtype=np.int32)
  elif setting == 6:
    cluster = np.linspace(0, n-1, num=16, dtype=np.int32)

  else:
    raise Exception('invalid setting %d' % setting)
  return cluster


def tsdfiter(paths, ind):
  for idx in ind:
    path = paths[idx]
    yield pyoctnet.Octree.create_from_bin(path).to_cdhw()

def tvl1iter(paths, ind, lam=1.5, tvl1_iters=500, truncation=1):
  for idx in ind:
    path = paths[idx]
    hist = pyoctnet.Octree.create_from_bin(path).to_cdhw()
    yield pyfusion.zach_tvl1_hist(hist, truncation, lam, tvl1_iters, init=None).squeeze()

def h5iter(paths, ind):
  idx = 0
  for path in paths:
    with h5py.File(path, 'r') as f:
      data = f['data']
      for bidx in range(data.shape[0]):
        if idx in ind:
          yield np.array(data[bidx]).squeeze()
        idx += 1
