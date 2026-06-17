import torch
import torch.nn as nn
import torch.nn.functional as F

import pyexr

import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage

import random
from random import randint

import os
import glob
import gc
import sys

eps = 0.00316

def preprocess_diffuse(diffuse, albedo):
  return diffuse / (albedo + eps)
def preprocess_specular(specular):
  assert(np.sum(specular < 0) == 0)
  return np.log(specular + 1)
def preprocess_diff_var(variance, albedo):
  #这里也有问题啊，论文中说将方差按照明度混合为单通道，那么应该是要先分别得到每个通道的方差再融合
  #但是数据的variance直接就是单通道的，可能直接来源于方差的明度混合，那么现在实际上是先混合再除
  luminance = 0.2126 * albedo[:, :, 0] + 0.7152 * albedo[:, :, 1] + 0.0722 * albedo[:, :, 2]
  denom = (luminance + eps) ** 2
  return variance / denom[:, :, np.newaxis]
def preprocess_spec_var(variance, specular):
  luminance = 0.2126 * specular[:, :, 0] + 0.7152 * specular[:, :, 1] + 0.0722 * specular[:, :, 2]
  denom = (luminance + eps) ** 2
  return variance / denom[:, :, np.newaxis]
def gradients(data):
  h, w, c = data.shape
  dX = data[:, 1:, :] - data[:, :w - 1, :]
  dY = data[1:, :, :] - data[:h - 1, :, :]
  # padding with zeros
  dX = np.concatenate((np.zeros([h,1,c], dtype=np.float32),dX), axis=1)
  dY = np.concatenate((np.zeros([1,w,c], dtype=np.float32),dY), axis=0)
  
  return np.concatenate((dX, dY), axis=2)

def remove_channels(data, channels):
  for c in channels:
    if c in data:
      del data[c]
    else:
      print("Channel {} not found in data!".format(c))

def read_buffer_bin(filepath, shape, dtype=np.float32):
    with open(filepath, 'rb') as f:
        size_bytes = f.read(8)
        stored_size = np.frombuffer(size_bytes, dtype=np.uint64)[0]
        data_bytes = f.read(stored_size)
    arr = np.frombuffer(data_bytes, dtype=dtype)
    # 这里按行主序 reshape，Vulkan 通常也是 row‑major
    return arr.reshape(shape)

def preprocess_input(samplePath, gtPath, debug=False):
  '''
script_dir = os.path.dirname(os.path.abspath(__file__))
  file_path = os.path.join(script_dir, samplePath)

  file = pyexr.open(file_path)
  data = file.get_all()

  #将每个通道的名字和数值类型输出出来
  if debug:
    for k, v in data.items():
      print(k, v.dtype)

  #将nan设置0，+inf设置为很大的值，-inf设置为很小的值
  for k, v in data.items():
    data[k] = np.nan_to_num(v)
  
  file_path = os.path.join(script_dir, gtPath)
  file_gt = pyexr.open(file_path)  #ground truth
  gt_data = file_gt.get_all()

  for k, v in gt_data.items():
    gt_data[k] = np.nan_to_num(v)

  #不要负数
  data['specular'] = np.maximum(data['specular'], 0)
  data['specularVariance'] = np.maximum(data['specularVariance'], 0)
  gt_data['specular'] = np.maximum(gt_data['specular'], 0)
  gt_data['specularVariance'] = np.maximum(gt_data['specularVariance'], 0)

  #保存albedo
  origAlbedo = data['albedo']

  #保存参考数据
  diff_ref = gt_data['diffuse'] #preprocess_diffuse(gt_data['diffuse'], gt_data['albedo'])
  spec_ref = preprocess_specular(gt_data['specular'])

  data['diffuse'] = preprocess_diffuse(data['diffuse'], data['albedo'])   # color / albedo
  data['diffuseVariance'] = preprocess_diff_var(data['diffuseVariance'], data['albedo'])  #diffV / albedo^2

  data['specularVariance'] = preprocess_spec_var(data['specularVariance'], data['specular'])  #specV / spec^2
  data['specular'] = preprocess_specular(data['specular'])  #log(spec + 1)

  data['depth'] = np.clip(data['depth'], 0, np.max(data['depth']))
  max_depth = np.max(data['depth'])
  if(max_depth != 0):  #归一化深度
    data['depth'] /= max_depth
    data['depthVariance'] /= max_depth * max_depth
  
  data['gradNormal'] = gradients(data['normal'][:, :, :3])
  data['gradDepth'] = gradients(data['depth'][:, :, :1])
  data['gradAlbedo'] = gradients(data['albedo'][:, :, :3])
  data['gradSpecular'] = gradients(data['specular'][:, :, :3])
  data['gradDiffuse'] = gradients(data['diffuse'][:, :, :3])

  data['diffuse'] = np.concatenate((data['diffuse'], data['diffuseVariance'], data['gradDiffuse']), axis=2)
  data['specular'] = np.concatenate((data['specular'], data['specularVariance'], data['gradSpecular']), axis=2)

  normalData = data['normal']
  data['normal'] = np.concatenate((data['normalVariance'], data['gradNormal']), axis=2)
  data['depth'] = np.concatenate((data['depthVariance'], data['gradDepth']), axis=2)
  data['albedo'] = np.concatenate((data['albedoVariance'], data['gradAlbedo']), axis=2)

  if debug:
    for k, v in data.items():
      print(k, v.shape, v.dtype)
  
  X_diff = np.concatenate((data['diffuse'],
                           data['normal'],
                           data['depth'],
                           data['albedo']), axis=2)
  X_spec = np.concatenate((data['specular'],
                           data['normal'],
                           data['depth'],
                           data['albedo']), axis=2)
  assert not np.isnan(X_diff).any()
  assert not np.isnan(X_spec).any()

  if debug:
    print("X_diff shape:", X_diff.shape)
    print(X_diff.dtype, X_spec.dtype)

  data['X_diff'] = X_diff
  data['X_spec'] = X_spec
  
  return {
    'input_diff': X_diff, 
    'input_spec': X_spec,
    'albedo': origAlbedo,
    'normal': normalData,
    'sampleColor': data['default'],

    'diff_ref': diff_ref,
    'spec_ref': spec_ref,
    'gtColor': gt_data['default'],
  }
  '''
  
  def load_bin(folder, name, shape):
        path = os.path.join(folder, f'{name}.bin')
        return read_buffer_bin(path, shape)
  sample_diff  = load_bin(samplePath, 'diff',    (27, 512, 512))
  sample_diff = np.transpose(sample_diff, (1, 2, 0))

  sample_spec  = load_bin(samplePath, 'spec',    (27, 512, 512))
  sample_spec = np.transpose(sample_spec, (1, 2, 0))

  sample_albedo= load_bin(samplePath, 'albedo',  (512, 512, 3))
  #sample_albedo = np.transpose(sample_albedo, (1, 2, 0))

  sample_normal= load_bin(samplePath, 'normal',  (3, 512, 512))
  sample_normal = np.transpose(sample_normal, (1, 2, 0))

  gt_diff      = load_bin(gtPath, 'diff',        (3, 512, 512))
  gt_diff = np.transpose(gt_diff, (1, 2, 0))

  gt_spec      = load_bin(gtPath, 'spec',        (3, 512, 512))
  gt_spec = np.transpose(gt_spec, (1, 2, 0))

  sample_spec_noLog = np.exp(sample_spec[:, :, 0:3]) - 1.0
  gt_spec_noLog = np.exp(gt_spec) - 1.0

  return {
    'input_diff': sample_diff, 
    'input_spec': sample_spec,
    'albedo': sample_albedo,
    'normal': sample_normal,
    'sampleColor': sample_diff[:, :, 0:3] * sample_albedo  + sample_spec_noLog,

    'diff_ref': gt_diff,
    'spec_ref': gt_spec,
    'gtColor': gt_diff + gt_spec_noLog,
    }


def show_data(data, figsize=(15, 15), normalize=False):
  if normalize:
    data = np.clip(data, 0, 1)**0.45454545
  plt.figure(figsize=figsize)
  imgplot = plt.imshow(data, aspect='equal')
  imgplot.axes.get_xaxis().set_visible(False)
  imgplot.axes.get_yaxis().set_visible(False)
  plt.show()
def show_data2(data, figsize=(15, 15), normalize=False, ax=None, show=True):
    if normalize:
        data = np.clip(data, 0, 1) ** 0.45454545
    if ax is None:
        # 旧行为：创建一个 figure，显示后阻塞
        _, ax = plt.subplots(figsize=figsize)
    ax.imshow(data, aspect='equal')
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    if show:
        plt.show()

'''
path = "C:/Users/fangzanbo/Desktop/FzbRenderer_nvvk/src/renderer/PathTracing_KPCNNDenoising/models_libtorch/train"
data = preprocess_input(path + "/staircase_32_7", path + "/staircase_8192_7", debug=True)

for k, v in data.items():
  print(k, "has nans:", np.isnan(v).any())
  
show_data(np.clip(data['input_diff'][:, :, :3], 0, 1)**0.45454545)
show_data(np.clip(data['gtColor'], 0, 1)**0.45454545)
#show_data(np.clip(data['input_diff'][:,:, :3], 0, 1)**0.45454545)
#show_data(data['input_diff'][4:7, :, :].transpose(1, 2, 0), normalize=False)
'''


#---------------------------------------------------makePatchs-----------------------------------------------------------
patch_size = 64 # patches are 64x64
n_patches = 400

def getVarianceMap(data, patch_size, relative=False):
  if data.ndim < 3:
    data = data[:, :, np.newaxis]
  
  mean = ndimage.uniform_filter(data, size=(patch_size, patch_size, 1))
  sqrmean = ndimage.uniform_filter(data**2, size=(patch_size, patch_size, 1))
  variance = np.maximum(sqrmean - mean**2, 0)
  
  #相对方差，数值大的部分方差自然大，但并不代表噪声比数值小的地方大
  if relative:
    variance = variance / np.maximum(mean**2, 1e-2)
  
  #取每个像素方差最大的通道值
  variance = variance.max(axis=2)
  variance = np.minimum(variance**(1.0/2.2), 1.0)

  return variance / variance.max()  #映射到0-1
def getImportanceMap(buffers, metrics, weights, patch_size):
  if len(metrics) != len(buffers): metrics = [metrics[0]] * len(buffers)
  if len(weights) != len(buffers): weights = [weights[0]] * len(buffers)
  
  impMap = None
  for buf, metric, weight in zip(buffers, metrics, weights):
    if metric == 'uniform': cur = np.ones(buf.shape[:2], dtype=np.float)
    elif metric == 'variance': cur = getVarianceMap(buf, patch_size)
    elif metric == 'relvar': cur = getVarianceMap(buf, patch_size, True)
    else: print('Unexpected metric:', metric)

    if impMap is None: impMap = cur * weight
    else: impMap += cur * weight
  
  return impMap / impMap.max()
def samplePatchesProg(img_dim, patch_size, n_samples, maxiter=5000):
  full_area = float(img_dim[0] * img_dim[1])
  sample_area = full_area / n_samples
  radius = np.sqrt(sample_area / np.pi)
  minsqrdist = (2*radius)**2  #点与点之间的最小距离

  #返回一点(x,y)与已有点集中点的最小平方距离
  def get_sqrdist(x, y, patches):
    if len(patches) == 0: return np.infty
    dist = patches - [x, y]
    return np.sum(dist**2, axis = 1).min()
  
  rate = 0.96
  patches = np.zeros((n_samples, 2), dtype = int)
  xmin, xmax = 0, img_dim[1] - patch_size[1] - 1  #这里(x,y)是矩形的左上角
  ymin, ymax = 0, img_dim[0] - patch_size[0] - 1
  for patch in range(n_samples):
    done = False
    while not done:
      for i in range(maxiter):  #尝试maxiter次
        x = randint(xmin, xmax)
        y = randint(ymin, ymax)
        sqrdist = get_sqrdist(x, y, patches[:patch, :])
        if sqrdist > minsqrdist:
          patches[patch, :] = [x, y]
          done = True
          break
      if not done:  #如果太拥挤则缩小距离
        radius *= rate
        minsqrdist = (2*radius)**2
  return patches
def prunePatches(shape, patches, patchsize, imp):
  pruned = np.empty_like(patches)
  
  def get_regions_list(shape, step):
    regions = []
    for y in range(0, shape[0], step):
      if y // step % 2 == 0: xrange = range(0, shape[1], step)  #偶数行从左到右；奇数行从右到左
      else: xrange = reversed(range(0, shape[1], step))
      for x in xrange: regions.append((x, x + step, y, y + step))
    return regions
  
  def split_patches(patches, region):
    cur = np.empty_like(patches)
    rem = np.empty_like(patches)
    ccount, rcount = 0, 0
    for i in range(patches.shape[0]):
      x, y = patches[i, 0], patches[i, 1]
      if region[0] <= x < region[1] and region[2] <= y < region[3]:
        cur[ccount, :] = [x, y]
        ccount += 1
      else: 
        rem[rcount, :] = [x, y]
        rcount += 1
    return cur[:ccount, :], rem[:rcount, :]
  
  rem = np.copy(patches)
  count, error = 0, 0
  for region in get_regions_list(shape, 4 * patchsize):
    cur, rem = split_patches(rem, region)
    for i in range(cur.shape[0]):
      x, y = cur[i, 0], cur[i, 1]
      if imp[y,x] - error > random.random():
        pruned[count, :] = [x, y]
        count += 1
        error += 1 - imp[y, x]
      else: error += 0 - imp[y, x]
  return pruned[:count, :]
def importanceSampling(data, debug=False):
  global patch_size, n_patches
  buffers = []
  for b in ['sampleColor', 'normal']: buffers.append(data[b])

  metrics = ['relvar', 'variance']
  weights = [1.0, 1.0]
  imp = getImportanceMap(buffers, metrics, weights, patch_size)

  if debug:
    print("Importance map:")
    plt.figure(figsize = (15, 15))
    imgplot = plt.imshow(imp)
    imgplot.axes.get_xaxis().set_visible(False)
    imgplot.axes.get_yaxis().set_visible(False)
    plt.show()
  
  patches = samplePatchesProg(buffers[0].shape[: 2], (patch_size, patch_size), n_patches)

  if debug:
    print("Patches:")
    plt.figure(figsize=(15, 10))
    plt.scatter(list(a[0] for a in patches), list(a[1] for a in patches))
    plt.show()

  pad = patch_size // 2
  pruned = np.maximum(0, prunePatches(buffers[0].shape[:2], patches + pad, patch_size, imp) - pad)

  if debug:
    print("After pruning:")
    plt.figure(figsize=(15, 10))
    plt.scatter(list(a[0] for a in pruned), list(a[1] for a in pruned))
    plt.show()
      
  return (pruned + pad)

#patches = importanceSampling(data, debug=False)

#patchSize is 65x65, follow the paper
def crop(data, pos, patch_size):
  half_patch = patch_size // 2
  sx, sy = half_patch, half_patch
  px, py = pos
  return {key: val[(py-sy):(py+sy+1),(px-sx):(px+sx+1),:] 
          for key, val in data.items()}
  #return {key: val[:, py-sy:py+sy+1, px-sx:px+sx+1]
  #        for key, val in data.items()}

'''
cropped = list(crop(data, tuple(pos), patch_size) for pos in patches)
print(len(cropped))
print("Sampling of chosen patches:")
for i in range(5):
  data_ = np.clip(cropped[np.random.randint(0, len(cropped))]['default'], 0, 1)**0.45454545
  plt.figure(figsize = (5,5))
  imgplot = plt.imshow(data_)
  imgplot.axes.get_xaxis().set_visible(False)
  imgplot.axes.get_yaxis().set_visible(False)
  plt.show()
'''
def get_cropped_patches(sample_file, gt_file):
  data = preprocess_input(sample_file, gt_file)
  patches = importanceSampling(data)
  cropped = list(crop(data, tuple(pos), patch_size) for pos in patches)
  return cropped

#---------------------------------------------------DataSet-----------------------------------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def to_torch_tensors(data):
  if isinstance(data, dict):
    for k, v in data.items():
      if not isinstance(v, torch.Tensor): data[k] = torch.from_numpy(v)
  elif isinstance(data, list):
    for i, v in enumerate(data):
      if not isinstance(v, torch.Tensor): data[i] = to_torch_tensors(v)
  return data
def send_to_device(data):
  if isinstance(data, dict):
    for k,v in data.items(): 
      if isinstance(v, torch.Tensor): data[k] = v.to(device)
  elif isinstance(data, list):
    for i, v in enumerate(data):
      if isinstance(v, torch.Tensor): data[i] = v.to(device)
  return data
def getsize(obj):
    marked = {id(obj)}
    obj_q = [obj]
    sz = 0

    while obj_q:
        sz += sum(map(sys.getsizeof, obj_q))

        # Lookup all the object reffered to by the object in obj_q.
        # See: https://docs.python.org/3.7/library/gc.html#gc.get_referents
        all_refr = ((id(o), o) for o in gc.get_referents(*obj_q))

        # Filter object that are already marked.
        # Using dict notation will prevent repeated objects.
        new_refr = {o_id: o for o_id, o in all_refr if o_id not in marked and not isinstance(o, type)}

        # The new obj_q will be the ones that were not marked,
        # and we will update marked with their ids so we will
        # not traverse them again.
        obj_q = new_refr.values()
        marked.update(new_refr.keys())

    return sz

class KPCNNDataset(torch.utils.data.Dataset):
  '''
    def __init__(self, folder):
    self.samples = []
    pattern = os.path.join(folder, "sample*.exr")
    for f in glob.glob(pattern):
        filename = os.path.basename(f)
        num = filename[len('sample'):filename.index('.')]
        sample_name = os.path.join(folder, f'sample{num}.exr')
        gt_name = os.path.join(folder, f'gt{num}.exr')
        print(sample_name, gt_name)
        self.samples.extend(get_cropped_patches(sample_name, gt_name))

    #for k, v in self.samples[0].items(): print(k, getsize(v))
    #print(getsize(self.samples) / 1024 / 1024, "MiB")

    #self.samples = to_torch_tensors(self.samples)
    #self.samples = send_to_device(self.samples)
  '''

  def __init__(self, sampleFolders, gtFolders):
    self.samples = []
    for sampleFolder, gtFolder in zip(sampleFolders, gtFolders):
        print(sampleFolder, gtFolder)
        self.samples.extend(get_cropped_patches(sampleFolder, gtFolder))



  def __len__(self):
    return len(self.samples)

  def __getitem__(self, idx):
    return self.samples[idx]
  
'''
dataset = KPCNNDataset("dataSet/train")
print("Left: noisy,    Right: reference")

def show_data_sbs(data1, data2, figsize=(15, 15)):
  f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=figsize)
  
  ax1.imshow(data1, aspect='equal')
  ax2.imshow(data2, aspect='equal')
  
  ax1.axis('off')
  ax2.axis('off')
  
  plt.show()
for i in range(len(dataset)):
  sample = dataset[np.random.randint(0, len(dataset))]
  
  data_sample = np.clip(sample['default'].cpu().numpy(), 0, 1)**0.45454545
  data_ref = np.clip(sample['finalGt'], 0, 1)**0.45454545
  show_data_sbs(data_sample, data_ref, figsize=(4, 2))
  
  if i == 6:
    break

input('按回车键退出...')

#for i, v in enumerate(cropped): torch.save(v, 'sample'+str(i+1)+'.pt')
'''
