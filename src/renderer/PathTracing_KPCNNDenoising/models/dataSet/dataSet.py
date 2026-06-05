import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

import pyexr

import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage

import random
from random import randint

import glob
import os
import time

def show_data(data, figsize=(15, 15), normalize=False):
  if normalize:
    data = np.clip(data, 0, 1)**0.45454545
  plt.figure(figsize=figsize)
  imgplot = plt.imshow(data, aspect='equal')
  imgplot.axes.get_xaxis().set_visible(False)
  imgplot.axes.get_yaxis().set_visible(False)
  plt.show()

def show_data_sbs(data1, data2, figsize=(15, 15)):
  f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=figsize)
  
  ax1.imshow(data1, aspect='equal')
  ax2.imshow(data2, aspect='equal')
  
  ax1.axis('off')
  ax2.axis('off')
  
  plt.show()

def show_channel(img, chan):
  data = img.get(chan)
  print("Channel:", chan)
  print("Shape:", data.shape)
  print(np.max(data), np.min(data))
  
  if chan in ["default", "diffuse", "albedo", "specular"]:
    data = np.clip(data, 0, 1)**0.45454545
  
  if chan in ["normal", "normalA"]:
    # normalize
    print("normalizing")
    for i in range(img.height):
      for j in range(img.width):
        data[i][j] = data[i][j] / np.linalg.norm(data[i][j])
    data = np.abs(data)
    
  if chan in ["depth", "visibility", "normalVariance"] and np.max(data) != 0:
    data /= np.max(data)
  
  if data.shape[2] == 1:
    print("Reshaping")
    data = data.reshape(img.height, img.width)

  show_data(data)

script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, "sample.exr")
file = pyexr.open(file_path)

print("Width:", file.width)
print("Height:", file.height)

print("Available channels:")
print(file.describe_channels())
  
print("Default channels:", file.channel_map['default'])

for chan in ["default", "albedo", "diffuse", "specular", "normal", "normalVariance", "depth", "visibility"]:
  show_channel(file, chan)