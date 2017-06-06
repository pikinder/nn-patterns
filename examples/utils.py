# Begin: Python 2/3 compatibility header small
# Get Python 3 functionality:
from __future__ import\
    absolute_import, print_function, division, unicode_literals
from future.utils import raise_with_traceback, raise_from
# catch exception with: except Exception as e
from builtins import range, map, zip, filter
from io import open
import six
# End: Python 2/3 compatability header small

import matplotlib.pyplot as plt
import numpy as np
import os
import PIL.Image


###############################################################################
# VGG16 utility functions
###############################################################################


VGG16_OFFSET = np.array([103.939, 116.779, 123.68])[:, np.newaxis, np.newaxis]


def preprocess(image):
    ret = image.copy()
    # Channels first.
    ret = ret.transpose(2, 0, 1)
    # To BGR.
    ret = ret[::-1, :, :]
    # Remove pixel-wise mean.
    ret -= VGG16_OFFSET
    return ret


###############################################################################
# Visualizations
###############################################################################


def original_image(x):
    """
    Revert VGG 16 preprocessing.
    """

    x = x.copy()
    x = x + VGG16_OFFSET
    x = x / 255.0
    # To RGB
    x = x[::-1,:,:]
    return x


def back_projection(x):
    x = x.copy()
    x /= np.max(np.abs(x)) # -1, 1
    x /= 2.0 # -0.5, 0.5
    x += 0.5 # 0, 1
    # To RGB
    x = x[::-1,:,:]
    return x


def heatmap(x):
    cmap = plt.cm.get_cmap('seismic') # 256 values.
    x_shape = x.shape
    x = x.sum(axis=0)
    x /= np.max(np.abs(x)) # -1,1
    x += 1. # 0, 2
    x *= 127.5 # 0, 255
    x = x.astype(np.int64)
    x_cmap = cmap(x.flatten())[:,:3].T
    return x_cmap.reshape(x_shape)


def put_into_big_image(x,big_image,i,j,n_padding):
    w = x.shape[1]
    h = x.shape[2]
    i = n_padding+(w+n_padding)*i
    j = n_padding+(h+n_padding)*j
    big_image[:,i:i+w,j:j+h] = x
    return big_image


###############################################################################
# Plot utility
###############################################################################


def load_image(path, size):
    ret = PIL.Image.open(path)
    ret = ret.resize((size, size))
    ret = np.asarray(ret, dtype=np.uint8).astype(np.float32)
    return ret


def get_imagenet_data():
    base_dir = os.path.dirname(__file__)
    with open(os.path.join(base_dir, "images", "ground_truth")) as f:
        ground_truth = {x.split()[0]: int(x.split()[1])
                        for x in f.readlines() if len(x.strip()) > 0}
    with open(os.path.join(base_dir, "images", "imagenet_label_mapping")) as f:
        image_label_mapping = {int(x.split(":")[0]): x.split(":")[1].strip()
                               for x in f.readlines() if len(x.strip()) > 0}

    images = [(load_image(os.path.join(base_dir, "images", f), 224),
               ground_truth[f])
              for f in os.listdir(os.path.join(base_dir, "images"))
              if f.endswith(".JPEG")]

    return images, image_label_mapping