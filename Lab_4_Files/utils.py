
"""Helper functions: You should not have to touch the following functions.
"""
import os
import cv2
import matplotlib
import PIL
import matplotlib.pyplot as plt
import numpy as np

from matplotlib import animation
from matplotlib.patches import Rectangle

from skimage import filters, img_as_float
from skimage.io import imread
from skimage.transform import pyramid_gaussian
from skimage import data, io, segmentation, color

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin
from sklearn.datasets import load_sample_image
from sklearn.utils import shuffle
from time import time
    
def load_frames_rgb(imgs_dir):
    
    frames = [cv2.cvtColor(cv2.imread(os.path.join(imgs_dir, frame)), cv2.COLOR_BGR2RGB) \
              for frame in sorted(os.listdir(imgs_dir))]
    return frames

def load_frames_as_float_gray(imgs_dir):
    frames = [img_as_float(imread(os.path.join(imgs_dir, frame), 
                                               as_gray=True)) \
              for frame in sorted(os.listdir(imgs_dir))]
    return frames

def load_bboxes(gt_path):
    bboxes = []
    with open(gt_path) as f:
        for line in f:
          
            x, y, w, h = line.split(',')
            #x, y, w, h = line.split()
            bboxes.append((int(x), int(y), int(w), int(h)))
    return bboxes

def animated_frames(frames):
    fig, ax = plt.subplots()
    fig.subplots_adjust(0,0,1,1)
    ax.axis('off')
    im = ax.imshow(frames[0])

    def animate(i):
        im.set_array(frames[i])
        return [im,]

    ani = animation.FuncAnimation(fig, animate, frames=len(frames),
                                  interval=80, blit=True)
    return ani

def animated_bbox(frames, bboxes, figsize=(10,8)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis('off')
    im = ax.imshow(frames[0])
    x, y, w, h = bboxes[0]
    bbox = ax.add_patch(Rectangle((x,y),w,h, linewidth=3,
                                  edgecolor='r', facecolor='none'))

    def animate(i):
        im.set_array(frames[i])
        bbox.set_bounds(*bboxes[i])
        return [im, bbox,]

    ani = animation.FuncAnimation(fig, animate, frames=len(frames),
                                  interval=200, blit=True)

    return ani

def animated_scatter(frames, trajs, figsize=(10,8)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis('off')
    im = ax.imshow(frames[0])
    scat = ax.scatter(trajs[0][:,1], trajs[0][:,0],
                      facecolors='none', edgecolors='r')

    def animate(i):
        im.set_array(frames[i])
        if len(trajs[i]) > 0:
            scat.set_offsets(trajs[i][:,[1,0]])
        else: # If no trajs to draw
            scat.set_offsets([]) # clear the scatter plot

        return [im, scat,]

    ani = animation.FuncAnimation(fig, animate, frames=len(frames),
                                  interval=60, blit=True)

    return ani

def kmeans(n_clusters, image):
    # Authors: Robert Layton <robertlayton@gmail.com>
    #          Olivier Grisel <olivier.grisel@ensta.org>
    #          Mathieu Blondel <mathieu@mblondel.org>
    #
    # License: BSD 3 clause


    
    # Convert to floats instead of the default 8 bits integer coding. Dividing by
    # 255 is important so that plt.imshow behaves works well on float data (need to
    # be in the range [0-1])
    image = np.array(image, dtype=np.float64)

    # Load Image and transform to a 2D numpy array.
    w, h, d = original_shape = tuple(image.shape)
    image_array = np.reshape(image, (w * h, d))

    t0 = time()
    image_array_sample = shuffle(image_array, random_state=0, n_samples=1_000)
    # image_array_sample = image_array
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(
        image_array_sample
    )

    # Get labels for all points

    labels = kmeans.predict(image_array).reshape(w, h)
    
    return labels