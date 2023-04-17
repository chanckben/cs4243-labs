# -*- coding: utf-8 -*-
"""NUS CS4243 Lab4.

"""

import numpy as np
import cv2
from skimage import filters
from scipy.ndimage.interpolation import shift


# TASK 1.1 #
def calcOpticalFlowHS(prevImg: np.array, nextImg: np.array, param_lambda: float, param_delta: float) -> np.array:
    """Computes a dense optical flow using the Horn–Schunck algorithm.
    
    The function finds an optical flow for each prevImg pixel using the Horn and Schunck algorithm [Horn81] so that: 
    
        prevImg(y,x) ~ nextImg(y + flow(y,x,2), x + flow(y,x,1)).


    Args:
        prevImg (np.array): First 8-bit single-channel input image.
        nextImg (np.array): Second input image of the same size and the same type as prevImg.
        param_lambda (float): Smoothness weight. The larger it is, the smoother optical flow map you get.
        param_delta (float): pre-set threshold for determing convergence between iterations.

    Returns:
        flow (np.array): Computed flow image that has the same size as prevImg and single 
            type (2-channels). Flow for (x,y) is stored in the third dimension.
        
    """
    # TASK 1.1 #
    Ix = filters.sobel_v(prevImg)
    Iy = filters.sobel_h(prevImg)

    # Compute frame differences
    It = nextImg - prevImg

    # Local average function
    def local_average(matrix):
        right_shift = shift(matrix, (0, 1), cval=0)
        left_shift = shift(matrix, (0, -1), cval=0)
        top_shift = shift(matrix, (-1, 0), cval=0)
        bottom_shift = shift(matrix, (1, 0), cval=0)

        return (right_shift + left_shift + top_shift + bottom_shift) / 4

    # Initialize flow fields u and v
    u = np.zeros(prevImg.shape)
    v = np.zeros(prevImg.shape)

    # Update equations
    while True:
        u_bar = local_average(u)
        v_bar = local_average(v)

        # the fraction term multiplied to Ix and Iy to determine the new u and v values in the update equations
        update_constant = (Ix*u_bar + Iy*v_bar + It) / (np.reciprocal(param_lambda) + Ix*Ix + Iy*Iy)
        u_updated = u_bar - update_constant*Ix
        v_updated = v_bar - update_constant*Iy

        # Check for convergence and break out of loop if convergence conditions are met
        # TODO: Check that difference < param_delta
        # u_diff = 
        # v_diff = 

    # TASK 1.1 #

    return flow
    
# TASK 1.2 #
def combine_and_normalize_features(feat1: np.array, feat2: np.array, gamma: float) -> np.array:
    """Combine two features together with proper normalization.

    Args:
        feat1 (np.array): of size (..., N1).
        feat2 (np.array): of size (..., N2).

    Returns:
        feats (np.array): combined features of size of size (..., N1+N2), with feat2 weighted by gamma.
        
    """
    # TASK 1.2 #

    # TASK 1.2 #
    
    return feats


def build_gaussian_kernel(sigma: int) -> np.array:

    def gaussianKernel(sigma):
        halfSize = int(np.ceil(3.0*sigma))
        kernel = np.zeros((2*halfSize+1, 1))
        s2 = sigma * sigma
        f = 1.0 / np.sqrt(2.0 * np.pi * s2)
        w2 = 1.0 / (2.0 * s2)
        for i in range(2*halfSize+1):
            p = i - halfSize
            kernel[i] = f * np.exp(-(p * p) * w2)
        return kernel

    g = gaussianKernel(sigma)

    kernel = g @ g.transpose()

    return kernel

def build_gaussian_derivative_kernel(sigma: int) -> np.array:
    
    def gaussianKernel(sigma):
        halfSize = int(np.ceil(3.0*sigma))
        kernel = np.zeros((2*halfSize+1, 1))
        s2 = sigma * sigma
        f = 1.0 / np.sqrt(2.0 * np.pi * s2)
        w2 = 1.0 / (2.0 * s2)
        for i in range(2*halfSize+1):
            p = i - halfSize
            kernel[i] = f * np.exp(-(p * p) * w2)
        return kernel
    
    def gaussianDerivativeKernel(sigma):
        halfSize = int(np.ceil(3.0*sigma))
        kernel = np.zeros((2*halfSize+1, 1))
        s2 = sigma * sigma
        f = 1.0 / np.sqrt(2.0 * np.pi * s2)
        w = 1.0 / (s2)
        w2 = 1.0 / (2.0 * s2)
        for i in range(2*halfSize+1):
            p = i - halfSize
            kernel[i] = - p * w * f * np.exp(-(p * p) * w2)
        return kernel

    dg = gaussianDerivativeKernel(sigma)
    g = gaussianKernel(sigma)


    kernel_y = dg @ g.transpose()
    kernel_x = g @ dg.transpose()
    
    return kernel_y, kernel_x


def build_LoG_kernel(sigma: int) -> np.array:
    
    def gaussianKernel(sigma):
        halfSize = int(np.ceil(3.0*sigma))
        kernel = np.zeros((2*halfSize+1, 1))
        s2 = sigma * sigma
        f = 1.0 / np.sqrt(2.0 * np.pi * s2)
        w2 = 1.0 / (2.0 * s2)
        for i in range(2*halfSize+1):
            p = i - halfSize
            kernel[i] = f * np.exp(-(p * p) * w2)
        return kernel

    g1 = gaussianKernel(sigma)

    kg1 = g1 @ g1.transpose()

    kernel = cv2.Laplacian(kg1, -1)

    
    return kernel

# TASK 2.1 #
def features_from_filter_bank(image, kernels):
    """Returns 17-dimensional feature vectors for the input image.

    Args:
        img (np.array): of size (..., 3).
        kernels (dict): dictionary storing gaussian, gaussian_derivative, and LoG kernels.

    Returns:
        feats (np.array): of size (..., 17).
        
    """
    # TASK 2.1 #


    # TASK 2.1 #
    return feats


# TASK 2.2 #
from sklearn.cluster import MiniBatchKMeans
from sklearn.neighbors import KDTree

class Textonization:
    def __init__(self, kernels, n_clusters=200):
        self.n_clusters = n_clusters
        self.kernels = kernels

    def training(self, training_imgs):
        """Takes all training images as input and stores the clustering centers for testing.

        Args:
            training_imgs (list): list of training images.
            
        """
        # TASK 2.2 #

        # TASK 2.2 #
        
        pass

    def testing(self, img):
        """Predict the texture label for each pixel of the input testing image. For each pixel in the test image, an ID from a learned texton dictionary can represent it. 

        Args:
            img (np.array): of size (..., 3).
            
        Returns:
            textons (np.array): of size (..., 1).
        
        """
        # TASK 2.2 #

        # TASK 2.2 #
        
        return textons

    
    
# TASK 2.3 #
def histogram_per_pixel(textons, window_size):
    """ Compute texton histogram by computing the distribution of texton indices within the window.

    Args:
        textons (np.array): of size (..., 1).
        
    Returns:
        hists (np.array): of size (..., 200).
    
    """
   
    # TASK 2.3 #

    # TASK 2.3 #
    
    return hists


