# -*- coding: utf-8 -*-
"""NUS CS4243 Lab4.

"""

import numpy as np
import cv2
from skimage import filters
from scipy.ndimage.filters import convolve

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
    def compute_gradients(firstImage, secondImage):
        firstImage = firstImage / 255
        secondImage = secondImage / 255

        # Kernels for finding gradients Ix, Iy, It
        kernel_x = np.array([[-1., 0., 1.], [-1., 0., 1.]]) / 4
        kernel_y = np.array([[-1., -1.], [0., 0.], [1., 1.]]) / 4
        kernel_t = np.array([[1., 1.], [1., 1.]]) / 4

        Ix = convolve(input=firstImage, weights=kernel_x, mode="constant")
        Iy = convolve(input=firstImage, weights=kernel_y, mode="constant")
        It = convolve(input=secondImage, weights=kernel_t, mode="constant") + convolve(
            input=firstImage, weights=-kernel_t, mode="constant"
        )

        I = [Ix, Iy, It]

        return I

    Ix, Iy, It = compute_gradients(prevImg, nextImg)

    avg_kernel = np.array([[0, 1/4, 0],
                            [1/4, 0, 1/4],
                            [0, 1/4, 0]], float)

    # Initialize flow fields u and v
    u = np.zeros(prevImg.shape)
    v = np.zeros(prevImg.shape)

    d= (1/param_lambda + Ix**2 + Iy**2)

    # Update equations
    while True:
        u_bar = convolve(u, avg_kernel, mode="constant")
        v_bar = convolve(v, avg_kernel, mode="constant")

        # the fraction term multiplied to Ix and Iy to determine the new u and v values in the update equations
        update_constant = (Ix*u_bar + Iy*v_bar + It) / d
        previous = u

        u = u_bar - update_constant*Ix
        v = v_bar - update_constant*Iy

        # Check for convergence and break out of loop if convergence conditions are met
        diff = np.linalg.norm(u - previous)
        if diff < param_delta:
            print("converged")
            break

    flow = np.stack((u, v), axis=2)
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
    def normalization(arr):
        return (arr - np.mean(arr)) / np.std(arr)

    r = feat1[..., 0]
    g = feat1[..., 1]
    b = feat1[..., 2]

    r_mean = normalization(r)
    g_mean = normalization(g)
    b_mean = normalization(b)

    u = feat2[..., 0]
    v = feat2[..., 1]

    u_mean = normalization(u)
    v_mean = normalization(v)

    feats = np.array((r_mean, g_mean, b_mean, u_mean, v_mean)).transpose((1,2,0))
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
    feats = []
    img_lab = cv2.cvtColor(image, cv2.COLOR_RGB2Lab)
    for k_gaus in kernels['gaussian']:
        for c in range(img_lab.shape[-1]):
            feats.append(cv2.filter2D(img_lab[:,:,c], -1, kernel=k_gaus))
    for k_dog in kernels['gaussian_derivative']:
        feats.append(cv2.filter2D(img_lab[:,:,0], -1, kernel=k_dog))
    for k_log in kernels['LoG']:
        feats.append(cv2.filter2D(img_lab[:,:,0], -1, kernel=k_log))
    feats = np.array(feats).transpose((1,2,0))
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
        # STEP 1: Apply filter bank to training images
        feature_img_lst = list(map(lambda img: features_from_filter_bank(img, self.kernels), training_imgs))
        feature_img_lst = list(map(lambda img: img.reshape((img.shape[0]*img.shape[1], img.shape[2])), feature_img_lst)) # flatten (w,h,17) to (w*h,17)
        feature_img_lst = np.concatenate(feature_img_lst, axis=0)
        # STEP 2: Cluster in feature space and store cluster centers
        mbk = MiniBatchKMeans(n_clusters=self.n_clusters) # default: batch_size=1024
        mbk.fit(feature_img_lst)
        cluster_centers = np.sort(mbk.cluster_centers_, axis=0)
        self.cc_tree = KDTree(cluster_centers) # store cluster centers in KDTree for testing
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
        # Filter image with filter bank and flatten
        feature_img = features_from_filter_bank(img, self.kernels)
        flat_img = feature_img.reshape((feature_img.shape[0]*feature_img.shape[1], feature_img.shape[2]))
        # Query tree and get predicted cluster centers
        textons = self.cc_tree.query(flat_img, return_distance=False)
        # Reshape textons back to original image shape
        textons = textons.reshape((feature_img.shape[0], feature_img.shape[1], 1))
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
    hists = np.zeros(textons.shape[:2] + (200,))
    half_size = (window_size - 1) // 2
    for i in range(textons.shape[0]):
        for j in range(textons.shape[1]):
            window = textons[i-half_size:i+half_size, j-half_size:j+half_size, :]
            hists[i,j] = np.bincount(window.flatten(), minlength=200)
    # TASK 2.3 #
    
    return hists


