""" CS4243 Lab 3: Feature Matching and Applications
"""
import numpy as np
from skimage import filters
from skimage.feature import corner_peaks
from scipy.spatial.distance import cdist
from scipy.ndimage.filters import convolve
from scipy.ndimage import gaussian_filter
from utils import pad, unpad
import math
import cv2
_COLOR_RED = (255, 0, 0)
_COLOR_GREEN = (0, 255, 0)
_COLOR_BLUE = (0, 0, 255)

_COLOR_RED = (255, 0, 0)
_COLOR_GREEN = (0, 255, 0)
_COLOR_BLUE = (0, 0, 255)



##### Part 1: Keypoint Detection, Description, and Matching #####

def harris_corners(img, window_size=3, k=0.04):
    '''
    Compute Harris corner response map. Follow the math equation
    R=Det(M)-k(Trace(M)^2).

    Hint:
        You may use the functions filters.sobel_v filters.sobel_h & scipy.ndimage.filters.convolve, 
        which are already imported above
        
    Args:
        img: Grayscale image of shape (H, W)
        window_size: size of the window function
        k: sensitivity parameter

    Returns:
        response: Harris response image of shape (H, W)
    '''

    H, W= img.shape
    window = np.ones((window_size, window_size))
    response = np.zeros((H, W))

    """ Your code starts here """
    Ix = filters.sobel_v(img)
    Iy = filters.sobel_h(img)

    A = convolve(Ix * Ix, window)
    B = convolve(Ix * Iy, window)
    C = convolve(Iy * Iy, window)

    det = A*C - B*B
    trace = A+C
    response = det - k * np.square(trace)    
    """ Your code ends here """ 
    
    return response

def naive_descriptor(patch):
    '''
    Describe the patch by normalizing the image values into a standard 
    normal distribution (having mean of 0 and standard deviation of 1) 
    and then flattening into a 1D array. 
    
    The normalization will make the descriptor more robust to change 
    in lighting condition.

    Args:
        patch: grayscale image patch of shape (h, w)
    
    Returns:
        feature: 1D array of shape (h * w)
    '''
    feature = []
    
    """ Your code starts here """
    mean = np.mean(patch)
    std = np.std(patch)
    feature_mat = (patch - mean) / (std + 0.0001)
    feature = feature_mat.flatten()
    """ Your code ends here """

    return feature

# GIVEN
def describe_keypoints(image, keypoints, desc_func, patch_size=16):
    '''
    Args:
        image: grayscale image of shape (H, W)
        keypoints: 2D array containing a keypoint (x, y) in each row
        desc_func: function that takes in an image patch and outputs
            a 1D feature vector describing the patch
        patch_size: size of a square patch at each keypoint
                
    Returns:
        desc: array of features describing the keypoints
    '''

    image.astype(np.float32)
    desc = []
    for i, kp in enumerate(keypoints):
        y, x = kp
        patch = image[np.max([0,y-(patch_size//2)]):y+((patch_size+1)//2),
                      np.max([0,x-(patch_size//2)]):x+((patch_size+1)//2)]
      
        desc.append(desc_func(patch))
   
    return np.array(desc)

# GIVEN
def make_gaussian_kernel(ksize, sigma):
    '''
    Good old Gaussian kernel.
    :param ksize: int
    :param sigma: float
    :return kernel: numpy.ndarray of shape (ksize, ksize)
    '''

    ax = np.linspace(-(ksize - 1) / 2., (ksize - 1) / 2., ksize)
    yy, xx = np.meshgrid(ax, ax)

    kernel = np.exp(-0.5 * (np.square(yy) + np.square(xx)) / np.square(sigma))

    return kernel / kernel.sum()


def simple_sift(patch):
    '''
    Your implementation does not need to exactly match the SIFT reference.
    Here are the key properties your (baseline) descriptor should have:
    (1) a 4x4 grid of cells, each length of 16/4=4. It is simply the
        terminology used in the feature literature to describe the spatial
        bins where gradient distributions will be described.
    (2) each cell should have a histogram of the local distribution of
        gradients in 8 orientations. Appending these histograms together will
        give you 4x4 x 8 = 128 dimensions.
    (3) Each feature should be normalized to unit length.

    Use the gradient orientation to determine the bin, and the gradient magnitude * weight from
    the Gaussian kernel as vote weight.

    Args:
        patch: grayscale image patch of shape (h, w)

    Returns:
        feature: 1D array of shape (128, )
    '''
    
    # You can change the parameter sigma, which has been default to 3
    weights = np.flipud(np.fliplr(make_gaussian_kernel(patch.shape[0],3)))
    
    histogram = np.zeros((4,4,8))
    
    """ Your code starts here """
    Ix = filters.sobel_v(patch).reshape(4,4,-1,4).swapaxes(1,2).reshape(4,4,4,4)
    Iy = filters.sobel_h(patch).reshape(4,4,-1,4).swapaxes(1,2).reshape(4,4,4,4)
    weight_arr = weights.reshape(4,4,-1,4).swapaxes(1,2).reshape(4,4,4,4)
    for i in range(4):
        for j in range(4):
            hist_bins = np.zeros(8, dtype=float)
            curr_Ix = Ix[i,j]
            curr_Iy = Iy[i,j]
            magnitude = np.sqrt(curr_Ix**2 + curr_Iy**2)
            orientation = np.arctan2(curr_Iy, curr_Ix)
            bin_index = (np.degrees(orientation + np.pi) // 45).astype(int)
            bin_index[bin_index == 8] = 0 # if angle is 360 degrees, change to 0 degrees
            vote_weight = magnitude * weight_arr[i,j]
            for idx, weight in zip(bin_index.flatten(), vote_weight.flatten()):
                hist_bins[idx] += weight
            histogram[i,j] = hist_bins
    feature = histogram.flatten()
    feature = feature / np.linalg.norm(feature)
    """ Your code ends here """

    return feature

def top_k_matches(desc1, desc2, k=2):
    '''
    Compute the Euclidean distance between each descriptor in desc1 versus all descriptors in desc2 (Hint: use cdist).
    For each descriptor Di in desc1, pick out k nearest descriptors from desc2, as well as the distances themselves.
    Example of an output of this function:
    
        [(0, [(18, 0.11414082134194799), (28, 0.139670625444803)]),
         (1, [(2, 0.14780585099287238), (9, 0.15420019834435536)]),
         (2, [(64, 0.12429203239414029), (267, 0.1395765079352806)]),
         ...<truncated>
    '''
    match_pairs = []
    
    """ Your code starts here """
    euclidean_distances = cdist(desc1, desc2, metric="euclidean") # returns m1xm2 matrix of euclidean distances between desc1 and desc2
    sorted_indices = np.argsort(euclidean_distances, axis=1) # return indices that would sort the values along horizontal axis
    sorted_euclidean_distances = np.take_along_axis(euclidean_distances, sorted_indices, axis=1) # same result as np.sort(euclidean_distances, axis=1)

    for i in range(0, desc1.shape[0]):
        match_pairs.append((i, [(sorted_indices[i][j], sorted_euclidean_distances[i][j]) for j in range(0, k)]))
    """ Your code ends here """

    return match_pairs

def ratio_test_match(desc1, desc2, match_threshold):
    '''
    Match two set of descriptors using the ratio test.
    Output should be a numpy array of shape (k,2), where k is the number of matches found. 
    In the following sample output:
        array([[  3,   0],
               [  5,  30],
               [ 11,   9],
               [ 18,   7],
               [ 24,   5],
               [ 30,  17],
               [ 32,  24],
               [ 46,  23], ... <truncated>
              )
              
        desc1[3] is matched with desc2[0], desc1[5] is matched with desc2[30], and so on.
    
    All other match functions will return in the same format as does this one.
    
    '''
    match_pairs = []
    top_2_matches = top_k_matches(desc1, desc2)
    
    """ Your code starts here """
    for match_tuple in top_2_matches:
        f1_idx, matches = match_tuple
        f_2a, f_2b = matches
        f_2a_idx, f_2a_value = f_2a
        _, f_2b_value = f_2b

        ratio = f_2a_value / f_2b_value

        if ratio < match_threshold:
            match_pairs.append([f1_idx, f_2a_idx])
    """ Your code ends here """

    # Modify this line as you wish
    match_pairs = np.array(match_pairs)
    return match_pairs

# GIVEN
def compute_cv2_descriptor(im, method=cv2.SIFT_create()):
    '''
    Detects and computes keypoints using one of the implementations in OpenCV
    You can use:
        cv2.SIFT_create()

    Do note that the keypoints coordinate is (col, row)-(x,y) in OpenCV. We have changed it to (row,col)-(y,x) for you. (Consistent with out coordinate choice)
    '''
    kpts, descs = method.detectAndCompute(im, None)
    
    keypoints = np.array([(kp.pt[1],kp.pt[0]) for kp in kpts])
    angles = np.array([kp.angle for kp in kpts])
    sizes = np.array([kp.size for kp in kpts])
    
    return keypoints, descs, angles, sizes

##### Part 2: Image Stitching #####

# GIVEN
def transform_homography(src, h_matrix, getNormalized = True):
    '''
    Performs the perspective transformation of coordinates

    Args:
        src (np.ndarray): Coordinates of points to transform (N,2)
        h_matrix (np.ndarray): Homography matrix (3,3)

    Returns:
        transformed (np.ndarray): Transformed coordinates (N,2)

    '''
    transformed = None

    input_pts = np.insert(src, 2, values=1, axis=1)
    transformed = np.zeros_like(input_pts)
    transformed = h_matrix.dot(input_pts.transpose())
    if getNormalized:
        transformed = transformed[:-1]/transformed[-1]
    transformed = transformed.transpose().astype(np.float32)
    
    return transformed


def compute_homography(src, dst):
    '''
    Calculates the perspective transform from at least 4 points of
    corresponding points using the **Normalized** Direct Linear Transformation
    method.

    Args:
        src (np.ndarray): Coordinates of points in the first image (N,2)
        dst (np.ndarray): Corresponding coordinates of points in the second
                          image (N,2)

    Returns:
        h_matrix (np.ndarray): The required 3x3 transformation matrix H.

    Prohibited functions:
        cv2.findHomography(), cv2.getPerspectiveTransform(),
        np.linalg.solve(), np.linalg.lstsq()
    '''
    h_matrix = np.eye(3, dtype=np.float64)
  
    """ Your code starts here """
    # Compute normalization matrices T and T_prime
    mx, my = np.mean(src, axis=0)
    mx_prime, my_prime = np.mean(dst, axis=0)
    sx, sy = np.std(src, axis=0) * 1/np.sqrt(2)
    sx_prime, sy_prime = np.std(dst, axis=0) * 1/np.sqrt(2)

    T = np.array([
        [1/sx, 0, -mx/sx],
        [0, 1/sy, -my/sy],
        [0, 0, 1]
    ])

    T_prime = np.array([
        [1/sx_prime, 0, -mx_prime/sx_prime],
        [0, 1/sy_prime, -my_prime/sy_prime],
        [0, 0, 1]
    ])

    # Add ones to matched keypoints
    p = np.concatenate((src, np.ones((src.shape[0], 1))), axis=1)
    p_prime = np.concatenate((dst, np.ones((dst.shape[0], 1))), axis=1)

    # Perform normalization
    q = T @ np.transpose(p)
    q_prime = T_prime @ np.transpose(p_prime)
    normalized_src = np.transpose(q)
    normalized_dst = np.transpose(q_prime)

    # Get matrix A
    A = None
    is_first_iteration = True
    for i in range(len(src)):
        normalized_p = normalized_src[i]
        normalized_p_prime = normalized_dst[i]

        x, y, _ = normalized_p
        x_prime, y_prime, _ = normalized_p_prime

        A_i = np.array([
            [-x, -y, -1, 0, 0, 0, x*x_prime, y*x_prime, x_prime],
            [0, 0, 0, -x, -y, -1, x*y_prime, y*y_prime, y_prime]
        ])

        if is_first_iteration:
            A = A_i
            is_first_iteration = False
        else:
            A = np.concatenate((A, A_i), axis=0)

    # Perform SVD and calculate h_matrix
    U, S, V_t = np.linalg.svd(A, full_matrices=False)
    min_singular_value_idx = np.argmin(S)
    corresponding_singular_vector = V_t[min_singular_value_idx]
    h_matrix = np.reshape(corresponding_singular_vector, (3,3))
    h_matrix = np.linalg.inv(T_prime) @ h_matrix @ T
    """ Your code ends here """

    return h_matrix

def ransac_homography(keypoints1, keypoints2, matches, sampling_ratio=0.5, n_iters=500, delta=20):
    """
    Use RANSAC to find a robust affine transformation

        1. Select random set of matches
        2. Compute affine transformation matrix
        3. Compute inliers
        4. Keep the largest set of inliers
        5. Re-compute least-squares estimate on all of the inliers

    Args:
        keypoints1: M1 x 2 matrix, each row is a point
        keypoints2: M2 x 2 matrix, each row is a point
        matches: N x 2 matrix, each row represents a match
            [index of keypoint1, index of keypoint 2]
        sampling_ratio: percentage of points selected at each iteration
        n_iters: the number of iterations RANSAC will run
        threshold: the threshold to find inliers

    Returns:
        H: a robust estimation of affine transformation from keypoints2 to
        keypoints1
    """
    N = matches.shape[0]
    n_samples = int(N * sampling_ratio)

    matched1_unpad = keypoints1[matches[:,0]]
    matched2_unpad = keypoints2[matches[:,1]]

    max_inliers = np.zeros(N)
    n_inliers = 0

    # RANSAC iteration start
    
    """ Your code starts here """
    for i in range(n_iters):
        # Get n_samples number of random correspondences
        n_sample_correspondences_indices = np.random.choice(N, n_samples, replace=False)

        # Calculate H matrix with random samples using DLT
        h_matrix = compute_homography(matched1_unpad[n_sample_correspondences_indices], matched2_unpad[n_sample_correspondences_indices])

        # Determine inlier count with calculated H matrix
        p_prime = transform_homography(matched1_unpad, h_matrix)
        differences = matched2_unpad - p_prime
        squared_distances = np.square(differences[:,0]) + np.square(differences[:,1])
        inliers = (squared_distances < np.square(delta))
        inlier_count = np.sum(inliers)

        # Keep H if largest number of inliers
        if inlier_count > n_inliers:
            max_inliers = inliers
            n_inliers = inlier_count

    # For best H with most inliers, recompute H using all inliers
    inliers1_unpad = keypoints1[matches[max_inliers][:,0]]
    inliers2_unpad = keypoints2[matches[max_inliers][:,1]]
    H = compute_homography(inliers1_unpad, inliers2_unpad)
    """ Your code ends here """
    
    return H, matches[max_inliers]

##### Part 3: Mirror Symmetry Detection #####

# GIVEN 
from skimage.feature import peak_local_max
def find_peak_params(hspace, params_list,  window_size=1, threshold=0.5):
    '''
    Given a Hough space and a list of parameters range, compute the local peaks
    aka bins whose count is larger max_bin * threshold. The local peaks are computed
    over a space of size (2*window_size+1)^(number of parameters)

    Also include the array of values corresponding to the bins, in descending order.
    '''
    assert len(hspace.shape) == len(params_list), \
        "The Hough space dimension does not match the number of parameters"
    for i in range(len(params_list)):
        assert hspace.shape[i] == len(params_list[i]), \
            f"Parameter length does not match size of the corresponding dimension:{len(params_list[i])} vs {hspace.shape[i]}"
    peaks_indices = peak_local_max(hspace.copy(), exclude_border=False, threshold_rel=threshold, min_distance=window_size)
    peak_values = np.array([hspace[tuple(peaks_indices[j])] for j in range(len(peaks_indices))])
    res = []
    res.append(peak_values)
    for i in range(len(params_list)):
        res.append(params_list[i][peaks_indices.T[i]])
    return res

# GIVEN
def angle_with_x_axis(pi, pj):  
    '''
    Compute the angle that the line connecting two points I and J make with the x-axis (mind our coordinate convention)
    Do note that the line direction is from point I to point J.
    '''
    # get the difference between point p1 and p2
    y, x = pi[0]-pj[0], pi[1]-pj[1] 
    
    if x == 0:
        return np.pi/2  
    
    angle = np.arctan(y/x)
    if angle < 0:
        angle += np.pi
    return angle

# GIVEN
def midpoint(pi, pj):
    '''
    Get y and x coordinates of the midpoint of I and J
    '''
    return (pi[0]+pj[0])/2, (pi[1]+pj[1])/2

# GIVEN
def distance(pi, pj):
    '''
    Compute the Euclidean distance between two points I and J.
    '''
    y,x = pi[0]-pj[0], pi[1]-pj[1] 
    return np.sqrt(x**2+y**2)

def shift_sift_descriptor(desc):
    '''
       Generate a virtual mirror descriptor for a given descriptor.
       Note that you have to shift the bins within a mini histogram, and the mini histograms themselves.
       e.g:
       Descriptor for a keypoint
       (the dimension is (128,), but here we reshape it to (16,8). Each length-8 array is a mini histogram.)
      [[  0.,   0.,   0.,   5.,  41.,   0.,   0.,   0.],
       [ 22.,   2.,   1.,  24., 167.,   0.,   0.,   1.],
       [167.,   3.,   1.,   4.,  29.,   0.,   0.,  12.],
       [ 50.,   0.,   0.,   0.,   0.,   0.,   0.,   4.],
       
       [  0.,   0.,   0.,   4.,  67.,   0.,   0.,   0.],
       [ 35.,   2.,   0.,  25., 167.,   1.,   0.,   1.],
       [167.,   4.,   0.,   4.,  32.,   0.,   0.,   5.],
       [ 65.,   0.,   0.,   0.,   0.,   0.,   0.,   1.],
       
       [  0.,   0.,   0.,   0.,  74.,   1.,   0.,   0.],
       [ 36.,   2.,   0.,   5., 167.,   7.,   0.,   4.],
       [167.,  10.,   0.,   1.,  30.,   1.,   0.,  13.],
       [ 60.,   2.,   0.,   0.,   0.,   0.,   0.,   1.],
       
       [  0.,   0.,   0.,   0.,  54.,   3.,   0.,   0.],
       [ 23.,   6.,   0.,   4., 167.,   9.,   0.,   0.],
       [167.,  40.,   0.,   2.,  30.,   1.,   0.,   0.],
       [ 51.,   8.,   0.,   0.,   0.,   0.,   0.,   0.]]
     ======================================================
       Descriptor for the same keypoint, flipped over the vertical axis
      [[  0.,   0.,   0.,   3.,  54.,   0.,   0.,   0.],
       [ 23.,   0.,   0.,   9., 167.,   4.,   0.,   6.],
       [167.,   0.,   0.,   1.,  30.,   2.,   0.,  40.],
       [ 51.,   0.,   0.,   0.,   0.,   0.,   0.,   8.],
       
       [  0.,   0.,   0.,   1.,  74.,   0.,   0.,   0.],
       [ 36.,   4.,   0.,   7., 167.,   5.,   0.,   2.],
       [167.,  13.,   0.,   1.,  30.,   1.,   0.,  10.],
       [ 60.,   1.,   0.,   0.,   0.,   0.,   0.,   2.],
       
       [  0.,   0.,   0.,   0.,  67.,   4.,   0.,   0.],
       [ 35.,   1.,   0.,   1., 167.,  25.,   0.,   2.],
       [167.,   5.,   0.,   0.,  32.,   4.,   0.,   4.],
       [ 65.,   1.,   0.,   0.,   0.,   0.,   0.,   0.],
       
       [  0.,   0.,   0.,   0.,  41.,   5.,   0.,   0.],
       [ 22.,   1.,   0.,   0., 167.,  24.,   1.,   2.],
       [167.,  12.,   0.,   0.,  29.,   4.,   1.,   3.],
       [ 50.,   4.,   0.,   0.,   0.,   0.,   0.,   0.]]
    '''
    
    """ Your code starts here """
    reshaped_desc = desc.reshape((4,4,8))
    flip_desc = np.flipud(reshaped_desc) # assuming the outer 4 arrays (axis=0) are 0,1,2,3, the flipped arrays are 3,2,1,0
    flip_desc_2d = flip_desc.reshape((flip_desc.shape[0]*flip_desc.shape[1], flip_desc.shape[2])) # flatten back to a 2d array
    flip_desc_2d[:,1:] = flip_desc_2d[:,-1:0:-1] # assuming each array (axis=0) contains 0,1,2,...,7, new array is 0,7,6,...,1
    res = flip_desc_2d.flatten() # flatten into a 128-element 1d array
    """ Your code ends here """
    
    return res

def create_mirror_descriptors(img):
    '''
    Return the output for compute_cv2_descriptor (which you can find in utils.py)
    Also return the set of virtual mirror descriptors.
    Make sure the virtual descriptors correspond to the original set of descriptors.
    '''
    
    """ Your code starts here """
    kps, descs, angles, sizes = compute_cv2_descriptor(img)
    mir_descs = np.apply_along_axis(shift_sift_descriptor, 1, descs)
    """ Your code ends here """
    
    return kps, descs, sizes, angles, mir_descs

def match_mirror_descriptors(descs, mirror_descs, threshold = 0.7):
    '''
    First use `top_k_matches` to find the nearest 3 matches for each keypoint. Then eliminate the mirror descriptor that comes 
    from the same keypoint. Perform ratio test on the two matches left. If no descriptor is eliminated, perform the ratio test 
    on the best 2. 
    '''
    three_matches = top_k_matches(descs, mirror_descs, k=3)

    match_result = []

    
    """ Your code starts here """
    for f1_idx, matches in three_matches:
        top_2_matches = list(filter(lambda match: match[0] != f1_idx, matches)) # eliminate mirror descriptor from same keypoint
        f_2a, f_2b = top_2_matches[0], top_2_matches[1]
        f_2a_idx, f_2a_value = f_2a
        _, f_2b_value = f_2b

        ratio = f_2a_value / f_2b_value

        if ratio < threshold:
            match_result.append([f1_idx, f_2a_idx])
    match_result = np.array(match_result)
    """ Your code ends here """
    
    return match_result


def find_symmetry_lines(matches, kps):
    '''
    For each pair of matched keypoints, use the keypoint coordinates to compute a candidate symmetry line.
    Assume the points associated with the original descriptor set to be I's, and the points associated with the mirror descriptor set to be
    J's.
    '''
    rhos = []
    thetas = []
    
    """ Your code starts here """
    for i, j in kps[matches]:
        mid_y, mid_x = midpoint(i, j)
        angle = angle_with_x_axis(i, j)
        rho = mid_x * np.cos(angle) + mid_y * np.sin(angle)
        thetas.append(angle)
        rhos.append(rho)
    """ Your code ends here """
    
    return rhos, thetas

def hough_vote_mirror(matches, kps, im_shape, window=1, threshold=0.5, num_lines=1):
    '''
    Hough Voting:
                 0<=thetas<= 2pi      , interval size = 1 degree
        -diagonal <= rhos <= diagonal , interval size = 1 pixel
    Feel free to vary the interval size.
    '''
    rhos, thetas = find_symmetry_lines(matches, kps)
    
    """ Your code starts here """
    DISTANCE_INTERVAL = 1
    THETA_INTERVAL = np.pi/180

    height, width = im_shape
    len_image_diagonal = np.sqrt(height**2 + width**2)

    DISTANCE_RANGE_LEN = 2*len_image_diagonal  # [-d, d]
    THETA_RANGE_LEN = np.pi  # [0, 𝜋]

    number_of_distance_bins = int(DISTANCE_RANGE_LEN / DISTANCE_INTERVAL)
    number_of_theta_bins = int(THETA_RANGE_LEN / THETA_INTERVAL)

    A = np.zeros((number_of_distance_bins, number_of_theta_bins))

    for rho, theta in zip(rhos, thetas):
        rho_bin_idx = int((rho + len_image_diagonal) / DISTANCE_INTERVAL)
        theta_bin_idx = int(theta / THETA_INTERVAL)
        A[rho_bin_idx, theta_bin_idx] += 1

    # Quantized theta values [0, 𝜋]
    theta_space = np.array([THETA_INTERVAL * i for i in range(number_of_theta_bins)])

    # Quantized distance (rho) values [-d, d]
    rho_space = np.array([DISTANCE_INTERVAL * i - int(len_image_diagonal) for i in range(number_of_distance_bins)])

    maxima = find_peak_params(A, [rho_space, theta_space], window, threshold)
    rho_values = []
    theta_values = []
    for i in range(num_lines):
        rho_values.append(maxima[1][i])
        theta_values.append(maxima[2][i])
    """ Your code ends here """
    
    return rho_values, theta_values




"""Helper functions: You should not have to touch the following functions.
"""
def trim(frame):
    if not np.sum(frame[0]):
        return trim(frame[1:])
    if not np.sum(frame[-1]):
        return trim(frame[:-2])
    if not np.sum(frame[:,0]):
        return trim(frame[:,1:])
    if not np.sum(frame[:,-1]):
        return trim(frame[:,:-2])
    return frame