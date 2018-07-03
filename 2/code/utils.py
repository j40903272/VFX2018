
# coding: utf-8

# In[ ]:


from pylab import *
import numpy as np
from scipy.ndimage import filters
import scipy
import matplotlib.pylab as pyp
from PIL import Image
import transform
import imgutil
import cv2


# In[ ]:


def HarrisCornerDetector(img, sigma=3):
    #win_size is the minimum number of pixels seperating the corners and image boundary
    
    # derivatives
    imx = zeros(img.shape)
    imy = zeros(img.shape)
    filters.gaussian_filter(img, (sigma, sigma), (0, 1), imx)
    filters.gaussian_filter(img, (sigma, sigma), (1, 0), imy)
    
    # compute components of the Harris matrix
    Ixx = filters.gaussian_filter(imx*imx,sigma)
    Ixy = filters.gaussian_filter(imx*imy,sigma)
    Iyy = filters.gaussian_filter(imy*imy,sigma)
    
    det = Ixx*Iyy - Ixy**2
    trace = Ixx + Iyy + 1e-10
    #R = det - 0.04*(trace**2)
    R = det / trace
    
    return R


# In[ ]:


def supression(R, win=5, thres=512):
    h, w = R.shape[0:2]
    
    R[:win, :] = 0
    R[-win:, :] = 0
    R[:, :win] = 0
    R[:, -win:] = 0
    
    # non-maximum suppression in 3x3 regions
    maxH = filters.maximum_filter(R, (win, win))
    R = R * (R == maxH)

    # sort points by strength and find their positions
    sortIdx = np.argsort(R.flatten())[::-1][:thres]
    y = sortIdx // w
    x = sortIdx % w

    # concatenate positions and values
    coords = np.vstack((x, y, R.flatten()[sortIdx])).T

    return coords


# In[ ]:


def plot_corners(img, coords): #Plots Harris corners on graph
    fig = figure(figsize=(15, 8))
    gray()
    imshow(img)
    plot(coords[:, 0], coords[:, 1], 'r*', markersize=5)
    axis('off')
    show()


# In[ ]:


def gen_descriptors(img, coords, win=5): #return pixel value
    descriptors = []
    for c in coords:
        x, y = int(c[1]), int(c[0])
        patch = img[x-win:x+win+1, y-win:y+win+1].flatten()
        # normalize the descriptor
        patch = (patch - patch.mean()) / patch.std()
        descriptors.append(patch)
    return np.array(descriptors)


# In[1]:


def match(d1, d2):
    n = len(d1)
    
    dists = scipy.spatial.distance.cdist(d1, d2)
    sort_idx = np.argsort(dists, 1)

    # find best indices and their distances
    best_idx = sort_idx[:, 0]
    best_distance = dists[np.r_[0: n], best_idx]

    # find second best indices and their distances
    second_best_idx = sort_idx[:, 1]
    second_best_distance = dists[np.r_[0:n], second_best_idx]

    # find the average of the second best distance
    mean = second_best_distance.mean()
    ratio = best_distance / mean

    # find the indices of the bestMatches for each descriptor
    desc1_match = np.argwhere(ratio < 0.5)
    desc2_match = best_idx[desc1_match]

    # put the matches in a single array and return as type int
    matches = np.hstack([desc1_match, desc2_match])
    
    return matches.astype(int)


# In[ ]:


def match2(d1, d2, thres=0.5, method='ncc'):
    # feature dim
    n = len(d1[0])
    
    if method == 'ncc': # normalized cross corelation
        d = -np.ones((len(d1), len(d2)))
        for i in range(len(d1)):
            for j in range(len(d2)):
                dist1 = (d1[i] - np.mean(d1[i])) / np.std(d1[i])
                dist2 = (d2[j] - np.mean(d2[j])) / np.std(d2[j])
                ncc_value = np.sum(dist1 * dist2) / (n-1)
                if ncc_value > thres:
                    d[i, j] = ncc_value

        ndx = np.argsort(-d)
        matchscores = ndx[:, 0]
        return matchscores # matched coords
    
    elif method == 'ed': # Euclidean Distance
        min_dist, max_dist = 100, 0
        distance = np.zeros((len(d1), len(d2)))
        for i in range(len(d1)):
            for j in range(len(d2)):
                dist = np.linalg.norm(d1[i]-d2[j])
                min_dist = min(min_dist, dist)
                max_dist = max(max_dist, dist)
                distance[i, j] = dist
        
        sort_dist = np.argsort(distance)[:, 0]
        min_dist = max(min_dist, thres)
        good_match = []
        for i, j in enumerate(sort_dist):
            if distance[i, j] < (min_dist*10):
                good_match.append([i, j])
        
        return np.array(good_match)


# In[ ]:


def appendimages(im1, im2): #the appended images displayed side by side for image mapping
    
    # select the image with the fewest rows and fill in enough empty rows
    rows1 = im1.shape[0]
    rows2 = im2.shape[0]
    
    if rows1 < rows2:
        im1 = concatenate((im1,zeros((rows2-rows1,im1.shape[1]))),axis=1)
    elif rows1 > rows2:
        im2 = concatenate((im2,zeros((rows1-rows2,im2.shape[1]))),axis=1)
    
    
    return concatenate((im2, im1), axis=1)

