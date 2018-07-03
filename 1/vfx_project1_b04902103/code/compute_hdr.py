
# coding: utf-8

# In[1]:


import os
import cv2
import random
import scipy.misc as spm
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from subprocess import call


# In[2]:


# Loading exposure images into a list
img_dir = './'
save_dir = img_dir + 'hdr/'
call(['mkdir', save_dir])
img_fn = [img_dir+'0'+str(i+112)+".JPG" for i in range(9)]
img_list = [spm.imread(fn) for fn in img_fn]
exposure_times = []
with open("exposure.txt", "r") as f:
    for line in f:
        exposure_times.append(1/float(line))
log_exposure_times = np.log(np.array(exposure_times, dtype=np.float32))


# In[3]:


# Align input images
alignMTB = cv2.createAlignMTB()
alignMTB.process(img_list, img_list)


# In[4]:


def linearWeight(pix_val):
    z_min, z_max = 0., 255.
    if pix_val <= (z_min + z_max) / 2:
        return pix_val - z_min
    return z_max - pix_val


# In[5]:


def sampleIntensities(imgs):
    z_min, z_max = 0, 255
    num_layer = len(imgs)
    num_sample = z_max - z_min + 1
    mid_img = imgs[num_layer // 2]
    intensity = np.zeros((num_sample, num_layer), dtype=np.uint8)

    for i in range(z_min, z_max + 1):
        rows, cols = np.where(mid_img == i)
        if len(rows) != 0:
            idx = random.randrange(len(rows))
            r, c = rows[idx], cols[idx]
            for j in range(num_layer):
                intensity[i, j] = imgs[j][r, c]
    
    return intensity


# In[6]:


def computeResponseCurve(samples, log_exp, smooth_lambda, weight_fn):
    z_min, z_max = 0, 255
    intensity_range = 255
    num_samples = samples.shape[0]
    num_imgs = len(log_exp)

    A = np.zeros((num_imgs * num_samples + intensity_range, num_samples + intensity_range + 1), dtype=np.float64)
    B = np.zeros((A.shape[0], 1), dtype=np.float64)

    # Include the data-fitting equations
    k = 0
    for i in range(num_samples):
        for j in range(num_imgs):
            z = samples[i, j]
            w = weight_fn(z)
            A[k, z] = w
            A[k, (intensity_range + 1) + i] = -w
            B[k, 0] = w * log_exp[j]
            k += 1

    # Include the smoothness equations
    for i in range(z_min + 1, z_max):
        w = weight_fn(i)
        A[k, i - 1] = w * smooth_lambda
        A[k, i    ] = -2 * w * smooth_lambda
        A[k, i + 1] = w * smooth_lambda
        k += 1

    # Fix the curve by setting its middle value to 0
    A[k, (z_max - z_min) // 2] = 1

    inv_A = np.linalg.pinv(A)
    x = np.dot(inv_A, B)

    g = x[0: intensity_range + 1]
    return g[:, 0]


# In[7]:


def computeRadianceMap(imgs, log_exp, response_curve, weight_fn):
    img_shape = imgs[0].shape
    rad_map = np.zeros(img_shape, dtype=np.float64)

    num_imgs = len(imgs)
    for i in range(img_shape[0]):
        for j in range(img_shape[1]):
            g = np.array([response_curve[imgs[k][i, j]] for k in range(num_imgs)])
            w = np.array([weight_fn(imgs[k][i, j]) for k in range(num_imgs)])
            W = np.sum(w)
            if W > 0:
                rad_map[i, j] = np.sum(w * (g - log_exp) / W)
            else:
                rad_map[i, j] = g[num_imgs // 2] - log_exp[num_imgs // 2]
    
    return rad_map


# In[8]:


def globalToneMapping(img, gamma):
    recovered = cv2.pow(img/255., 1.0/gamma)
    #recovered = cv2.pow(img, 1.0/gamma)
    return recovered


# In[9]:


def colorAdjustment(img, T):
    m, n, c = img.shape
    output = np.zeros((m, n, c))
    for ch in range(c):
        img_avg, T_avg = np.average(img[:, :, ch]), np.average(T[:, :, ch])
        output[..., ch] = img[..., ch] * (T_avg / img_avg)

    return output


# In[10]:


def computeHDR(imgs, log_exp, smooth_lambda=100., gamma=0.6):
    num_channels = imgs[0].shape[2]
    hdr_img = np.zeros(imgs[0].shape, dtype=np.float64)

    for channel in range(num_channels):
        layer_stack = [img[:, :, channel] for img in imgs]
        intensity_samples = sampleIntensities(layer_stack)
        response_curve = computeResponseCurve(intensity_samples, log_exp, smooth_lambda, linearWeight)
        response_curve_list.append(response_curve)
        rad_map = computeRadianceMap(layer_stack, log_exp, response_curve, linearWeight)
        rad_map_list.append(rad_map)
        hdr_img[..., channel] = cv2.normalize(rad_map, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        
    tone_img = globalToneMapping(hdr_img, gamma)
    template = imgs[len(imgs)//2]
    img_tuned = colorAdjustment(tone_img, template)
    output = cv2.normalize(img_tuned, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    return hdr_img, output.astype(np.uint8)


# In[11]:


def clip(img):
    img[np.where(np.isnan(img))] = 0.0
    img[np.where(np.isinf(img))] = 0.0
    return img


# In[12]:


def combine_ldr(stack, exp_time):
    r, c, col, n = stack.shape
    img_out = np.zeros((r, c, col))
    total_weight = np.zeros((r, c, col))
    for i in range(n):
        tmp_stack = np.power(stack[:,:,:,i] / 255.0, 2.2)
        tmp_weight = np.ones(tmp_stack.shape)
        img_out = img_out + (tmp_weight * tmp_stack) / exp_time[i]
        total_weight = total_weight + tmp_weight
        
    return clip(img_out / total_weight)


# In[13]:


def simpleHDR(imgs, exp_times):
    n_img = len(exp_times)
    if n_img == 0:
        print ('Input images and exposure times are invalid')
        return

    h, w, col = imgs[0].shape
    stack = np.zeros((h, w, col, n_img))
    for i in range(n_img):
        stack[:,:,:,i] = imgs[i]
    return combine_ldr(stack, np.exp(exp_times) + 1.0)


# In[ ]:


response_curve_list = []
rad_map_list = []


# In[ ]:


before_tone, HDR = computeHDR(img_list, log_exposure_times)


# In[ ]:


out = cv2.normalize(before_tone, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
#plt.imshow(out)
plt.imsave(save_dir+'before_tone.jpg', out)


# In[ ]:


#plt.imshow(HDR)
#plt.imsave(save_dir+'HDR.jpg', HDR)


# In[ ]:


rad_map = np.zeros(img_list[0].shape, dtype=np.float64)
for i in range(3):
    rad_map[..., i] = cv2.normalize(rad_map_list[i], None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
tone_img = globalToneMapping(rad_map, 0.9)
template = img_list[len(img_list)//2]
img_tuned = colorAdjustment(tone_img, template)
output = cv2.normalize(img_tuned, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
#plt.imshow(output.astype(np.uint8))
plt.imsave(save_dir+'HDR.jpg', HDR)


# In[66]:


hdr = simpleHDR(img_list, exposure_times)


# In[73]:


#plt.imshow(hdr)



# In[ ]:


def log_mean(img):
    delta = 1.0e-6
    img_delta = np.log(img + delta)
    return np.exp(np.average(img_delta))


# In[67]:


def lum(img):
    l = 0.2126 * img[:,:,0] +         0.7152 * img[:,:,1] +         0.0722 * img[:,:,2]
    return l


# In[ ]:


def drago(img, Ld_max = 100.0, p = 0.95):
    L = lum(img)
    Lwa = log_mean(L)
    Lwa = Lwa / ((1.0 + p - 0.85) ** 5.0)
    LMax = np.max(L)

    L_wa = L / Lwa
    LMax_wa = LMax / Lwa

    c1 = np.log(p) / np.log(0.5)
    c2 = (Ld_max / 100.0) / np.log10(1.0 + LMax_wa)
    Ld = c2 * np.log(1.0 + L_wa) / np.log(2.0 + 8.0 * ((L_wa / LMax_wa) ** c1))

    ret = np.zeros(img.shape)
    for c in range(3):
        ret[:,:,c] = clip(img[:,:,c] * Ld / L)

    ret = np.maximum(ret, 0.0)
    ret = np.minimum(ret, 1.0)
    return ret


# In[ ]:


tm = drago(out)
tm = gamma(tm, 1.0 / 0.6)
plt.imsave(save_dir+'drago.jpg', tm)
plt.imshow(tm)


# In[68]:


def rec_filter_horizontal(I, D, sigma):
    a = np.exp(-np.sqrt(2.0) / sigma)

    F = I.copy()
    V = np.power(a, D)

    h, w, num_channels = I.shape

    for i in range(1,w):
        for c in range(num_channels):
            F[:,i,c] = F[:,i,c] + V[:,i] * (F[:,i-1,c] - F[:,i,c])

    for i in range(w-2,-1,-1):
        for c in range(num_channels):
            F[:,i,c] = F[:,i,c] + V[:,i+1] * (F[:,i+1,c] - F[:,i,c])

    return F


# In[69]:


def bilateral(I, sigma_s, sigma_r, num_iterations=5, J=None):
    if I.ndim == 3:
        img = I.copy()
    else:
        h, w = I.shape
        img = I.reshape((h, w, 1))

    if J is None:
        J = img

    if J.ndim == 2:
        h, w = J.shape
        J = np.reshape(J, (h, w, 1))

    h, w, num_channels = J.shape

    dIcdx = np.diff(J, n=1, axis=1)
    dIcdy = np.diff(J, n=1, axis=0)

    dIdx = np.zeros((h, w))
    dIdy = np.zeros((h, w))

    for c in range(num_channels):
        dIdx[:,1:] = dIdx[:,1:] + np.abs(dIcdx[:,:,c])
        dIdy[1:,:] = dIdy[1:,:] + np.abs(dIcdy[:,:,c])

    dHdx = (1.0 + sigma_s / sigma_r * dIdx)
    dVdy = (1.0 + sigma_s / sigma_r * dIdy)

    dVdy = dVdy.T

    N = num_iterations
    F = img.copy()

    sigma_H = sigma_s

    for i in range(num_iterations):
        sigma_H_i = sigma_H * np.sqrt(3.0) * (2.0 ** (N - (i + 1))) / np.sqrt(4.0 ** N - 1.0)

        F = rec_filter_horizontal(F, dHdx, sigma_H_i)
        F = np.swapaxes(F, 0, 1)
        F = rec_filter_horizontal(F, dVdy, sigma_H_i)
        F = np.swapaxes(F, 0, 1)

    return F


# In[70]:


def bilateral_separation(img, sigma_s=0.02, sigma_r=0.4):
    r, c = img.shape

    sigma_s = max(r, c) * sigma_s

    img_log = np.log10(img + 1.0e-6)
    img_fil = bilateral(img_log, sigma_s, sigma_r)

    base = 10.0 ** (img_fil) - 1.0e-6

    base[base <= 0.0] = 0.0

    base = base.reshape((r, c))
    detail = clip(img / base)

    return base, detail


# In[71]:


'''
Durand and Dorsey SIGGGRAPH 2002,
"Fast Bilateral Fitering for the display of high-dynamic range images"
'''
def durand(img, target_contrast=5.0):
    L = lum(img)
    tmp = np.zeros(img.shape)
    for c in range(3):
        tmp[:,:,c] = clip(img[:,:,c] / L)

    Lbase, Ldetail = bilateral_separation(L)

    log_base = np.log10(Lbase)

    max_log_base = np.max(log_base)
    log_detail = np.log10(Ldetail)
    compression_factor = np.log(target_contrast) / (max_log_base - np.min(log_base))
    log_absolute = compression_factor * max_log_base

    log_compressed = log_base * compression_factor + log_detail - log_absolute

    output = np.power(10.0, log_compressed)

    ret = np.zeros(img.shape)
    for c in range(3):
        ret[:,:,c] = tmp[:,:,c] * output

    ret = np.maximum(ret, 0.0)
    ret = np.minimum(ret, 1.0)

    return ret


# In[72]:


def gamma(L, g):
    return np.power(L, g)


# In[74]:


tm = durand(hdr)
tm = gamma(tm, 1.0 / 2.2)
#plt.imshow(tm)
plt.imsave(save_dir+'gamma correction.jpg', hdr)


# In[49]:


tm = durand(out)
tm = gamma(tm, 1.0 / 0.6)
plt.imsave(save_dir+'hdr+gamma correction+durand.jpg', tm)
#plt.imshow(tm)

tm = durand(out)
plt.imsave(save_dir+'hdr+durand.jpg', tm)



'''
plt.clf()
plt.subplot(111)
e = list(range(len(response_curve_list[0])))
plt.plot(e, response_curve_list[0], 'b', label='b')
plt.plot(e, response_curve_list[1], 'g', label='g')
plt.plot(e, response_curve_list[2], 'r', label='r')
plt.legend()
#plt.savefig('response_curve.png')
plt.show()


# In[ ]:


plt.clf()
e = list(range(len(response_curve_list[0])))
plt.subplot(221)
plt.plot(e, response_curve_list[0], 'b', label='b')
plt.legend()
plt.subplot(222)
plt.plot(e, response_curve_list[1], 'g', label='g')
plt.legend()
plt.subplot(223)
plt.plot(e, response_curve_list[2], 'r', label='r')
plt.legend()
#plt.savefig('response_curve_separate.png')
plt.show()
'''
