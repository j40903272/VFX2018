
# coding: utf-8

# In[1]:


import cv2
import numpy as np
import scipy.misc as spm
from subprocess import call


# In[2]:


#To downsample a grayscale image by a factor of 2
def ImageShrink2(img):
    return cv2.resize(img, (0,0), fx = 0.5, fy = 0.5)


# In[3]:


#To compute two bitmaps for a grayscale image
#tb: the median-thresholding bitmap
#eb: the edge bitmap
def ComputeBitmaps(img):
    
    med_thresh = np.median(img.flatten('F'), axis = 0)
    #counts = np.bincount(img.flatten('F'))
    #med_thresh = np.argmax(counts)
    
    ret, tb = cv2.threshold(img, med_thresh, 255, type = 0)
    
    #eb = cv2.inRange(np.array(img), med_thresh - 4, med_thresh + 4)
    eb = np.array(img)
    eb[np.where(eb <= med_thresh - 4)] = 0
    eb[np.where(eb > med_thresh + 4)] = 255
    print (type(tb), type(eb))
    return tb, eb


# In[4]:


def BitmapXOR(bm1, bm2):
    bm_ret = np.bitwise_xor(bm1, bm2)
    return bm_ret


# In[5]:


def BitmapAND(bm1, bm2):
    bm_ret = np.bitwise_and(bm1, bm2)
    return bm_ret


# In[6]:


def BitmapTotal(bm):
    return np.count_nonzero(bm)


# In[7]:


#Main function of MTB algorithm
def GetExpShift(im1, im2, shift_bits):
    
    cur_shift_x = cur_shift_y = 0
    shift_ret_x = shift_ret_y = 0
    if shift_bits > 0:
        sml_im1 = ImageShrink2(im1)
        sml_im2 = ImageShrink2(im2)
        cur_shift_x, cur_shift_y = GetExpShift(sml_im1, sml_im2, shift_bits-1)
        cur_shift_x = cur_shift_x * 2
        cur_shift_y = cur_shift_y * 2
    else:
        cur_shift_x = cur_shift_y = 0
        
    tb1, eb1 = ComputeBitmaps(im1)
    tb2, eb2 = ComputeBitmaps(im2)
    min_err = im1.shape[0] * im1.shape[1]
    
    for i in range(-1, 2):
        for j in range(-1, 2):
            xs = cur_shift_x + i
            ys = cur_shift_y + j
            shifted_tb2 = ImageShift(tb2, xs, ys)
            shifted_eb2 = ImageShift(eb2, xs, ys)
            diff_b = BitmapXOR(tb1, shifted_tb2)
            diff_b = BitmapAND(diff_b, eb1)
            diff_b = BitmapAND(diff_b, shifted_tb2)
            
            err = BitmapTotal(diff_b)
            if err < min_err:
                shift_ret_x = xs
                shift_ret_y = ys
                min_err = err
                
    return shift_ret_x, shift_ret_y


# In[8]:


#To shift a real color image
def ImageShift(img, x, y):
    num_rows, num_cols = img.shape[:2]
    translation_matrix = np.float32([ [1, 0, x], [0, 1, y] ])
    return cv2.warpAffine(img, translation_matrix, (num_cols, num_rows))


# In[9]:


#To align an array of color images
def AlignImages(img_list):
    if len(img_list) <= 1:
        return img_list
    
    #get grayscale images
    gray_list = [None] * len(img_list)
    for i in range(len(img_list)):
        gray_list[i] = cv2.cvtColor(img_list[i], cv2.COLOR_BGR2GRAY)
        
    #align with the first img
    for i in range(1, len(img_list)):
        shift_x, shift_y = GetExpShift(gray_list[0], gray_list[i], 4)
        img_list[i] = ImageShift(img_list[i], shift_x, shift_y)
    return img_list


# In[10]:


## # Loading exposure images into a list
img_dir = './imageset8/'
save_dir = img_dir + 'bitmaps/'
call(['mkdir', save_dir])
img_fn = [img_dir+'0'+str(i+112)+".JPG" for i in range(9)]
img_list = [spm.imread(fn) for fn in img_fn]

#align_list = AlignImages(img_list)


img_list[3] = cv2.cvtColor(img_list[3], cv2.COLOR_BGR2GRAY)
tb, eb = ComputeBitmaps(img_list[3])
cv2.imwrite(save_dir+'tb.jpg', tb)
cv2.imwrite(save_dir+'eb.jpg', eb)
ret = BitmapXOR(tb, eb)
cv2.imwrite(save_dir+'mask.jpg', ret)

