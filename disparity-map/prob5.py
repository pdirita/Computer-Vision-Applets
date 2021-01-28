## Default modules imported. Import more if you need to.

import numpy as np
import scipy.ndimage.interpolation as spi

#########################################
### Hamming distance computation
### You can call the function hamdist with two
### uint32 bit arrays of the same size. It will
### return another array of the same size with
### the elmenet-wise hamming distance.
hd8bit = np.zeros((256,))
for i in range(256):
    v = i
    for k in range(8):
        hd8bit[i] = hd8bit[i] + v%2
        v=v//2


def hamdist(x,y):
    dist = np.zeros(x.shape)
    g = x^y
    for i in range(4):
        dist = dist + hd8bit[g%256]
        g = g // 256
    return dist
#########################################



## Fill out these functions yourself

# Compute a 5x5 census transform of the grayscale image img.
# Return a uint32 array of the same shape
def census(img):

    W = img.shape[1]
    H = img.shape[0]
    
    c = np.zeros([H,W],dtype=np.uint32)
    m = np.max(img)
    # loop over just the kernel like in homework 1
    for R in range(-2,3):
        for C in range(-2,3):
            ind = (R,C)
            if ind == (0,0):
                continue
            inten = np.where(img > spi.shift(img, ind, cval=m), 1., 0.)
            c = np.left_shift(np.bitwise_or(c, inten.astype(np.uint32)),1)
    return c
    

# Given left and right image and max disparity D_max, return a disparity map
# based on matching with  hamming distance of census codes. Use the census function
# you wrote above.
#
# d[x,y] implies that left[x,y] matched best with right[x-d[x,y],y]. Disparity values
# should be between 0 and D_max (both inclusive).
def smatch(left,right,dmax):
    cRight = census(right)
    cLeft = census(left)
    dmax = dmax + 1 #inclusive
    ham = []
    shifts = []
    for i in range(dmax):
        ind = (0, i)
        shifts.append(spi.shift(cRight, ind, cval=25))
    for i in range(dmax):
        ham.append(hamdist(cLeft, shifts[i]))

    return np.argmin(np.stack(ham,axis=2),axis=2)

    
########################## Support code below

from skimage.io import imread, imsave
from os.path import normpath as fn # Fixes window/linux path conventions
import matplotlib.cm as cm
import warnings
warnings.filterwarnings('ignore')


left = imread(fn('inputs/left.jpg'))
right = imread(fn('inputs/right.jpg'))

d = smatch(left,right,40)

# Map to color and save
dimg = cm.jet(np.minimum(1,np.float32(d.flatten())/20.))[:,0:3]
dimg = dimg.reshape([d.shape[0],d.shape[1],3])
imsave(fn('outputs/prob5.png'),dimg)
