## Default modules imported. Import more if you need to.

import numpy as np


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

def census(img):
    W = img.shape[1]
    H = img.shape[0]

    c = np.zeros([H, W], dtype=np.uint32)
    m = np.max(img)
    # loop over just the kernel like in homework 1
    for R in range(-2, 3):
        for C in range(-2, 3):
            ind = (R, C)
            if ind == (0, 0):
                continue
            inten = np.where(img > shift(img, ind, const=m), 1., 0.)
            c = np.left_shift(np.bitwise_or(c, inten.astype(np.uint32)), 1)
    return c

def shift(img, ind, const=0):
    x = ind[0]
    y = ind[1]
    xshift = np.roll(img, x, axis=0)
    if x >= 0: xshift[0:x,:] = const
    else: xshift[x:,:] = const
    yshift = np.roll(xshift, y, axis=1)
    if y >= 0: yshift[:,0:y] = const
    else: yshift[:,y:] = const
    return yshift

def buildcv(left,right,dmax):
    leftCen, rightCen = census(left), census(right)
    dmax = dmax + 1
    hams = [hamdist(leftCen, shift(rightCen, (0, x), 24)) for x in range(dmax)]
    return np.stack(hams, axis=2)


# Implement the forward-backward viterbi method to smooth
# only along horizontal lines. Assume smoothness cost of
# 0 if disparities equal, P1 if disparity difference <= 1, P2 otherwise.
#
# Function takes in cost volume cv, and values of P1 and P2
# Return the disparity map
def viterbilr(cv,P1,P2):
    H = cv.shape[0]
    W = cv.shape[1]
    D = cv.shape[2]

    d = np.zeros((H,W), dtype=np.int32)
    c = np.zeros_like(cv)
    z = c.astype(np.int32)
    c[:,0,:] = cv[:,0,:] # this is c^bar

    S = np.ones((D,D)) * P2 - P2 * np.eye(D) - (P2-P1)*(shift(np.eye(D), (0,1),0)) - (P2-P1)*(shift(np.eye(D), (0,-1), 0))
    Sx, Sy = S.shape
    S = np.repeat(S.reshape(1,Sx,Sy), H, axis=0)

    #forward
    for x in range(0,W-1):
        z[:,x+1,:] = np.argmin(S[:,:,:] + c[:,x:x+1,:], axis=2).astype(np.int32)
        c[:,x+1,:] = cv[:,x+1,:] + np.min(S[:,:,:] + c[:,x:x+1,:], axis=2)

    #backward
    d[:, W-1] = np.argmin(c[:,W-1,:], axis=1)
    for x in range(W-1,0,-1):
        dx = [i for i in range(H)]
        dy = [x for i in range(H)]
        dz = d[:, x]
        d[:,x-1] = z[dx,dy,dz]
    return d
    
    
########################## Support code below

from skimage.io import imread, imsave
from os.path import normpath as fn # Fixes window/linux path conventions
import matplotlib.cm as cm
import warnings
warnings.filterwarnings('ignore')


left = np.float32(imread(fn('inputs/left.jpg')))/255.
right = np.float32(imread(fn('inputs/right.jpg')))/255.

left_g = np.mean(left,axis=2)
right_g = np.mean(right,axis=2)
                   
cv = buildcv(left_g,right_g,50)
d = viterbilr(cv,0.5,16)

# Map to color and save
dimg = cm.jet(np.minimum(1,np.float32(d.flatten())/50.))[:,0:3]
dimg = dimg.reshape([d.shape[0],d.shape[1],3])
imsave(fn('outputs/prob3a.jpg'),dimg)
