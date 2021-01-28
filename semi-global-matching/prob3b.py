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

# computes c~ (c^tilde) at index [:,i,:] by doing c^bar[:,i,:] - min(c^bar[:,i,:)
def computeCT(ct,cb,i):
    cMin = np.min(cb[:, i, :], axis=1, keepdims=True)
    ct[:,i,:] = cb[:,i,:] - cMin

# modified code from problem 3a
# efficiently computes augmented cost volume on an image given cv, P1, P2
def viterbiSGM(cv,P1,P2):
    H = cv.shape[0]
    W = cv.shape[1]
    D = cv.shape[2]

    c = np.zeros_like(cv) # this is c^bar
    ct = np.zeros_like(cv) # this is c^tilde

    c[:,0,:] = cv[:,0,:]
    computeCT(ct,cv,0)

    for x in range(0,W-1):
        valArray = np.zeros((H,D,4))
        ctPoint = ct[:,x,:]
        valArray[:,:,0] = P2
        valArray[:,:,1] = P1 + shift(ctPoint, (0,1), 24) # P1 + ct[x-1,d+1], so we have to shift 1 horizontally
        valArray[:,:,2] = P1 + shift(ctPoint, (0,-1), 24) # P1 + ct[x-1,d-1], so we have to shift -1 horizontally
        valArray[:,:,3] = ctPoint

        c[:,x+1,:] = cv[:,x+1,:] + np.min(valArray, axis=2) # c^bar = c[x,d] + min(valArray)
        computeCT(ct,c,x+1)
    return c



# Do SGM. First compute the augmented / smoothed cost volumes along 4
# directions (LR, RL, UD, DU), and then compute the disparity map as
# the argmin of the sum of these cost volumes. 
def SGM(cv,P1,P2):
    #cvLR is just cv
    cvRL = np.fliplr(cv)
    cvUD = np.swapaxes(cv, 0, 1)
    cvDU = np.swapaxes(np.flipud(cv), 0, 1)

    LR = viterbiSGM(cv,P1,P2)
    RL = np.fliplr(viterbiSGM(cvRL,P1,P2))
    UD = np.swapaxes(viterbiSGM(cvUD,P1,P2), 0, 1)
    DU = np.flipud(np.swapaxes(viterbiSGM(cvDU, P1, P2), 0, 1))

    cvSum = LR + RL + UD + DU

    return np.argmin(cvSum,axis=2)

    
    
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
d = SGM(cv,0.5,16)

# Map to color and save
dimg = cm.jet(np.minimum(1,np.float32(d.flatten())/50.))[:,0:3]
dimg = dimg.reshape([d.shape[0],d.shape[1],3])
imsave(fn('outputs/prob3b.jpg'),dimg)
