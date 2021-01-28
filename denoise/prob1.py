## Default modules imported. Import more if you need to.

import numpy as np
from skimage.io import imread, imsave

## Fill out these functions yourself

# Copy from Pset1/Prob6 
def im2wv(img,nLev):
    if nLev == 0:
        return [img]
    hA = (img[0::2,:] + img[1::2,:])/2
    hB = (-img[0::2,:] + img[1::2,:])/2
    L = hA[:,0::2] + hA[:,1::2]
    h1 = hB[:,0::2] + hB[:,1::2]
    h2 = -hA[:,0::2] + hA[:,1::2]
    h3 = -hB[:,0::2] + hB[:,1::2]
    
    return [[h1,h2,h3]] + im2wv(L, nLev-1)


# Copy from Pset1/Prob6 
def wv2im(pyr):
    while len(pyr) > 1:
        L0 = pyr[-1]
        Hs = pyr[-2]
        H1 = Hs[0]
        H2 = Hs[1]
        H3 = Hs[2]

        sz = L0.shape
        L = np.zeros([sz[0]*2,sz[1]*2], dtype=np.float32)

        L[::2,::2] = (L0-H1-H2+H3)/2.
        L[1::2,::2] = (L0+H1-H2-H3)/2.
        L[::2,1::2] = (L0-H1+H2-H3)/2.
        L[1::2,1::2] = (L0+H1+H2+H3)/2.
        pyr = pyr[:-2] + [L]
    return pyr[0]


# Fill this out
# You'll get a numpy array/image of coefficients y
# Return corresponding coefficients x (same shape/size)
# that minimizes (x - y)^2 + lmbda * abs(x)
def denoise_coeff(y,lmbda):
    # x = .5 * (2 * y - sign(x)*lmbda)
    x1 = .5 * (2 * y - lmbda)
    x2 = .5 * (2 * y + lmbda)
    # plug in both possible values for x
    x1a = (x1-y)**2+lmbda*np.abs(x1)
    x2a = (x2-y)**2+lmbda*np.abs(x2)
    # return array where x is the value that minimizes the objective function at each point
    return np.where(x1a<x2a, x1, x2)

########################## Support code below

from os.path import normpath as fn # Fixes window/linux path conventions
import warnings
warnings.filterwarnings('ignore')


# Utility functions to clip intensities b/w 0 and 1
# Otherwise imsave complains
def clip(im):
    return np.maximum(0.,np.minimum(1.,im))



############# Main Program

lmain = .25

img = np.float32(imread(fn('inputs/p1.png')))/255.

pyr = im2wv(img,4)
for i in range(len(pyr)-1):
    for j in range(2):
        pyr[i][j] = denoise_coeff(pyr[i][j],lmain/(2**i))
    pyr[i][2] = denoise_coeff(pyr[i][2],np.sqrt(2)*lmain/(2**i))
    
im = wv2im(pyr)        
imsave(fn('outputs/prob1.png'),clip(im))
