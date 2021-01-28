## Default modules imported. Import more if you need to.

import numpy as np
import math
from skimage.io import imread, imsave

# Fill this out
# X is input color image
# K is the support of the filter (2K+1)x(2K+1)
# sgm_s is std of spatial gaussian
# sgm_i is std of intensity gaussian
def bfilt(X,K,sgm_s,sgm_i):
    # n1 ranges from n2-[K,K] to n2 + [K,K]
    Y = np.zeros_like(X)
    Xpad = np.pad(X, ((K, K), (K, K), (0, 0)), 'constant').astype(np.float32)
    rows,cols,channels = X.shape
    wtNorm = np.zeros((rows,cols))
    for r in range(-K,K+1):
        for c in range(-K,K+1):
            window = Xpad[K+r:K+r+rows, K+c:K+c+cols,:]
            b1 = np.exp(-(r**2+c**2)/(2*sgm_s**2))
            norm = np.linalg.norm(X-window, axis=2)
            b2 = np.exp(-norm / (2*sgm_i**2))
            wt = b1 * b2
            for i in range(3):
                Y[:,:,i] += wt * window[:,:,i]
            
            wtNorm += wt
    
    for i in range(3):
        Y[:,:,i] /= wtNorm

    return Y


########################## Support code below

def clip(im):
    return np.maximum(0.,np.minimum(1.,im))

from os.path import normpath as fn # Fixes window/linux path conventions
import warnings
warnings.filterwarnings('ignore')

img1 = np.float32(imread(fn('inputs/p4_nz1.jpg')))/255.
img2 = np.float32(imread(fn('inputs/p4_nz2.jpg')))/255.

K=9

print("Creating outputs/prob4_1_a.jpg")
im1A = bfilt(img1,K,2,0.5)
imsave(fn('outputs/prob4_1_a.jpg'),clip(im1A))


print("Creating outputs/prob4_1_b.jpg")
im1B = bfilt(img1,K,4,0.25)
imsave(fn('outputs/prob4_1_b.jpg'),clip(im1B))

print("Creating outputs/prob4_1_c.jpg")
im1C = bfilt(img1,K,16,0.125)
imsave(fn('outputs/prob4_1_c.jpg'),clip(im1C))

# Repeated application
print("Creating outputs/prob4_1_rep.jpg")
im1D = bfilt(img1,K,2,0.125)
for i in range(8):
    im1D = bfilt(im1D,K,2,0.125)
imsave(fn('outputs/prob4_1_rep.jpg'),clip(im1D))

# Try this on image with more noise    
print("Creating outputs/prob4_2_rep.jpg")
im2D = bfilt(img2,2,8,0.125)
for i in range(16):
    im2D = bfilt(im2D,K,2,0.125)
imsave(fn('outputs/prob4_2_rep.jpg'),clip(im2D))
