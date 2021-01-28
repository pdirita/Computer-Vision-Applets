## Default modules imported. Import more if you need to.

import numpy as np
from skimage.io import imread, imsave
from skimage.util import view_as_windows

## Fill out these functions yourself
u = [
        [1,1,1,1],
        [-1,1,-1,1],
        [-1,-1,1,1],
        [1,-1,-1,1]
    ]
unitary = .5 * np.asarray(u)
uInv = np.linalg.inv(unitary)

def im2wv(img,nLev):
    if nLev==0:
        return [img]

    window = view_as_windows(img, (2,2), 2)
    wx = window.shape[-1]
    wy = window.shape[-2]
    window = window.reshape(-1, wx, wy)
    a = [[],[],[],[]]
    for w in window:
        wp = w.reshape(4,1,order="F")
        l = np.matmul(unitary,wp)
        #print(l)
        a[0].append(l[0][0])
        a[1].append(l[1][0])
        a[2].append(l[2][0])
        a[3].append(l[3][0])
    for i in range(4):
        a[i] = np.array(a[i]).reshape(img.shape[0]//2, img.shape[1]//2)

    return [a[1:]] + im2wv(a[0],nLev-1)

    


def wv2im(pyr):
    img = pyr[-1]
    H = pyr[:-1]
    if len(H)==0:
        return img
    Hfinal = H[-1]
    h1,h2,h3 = Hfinal
    window = np.array([img.reshape(-1),h1.reshape(-1),h2.reshape(-1),h3.reshape(-1)])
    transform = np.matmul(uInv, window)

    X = [x.reshape((2,2), order="F") for x in transform.T]
    X = np.array(X).reshape((*img.shape,2,2))
    concat1 = [np.concatenate(x, axis=1) for x in X]
    X = np.concatenate(concat1, axis=0)
    H[-1] = X
    return wv2im(H)



########################## Support code below

from os.path import normpath as fn # Fixes window/linux path conventions
import warnings
warnings.filterwarnings('ignore')


# Utility functions to clip intensities b/w 0 and 1
# Otherwise imsave complains
def clip(im):
    return np.maximum(0.,np.minimum(1.,im))


# Visualize pyramid like in slides
def vis(pyr, lev=0):
    if len(pyr) == 1:
        return pyr[0]/(2**lev)

    sz=pyr[0][0].shape
    sz1 = [sz[0]*2,sz[1]*2]
    img = np.zeros(sz1,dtype=np.float32)

    img[0:sz[0],0:sz[1]] = vis(pyr[1:],lev+1)

    # Just scale / shift gradient images for visualization
    img[sz[0]:,0:sz[1]] = pyr[0][0]*(2**(1-lev))+0.5
    img[0:sz[0],sz[1]:] = pyr[0][1]*(2**(1-lev))+0.5
    img[sz[0]:,sz[1]:] = pyr[0][2]*(2**(1-lev))+0.5

    return img



############# Main Program


img = np.float32(imread(fn('inputs/p6_inp.jpg')))/255.

# Visualize pyramids
pyr = im2wv(img,1)
imsave(fn('outputs/prob6a_1.jpg'),clip(vis(pyr)))

pyr = im2wv(img,2)
imsave(fn('outputs/prob6a_2.jpg'),clip(vis(pyr)))

pyr = im2wv(img,3)
imsave(fn('outputs/prob6a_3.jpg'),clip(vis(pyr)))

# Inverse transform to reconstruct image
im = clip(wv2im(pyr))
imsave(fn('outputs/prob6b.jpg'),im)

# Zero out some levels and reconstruct
for i in range(len(pyr)-1):

    for j in range(3):
        pyr[i][j][...] = 0.

    im = clip(wv2im(pyr))
    imsave(fn('outputs/prob6b_%d.jpg' % i),im)
