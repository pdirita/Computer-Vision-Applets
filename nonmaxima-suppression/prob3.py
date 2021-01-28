## Default modules imported. Import more if you need to.

import numpy as np
from skimage.io import imread, imsave
from scipy.signal import convolve2d as conv2

# Different thresholds to try
T0 = 1.0
T1 = 1.25
T2 = 1.5


########### Fill in the functions below

# Return magnitude, theta of gradients of X
def grads(X):
    #placeholder
    H = np.zeros(X.shape,dtype=np.float32)
    theta = np.zeros(X.shape,dtype=np.float32)

    dxl = [
        [1, 0, -1],
        [2, 0, -2],
        [1, 0, -1]
    ]
    dyl = [
        [1, 2, 1],
        [0, 0, 0],
        [-1, -2, -1]
    ]
    Dx = np.asarray(dxl)
    Dy = np.asarray(dyl)
    
    Gx = conv2(X, Dx, "same", "symm")
    Gy = conv2(X, Dy, "same", "symm")

    H = (Gx ** 2 + Gy ** 2) ** .5
    theta = np.arctan2(Gy,Gx)
    H = H.astype('float32')
    theta = theta.astype('float32')
    
    return H,theta

def nms(E, H, theta):
    orient = theta * 180. / np.pi
    orient[orient < 0] += 180

    E2 = np.zeros_like(E)

    deg0_0 = np.where(orient >= 0, 1, 0)
    deg0_1 = np.where(orient < 22.5, 1, 0)
    deg0_a = np.logical_and(deg0_0, deg0_1)
    deg0_2 = np.where(orient >= 175.5, 1, 0)
    deg0_3 = np.where(orient <= 180, 1, 0)
    deg0_b = np.logical_and(deg0_2, deg0_3)
    deg0 = np.logical_and(np.logical_or(deg0_a, deg0_b), E != 0)
    deg45_0 = np.where(22.5 <= orient, 1, 0)
    deg45_1 = np.where(orient < 67.5, 1, 0)
    deg45 = np.logical_and(np.logical_and(deg45_0, deg45_1), E != 0)
    deg90_0 = np.where(67.5 <= orient, 1, 0)
    deg90_1 = np.where(orient < 112.5, 1, 0)
    deg90 = np.logical_and(np.logical_and(deg90_0, deg90_1), E != 0)
    deg135_0 = np.where(112.5 <= orient, 1, 0)
    deg135_1 = np.where(orient < 157.5, 1, 0)
    deg135 = np.logical_and(np.logical_and(deg135_0, deg135_1), E != 0)

    X1 = deg0 * np.roll(H, (1,0))
    X2 = deg0 * np.roll(H, (-1, 0))
    E2 = np.logical_or(E2, np.logical_and(np.where(deg0 * H > X1, 1, 0), np.where(deg0 * H > X2, 1, 0)))

    X1 = deg45 * np.roll(H, (1, 1))
    X2 = deg45 * np.roll(H, (-1, -1))
    E2 = np.logical_or(E2, np.logical_and(np.where(deg45 * H > X1, 1, 0), np.where(deg45 * H > X2, 1, 0)))

    X1 = deg90 * np.roll(H, (0, 1))
    X2 = deg90 * np.roll(H, (0, -1))
    E2 = np.logical_or(E2, np.logical_and(np.where(deg90 * H > X1, 1, 0), np.where(deg90 * H > X2, 1, 0)))

    X1 = deg135 * np.roll(H, (1, -1))
    X2 = deg135 * np.roll(H, (-1, 1))
    E2 = np.logical_or(E2, np.logical_and(np.where(deg135 * H > X1, 1, 0), np.where(deg135 * H > X2, 1, 0)))

    return np.float32(E2)

########################## Support code below

from os.path import normpath as fn # Fixes window/linux path conventions
import warnings
warnings.filterwarnings('ignore')

img = np.float32(imread(fn('inputs/p3_inp.jpg')))/255.

H,theta = grads(img)

imsave(fn('outputs/prob3_a.jpg'),H/np.max(H[:]))

## Part b

E0 = np.float32(H > T0)
E1 = np.float32(H > T1)
E2 = np.float32(H > T2)

imsave(fn('outputs/prob3_b_0.jpg'),E0)
imsave(fn('outputs/prob3_b_1.jpg'),E1)
imsave(fn('outputs/prob3_b_2.jpg'),E2)

E0n = nms(E0,H,theta)
E1n = nms(E1,H,theta)
E2n = nms(E2,H,theta)

imsave(fn('outputs/prob3_b_nms0.jpg'),E0n)
imsave(fn('outputs/prob3_b_nms1.jpg'),E1n)
imsave(fn('outputs/prob3_b_nms2.jpg'),E2n)
