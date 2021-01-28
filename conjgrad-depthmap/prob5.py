## Default modules imported. Import more if you need to.

import numpy as np
from scipy.signal import convolve2d as conv2
from skimage.io import imread, imsave


## Fill out these functions yourself

# Kernpad function from pSet1
def kernpad(K,size):
    Ko = np.zeros(size,dtype=np.float32)
    ks = K.shape
    hk = (ks[0]-1)//2
    wk = (ks[1]-1)//2
    Ko[:(hk+1),:(wk+1)] = K[hk:,wk:]
    Ko[:(hk+1),-wk:] = K[hk:,:wk]
    Ko[-hk:,:(wk+1)] = K[:hk,wk:]
    Ko[-hk:,-wk:] = K[:hk,:wk]
    return Ko

# Multiplication function for conjugate gradient
# vector-vector inner products will just be element-wise multiply followed by sum over all pixels
def mult(A,B):
    return np.sum(np.inner(A,B.T))



# Inputs:
#    nrm: HxWx3. Unit normal vectors at each location. All zeros at mask == 0
#    mask: A 0-1 mask of size HxW, showing where observed data is 'valid'.
#    lmda: Scalar value of lambda to be used for regularizer weight as in slides.
#
# Returns depth map Z of size HxW.
#
# Be careful about division by 0.
#
# Implement using conjugate gradient, with a weight = 0 for mask == 0, and proportional
# to n_z^2 elsewhere. See slides.

def ntod(nrm, mask, lmda):
    # manually pad out kernel matrices to square to make things easier
    fx = np.asarray([[.5,0.,-.5], [0.,0.,0.], [0.,0.,0.]])
    fy = np.asarray([[-.5,0.,0.], [0.,0.,0.], [.5,0.,0.]])
    fr = np.asarray([[-1/9., -1/9., -1/9.],[-1/9., 8/9., -1/9.],[-1/9., -1/9., -1/9.]])

    fxBar = np.flip(fx)
    fyBar = np.flip(fy)
    frBar = np.flip(fr)

    nx = nrm[:,:,0]
    ny = nrm[:,:,1]
    nz = nrm[:,:,2]

    # Ensure gx, gy, w are consistent with valid mask area
    gx = np.where(mask!=0, -1 * nx / nz, 0)
    gy = np.where(mask!=0, -1 * ny / nz, 0)
    w = np.where(mask != 0, nz ** 2, 0)

    # Init. parameters for conjugate gradient
    Z = np.zeros_like(mask) 
    numIters = 100
    k = 0
    b = conv2((gx * w), fxBar, "same") + conv2((gy * w), fyBar, "same")
    r = b
    p = r

    while k < numIters:
        Qp = conv2((conv2(p, fx, "same") * w), fxBar, "same") + conv2((conv2(p, fy, "same") * w), fyBar, "same") + lmda * conv2((conv2(p, fr, "same")), frBar, "same")
        alpha = mult(r.T, r) / mult(p.T, Qp)
        Z = Z + (alpha * p)
        rOld = r
        r = r - (alpha * Qp)
        beta = mult(r.T, r) / mult(rOld.T, rOld)
        p = r + (beta * p)
        k += 1

    return Z


########################## Support code below

from os.path import normpath as fn # Fixes window/linux path conventions
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D


#### Main function

# nrm = imread(fn('inputs/phstereo/true_normals.png'))

# Un-comment  next line to read your output instead
nrm = imread(fn('outputs/prob3_nrm.png'))


mask = np.float32(imread(fn('inputs/phstereo/mask.png')) > 0)

nrm = np.float32(nrm/255.0)
nrm = nrm*2.0-1.0
nrm = nrm * mask[:,:,np.newaxis]


# Main Call
Z = ntod(nrm,mask,1e-7)


# Plot 3D shape

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x,y = np.meshgrid(np.float32(range(nrm.shape[1])),np.float32(range(nrm.shape[0])))
x = x - np.mean(x[:])
y = y - np.mean(y[:])

Zmsk = Z.copy()
Zmsk[mask == 0] = np.nan
Zmsk = Zmsk - np.nanmedian(Zmsk[:])

lim = 100
ax.plot_surface(x,-y,Zmsk, \
                linewidth=0,cmap=cm.inferno,shade=True,\
                vmin=-lim,vmax=lim)

ax.set_xlim3d(-450,450)
ax.set_ylim3d(-450,450)
ax.set_zlim3d(-450,450)

plt.show()
