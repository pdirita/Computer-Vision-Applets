## Default modules imported. Import more if you need to.

import numpy as np
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

# Inputs:
#    nrm: HxWx3. Unit normal vectors at each location. All zeros at mask == 0
#    mask: A 0-1 mask of size HxW, showing where observed data is 'valid'.
#    lmda: Scalar value of lambda to be used for regularizer weight as in slides.
#
# Returns depth map Z of size HxW.
#
# Be careful about division by 0.
#
# Implement in Fourier Domain / Frankot-Chellappa
def ntod(nrm, mask, lmda):
    # manually pad out kernel matrices to square to make things easier
    # ensures kernpad won't give an index error
    fx = kernpad(np.asarray([[.5,0.,-.5], [0.,0.,0.], [0.,0.,0.]]), mask.shape)
    fy = kernpad(np.asarray([[-.5,0.,0.], [0.,0.,0.], [.5,0.,0.]]), mask.shape)
    fr = kernpad(np.asarray([[-1/9., -1/9., -1/9.],[-1/9., 8/9., -1/9.],[-1/9., -1/9., -1/9.]]), mask.shape)
    
    nx = nrm[:,:,0]
    ny = nrm[:,:,1]
    nz = nrm[:,:,2]
    
    # Ensure gx, gy are consistent with valid mask area
    gx = np.where(mask!=0, -1 * nx / nz, 0)
    gy = np.where(mask!=0, -1 * ny / nz, 0)

    # Get fourier transforms of all arrays
    Fx = np.fft.fft2(fx)
    Fy = np.fft.fft2(fy)
    Fr = np.fft.fft2(fr)
    Gx = np.fft.fft2(gx)
    Gy = np.fft.fft2(gy)

    # Compute F and set [0,0] to 0 explicitly
    F = ( np.conj(Fx) * Gx + np.conj(Fy) * Gy) / (np.absolute(Fx)**2 + np.absolute(Fy)**2 + lmda * np.absolute(Fr)**2)
    F[0,0] = 0.

    # Images are real-valued, so we only care about real part
    return np.real(np.fft.ifft2(F))


########################## Support code below

from os.path import normpath as fn # Fixes window/linux path conventions
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D


#### Main function

nrm = imread(fn('inputs/phstereo/true_normals.png'))

# Un-comment  next line to read your output instead
# nrm = imread(fn('outputs/prob3_nrm.png'))


mask = np.float32(imread(fn('inputs/phstereo/mask.png')) > 0)

nrm = np.float32(nrm/255.0)
nrm = nrm*2.0-1.0
nrm = nrm * mask[:,:,np.newaxis]


# Main Call
Z = ntod(nrm,mask,1e-6)


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
