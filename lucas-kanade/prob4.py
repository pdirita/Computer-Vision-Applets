## Default modules imported. Import more if you need to.
import numpy as np
from scipy.signal import convolve2d as conv2

# Use these as the x and y derivative filters
fx = np.float32([[1,0,-1]]) * np.float32([[1,1,1]]).T / 6.
fy = fx.T


# Compute optical flow using the lucas kanade method
# Use the fx, fy, defined above as the derivative filters
# and compute derivatives on the average of the two frames.
# Also, consider (x',y') values in a WxW window.
# Return two image shape arrays u,v corresponding to the
# horizontal and vertical flow.
def lucaskanade(f1,f2,W):
    K = np.ones((W,W)) # kernel of ones in order to use convolution for quick sum
    avg = (f1 + f2) / 2.0
    T = f2 - f1
    stab = .00001

    X = conv2(avg, fx, mode="same")
    Y = conv2(avg, fy, mode="same")

    A11 = conv2(np.square(X), K, mode="same") + stab
    A22 = conv2(np.square(Y), K, mode="same") + stab
    A12 = conv2(X*Y, K, mode="same")
    A21 = conv2(Y*X, K, mode="same")

    # compute inverse of 2x2 matrix pointwise
    det = (A11 * A22) - (A12 * A21)
    A11_i = A22 / det
    A22_i = A11 / det
    A12_i = (-1 * A12) / det
    A21_i = (-1 * A21) / det


    B1 = -1 * conv2(X*T, K, mode="same")
    B2 = -1 * conv2(Y*T, K, mode="same")

    u = A11_i * B1 + A12_i * B2
    v = A21_i * B1 + A22_i * B2


    return u,v
    
########################## Support code below

from skimage.io import imread, imsave
from os.path import normpath as fn # Fixes window/linux path conventions
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


f1 = np.float32(imread(fn('inputs/frame10.jpg')))/255.
f2 = np.float32(imread(fn('inputs/frame11.jpg')))/255.

u,v = lucaskanade(f1,f2,11)


# Display quiver plot by downsampling
x = np.arange(u.shape[1])
y = np.arange(u.shape[0])
x,y = np.meshgrid(x,y[::-1])
plt.quiver(x[::8,::8],y[::8,::8],u[::8,::8],-v[::8,::8],pivot='mid')

plt.show()
