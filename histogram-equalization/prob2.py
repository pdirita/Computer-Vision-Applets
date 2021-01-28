## Default modules imported. Import more if you need to.

import numpy as np
from skimage.io import imread, imsave

# Fill this out
# X is input 8-bit grayscale image
# Return equalized image with intensities from 0-255
def histeq(X):
    # turn image into np array
    i = np.asarray(X)
    # flatten into 1 dimensional array
    flt = i.flatten()
    # get counts of unique values in array (get histogram)
    hist, counts = np.unique(flt, return_counts=True)
    # normalize frequencies of unique values
    countNorm = counts / i.size
    # get cumulative sum of normalized unique values and multiply by maximum value from original image
    cSum = np.cumsum(countNorm)
    maxMul = cSum * np.max(i)
    # round back to integer values
    iEq = np.round(maxMul)
    i2 = np.copy(i)
    
    for k in range(len(i)):
        for j in range(len(i[0])):
            i2[k][j] = iEq[np.where(hist == i[k][j])]
    return i2

    

########################## Support code below

from os.path import normpath as fn # Fixes window/linux path conventions
import warnings
warnings.filterwarnings('ignore')

img = imread(fn('inputs/p2_inp.jpg'))

out = histeq(img)

out = np.maximum(0,np.minimum(255,out))
out = np.uint8(out)
imsave(fn('outputs/prob2.jpg'),out)
