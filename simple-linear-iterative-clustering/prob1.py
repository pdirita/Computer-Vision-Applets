## Default modules imported. Import more if you need to.
### Problem designed by Abby Stylianou

import numpy as np
from scipy.signal import convolve2d as conv2

def get_cluster_centers(im,num_clusters):
    # Implement a method that returns an initial grid of cluster centers. You should first
    # create a grid of evenly spaced centers (hint: np.meshgrid), and then use the method
    # discussed in class to make sure no centers are initialized on a sharp boundary.
    # You can use the get_gradients method from the support code below.

    #inds = np.indices((im.shape[0],im.shape[1]))
    S = np.sqrt(im.shape[0] * im.shape[1] / num_clusters)
    xGrid, yGrid = np.meshgrid(np.arange(0,im.shape[1],S).astype(np.int32), np.arange(0,im.shape[0],S).astype(np.int32))
    grads = get_gradients(im)
    xGrid_r = xGrid.ravel()
    yGrid_r = yGrid.ravel()
    coords = []
    rollGrad = []
    for i in range(-1,2):
        for j in range(-1,2):
            coords.append((i,j))
            roll = np.roll(grads,(i,j,0))
            rollGrad.append(roll)
    gradStack = np.stack(rollGrad, axis=2)
    mins = np.argmin(gradStack, axis=2)
    cluster_centers = np.stack([yGrid_r, xGrid_r], axis=1)
    minCoords = mins[yGrid_r, xGrid_r]
    cluster_centers -= np.array([coords[k] for k in minCoords])
    return cluster_centers


def slic(im,num_clusters,cluster_centers):
    # Implement the slic function such that all pixels assigned to a label
    # should be close to each other in squared distance of augmented vectors.
    # You can weight the color and spatial components of the augmented vectors
    # differently. To do this, experiment with different values of spatial_weight.
    h,w,c = im.shape
    spatial_weight = 5
    clusters = np.zeros((h,w))
    S = int(np.sqrt(h*w/num_clusters))
    centerX = cluster_centers[:, 0].ravel()
    centerY = cluster_centers[:, 1].ravel()
    xGrid, yGrid = np.meshgrid(np.arange(0, h, 1).astype(np.int32),np.arange(0, w, 1).astype(np.int32))
    spatialX = spatial_weight * xGrid.reshape((h,w,1))
    spatialY = spatial_weight * yGrid.reshape((h,w,1))
    spatialXY = np.append(spatialY,spatialX, axis=2)
    aug = np.append(im, spatialXY, axis=2)
    augpts = aug[centerX,centerY]
    labels = np.arange(0, num_clusters)
    minErr = float("inf")
    i = 0
    while True:
        currMin = np.inf + np.ones((h,w))
        err = 0
        for ck in range(num_clusters):
            ya=0
            yb=h
            xa=0
            xb=w
            ypts = int(augpts[ck, 3] / spatial_weight)
            xpts = int(augpts[ck, 4] / spatial_weight)
            if (ypts - S) > ya: ya = (ypts-S)
            if (ypts+S+1) < yb: yb = (ypts+S+1)
            if (xpts-S) > xa: xa = (xpts-S)
            if (xpts+S+1) < xb: xb = (xpts+S+1)

            part = aug[ya:yb, xa:xb, :]
            distPart = currMin[ya:yb,xa:xb]
            #ptTest=augpts[ck,:]
            pt = augpts[ck,:].reshape((1,1,5))
            d = np.sum(np.square(part - pt), axis=2)
            validRgn = np.where(d >= distPart,0,1)
            clusters[ya:yb,xa:xb] *= 1-validRgn
            label = labels[ck]
            clusters[ya:yb,xa:xb] += validRgn * label

            currMin[ya:yb,xa:xb] = np.minimum(distPart, d)
        for ck in range(num_clusters):
            validRgn = np.where(clusters != labels[ck], 0, 1)
            validCount = np.sum(validRgn)
            vh,vw = validRgn.shape
            if validCount == 0:
                print("No change in clusters on this iteration")
            else:
                mean = augpts[ck,:]
                errMsk = validRgn.reshape((vh,vw,1))
                errImg = aug * errMsk
                reduce = np.sum(errImg, axis=0)
                meanP = np.sum(reduce,axis=0) / validCount
                diff = np.abs(meanP-mean)
                err += np.sum(diff)
                augpts[ck,:] = meanP
        # minErr = min(err, minErr)
        # if err < 1:
        #     return clusters
        # else:
        #     print(minErr)
        #     print(i)
        #     i += 1
        #     if i % 1000 == 0:
        #         print(i)

        if err < 1:
            return clusters
        else:
            print(err)

    return clusters

########################## Support code below

from skimage.io import imread, imsave
from os.path import normpath as fn # Fixes window/linux path conventions
import matplotlib.cm as cm
import warnings
warnings.filterwarnings('ignore')

# Use get_gradients (code from pset1) to get the gradient of your image when initializing your cluster centers.
def get_gradients(im):
    if len(im.shape) > 2:
        im = np.mean(im,axis=2)
    df = np.float32([[1,0,-1]])
    sf = np.float32([[1,2,1]])
    gx = conv2(im,sf.T,'same','symm')
    gx = conv2(gx,df,'same','symm')
    gy = conv2(im,sf,'same','symm')
    gy = conv2(gy,df.T,'same','symm')
    return np.sqrt(gx*gx+gy*gy)

# normalize_im normalizes our output to be between 0 and 1
def normalize_im(im):
    im += np.abs(np.min(im))
    im /= np.max(im)
    return im

# create an output image of our cluster centers
def create_centers_im(im,centers):
    for center in centers:
        im[center[0]-2:center[0]+2,center[1]-2:center[1]+2] = [255.,0.,255.]
    return im

im = np.float32(imread(fn('inputs/lion.jpg')))

num_clusters = [25,49,64,81,100]
for num_clusters in num_clusters:
    cluster_centers = get_cluster_centers(im,num_clusters)
    imsave(fn('outputs/prob1a_' + str(num_clusters)+'_centers.jpg'),normalize_im(create_centers_im(im.copy(),cluster_centers)))
    out_im = slic(im,num_clusters,cluster_centers)

    Lr = np.random.permutation(num_clusters)
    out_im = Lr[np.int32(out_im)]
    dimg = cm.jet(np.minimum(1,np.float32(out_im.flatten())/float(num_clusters)))[:,0:3]
    dimg = dimg.reshape([out_im.shape[0],out_im.shape[1],3])
    imsave(fn('outputs/prob1b_'+str(num_clusters)+'.jpg'),normalize_im(dimg))
