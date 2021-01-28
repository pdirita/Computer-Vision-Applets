### EDF --- An Autograd Engine for instruction
## (based on joint discussions with David McAllester)

import numpy as np
from numpy import newaxis
# Global list of different kinds of components
ops = []
params = []
values = []


# Global forward
def Forward():
    for c in ops: c.forward()

# Global backward    
def Backward(loss):
    for c in ops:
        c.grad = np.zeros_like(c.top)
    for c in params:
        c.grad = np.zeros_like(c.top)

    loss.grad = np.ones_like(loss.top)
    for c in ops[::-1]: c.backward() 

# SGD
def SGD(lr):
    for p in params:
        p.top = p.top - lr*p.grad


## Fill this out        
def init_momentum():
    for x in params: x.velocity = 0


## Fill this out
def momentum(lr,mom=0.9):
    for x in params:
        x.velocity = mom * x.velocity + lr * x.grad
        x.top -= x.velocity


###################### Different kinds of nodes

# Values (Inputs)
class Value:
    def __init__(self):
        values.append(self)

    def set(self,value):
        self.top = np.float32(value).copy()

# Parameters (Weights we want to learn)
class Param:
    def __init__(self):
        params.append(self)

    def set(self,value):
        self.top = np.float32(value).copy()


### Operations

# Add layer (x + y) where y is same shape as x or is 1-D
class add:
    def __init__(self,x,y):
        ops.append(self)
        self.x = x
        self.y = y

    def forward(self):
        self.top = self.x.top + self.y.top

    def backward(self):
        if self.x in ops or self.x in params:
            self.x.grad = self.x.grad + self.grad

        if self.y in ops or self.y in params:
            if len(self.y.top.shape) < len(self.grad.shape):
                ygrad = np.sum(self.grad,axis=tuple(range(len(self.grad.shape)-1)))
            else:
                ygrad= self.grad
            self.y.grad = self.y.grad + ygrad

# Matrix multiply (fully-connected layer)
class matmul:
    def __init__(self,x,y):
        ops.append(self)
        self.x = x
        self.y = y

    def forward(self):
        self.top = np.matmul(self.x.top,self.y.top)

    def backward(self):
        if self.x in ops or self.x in params:
            self.x.grad = self.x.grad + np.matmul(self.y.top,self.grad.T).T
        if self.y in ops or self.y in params:
            self.y.grad = self.y.grad + np.matmul(self.x.top.T,self.grad)


# Rectified Linear Unit Activation            
class RELU:
    def __init__(self,x):
        ops.append(self)
        self.x = x

    def forward(self):
        self.top = np.maximum(self.x.top,0)

    def backward(self):
        if self.x in ops or self.x in params:
            self.x.grad = self.x.grad + self.grad * (self.top > 0)


# Reduce to mean
class mean:
    def __init__(self,x):
        ops.append(self)
        self.x = x

    def forward(self):
        self.top = np.mean(self.x.top)

    def backward(self):
        if self.x in ops or self.x in params:
            self.x.grad = self.x.grad + self.grad*np.ones_like(self.x.top) / np.float32(np.prod(self.x.top.shape))



# Soft-max + Loss (per-row / training example)
class smaxloss:
    def __init__(self,pred,gt):
        ops.append(self)
        self.x = pred
        self.y = gt

    def forward(self):
        y = self.x.top
        y = y - np.amax(y,axis=1,keepdims=True)
        yE = np.exp(y)
        yS = np.sum(yE,axis=1,keepdims=True)
        y = y - np.log(yS); yE = yE / yS

        truey = np.int64(self.y.top)
        self.top = -y[range(len(truey)),truey]
        self.save = yE

    def backward(self):
        if self.x in ops or self.x in params:
            truey = np.int64(self.y.top)
            self.save[range(len(truey)),truey] = self.save[range(len(truey)),truey] - 1.
            self.x.grad = self.x.grad + np.expand_dims(self.grad,-1)*self.save
        # No backprop to labels!    

# Compute accuracy (for display, not differentiable)        
class accuracy:
    def __init__(self,pred,gt):
        ops.append(self)
        self.x = pred
        self.y = gt

    def forward(self):
        truey = np.int64(self.y.top)
        self.top = np.float32(np.argmax(self.x.top,axis=1)==truey)

    def backward(self):
        pass


# Downsample by 2    
class down2:
    def __init__(self,x):
        ops.append(self)
        self.x = x
        
    def forward(self):
        self.top = self.x.top[:,::2,::2,:]

    def backward(self):
        if self.x in ops or self.x in params:
            grd = np.zeros_like(self.x.top)
            grd[:,::2,::2,:] = self.grad
            self.x.grad = self.x.grad + grd


# Flatten (conv to fc)
class flatten:
    def __init__(self,x):
        ops.append(self)
        self.x = x
        
    def forward(self):
        self.top = np.reshape(self.x.top,[self.x.top.shape[0],-1])

    def backward(self):
        if self.x in ops or self.x in params:
            self.x.grad = self.x.grad + np.reshape(self.grad,self.x.top.shape)
            
# Convolution Layer
## Fill this out
class conv2:

    def __init__(self,x,k):
        ops.append(self)
        self.x = x
        self.k = k

    def forward(self):
        img = self.x.top
        knl = self.k.top
        b = img.shape[0]
        y = img.shape[1]
        x = img.shape[2]
        c1 = img.shape[3]
        ky = knl.shape[0]
        kx = knl.shape[1]
        c2 = knl.shape[3]
        g = np.zeros((b,y,x,c2))

        for Ky in range(ky):
            for Kx in range(kx):
                rolled = np.roll(img, (0, -Ky, -Kx, 0))
                rolled_newDim = rolled[:,:,:,:,newaxis]
                f = np.concatenate([rolled_newDim for x in range(c2)], axis=4)
                h = knl[Ky, Kx, :, :].reshape(1,1,1,c1,c2)
                g += np.sum(f * h, axis=3)
        self.top = g[:, :-ky+1, :-kx+1, :]





    def backward(self):
        img = self.x.top
        knl = self.k.top
        grad = self.grad
        b = img.shape[0]
        y = img.shape[1]
        x = img.shape[2]
        c1 = img.shape[3]
        ky = knl.shape[0]
        kx = knl.shape[1]
        c2 = knl.shape[3]

        if self.x in ops or self.x in params:
            kFlip = np.flip(knl, (0, 1))
            gImg = np.zeros((b, ky+y+1, kx+x+1, c1))
            padgrad = np.pad(grad, ((0,0), (ky, ky), (kx,kx), (0,0)), mode="constant")
            for Ky in range(min(ky, y)):
                for Kx in range(min(kx, x)):
                    rolled = np.roll(padgrad, (0, -Ky, -Kx, 0))
                    rolled_newDim = rolled[:, :, :, :, newaxis]
                    f = np.concatenate([rolled_newDim for x in range(c1)], axis=4)
                    fSwap = np.swapaxes(f, 3,4)
                    k = np.repeat(np.expand_dims(kFlip[Ky,Kx,:,:].reshape(1,1,c1,c2), axis=0), b, axis=0)
                    gImg += np.sum(fSwap * k, axis=4)
            self.x.grad = gImg[:, ky:-ky+1, kx:-kx+1, :]

        if self.k in ops or self.k in params:
            gKnl = np.zeros((ky,kx,c1,c2))
            dim1 = y-ky+1
            dim2 = x-kx+1
            for Ky in range(min(ky,y)):
                for Kx in range(min(kx,x)):
                    rolled = np.roll(img, (0, -Ky, -Kx, 0))
                    rolled_sliced = rolled[:, :dim1, :dim2, :]
                    rs_newDim = rolled_sliced[:,:,:,:, newaxis]
                    f = np.concatenate([rs_newDim for x in range(c2)], axis=4)
                    grad_newDim = grad[:,:,:,:,newaxis]
                    fp = np.concatenate([grad_newDim for x in range(c1)], axis=4)
                    fpSwap = np.swapaxes(fp, 3, 4)
                    reduce = np.sum(f * fpSwap, axis=0)
                    reduce = np.sum(reduce, axis=0)
                    gKnl[Ky, Kx, :, :] = np.sum(reduce, axis=0)
            self.k.grad = gKnl



