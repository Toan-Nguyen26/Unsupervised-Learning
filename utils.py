import numpy as np
import math
import pandas as pd


class MSELoss:      # For Reference
    def __init__(self):
        # Buffers to store intermediate results.
        self.p = None
        self.g = None

    def __call__(self, y_pred, y_gt):
        y_pred,y_gt = np.array(y_pred),np.array(y_gt)
        self.p = y_pred
        self.g = y_gt

        # MSE = 0.5 x (GT - Prediction)^2
        loss = 0.5 * np.power([y_gt[i] - y_pred[i] for i in range(len(y_gt))], 2)
        return loss

    def __grad__(self):
        # Derived by calculating dL/dy_pred
        gradient = [-1*(self.g[i] - self.p[i]) for i in range(len(self.g))]
        # We are creating and emptying buffers to emulate computation graphs in
        # Modern ML frameworks such as Tensorflow and Pytorch. It is not required.
        self.p = None
        self.g = None

        return gradient


class CrossEntropyLoss: 
    def __init__(self):
        # Buffers to store intermediate results.
        self.p = None
        self.g = None

    def __call__(self, y_pred, y_gt):
        y_pred,y_gt = np.array(y_pred),np.array(y_gt)
        self.p,self.g = y_pred,y_gt
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        loss = -y_gt * np.log(y_pred) - (1 - y_gt) * np.log(1 - y_pred)
        return loss

    def __grad__(self):
        p = np.clip(self.p, 1e-15, 1 - 1e-15)
        gradient = - (self.g / p) + (1 - self.g) / (1 - p)
        
        self.p = None
        self.g = None
        return gradient
    

class SoftmaxActivation:
    def __init__(self):
        self.y = None

    def __call__(self, y):
        y = np.array(y)
        self.y = y
        
        norm = y - np.max(y,axis=1,keepdims=True)
        exps = np.exp(norm)
        return exps / np.sum(exps,axis=1,keepdims=True)
                                                        
    def __grad__(self):
        s = SoftmaxActivation()(self.y)
        grad = s*(1-s)
        
        self.y = None
        return grad

class SigmoidActivation:
    def __init__(self):
        self.y = None

    def __call__(self, y):
        y = np.array(y)
        self.y = y
        return np.where(y >= 0, 1 / (1 + np.exp(-y)), np.exp(y) / (1 + np.exp(y)))

    def __grad__(self):
        g = 1 / (1 + np.exp(-self.y))
        grad = g*(1-g)
        
        self.y = None
        return grad

class ReLUActivation:
    def __init__(self):
        self.y = None

    def __call__(self, y):
        y = np.array(y)
        self.y = y
        y = np.maximum(y, 0)
        return y

    def __grad__(self):
        # dy/dz = 1 if z was > 0 or dy/dz = 0 if z was <= 0
        gradient = np.where(self.y > 0, 1, 0)
        
        self.y = None
        return gradient

def accuracy_score(y_true, y_pred):
    """ Compare y_true to y_pred and return the accuracy """
    accuracy = np.sum(np.array(y_true) == np.array(y_pred), axis=0) / len(y_true)
    return accuracy