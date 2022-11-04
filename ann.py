import numpy as np
import math
import pandas as pd


class MSELoss:      # For Reference
    def __init__(self):
        # Buffers to store intermediate results.
        self.current_prediction = None
        self.current_gt = None
        pass

    def __call__(self, y_pred, y_gt):

        self.current_prediction = y_pred
        self.current_gt = y_gt

        # MSE = 0.5 x (GT - Prediction)^2
        loss = 0.5 * np.power([y_gt[i] - y_pred[i] for i in range(len(y_gt))], 2)
        return loss

    def __grad__(self):
        # Derived by calculating dL/dy_pred
        gradient = [-1*(self.current_gt[i] - self.current_prediction[i]) for i in range(len(self.current_gt))]
        # We are creating and emptying buffers to emulate computation graphs in
        # Modern ML frameworks such as Tensorflow and Pytorch. It is not required.
        self.current_prediction = None
        self.current_gt = None

        return gradient


class CrossEntropyLoss:  
    def __init__(self):
        # Buffers to store intermediate results.
        self.current_prediction = None
        self.current_gt = None

    def __call__(self, y_pred, y_gt):
        self.current_prediction = y_pred
        self.current_gt = y_gt
        
        loss = [-np.log(y_pred[i]) if y_gt[i] == 1 else -np.log(1-y_pred[i]) for i in range(len(y_pred))]
        return loss

    # x is the output from the fully connected layer
    # y is the labels
    def __grad__(self):
        m = len(self.current_gt)
        grad = SoftmaxActivation()(self.current_prediction)
        grad[range(m)] -= 1
        grad = grad/m
        
        self.current_prediction = None
        self.current_gt = None
        
        return grad


class SoftmaxActivation:
    def __init__(self):
        self.y = None
        pass

    def __call__(self, y):
        self.y = np.array(y)
        vals = np.exp(self.y - np.max(self.y)) / np.sum(np.exp(self.y - np.max(self.y)))
        return vals
                                                        
    def __grad__(self):
        # We were using a heuristic (jacobian[column j] / jacobian[i][j] and kept getting overflow)
        # Defaulted to just using sigmoid's
        y_exp = np.array([x or 1000000000 for x in np.exp(-self.y)])
        grad = 1/(1 + y_exp) - (1 - 1/(1 + y_exp))
        return grad


class SigmoidActivation:
    def __init__(self):
        self.y = None
        pass

    def __call__(self, y):
        self.y = np.array(y)
        y_exp = np.array([x or 1000000000 for x in np.exp(-self.y)])
        vals = 1/(1 + y_exp)
        return vals

    def __grad__(self):
        y_exp = np.array([x or 1000000000 for x in np.exp(-self.y)])
        grad = 1/(1 + y_exp) - (1 - 1/(1 + y_exp))
        return grad

class ReLUActivation:
    def __init__(self):
        self.z = None
        pass

    def __call__(self, z):
        # y = f(z) = max(z, 0) -> Refer to the computational model of an Artificial Neuron
        self.z = z
        y = np.maximum(z, 0)
        return y

    def __grad__(self):
        # dy/dz = 1 if z was > 0 or dy/dz = 0 if z was <= 0
        gradient = np.where(self.z > 0, 1, 0)
        return gradient
    
class NoActivation:
    def __init__(self):
        self.z = None
        
    def __call__(self,z):
        self.z = z
        return self.z
    
    def __grad__(self):
        return self.z


def accuracy_score(y_true, y_pred):
    """ Compare y_true to y_pred and return the accuracy """
    accuracy = np.sum(y_true == y_pred, axis=0) / len(y_true)
    return accuracy
