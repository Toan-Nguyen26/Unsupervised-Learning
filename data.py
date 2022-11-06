import os, sys
import numpy as np
from sklearn import datasets
import random
from matplotlib import pyplot as plt

def visualizeObservation(n):
    plt.matshow(n) 
    plt.show()
    
def readDataLabels(): 
    #read in the data and the labels to feed into the ANN
    data = datasets.load_digits()
    images = data.images
    X = data.data
    y = data.target

    return images,X,y

def to_categorical(y):
    #Convert the nominal y values tocategorical
    cats = set(y)
    if len(cats) < 3:
        return [cat == cats[0] for cat in cats]
        
    dummies = {cat : [0 for i in range(len(y))]}
    for i,cat in enumerate(arr):
        dummies[cat][i] = 1
        
    return list(dummies.values())

def train_test_split(data,labels,n=0.8):
    #split data in training and testing sets
    index = list(range(len(data)))
    kept = []
    n = round(len(data)*n)
    for i in range(n):
        idx = round(random.random()*(len(index)-1))
        kept.append(index.pop(idx))
    
    X,Y = data[kept],labels[kept]
    xTest,yTest = data[index],labels[index]
    return X,Y,xTest,yTest

def normalize_data(X): 
    l2 = np.atleast_1d(np.linalg.norm(X, ord=2, axis=1))
    l2[l2 == 0] = 1
    return X / np.expand_dims(l2, axis = 1)