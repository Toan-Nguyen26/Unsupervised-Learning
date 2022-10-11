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
    return y

def train_test_split(data,labels,n=0.8):
    #split data in training and testing sets
    index = list(range(len(data)))
    kept = []
    n = round(len(data)*n)
    for i in range(n):
        idx = round(random.random()*(len(index)-1))
        kept.append(index.pop(idx))
    return [data[i] for i in kept],[labels[i] for i in kept]

def normalize_data(data): #TODO
    # normalize/standardize the data
    return