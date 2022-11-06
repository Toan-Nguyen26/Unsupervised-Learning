import os, sys
import numpy as np
import math
import random
from matplotlib import pyplot as plt

from data import readDataLabels, normalize_data, train_test_split, to_categorical, visualizeObservation
from utils import MSELoss, CrossEntropyLoss, SoftmaxActivation, SigmoidActivation, ReLUActivation, accuracy_score

class ANN:
    def __init__(self, num_input_features, num_hidden_units, num_outputs, hidden_unit_activation, output_activation, loss_function, momentum=False):
        #hidden = [5,6,7]
        if type(num_hidden_units) is int:
            num_hidden_units = [num_hidden_units]
        if type(hidden_unit_activation) is not list:
            hidden_unit_activation = [hidden_unit_activation for x in num_hidden_units]
        assert len(num_hidden_units) == len(hidden_unit_activation)
        
        self.num_input_features = num_input_features
        self.num_hidden_units = num_hidden_units
        self.num_outputs = num_outputs

        self.hidden_unit_activation = hidden_unit_activation
        self.output_activation = output_activation
        self.loss_function = loss_function
        
        self.weights,self.biases,self.Zs,self.As = None,None,None,None
        self.momentum = momentum
        
        self.initialize_weights()

    def initialize_weights(self):
        self.weights = [np.random.uniform(0,1,(self.num_input_features,self.num_hidden_units[0]))]
        self.biases = [np.zeros((1,self.num_hidden_units[0]))]
        self.activations = [h() for h in self.hidden_unit_activation] + [self.output_activation()]
        for i in range(len(self.num_hidden_units)-1):
            self.weights.append(np.random.uniform(0,1,(self.num_hidden_units[i],self.num_hidden_units[i+1])))
            self.biases.append(np.zeros((1,self.num_hidden_units[i+1])))
        self.weights.append(np.random.uniform(0,1,(self.num_hidden_units[-1],self.num_outputs)))
        self.biases.append(np.zeros((1,self.num_outputs)))
        
        self.GWs = [0 for i in range(len(self.weights))]
        self.GBs = self.GWs.copy()

    def forward(self,inputs):
        self.As = []
        for bias,weight,activation in zip(self.biases,self.weights,self.activations):
            self.As.append(inputs)
            z = inputs.dot(weight) + bias
            inputs = activation(z)
        return inputs

    def backward(self,da,learningRate,decayFactor):
        for i,activation,a in list(zip(range(len(self.weights)),self.activations,self.As))[::-1]:
            dz = da * activation.__grad__()
            dw = a.T.dot(dz)
            db = np.sum(dz,axis=0)
            
            da = dz.dot(self.weights[i].T)
            
            self.update_weight(i,dw,db,learningRate,decayFactor)
            
    def update_weight(self,i,dw,db,learningRate,decayFactor):
        #straight update
        if not self.momentum:
            self.weights[i] -= learningRate*dw
            self.biases[i] -= learningRate*db
        else: #Momentum AGD
            self.GWs[i] = decayFactor*self.GWs[i] + learningRate*dw
            self.GBs[i] = decayFactor*self.GBs[i] + learningRate*db
            
            self.weights[i] -= self.GWs[i]
            self.biases[i] -= self.GBs[i]
            
                
    def train(self, X, Y, testX, testY, learning_rate=0.001, num_epochs=1000, decayFactor=0.9):
        lossF = self.loss_function()
        self.losses = []
        self.accuracies = []
        for epoch in range(num_epochs):
            preds = self.forward(X)
            
            loss = np.sum(lossF.__call__(preds,Y))
            self.backward(lossF.__grad__(),learning_rate,decayFactor)

            self.losses.append(loss)
            self.accuracies.append(self.test(testX,testY))
            
    def test(self, test_x, test_y):
        # Get predictions from test dataset
        preds = [list(x) for x in self.forward(test_x)]
        test_y = [list(x) for x in test_y]

        preds = [x.index(max(x)) for x in preds]
        labels = [x.index(max(x)) for x in test_y]

        # Calculate the prediction accuracy, see utils.py
        return accuracy_score(labels,preds)

def main(argv):

    # Load dataset
    images,X,yRaw = readDataLabels()   # dataset[0] = X, dataset[1] = y
    
    n = len(set(yRaw))
    Y = np.array([[1.0 if i == y else 0.0 for i in range(n)] for y in yRaw])
    visualizeObservation(images[9])
    print(yRaw[9],Y[9])
    
    X = normalize_data(X)
    
    # Split data into train and test split. call function in data.py
    X,Y,testX,testY = train_test_split(X,Y)
    
    # train base model
    ann1 = ANN(64,16,10,SigmoidActivation,SoftmaxActivation,
      CrossEntropyLoss)
    ann1.train(X,Y,testX,testY)

    # train momentum model
    ann2 = ANN(64,16,10,SigmoidActivation,SoftmaxActivation,
      CrossEntropyLoss,momentum=True)
    ann2.train(X,Y,testX,testY)
    

    #plot accuracies
    plt.plot(ann1.accuracies,label='base')
    plt.plot(ann2.accuracies,label='momentum')
    
    # Call ann->test().. to get accuracy in test set and print it.
    accuracy1 = ann1.test(testX,testY)
    accuracy2 = ann2.test(testX,testY)
    print(accuracy1,accuracy2)

if __name__ == "__main__":
    main(sys.argv)
