import os, sys
import numpy as np
import math
import random

#from data import readDataLabels, normalize_data, train_test_split, to_categorical, visualizeObservation
#from utils import accuracy_score

# Create an MLP with 8 neurons
# Input -> Hidden Layer -> Output Layer -> Output
# Neuron = f(w.x + b)
# Do forward and backward propagation

mode = 'train'      # train/test... Optional mode to avoid training incase you want to load saved model and test only.

def interval(num1,num2): #interval(0,1)
    n = min(num1,num2)
    x = max(num1,num2)
    return n + random.random()*(x-n)

class Layer:
    def __init__(self,size,activation):
        self.size = size
        self.activation = activation
        self.nodes = [Node() for x in range(self.size)]
        
        self.bias = None
        self.fromEdges = None
        self.toEdges = None
        self.initialValues = None
        
    def setFromEdges(self,edges):
        self.fromEdges = edges
        
    def setToEdges(self,edges):
        self.toEdges = edges
        
    def activate(self):
        self.initialValues = [node.value for node in self.nodes]
        vals = self.activation([node.value for node in self.nodes])
        for i,node in enumerate(self.nodes):
            node.value = vals[i]
            
    def reset(self):
        for node in self.nodes:
            node.reset()
        
class Node:
    def __init__(self):
        self.value = 0
    
    def reset(self):
        self.value = 0
        
class Edge:
    def __init__(self,input,output,weight):
        self.input = input
        self.output = output
        self.weight = weight
        
    def apply(self):
        if self.input is None:
            self.output.value += self.weight
            return
        self.output.value += self.input.value * self.weight

class ANN:
    def __init__(self, num_input_features, num_hidden_units, num_outputs, hidden_unit_activation, output_activation, loss_function):
        #hidden = [5,6,7]
        if type(num_hidden_units) is int:
            num_hidden_units = [num_hidden_units]
        
        self.num_input_features = num_input_features
        self.num_hidden_units = num_hidden_units
        self.num_outputs = num_outputs

        self.hidden_unit_activation = hidden_unit_activation
        self.output_activation = output_activation
        self.loss_function = loss_function
        
        self.layers = []
        self.layers.append(Layer(num_input_features,NoActivation()))
        for hidden in num_hidden_units:
            self.layers.append(Layer(hidden,hidden_unit_activation()))
        self.layers.append(Layer(num_outputs,output_activation()))


    def initialize_weights(self,edgeWts=None):
        self.edges = []
        for i in range(len(self.layers)-1):
            layer1 = self.layers[i]
            layer2 = self.layers[i+1]
            
            #Using He initialization
            edges = [[Edge(node1,node2,interval(layer2.size,layer1.size)*np.sqrt(2/layer1.size)) 
                      for node2 in layer2.nodes] for node1 in layer1.nodes]
            layer1.bias = [Edge(None,node2,interval(layer2.size,layer1.size)*np.sqrt(2/layer1.size)) for node2 in layer2.nodes]
            
            layer1.setToEdges(edges)
            layer2.setFromEdges(edges)
            self.edges.append([y for x in edges for y in x])
            
        if edgeWts is not None:
            for i,edge in enumerate(self.edges):
                edge.weight = edgeWts[i]

    def forward(self,inputs):
        for i,inp in enumerate(inputs):
            self.layers[0].nodes[i].value = inp
        for layer in self.layers:
            layer.activate()
            if layer.toEdges is not None:
                for nodeArray in layer.toEdges:
                    for edge in nodeArray:
                        edge.apply()
            if layer.bias is not None:
                for edge in layer.bias:
                    edge.apply()

        outputs = [output.value for output in self.layers[-1].nodes]
        self.reset()
        return outputs

    def backward(self,incomingLoss,learningRate):
        for layer in reversed(self.layers):
            # Adjust biases
            if layer.bias is not None:
                biases = [layer.bias[i].weight - incomingLoss[i]*learningRate for i in range(len(layer.bias))]
                for i,bias in enumerate(layer.bias):
                    bias.weight = biases[i]
            
            # Reverse edges
            if layer.toEdges is not None:
                error = np.dot([[x]*len(incomingLoss) for x in np.array(layer.initialValues).T],incomingLoss)
                
                edges = [[x.weight for x in y] for y in layer.toEdges]
                incomingLoss = np.dot(incomingLoss,np.array(edges).T)
                for i,edgeArray in enumerate(layer.toEdges):
                    for j,edge in enumerate(edgeArray):
                        edge.weight = edges[i][j] - learningRate*error[i]
            
            # Reverse activation
            incomingLoss = incomingLoss * layer.activation.__grad__()
                
    def train(self, X, Y, testX, testY, learning_rate=0.01, num_epochs=25):
        lossF = self.loss_function()
        for epoch in range(num_epochs):
            print(epoch)
            losses = []
            for i in range(len(X)):
                sample = X[i]
                label = Y[i]
                label = [1 if i == label else 0 for i in range(self.num_outputs)]
                
                output = self.forward(sample)
                loss = lossF.__call__(output,label)
                grad = lossF.__grad__()
                self.backward(grad,learning_rate)
                losses.append(sum(loss))
            
            print(self.test(testX,testY))
            print(sum(losses)/len(losses))

    def test(self, test_x, test_y):
        # Get predictions from test dataset
        predictions = []
        for observation in test_x:
            pred = self.forward(observation)
            pred = pred.index(max(pred))
            predictions.append(pred)
        # Calculate the prediction accuracy, see utils.py
        return accuracy_score(test_y,predictions)
    
    def reset(self):
        for layer in self.layers:
            layer.reset()
                
    def __str__(self):
        string = ''
        string += str(num_input_features) + ',' + str(num_hidden_units) + ',' + str(num_outputs) + '\n'
        for edge in self.edges:
            string += str(edge.weight) + ','
        return string


def main(argv):

    # Load dataset
    images,X,Y = readDataLabels()   # dataset[0] = X, dataset[1] = y
    visualizeObservation(images[9])
    X = normalize_data(X)
    
    # Split data into train and test split. call function in data.py
    X,Y,testX,testY = train_test_split(X,Y)
    
    # call ann->train()... Once trained, try to store the model to avoid re-training everytime
    if mode == 'train':
        ann = ANN(64,[16],10,SigmoidActivation,SoftmaxActivation,
          CrossEntropyLoss)
        ann.initialize_weights()
        ann.train(X,Y,testX,testY)
    else:
        # Call loading of trained model here, if using this mode (Not required, provided for convenience)
        with open('model.txt') as file:
            model = file.readlines()
        structure = model[0]
        weights = model[1]
        ann = ANN(structure[0],structure[1],structure[2],SigmoidActivation,SoftmaxActivation,
          MSELoss)#CrossEntropy or MSE
        ann.initialize_weights(f[1])
        ann.train(X,Y,testX,testY)

    # Call ann->test().. to get accuracy in test set and print it.
    accuracy = ann.test(testX,testY)
    print(accuracy)

if __name__ == "__main__":
    main(sys.argv)
