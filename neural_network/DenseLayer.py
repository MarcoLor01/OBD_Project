# This class contains the implementations and the method of a single dense layer
import numpy as np


class DenseLayer:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.01 * np.random.randn(n_inputs,
                                              n_neurons)  # 0.01 * Random value from a normal distribution with average = 0 and st. dev = 1
        self.biases = np.zeros((1, n_neurons))
        self.inputs = None
        self.output = None
        self.dweights = None
        self.dbiases = None
        self.dinputs = None

    def forward(self, inputs):
        # ** inputs = (m,n), weights = (n, p) with p number of units in the output (like the next level), bias = (p), output = (m,p)
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        self.dinputs = np.dot(dvalues, self.weights.T)
