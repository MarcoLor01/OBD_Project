# This class contains the implementations and the method of a single dense layer
import numpy as np


class DenseLayer:
    def __init__(self, n_inputs, n_neurons, l1_regularization_bias=0, l1_regularization_weights=0, l2_regularization_weights=0, l2_regularization_bias=0, first_layer = False):

        self.weights = 0.1 * np.random.randn(n_inputs, n_neurons)  # 0.01 * Random value from a normal distribution with average = 0 and st. dev = 1
        self.biases = np.zeros((1, n_neurons))
        self.l1_regularization_bias = l1_regularization_bias
        self.l1_regularization_weights = l1_regularization_weights
        self.l2_regularization_weights = l2_regularization_weights
        self.l2_regularization_bias = l2_regularization_bias
        self.inputs = None
        self.output = None
        self.dweights = None
        self.dbiases = None
        self.dinputs = None

    def forward(self, inputs, training):
        # ** inputs = (m,n), weights = (n, p) with p number of units in the output (like the next level), bias = (p), output = (m,p)
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)

        if self.l1_regularization_weights > 0:
            dl1_weights = np.ones_like(self.weights)
            dl1_weights[self.weights < 0] = -1
            self.dweights += self.l1_regularization_weights * dl1_weights

        if self.l1_regularization_bias > 0:
            dl1_bias = np.ones_like(self.biases)
            dl1_bias[self.biases < 0] = -1
            self.dbiases += self.l1_regularization_bias * dl1_bias

        if self.l2_regularization_weights > 0:
            self.dweights += 2 * self.l2_regularization_weights * self.weights

        if self.l2_regularization_bias > 0:
            self.dbiases += 2 * self.l2_regularization_bias * self.biases

        self.dinputs = np.dot(dvalues, self.weights.T)

