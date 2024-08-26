# This class contains the implementations and the method of a single dense layer
import numpy as np


class DenseLayer:
    def __init__(self, n_inputs, n_neurons, initialization=None, l1_regularization_bias=0, l1_regularization_weights=0,
                 l2_regularization_weights=0, l2_regularization_bias=0):
        if initialization == "He":
            self.weights = initialize_he(n_inputs, n_neurons)
        elif initialization == "Glorot":
            self.weights = initialize_glorot(n_inputs, n_neurons)
        else:
            self.weights = 0.1 * np.random.randn(n_inputs, n_neurons)

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

    def __str__(self):
        if self.l1_regularization_weights != 0:
            return f"Layer Dense di dimensione: ({self.weights.shape[0]}, {self.weights.shape[1]}) con regolarizzazione L1 di valori - Weights: {self.l1_regularization_weights}, Bias: {self.l1_regularization_bias}"
        elif self.l2_regularization_weights != 0:
            return f"Layer Dense di dimensione: ({self.weights.shape[0]}, {self.weights.shape[1]}) con regolarizzazione L2 di valori - Weights: {self.l2_regularization_weights}, Bias: {self.l2_regularization_bias}"
        else:
            return f"Layer Dense di dimensione: ({self.weights.shape[0]}, {self.weights.shape[1]}) senza regolarizzazione"

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


def initialize_he(n_inputs, n_neurons):
    stddev = np.sqrt(2 / n_inputs)
    weights = stddev * np.random.randn(n_inputs, n_neurons)
    return weights


def initialize_glorot(n_inputs, n_neurons):
    limit = np.sqrt(6 / (n_inputs + n_neurons))
    weights = np.random.uniform(-limit, limit, size=(n_inputs, n_neurons))
    return weights
