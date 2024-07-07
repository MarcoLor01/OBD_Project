# This class contains the implementations and the method of a single dense layer
import numpy as np


class dense_layer:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.01 * np.random.randn(n_inputs,
                                              n_neurons)  # 0.01 * Random value from a normal distribution with average = 0 and st. dev = 1
        self.biases = np.zeros((1, n_neurons))
        self.output = None

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases
