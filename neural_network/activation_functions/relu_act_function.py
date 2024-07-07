import numpy as np

# Implementation of relu activation function


class relu:

    def __init__(self):
        self.inputs = None
        self.output = None
        self.dinputs = None

    def forward(self, input_units):
        self.inputs = input_units
        self.output = np.maximum(0, input_units)  # Element-wise maximum

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0

