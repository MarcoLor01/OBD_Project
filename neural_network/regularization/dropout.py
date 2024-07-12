import numpy as np


class dropout:

    def __init__(self, rate):
        self.rate = 1 - rate
        self.inputs = None
        self.mask = None
        self.outputs = None
        self.dinputs = None

    def forward(self, inputs):
        self.inputs = inputs
        self.mask = np.random.binomial(1, self.rate, size=inputs.shape) / self.rate
        self.outputs = self.inputs * self.mask

    def backward(self, dvalues):
        self.dinputs = dvalues * self.mask
        