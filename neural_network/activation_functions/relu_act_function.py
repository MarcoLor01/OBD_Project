import numpy as np

# Implementation of relu activation function


class relu:

    def __init__(self):
        self.output = None

    def forward(self, input_units):
        self.output = np.maximum(0, input_units)  # Element-wise maximum
