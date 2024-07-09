import numpy as np


class adagrad:  # Adaptive gradient

    def __init__(self, learning_rate=1.0, decay=0., epsilon=1e-7):
        self.learning_rate = learning_rate
        self.decay = decay
        self.iteration = 0
        self.current_learning_rate = learning_rate
        self.epsilon = epsilon

    def decay_learning_rate_step(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay * self.iteration))


    def update_weights(self, layer):
        if not hasattr(layer, "adagrad_weights"):
            layer.adagrad_weights = np.zeros_like(layer.weights)
            layer.adagrad_biases = np.zeros_like(layer.biases)

        layer.adagrad_weights += layer.dweights**2
        layer.adagrad_biases += layer.dbiases**2

        layer.weights += -self.current_learning_rate * (layer.dweights / (np.sqrt(layer.adagrad_weights) + self.epsilon))
        layer.biases += -self.current_learning_rate * (layer.dbiases / (np.sqrt(layer.adagrad_biases) + self.epsilon))

    def post_step_learning_rate(self):
        self.iteration += 1
