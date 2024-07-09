import numpy as np


class Sgd:

    def __init__(self, learning_rate=1.0, decay=0., momentum=0.):
        self.learning_rate = learning_rate
        self.decay = decay
        self.iteration = 0
        self.current_learning_rate = learning_rate
        self.momentum = momentum  # Decay factor for momentum

    def decay_learning_rate_step(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate / (1. + self.decay * self.iteration)

    def update_weights(self, layer):

        if self.momentum:

            # If is the first time, create the momentum attr in the layer
            if not hasattr(layer, 'weight_momentums'):
                layer.weight_momentums = np.zeros_like(layer.weights)
                layer.bias_momentums = np.zeros_like(layer.biases)

            # Here we are going to calculate the value of the momentums
            weights_update = self.momentum * layer.weight_momentums - self.current_learning_rate * layer.dweights
            biases_update = self.momentum * layer.bias_momentums - self.current_learning_rate * layer.dbiases

            # Update the value
            layer.weight_momentums = weights_update
            layer.biases_momentums = biases_update

        else:  # Now we are in a vanilla sgd
            weights_update = - self.current_learning_rate * layer.dweights
            biases_update = - self.current_learning_rate * layer.dbiases

        layer.weights += weights_update
        layer.biases += biases_update


    def post_step_learning_rate(self):
        self.iteration += 1
