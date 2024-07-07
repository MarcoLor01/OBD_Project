
class sgd:

    def __init__(self, learning_rate=1.0):
        self.learning_rate = learning_rate

    def update_weights(self, layer):
        layer.weights += -self.learning_rate * layer.dweights
        layer.biases += -self.learning_rate * layer.dbiases
