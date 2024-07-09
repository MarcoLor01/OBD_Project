from abc import ABC, abstractmethod
import numpy as np

# Common loss function


def regularization_loss(layer):

    reg_loss = 0

    # L1 regularization - weights
    if layer.l1_regularization_weights > 0:
        reg_loss += layer.weight_regularizer_l1 * np.sum(np.abs(layer.weights))

    # L2 regularization - weights
    if layer.l2_regularization_weights > 0:
        reg_loss += layer.weight_regularizer_l2 * np.sum(layer.weights * layer.weights)

    # L1 regularization - biases
    if layer.l1_regularization_bias > 0:
        reg_loss += layer.bias_regularizer_l1 * np.sum(np.abs(layer.biases))


    # L2 regularization - biases
    if layer.l2_regularization_bias > 0:
        reg_loss += layer.bias_regularizer_l2 * np.sum(layer.biases * layer.biases)

    return reg_loss


class Loss(ABC):
    @abstractmethod
    def forward(self, output, target_class):
        pass

    def calculate(self, output, target_class):
        sample_losses = self.forward(output, target_class)
        data_loss = np.mean(sample_losses)
        return data_loss

