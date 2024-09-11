from abc import ABC, abstractmethod
import numpy as np
import pandas as pd


# Common loss function


class Loss(ABC):
    def __init__(self):
        self.accumulated_count = None
        self.accumulated_sum = None
        self.trainable = None

    @abstractmethod
    def forward(self, output, target_class):
        pass

    def set_trainable(self, trainable_layers):
        self.trainable = trainable_layers

    def calculate(self, output, target_class, include_reg=False):

        sample_losses = self.forward(output, target_class)
        data_loss = np.mean(sample_losses)
        
        self.accumulated_sum += np.sum(sample_losses)
        self.accumulated_count += len(sample_losses)
        
        if not include_reg:
            return data_loss

        return data_loss, self.regularization_loss()

    def calculated_accumulated(self, *, include_reg=False):
        data_loss = self.accumulated_sum / self.accumulated_count
        if not include_reg:
            return data_loss

        return data_loss, self.regularization_loss()

    def regularization_loss(self):

        reg_loss = 0
        for layer in self.trainable:

            if layer.l1_regularization_weights > 0:
                reg_loss += layer.l1_regularization_weights * np.sum(np.abs(layer.weights))

            # L2 regularization - weights
            if layer.l2_regularization_weights > 0:
                reg_loss += layer.l2_regularization_weights * np.sum(layer.weights * layer.weights)

            # L1 regularization - biases
            if layer.l1_regularization_bias > 0:
                reg_loss += layer.l1_regularization_bias * np.sum(np.abs(layer.biases))

            # L2 regularization - biases
            if layer.l2_regularization_bias > 0:
                reg_loss += layer.l2_regularization_bias * np.sum(layer.biases * layer.biases)

        return reg_loss

    def new_pass(self):
        self.accumulated_sum = 0
        self.accumulated_count = 0
