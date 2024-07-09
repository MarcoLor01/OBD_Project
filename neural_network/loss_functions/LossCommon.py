from abc import ABC, abstractmethod
import numpy as np

# Common loss function


class Loss(ABC):
    @abstractmethod
    def forward(self, output, target_class):
        pass

    def calculate(self, output, target_class):
        sample_losses = self.forward(output, target_class)
        data_loss = np.mean(sample_losses)
        return data_loss
