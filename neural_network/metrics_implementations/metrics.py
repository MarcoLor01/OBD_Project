import numpy as np


# Implementation of accuracy
def accuracy_metric(predictions, target_class):
    if len(target_class.shape) == 2:
        target_class = np.argmax(target_class, axis=1)
    return np.mean(predictions == target_class)

