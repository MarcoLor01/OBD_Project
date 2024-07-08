import numpy as np
from nnfs.datasets import spiral_data
from neural_network.activation_functions.relu_act_function import relu
from neural_network.dense_layer import dense_layer
from neural_network.loss_functions.softmax_cross_entropy import softmax_cross_entropy
from neural_network.optimizers.sgd import sgd
from neural_network.metrics_implementations.metrics import accuracy_metric
from utils.graphic import plot_training_metrics

X, y = spiral_data(samples=100, classes=3)

dense1 = dense_layer(2, 64)
activation1 = relu()
dense2 = dense_layer(64, 3)
loss = softmax_cross_entropy()
optimizer = sgd(decay=1e-3, momentum=0.9)

loss_history = []
accuracy_history = []

for epoch in range(0, 20001):

    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    loss_result = loss.forward(dense2.output, y)

    predictions = np.argmax(loss.output, axis=1)
    accuracy = accuracy_metric(predictions, y)

    if not epoch % 100:
        print(f'epoch: {epoch}, ' +
              f'acc: {accuracy:.3f}, ' +
              f'loss: {loss_result:.3f}, ' +
              f'actual learning rate: {optimizer.current_learning_rate:.4f}'
              )
        loss_history.append(loss_result)
        accuracy_history.append(accuracy)


    loss.backward(loss.output, y)
    dense2.backward(loss.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)

    optimizer.decay_learning_rate_step()
    optimizer.update_weights(dense1)
    optimizer.update_weights(dense2)
    optimizer.post_step_learning_rate()

plot_training_metrics(loss_history, accuracy_history=accuracy_history)
