from sklearn.model_selection import train_test_split

from dataset.fashion_mnist_classification.preprocess_fashion_mnist import fashion_mnist_dataset
from dataset.fashion_mnist_classification.preprocess_fashion_mnist import shuffle_data
from neural_network.DenseLayer import DenseLayer
from neural_network.Model import Model
from neural_network.activation_functions.ReluActivationFunction import Relu
from neural_network.activation_functions.SoftmaxActivationFunction import Softmax
from neural_network.loss_functions.LossCategoricalCrossEntropy import LossCategoricalCrossEntropy
from neural_network.metrics_implementations.AccuracyCategorical import AccuracyCategorical
from neural_network.optimizers.Adam import Adam
from neural_network.RandomizedSearch import randomizedSearchCV
from neural_network.RandomizedSearch import print_model
from neural_network.regularization.Dropout import Dropout
from neural_network.regularization.EarlyStopping import EarlyStopping
from utils.Graphic import plot_training_metrics

X_train, y_train, X_test, y_test = fashion_mnist_dataset()
X_train, y_train = shuffle_data(X_train, y_train)
X_test, y_test = shuffle_data(X_test, y_test)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Modello di base, descritto all'interno della relazione

n_inputs = X_train.shape[1]
n_output = 10
n_neurons_first_hidden_layer = int(2/3 * n_inputs + n_output)
n_neurons_second_hidden_layer = int(n_neurons_first_hidden_layer / 2)

model = Model()
model.add_layer(DenseLayer(X_train.shape[1], n_neurons_first_hidden_layer, initialization="He"))
model.add_layer(Relu())
model.add_layer(DenseLayer(n_neurons_first_hidden_layer, n_neurons_second_hidden_layer, initialization="He"))
model.add_layer(Relu())
model.add_layer(DenseLayer(n_neurons_second_hidden_layer, n_output, initialization="He"))
model.add_layer(Softmax())


model.set(
    loss=LossCategoricalCrossEntropy(),
    optimizer=Adam(decay=5e-5),
    accuracy=AccuracyCategorical(),
    early_stopping=EarlyStopping(patience=10, min_delta=0.1)
)

model.finalize()
print_model(-1, model)
loss_history, accuracy_history, val_loss_history, val_accuracy_history = model.train(X_train, y_train, val_data=(X_val, y_val), epochs=100, batch_size=128, print_every=100)

confidences = model.predict(X_test[:20])

predictions = model.output_activation.predictions(confidences)
print(predictions)
print(y_test[:20])

plot_training_metrics(loss_history, accuracy_history)
plot_training_metrics(val_loss_history, val_accuracy_history)
# randomized_search_parameters = {
#     "metrics": ["AccuracyCategorical"],
#     "optimizers": {
#         "Adagrad": [{"learning_rate": [0.01, 0.001]}],
#         "Adam": [{"learning_rate": [0.01, 0.001]}],
#         "RmsProp": [{"learning_rate": [0.01, 0.001]}],
#         "Sgd": [{"learning_rate": [0.01, 0.001]}, {"momentum": [0.01, 0.001]}]
#     },
#     "regularization": [
#         {"Dropout": {"rate": [0.2, 0.3, 0.4, 0.5]}},
#         {"L2": {"weights": [0.01, 0.001, 0.0001], "bias": [0.01, 0.001, 0.0001]}},
#         {"L1": {"weights": [0.01, 0.001, 0.0001], "bias": [0.01, 0.001, 0.0001]}}
#     ],
#     "number_of_layers": [2, 3, 4],
#     "neurons_per_layer": [64, 128, 256, 512],
#     "early_stopping": {"patience": 5, "min_delta": 0.01}
# }



# best_model = randomizedSearchCV(X_train, y_train,
#                                 val_data=(X_test, y_test),
#                                 parameters=randomized_search_parameters,
#                                 combination=10,
#                                 number_of_classes=10,
#                                 epochs=1,
#                                 batch_size=64
#                                 )
# print_model(0, best_model)
