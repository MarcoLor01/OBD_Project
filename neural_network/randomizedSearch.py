import random
from neural_network.Model import Model
from neural_network.activation_functions.LinearActivation import ActivationLinear
from neural_network.activation_functions.SigmoidActivationFunction import Sigmoid
from neural_network.activation_functions.SoftmaxActivationFunction import Softmax
from neural_network.loss_functions.LossBinaryCrossEntropy import LossBinaryCrossEntropy
from neural_network.loss_functions.LossCategoricalCrossEntropy import LossCategoricalCrossEntropy
from neural_network.loss_functions.LossMSE import Mse
from neural_network.loss_functions.LossMAE import Mae
from neural_network.metrics_implementations.F1_score import F1Score
from neural_network.metrics_implementations.Accuracy import Accuracy
from neural_network.regularization import Dropout, EarlyStopping
from neural_network.optimizers import Adagrad, Adam, RmsProp, Sgd


def randomizedSearchCV(X, y, *, parameters, combination, number_of_classes):
    # Estrazione dei parametri dalla lista
    list_of_optimizers = parameters["optimizers"]
    list_of_regularizers = parameters["regularization"]
    number_of_layers = parameters["number_of_layers"]
    metrics = parameters["metrics"]
    neurons_per_layer = parameters["neurons_per_layer"]
    dropout_rate = parameters["dropout_rate"]
    optimizer_params = parameters["optimizer_params"]

    for i in range(combination):
        random_optimizer = random.choice(list_of_optimizers)
        random_metric = random.choice(metrics)
        random_optimizer_params = random.choice(optimizer_params[random_optimizer])

        model = Model()

        if number_of_classes > 2:
            loss = LossCategoricalCrossEntropy()
            activation_function = Softmax()
        elif number_of_classes == 2:
            loss = LossBinaryCrossEntropy()
            activation_function = Sigmoid()
        else:
            loss = random.choice([Mse(), Mae()])
            activation_function = ActivationLinear()

        optimizer = obtain_optimizer(random_optimizer)
        accuracy = set_accuracy(random_metric)

        model.set(loss=loss, optimizer=optimizer, accuracy=accuracy)

        num_layers = random.choice(number_of_layers)
        layers = []

        for j in range(num_layers):

            num_neurons = random.choice(neurons_per_layer)

            apply_regularization = random.choice([True, False])
            regularizer_name = random.choice(list_of_regularizers) if apply_regularization else None
            regularizer = obtain_regularizer(regularizer_name,
                                             random.choice(dropout_rate) if regularizer_name == "Dropout" else None)

            layers.append({
                "neurons": num_neurons,
                "regularizer": regularizer,
            })

        # Aggiungi i layer al modello
        for idx, layer in enumerate(layers):
            model.add_layer(layer["neurons"])

        # Stampa la configurazione scelta casualmente
        print("Configurazione scelta casualmente:")
        print(f"Optimizer: {random_optimizer} con parametri {random_optimizer_params}")
        print(f"Metrica: {random_metric}")
        print(f"Numero di layer: {num_layers}")

        for idx, layer in enumerate(layers):
            print(f"Layer {idx + 1}: {layer['neurons']} neuroni, "
                  f"Regularizer: {layer['regularizer']}")

        # Addestra e valuta il modello (aggiungi il codice per questo passaggio)




def obtain_optimizer(name):
    if name == "Adagrad":
        return Adagrad
    elif name == "Adam":
        return Adam
    elif name == "RMSprop":
        return RmsProp
    elif name == "SGD":
        return Sgd
    else:
        raise ValueError(f"Optimizer {name} not recognized")


def set_accuracy(metric_name):
    if metric_name == "F1Score":
        return F1Score()
    elif metric_name == "Accuracy":
        return Accuracy()
    else:
        raise ValueError(f"Metric {metric_name} not recognized")


def obtain_regularizer(name, rate):
    if name == "Dropout":
        return Dropout.Dropout(rate)

    # For possible extensions

