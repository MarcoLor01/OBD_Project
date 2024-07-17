from neural_network.DenseLayer import DenseLayer
from neural_network.Model import Model
from neural_network.activation_functions.LinearActivation import ActivationLinear
from neural_network.activation_functions.ReluActivationFunction import Relu
from neural_network.activation_functions.SigmoidActivationFunction import Sigmoid
from neural_network.activation_functions.SoftmaxActivationFunction import Softmax
from neural_network.loss_functions.LossBinaryCrossEntropy import LossBinaryCrossEntropy
from neural_network.loss_functions.LossCategoricalCrossEntropy import LossCategoricalCrossEntropy
from neural_network.loss_functions.LossMSE import Mse
from neural_network.loss_functions.LossMAE import Mae
from neural_network.metrics_implementations.AccuracyCategorical import AccuracyCategorical
from neural_network.metrics_implementations.AccuracyRegression import AccuracyRegression
from neural_network.metrics_implementations.F1_score import F1Score
from neural_network.optimizers.Adam import Adam
from neural_network.optimizers.Sgd import Sgd
from neural_network.optimizers.Adagrad import Adagrad
from neural_network.optimizers.RmsProp import Rmsprop
from neural_network.regularization.Dropout import Dropout
import random

from neural_network.regularization.EarlyStopping import EarlyStopping


def randomizedSearchCV(X, y, *, val_data, parameters, combination, number_of_classes, epochs, batch_size):
    optimizer_choice = select_random_optimizer(parameters)
    number_of_layers = select_number_layer(parameters)
    X_val, y_val = val_data
    if number_of_classes > 2:
        loss = LossCategoricalCrossEntropy()
        activation_function = Softmax()
    elif number_of_classes == 2:
        loss = LossBinaryCrossEntropy()
        activation_function = Sigmoid()
    else:
        loss = random.choice([Mse(), Mae()])
        activation_function = ActivationLinear()

    accuracy = set_accuracy(parameters['metrics'], number_of_classes)

    for i in range(combination):
        next_input = X.shape[1]
        random.seed(i)
        model = Model()

        for j in range(number_of_layers):
            neurons_per_layer = select_number_neurons(parameters)
            regularizers = select_random_regularization(parameters)

            if j == number_of_layers - 1:
                neurons_per_layer = number_of_classes

            dropout, next_layer = instance_regularizers(regularizers, next_input, neurons_per_layer)
            print(next_layer)
            if next_layer is not None:
                model.add_layer(next_layer)
                next_input = neurons_per_layer
            else:
                model.add_layer(DenseLayer(next_input, neurons_per_layer))
                next_input = neurons_per_layer
            if j != number_of_layers - 1:
                model.add_layer(Relu())
            if dropout is not None:
                model.add_layer(dropout)

        model.add_layer(activation_function)
        early_stopping = get_early_stopping_instance(parameters)
        if early_stopping is not None:
            model.set(loss=loss, optimizer=optimizer_choice, accuracy=accuracy, early_stopping=early_stopping)
        else:
            model.set(loss=loss, optimizer=optimizer_choice, accuracy=accuracy)
        model.finalize()
        print_model(i, model)
        model.train(X, y, val_data=(X_val, y_val), epochs=epochs, batch_size=batch_size, print_every=100)


def get_early_stopping_instance(parameters):
    early_stopping = parameters.get("early_stopping", {})
    patience = early_stopping.get("patience", 5)  # Valore di default per patience
    min_delta = early_stopping.get("min_delta", 0)  # Valore di default per min_delta

    if not isinstance(patience, int) or not isinstance(min_delta, (int, float)):
        raise ValueError(
            "Invalid types for Early Stopping Parameters: patience should be int, min_delta should be int or float")

    early_stopping_instance = EarlyStopping(patience=patience, min_delta=min_delta)

    return early_stopping_instance


def select_number_layer(parameters):
    numbers_of_layer = parameters['number_of_layers']
    number_layer_choice = random.choice(numbers_of_layer)
    return number_layer_choice


def select_number_neurons(parameters):
    numbers_of_neurons = parameters['neurons_per_layer']
    number_neurons_choice = random.choice(numbers_of_neurons)
    return number_neurons_choice


def select_random_optimizer(parameters):
    optimizers = parameters['optimizers']
    optimizer_name_algorithm = random.choice(list(optimizers.keys()))

    combined_params = {}
    for param_set in optimizers[optimizer_name_algorithm]:
        for param, values in param_set.items():
            if param not in combined_params:
                combined_params[param] = values
            else:
                combined_params[param].extend(values)

    selected_parameters = {param: random.choice(values) for param, values in combined_params.items()}

    if optimizer_name_algorithm == 'Adagrad':
        optimizer_instance = Adagrad(**selected_parameters)
    elif optimizer_name_algorithm == 'Adam':
        optimizer_instance = Adam(**selected_parameters)
    elif optimizer_name_algorithm == 'RmsProp':
        optimizer_instance = Rmsprop(**selected_parameters)
    elif optimizer_name_algorithm == 'Sgd':
        optimizer_instance = Sgd(**selected_parameters)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name_algorithm}")

    return optimizer_instance


def print_model(iteration, model):
    print(f"Combinazione numero: {iteration}")
    for i, layer in enumerate(model.layers):
        print("----------------------------")
        print(f"Layer numero {i}: {layer}")
    print("----------------------------")


def set_accuracy(metric_name, number_of_classes):
    if len(metric_name) != 1:
        raise ValueError(f"You can insert only 1 metric")

    metric = metric_name[0]
    if metric == "F1Score":
        return F1Score()
    elif metric in ["Accuracy", "AccuracyCategorical"]:
        if number_of_classes > 1:
            return AccuracyCategorical()
        else:
            return AccuracyRegression()
    elif metric == "AccuracyRegression":
        return AccuracyRegression()
    else:
        raise ValueError(f"Metric {metric} not recognized")


def select_random_regularization(parameters):
    regularization = parameters['regularization']
    selected_regularization = {}

    for reg_dict in regularization:
        for reg_technique, params in reg_dict.items():
            selected_params = {}
            for param, values in params.items():
                selected_params[param] = random.choice(values)
            selected_regularization[reg_technique] = selected_params

    return selected_regularization

## Cambiare questo
def instance_regularizers(selected_regularization, n_inputs, n_neurons):
    dropout = None
    layer_this = None
    is_inserted = False
    for reg_technique, params in selected_regularization.items():
        if reg_technique == "Dropout" and random.choice([True, False]):
            dropout_rate = params["rate"]
            dropout = Dropout(dropout_rate)
        if reg_technique == "L2" and random.choice([True, False]) and is_inserted is False:
            weights_reg = params.get("weights")
            bias_reg = params.get("bias")
            if weights_reg is not None and bias_reg is not None:
                layer_this = DenseLayer(n_inputs=n_inputs, n_neurons=n_neurons,
                                        l2_regularization_weights=weights_reg,
                                        l2_regularization_bias=bias_reg)
                is_inserted = True
            else:
                raise ValueError("Both weights and bias regularization values must be provided for L2.")
        elif reg_technique == "L1" and random.choice([True, False]) and is_inserted is False:
            weights_reg = params.get("weights")
            bias_reg = params.get("bias")
            if weights_reg is not None and bias_reg is not None:
                layer_this = DenseLayer(n_inputs=n_inputs, n_neurons=n_neurons,
                                        l1_regularization_weights=weights_reg,
                                        l1_regularization_bias=bias_reg)
                is_inserted = True
            else:
                raise ValueError("Both weights and bias regularization values must be provided for L1.")

    return dropout, layer_this
