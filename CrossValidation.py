import time
from itertools import product

import numpy as np
from neural_network.DenseLayer import DenseLayer
from neural_network.Model import Model
from neural_network.activation_functions.LinearActivation import ActivationLinear
from neural_network.activation_functions.ReluActivationFunction import Relu
from neural_network.activation_functions.SigmoidActivationFunction import Sigmoid
from neural_network.activation_functions.SoftmaxActivationFunction import Softmax
from neural_network.activation_functions.TanhActivationFunction import Tanh
from neural_network.loss_functions.LossBinaryCrossEntropy import LossBinaryCrossEntropy
from neural_network.loss_functions.LossCategoricalCrossEntropy import LossCategoricalCrossEntropy
from neural_network.loss_functions.LossMSE import Mse
from neural_network.metrics_implementations.AccuracyCategorical import AccuracyCategorical
from neural_network.metrics_implementations.AccuracyRegression import AccuracyRegression
from neural_network.metrics_implementations.F1_score import compare_test_multiclass
from neural_network.optimizers.Adagrad import Adagrad
from neural_network.optimizers.RmsProp import Rmsprop
from neural_network.optimizers.Sgd import Sgd
from neural_network.optimizers.Adam import Adam
from neural_network.regularization.Dropout import Dropout
from neural_network.regularization.EarlyStopping import EarlyStopping

from colorama import Fore, Style, init

init(autoreset=True)

best_model = {
    'model': Model,
    'f1_metric': 0.0,
    'accuracy': 0.0,
    'precision': 0.0,
    'recall': 0.0
}


def print_model(iteration, model, combination, number_of_combinations):
    if iteration >= 0:
        print(f"\n{Fore.BLUE + '=' * 40}")
        print(f"{Fore.BLUE}Combinazione Numero: {iteration + 1}")
        print(f"{Fore.BLUE + '=' * 40}\n")
    print(f"{Fore.CYAN}\n Modello {combination}/{number_of_combinations}")
    print(f"{Fore.CYAN}\n ===== STRUTTURA DELLA RETE NEURALE =====\n")
    model_opt = model.optimizer
    for i, layer in enumerate(model.layers):
        layer_type = type(layer).__name__
        if hasattr(layer, 'weights'):
            dimensions = layer.weights.shape
            layer_description = f"{str(layer)}"
        elif hasattr(layer, 'rate'):
            layer_description = f"{str(layer)}"
        else:
            layer_description = f"Attivazione {str(layer_type)}"

        print(f"{Fore.CYAN}{i}: {layer_description}")
        print(f"{Fore.CYAN + '-' * 40}")
    print(f"\n{Fore.CYAN} {str(model_opt)}")

    print(Style.RESET_ALL)


def k_fold_indices(data, k):  # k = number of folds
    fold_size = len(data) // k
    indices = np.arange(len(data))
    folds = []
    for i in range(k):
        test_indices = indices[i * fold_size: (i + 1) * fold_size]
        train_indices = np.concatenate([indices[:i * fold_size], indices[(i + 1) * fold_size:]])
        folds.append((train_indices, test_indices))
    return folds


def k_fold_cross_validation(X, y, number_of_folds, n_output, layer_neurons, activation_function, regularizer, optimizer,
                            use_dropout, number_of_combinations, combination):
    accuracy_score = []
    precision_score = []
    recall_score = []
    f1_score_score = []

    fold_indices = k_fold_indices(X, number_of_folds)
    print_check = False
    for i, (train_indices, test_indices) in enumerate(fold_indices):
        X_train, y_train = X[train_indices], y[train_indices]
        X_test, y_test = X[test_indices], y[test_indices]

        model = model_creation(X_train.shape[1], n_output, layer_neurons, activation_function[0], regularizer,
                               optimizer, use_dropout)


        if print_check is False:
            print_model(-1, model, combination, number_of_combinations)
            print_check = True
        print(f"\n========== Combination {i + 1} ==========\n")
        start_time = time.time()

        model.train(X_train, y_train, epochs=20, batch_size=128, print_every=100)
        end_time = time.time()
        print("Tempo impiegato per il training di questo modello: ", end_time - start_time)
        y_pred = model.predict(X_test)

        accuracy, precision, recall, f1_score = compare_test_multiclass(y_pred, y_test)

        # Append the fold score to the list of scores
        accuracy_score.append(accuracy)
        precision_score.append(precision)
        recall_score.append(recall)
        f1_score_score.append(f1_score)
    accuracy_mean = np.mean(accuracy_score)
    precision_mean = np.mean(precision_score)
    recall_mean = np.mean(recall_score)
    f1_mean = np.mean(f1_score_score)

    print("\n========== K-Fold Cross-Validation Result ==========\n ")
    print("Mean Accuracy: ", accuracy_mean)
    print("Mean Precision: ", precision_mean)
    print("Mean Recall: ", recall_mean)
    print("Mean F1_Score: ", f1_mean)

    return model, accuracy_mean, precision_mean, recall_mean, f1_mean


def model_creation(X_train_shape, n_output, layer_neurons, activation_function, regularizer, optimizer, use_dropout):
    initialization = ''
    act_func_1 = None
    act_func_2 = None
    loss_function = None
    accuracy_metric = None
    if activation_function == 'relu':
        act_func_1 = Relu()
        act_func_2 = Relu()
        initialization = "He"
    else:
        act_func_1 = Tanh()
        act_func_2 = Tanh()
        initialization = "Glorot"
    if n_output > 2:
        output_function = Softmax()
        loss_function = LossCategoricalCrossEntropy()
        accuracy_metric = AccuracyCategorical()
    elif n_output == 2:
        output_function = Sigmoid()
        loss_function = LossBinaryCrossEntropy()
        accuracy_metric = AccuracyCategorical()
    else:
        output_function = ActivationLinear()
        loss_function = Mse()
        accuracy_metric = AccuracyRegression()

    model = Model()
    if regularizer == "l1":

        model.add_layer(
            DenseLayer(X_train_shape, layer_neurons[0], initialization=initialization, l1_regularization_weights=0.001,
                       l1_regularization_bias=0.001))
        model.add_layer(act_func_1)
        check_dropout(model, use_dropout)
        model.add_layer(DenseLayer(layer_neurons[0], layer_neurons[1], initialization=initialization,
                                   l1_regularization_weights=0.001, l1_regularization_bias=0.001))
        model.add_layer(act_func_2)
        check_dropout(model, use_dropout)

        model.add_layer(
            DenseLayer(layer_neurons[1], n_output, initialization=initialization, l1_regularization_weights=0.001,
                       l1_regularization_bias=0.001))
        model.add_layer(output_function)

    elif regularizer == "l2":
        model.add_layer(
            DenseLayer(X_train_shape, layer_neurons[0], initialization=initialization, l2_regularization_weights=0.001,
                       l2_regularization_bias=0.001))
        model.add_layer(act_func_1)
        check_dropout(model, use_dropout)

        model.add_layer(DenseLayer(layer_neurons[0], layer_neurons[1], initialization=initialization,
                                   l2_regularization_weights=0.001, l2_regularization_bias=0.001))
        model.add_layer(act_func_2)
        check_dropout(model, use_dropout)
        model.add_layer(
            DenseLayer(layer_neurons[1], n_output, initialization=initialization, l2_regularization_weights=0.001,
                       l2_regularization_bias=0.001))
        model.add_layer(output_function)

    else:
        # No regularization (None)
        model.add_layer(DenseLayer(X_train_shape, layer_neurons[0], initialization=initialization))
        model.add_layer(act_func_1)
        check_dropout(model, use_dropout)
        model.add_layer(DenseLayer(layer_neurons[0], layer_neurons[1], initialization=initialization))
        model.add_layer(act_func_2)
        check_dropout(model, use_dropout)
        model.add_layer(DenseLayer(layer_neurons[1], n_output, initialization=initialization))
        model.add_layer(output_function)

    opt_alg = None

    if optimizer == "adam":
        opt_alg = Adam(decay=0.0001, learning_rate=0.01)
    elif optimizer == "rmsprop":
        opt_alg = Rmsprop(decay=0.0001, learning_rate=0.01)
    elif optimizer == "adagrad":
        opt_alg = Adagrad(learning_rate=0.01)
    elif optimizer == "sgd":
        opt_alg = Sgd(decay=0.0001, learning_rate=0.01)
    elif optimizer == "sgd_momentum":
        opt_alg = Sgd(decay=0.0001, momentum=0.9, learning_rate=0.01)

    model.set(
        loss=loss_function,
        optimizer=opt_alg,
        accuracy=accuracy_metric,
        early_stopping=EarlyStopping(patience=6, min_delta=0.001)
    )

    model.finalize()
    return model


layer_combination = [[512, 256], [1024, 512]]
regularizers = ["l1", "l2", None]
optimizers = ["adam", "rmsprop", "adagrad"]
dropout = [True, False]
activation_functions = ["relu", "tanh"]


def validation(X_train, y_train, n_output, number_of_folders):
    all_combinations = product(layer_combination, activation_functions, regularizers, optimizers, dropout)
    i = 1
    start_time = time.time()
    for combination in all_combinations:
        numbers_of_neurons, activation_function, regularizer, optimizer, use_dropout = combination
        model, accuracy, precision, recall, f1_score = k_fold_cross_validation(X_train, y_train, number_of_folders,
                                                                               n_output,
                                                                               numbers_of_neurons, activation_functions,
                                                                               regularizer, optimizer, use_dropout,
                                                                               calculate_combinations_count(), i)
        i += 1

        if (f1_score > best_model['f1_metric'] or
                (f1_score == best_model['f1_metric'] and precision > best_model['precision']) or
                (f1_score == best_model['f1_metric'] and precision == best_model['precision'] and recall > best_model[
                    'recall']) or
                (f1_score == best_model['f1_metric'] and precision == best_model['precision'] and recall == best_model[
                    'recall'] and accuracy > best_model['accuracy'])):
            best_model['f1_metric'] = f1_score
            best_model['accuracy'] = accuracy
            best_model['precision'] = precision
            best_model['recall'] = recall
            best_model['model'] = model

        print("\n========== Best Model ==========\n")
        print(f"Il miglior modello fin'ora ha i seguenti parametri:")
        print(f"Accuracy: {best_model['accuracy']}")
        print(f"Precision: {best_model['precision']}")
        print(f"Recall: {best_model['recall']}")
        print(f"F1 Score: {best_model['f1_metric']}")
    end_time = time.time()
    print("Tempo impiegato per la C-V: ", end_time - start_time)
    return best_model


def calculate_combinations_count():
    # Calcola il numero totale di combinazioni
    all_combinations = list(product(layer_combination, activation_functions, regularizers, optimizers, dropout))
    return len(all_combinations)

def check_dropout(model, dropout_bool):
    if dropout_bool is True:
        model.add_layer(Dropout(0.2))
