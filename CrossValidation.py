from itertools import product

import numpy as np
from neural_network.DenseLayer import DenseLayer
from neural_network.Model import Model
from neural_network.activation_functions.LinearActivation import ActivationLinear
from neural_network.activation_functions.ReluActivationFunction import Relu
from neural_network.activation_functions.SigmoidActivationFunction import Sigmoid
from neural_network.activation_functions.SoftmaxActivationFunction import Softmax
from neural_network.loss_functions.LossBinaryCrossEntropy import LossBinaryCrossEntropy
from neural_network.loss_functions.LossCategoricalCrossEntropy import LossCategoricalCrossEntropy
from neural_network.loss_functions.LossMSE import Mse
from neural_network.metrics_implementations.AccuracyCategorical import AccuracyCategorical
from neural_network.metrics_implementations.AccuracyRegression import AccuracyRegression
from neural_network.metrics_implementations.F1_score import compare_test_multiclass
from neural_network.optimizers.Adam import Adam
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

    for i, layer in enumerate(model.layers):
        layer_type = type(layer).__name__
        if hasattr(layer, 'weights'):
            dimensions = layer.weights.shape
            layer_description = f"{layer_type} di dimensione: {dimensions}"
        else:
            layer_description = f"Attivazione {layer_type}"

        print(f"{Fore.CYAN}{i}: {layer_description}")
        print(f"{Fore.CYAN + '-' * 40}")

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
        model = model_creation(X_train.shape[1], n_output, layer_neurons, activation_function[0], regularizer, optimizer,
                               use_dropout)
        if print_check is False:
            print_model(-1, model, combination, number_of_combinations)
            print_check = True
        print(f"\n========== Combination {i + 1} ==========\n")
        model.train(X_train, y_train, epochs=1, batch_size=128, print_every=100)

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
    # else:
    #    act_func = Tanh
    #    initialization = "Glorot"
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
        accuracy_metric = AccuracyRegression() #Da cambiare

    model = Model()
    model.add_layer(DenseLayer(X_train_shape, layer_neurons[0], initialization=initialization))
    model.add_layer(act_func_1)
    model.add_layer(DenseLayer(layer_neurons[0], layer_neurons[1], initialization=initialization))
    model.add_layer(act_func_2)
    model.add_layer(DenseLayer(layer_neurons[1], n_output, initialization=initialization))
    model.add_layer(output_function)

    model.set(
        loss=loss_function,
        optimizer=Adam(decay=5e-5), #Sistemare
        accuracy=accuracy_metric,
        early_stopping=EarlyStopping(patience=6, min_delta=0.1)
    )

    model.finalize()
    return model



# layer_combination = [[512, 256], [256, 128], [128, 64]]
layer_combination = [[512, 256]]
# regularizers = ["l1", "l2", None]
regularizers = [None]
#optimizers = ["adam", "rmsprop", "sgd", "adagrad"]
optimizers =["adam"]
#dropout = [True, False]
dropout = [False]
activation_functions = ["relu"]


def validation(X_train, y_train, n_output, number_of_folders):
    all_combinations = product(layer_combination, activation_functions, regularizers, optimizers, dropout)
    i = 1

    for combination in all_combinations:
        numbers_of_neurons, activation_function, regularizer, optimizer, use_dropout = combination
        model, accuracy, precision, recall, f1_score = k_fold_cross_validation(X_train, y_train, number_of_folders, n_output,
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

    return best_model

def calculate_combinations_count():
    # Calcola il numero totale di combinazioni
    all_combinations = list(product(layer_combination, activation_functions, regularizers, optimizers, dropout))
    return len(all_combinations)



#TODO
# 1) Capire quale è il miglor decay per l'ottimizzatore, e fare tutte le combinazioni
# 2) Funzione di attivazione passa una lista, ho messo [0] per ora ma è da aggiustare
# 3) Gestire le regolarizzazioni