import numpy as np

from CrossValidation import validation, print_best_model
from dataset.fashion_mnist_classification.preprocess_fashion_mnist import fashion_mnist_dataset
from dataset.fashion_mnist_classification.preprocess_fashion_mnist import shuffle_data
from neural_network.metrics_implementations.F1_score import compare_test_multiclass
from utils.Graphic import print_confusion_matrix

if __name__ == "__main__":

    X_train, y_train, X_test, y_test = fashion_mnist_dataset()
    X_train, y_train = shuffle_data(X_train, y_train)
    X_test, y_test = shuffle_data(X_test, y_test)

    best_model = validation(X_train, y_train, n_output=10, number_of_folders=5, epochs=20, multithread=True)
    print_best_model(best_model['model'])
    y_pred = best_model['model'].predict(X_test)
    y_pred = np.argmax(y_pred, axis=1)
    accuracy, precision, recall, f1_score = compare_test_multiclass(y_pred, y_test)

    print("\n========== Stats of the Best Model on the Test Set ==========\n ")
    print("Mean Accuracy: ", accuracy)
    print("Mean Precision: ", precision)
    print("Mean Recall: ", recall)
    print("Mean F1_Score: ", f1_score)

    class_labels = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    print_confusion_matrix(y_test, y_pred, class_labels)

