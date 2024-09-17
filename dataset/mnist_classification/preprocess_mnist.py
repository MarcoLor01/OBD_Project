import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def mnist_dataset():
    train_df = pd.read_csv('dataset/mnist_classification/mnist_train.csv')
    test_df = pd.read_csv('dataset/mnist_classification/mnist_test.csv')

    X_train = train_df.drop(columns=['label']).values
    y_train = train_df['label'].values
    print_stats(X_train, y_train)
    X_test = test_df.drop(columns=['label']).values
    y_test = test_df['label'].values
    X_train = X_train / 255.0
    X_test = X_test / 255.0
    print("Shape y_train: ", y_train.shape)
    return X_train, y_train, X_test, y_test


def print_stats(X_train, y_train):
    # Istogramma delle etichette (frequenza dei numeri nel dataset)
    plt.figure(figsize=(8, 6))
    plt.hist(y_train, bins=np.arange(11)-0.5, edgecolor='k', alpha=0.7)
    plt.xticks(np.arange(10))
    plt.title('Distribuzione delle Etichette nel Dataset MNIST', fontsize=16)
    plt.xlabel('Cifre', fontsize=14)
    plt.ylabel('Frequenza', fontsize=14)
    plt.grid(True)
    plt.show()

    # Stampa alcuni esempi di immagini dal dataset
    fig, axes = plt.subplots(2, 5, figsize=(10, 5))
    fig.suptitle('Esempi di Immagini del Dataset MNIST', fontsize=16)
    for i, ax in enumerate(axes.flat):
        img = X_train[i].reshape(28, 28)  # Reshape a 28x28 pixel
        ax.imshow(img, cmap='gray')
        ax.set_title(f'Etichetta: {y_train[i]}')
        ax.axis('off')
    plt.show()
