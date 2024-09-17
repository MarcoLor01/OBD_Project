import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.utils import shuffle

EPOCHS = 10
BATCH_SIZE = 128


def fashion_mnist_dataset():
    train_df = pd.read_csv('dataset/fashion_mnist_classification/fashion-mnist_train.csv')
    test_df = pd.read_csv('dataset/fashion_mnist_classification/fashion-mnist_test.csv')
    X_train = train_df.drop(columns=['label']).values
    print("SHAPE: ", X_train.shape)
    y_train = train_df['label'].values
    X_test = test_df.drop(columns=['label']).values
    y_test = test_df['label'].values
    X_train = X_train / 255.0
    X_test = X_test / 255.0

    train_data = pd.DataFrame(X_train)
    train_data['label'] = y_train

    stratified_sample = train_data.groupby('label').apply(
        lambda x: x.sample(frac=40000 / len(train_data), random_state=42)).reset_index(drop=True)

    X_train_sample = stratified_sample.drop(columns=['label']).values
    y_train_sample = stratified_sample['label'].values
    return X_train_sample, y_train_sample, X_test, y_test



def shuffle_data(X_train, y_train):
    X_train, y_train = shuffle(X_train, y_train, random_state=42)
    return X_train, y_train


def total_step(X):
    steps = X.shape[0] // BATCH_SIZE
    if steps * BATCH_SIZE < X.shape[0]:
        steps += 1
    return steps


def print_fashion_mnist_stats(X_train, y_train):
    # Genera un istogramma della distribuzione delle classi
    plt.figure(figsize=(8, 6))
    plt.hist(y_train, bins=np.arange(11) - 0.5, edgecolor='black', alpha=0.7)
    plt.xticks(np.arange(10))
    plt.title("Distribuzione delle Classi nel Dataset Fashion MNIST", fontsize=16)
    plt.xlabel("Classe", fontsize=14)
    plt.ylabel("Frequenza", fontsize=14)
    plt.grid(True)
    plt.show()

    # Definizione delle etichette delle classi per Fashion MNIST
    class_labels = ['T-shirt/Top', 'Pantaloni', 'Pullover', 'Vestito', 'Cappotto',
                    'Sandalo', 'Camicia', 'Sneaker', 'Borsa', 'Stivale']

    # Mostra alcuni esempi dal dataset
    num_examples = 6
    plt.figure(figsize=(12, 8))

    for i in range(num_examples):
        plt.subplot(2, 3, i + 1)
        img = X_train[i].reshape(28, 28)  # Riorganizza il vettore in un'immagine 28x28
        plt.imshow(img, cmap='gray')
        plt.title(f"Classe: {class_labels[y_train[i]]}")
        plt.axis('off')

    plt.tight_layout()
    plt.show()


def print_label_frequencies(y_train, y_test):
    # Conta le occorrenze delle classi nel dataset di training
    unique_train, counts_train = np.unique(y_train, return_counts=True)
    unique_test, counts_test = np.unique(y_test, return_counts=True)

    print("Frequenza delle classi nel dataset di Training:")
    for label, count in zip(unique_train, counts_train):
        print(f"Classe {label}: {count} occorrenze")

    print("\nFrequenza delle classi nel dataset di Test:")
    for label, count in zip(unique_test, counts_test):
        print(f"Classe {label}: {count} occorrenze")