import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.utils import shuffle

EPOCHS = 10


def fashion_mnist_dataset():
    train_df = pd.read_csv('dataset/fashion_mnist_classification/fashion-mnist_train.csv')
    test_df = pd.read_csv('dataset/fashion_mnist_classification/fashion-mnist_test.csv')
    X_train = train_df.drop(columns=['label']).values
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


def print_label_frequencies(y_train, y_test):
    unique_train, counts_train = np.unique(y_train, return_counts=True)
    unique_test, counts_test = np.unique(y_test, return_counts=True)

    print("Frequenza delle classi nel dataset di Training:")
    for label, count in zip(unique_train, counts_train):
        print(f"Classe {label}: {count} occorrenze")

    print("\nFrequenza delle classi nel dataset di Test:")
    for label, count in zip(unique_test, counts_test):
        print(f"Classe {label}: {count} occorrenze")
