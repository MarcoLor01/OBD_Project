import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def mnist_dataset():
    train_df = pd.read_csv('dataset/mnist_classification/mnist_train.csv')
    test_df = pd.read_csv('dataset/mnist_classification/mnist_test.csv')

    X_train = train_df.drop(columns=['label']).values
    y_train = train_df['label'].values
    # print_stats(X_train, y_train)
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

