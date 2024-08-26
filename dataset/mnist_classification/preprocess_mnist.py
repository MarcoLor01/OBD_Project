import pandas as pd


def mnist_dataset():
    train_df = pd.read_csv('dataset/mnist_classification/mnist_train.csv')
    test_df = pd.read_csv('dataset/mnist_classification/mnist_test.csv')

    X_train = train_df.drop(columns=['label']).values
    y_train = train_df['label'].values
    X_test = test_df.drop(columns=['label']).values
    y_test = test_df['label'].values
    X_train = X_train / 255.0
    X_test = X_test / 255.0
    return X_train, y_train, X_test, y_test
