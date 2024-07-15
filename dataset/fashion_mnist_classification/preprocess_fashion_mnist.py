import pandas as pd
from matplotlib import pyplot as plt
from sklearn.utils import shuffle

EPOCHS = 10
BATCH_SIZE = 128


def fashion_mnist_dataset():
    train_df = pd.read_csv('dataset/fashion_mnist_classification/fashion-mnist_train.csv')
    test_df = pd.read_csv('dataset/fashion_mnist_classification/fashion-mnist_test.csv')

    X_train = train_df.drop(columns=['label']).values
    y_train = train_df['label'].values
    X_test = test_df.drop(columns=['label']).values
    y_test = test_df['label'].values
    X_train = X_train / 255.0
    X_test = X_test / 255.0
    return X_train, y_train, X_test, y_test


def plot_image(image, label):
    image = image.reshape(28, 28)
    plt.imshow(image, cmap='gray')
    plt.title(f'Label: {label}')
    plt.axis('off')
    plt.show()


def shuffle_data(X_train, y_train):
    X_train, y_train = shuffle(X_train, y_train, random_state=42)
    return X_train, y_train


def total_step(X):
    steps = X.shape[0] // BATCH_SIZE
    if steps * BATCH_SIZE < X.shape[0]:
        steps += 1
    return steps

