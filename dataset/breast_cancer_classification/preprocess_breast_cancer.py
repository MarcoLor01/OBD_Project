import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def breast_cancer_dataset():
    df = pd.read_csv('dataset/breast_cancer_classification/breast-cancer.csv')
    df = df.drop(columns=['id'], axis=1)
    pd.set_option('future.no_silent_downcasting', True)
    df['diagnosis'] = df['diagnosis'].replace({'M': 1, 'B': 0})

    df = df.drop(
        columns=['fractal_dimension_se', 'symmetry_se', 'texture_se', 'fractal_dimension_mean', 'smoothness_se'],
        axis=1)

    X = df.drop('diagnosis', axis=1)
    output = df['diagnosis']

    X_train, X_test, y_train, y_test = train_test_split(X, output, test_size=0.2, random_state=42, stratify=output)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    y_train = y_train.values.reshape(-1, 1)
    y_test = y_test.values.reshape(-1, 1)

    return X_train, X_test, y_train, y_test
