import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler


def flight_price_dataset():
    # Carica il dataset
    df = pd.read_csv('dataset/flight_price_prediction/flight_price.csv')
    df.drop(["Unnamed: 0", "flight"], axis=1, inplace=True)
    # df = df.head(80000)
    # Trasformazione delle etichette
    le = LabelEncoder()
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = le.fit_transform(df[col])

    q1 = df["duration"].quantile(0.25)
    q3 = df["duration"].quantile(0.75)
    iqr = q3 - q1

    upper_limit = q3 + (1.5 * iqr)
    lower_limit = q1 - (1.5 * iqr)

    df = df.loc[(df["duration"] < upper_limit) & (df["duration"] > lower_limit)]

    high_price_data = df[df['price'] > 80000]

    # Duplica questi esempi per bilanciare il dataset
    df = pd.concat([df, high_price_data], axis=0)
    df = pd.concat([df, high_price_data], axis=0)
    df = pd.concat([df, high_price_data], axis=0)
    # Suddivisione con label
    X = df.drop(["price"], axis=1).values
    y = df['price'].values.reshape(-1, 1)  # Reshape di y per essere 2D
    num_above_100000 = np.sum(y > 100000)
    print("Numero biglietti sopra i 100000 euro: ", num_above_100000)

    num_instances = y.shape
    print("Numero di istanze in y: ", num_instances, X.shape)
    # Divisione dei dati in training e test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    under_80 = 0
    over_80 = 0
    for i in range(len(y_train)):
        if y_train[i] > 80000:
            over_80 += 1
        else:
            under_80 += 1
    # Numero di istanze in y
    print("Numero di prezzi sopra gli 80000 euro: ", over_80)
    print("Numero di prezzi sotto gli 80000 euro: ", under_80)

    mmscaler_X = MinMaxScaler()
    X_train = mmscaler_X.fit_transform(X_train)
    X_val = mmscaler_X.transform(X_val)
    X_test = mmscaler_X.transform(X_test)

    mmscaler_y_min = MinMaxScaler()
    mmscaler_y = StandardScaler()
    y_train = mmscaler_y_min.fit_transform(y_train)
    y_val = mmscaler_y_min.transform(y_val)
    y_test = mmscaler_y_min.transform(y_test)

    return X_train, y_train, X_val, y_val, X_test, y_test, mmscaler_X, mmscaler_y_min

