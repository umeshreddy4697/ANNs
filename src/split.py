def train_test_split_time(X, y, test_ratio=0.2):
    """
    Splits time-series data by order.
    Past -> train
    Future -> test
    """
    split_index = int(len(X) * (1 - test_ratio))

    X_train = X[:split_index]
    y_train = y[:split_index]

    X_test = X[split_index:]
    y_test = y[split_index:]

    return X_train, X_test, y_train, y_test