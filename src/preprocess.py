from sklearn.preprocessing import MinMaxScaler
import numpy as np


def scale_data(X, y):
    """
    Scale inputs (X) and target (y) separately.
    y is ALWAYS treated as a single-column target.
    """
    if X.ndim == 3:
        X = X.reshape(X.shape[0], -1)

    # Ensure y is (N,1)
    y = y.reshape(-1, 1)

    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y)

    return X_scaled, y_scaled, scaler_X, scaler_y