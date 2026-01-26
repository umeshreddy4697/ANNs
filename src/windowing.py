import numpy as np


def create_windows(feature_matrix, window_size):
    """
    feature_matrix shape: (N, num_features)

    X shape: (samples, window_size, num_features)
    y shape: (samples, 1)  → next-day RETURN only
    """
    X, y = [], []

    for i in range(len(feature_matrix) - window_size):
        X.append(feature_matrix[i:i + window_size])

        # Target = NEXT-DAY RETURN (first column only)
        y.append(feature_matrix[i + window_size][0])

    return np.array(X), np.array(y).reshape(-1, 1)