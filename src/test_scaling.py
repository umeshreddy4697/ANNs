from load_data import load_close_prices
from windowing import create_windows
from preprocess import scale_data

prices = load_close_prices("AAPL")
X, y = create_windows(prices, window_size=5)

print("X shape before scaling:", X.shape)
print("y shape before scaling:", y.shape)

Xs, ys, _, _ = scale_data(X, y)

print("X shape after scaling:", Xs.shape)
print("y shape after scaling:", ys.shape)