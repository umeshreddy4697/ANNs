from load_data import load_close_prices
from windowing import create_windows
from preprocess import scale_data
from split import train_test_split_time

prices = load_close_prices("AAPL")
X, y = create_windows(prices, window_size=5)

X, y, _, _ = scale_data(X, y)

X_train, X_test, y_train, y_test = train_test_split_time(X, y, test_ratio=0.2)

print("Total samples:", len(X))
print("Train samples:", len(X_train))
print("Test samples:", len(X_test))

print("Last train X sample:", X_train[-1])
print("First test X sample:", X_test[0])