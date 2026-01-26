from load_data import load_close_prices
from windowing import create_windows

prices = load_close_prices("AAPL")
WINDOW_SIZE = 5

X, y = create_windows(prices, WINDOW_SIZE)

print("Total samples:", len(X))
print("One sample X:", X[0])
print("One sample y:", y[0])