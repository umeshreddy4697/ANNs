import torch

from load_data import load_close_prices
from windowing import create_windows
from preprocess import scale_data
from split import train_test_split_time
from model import StockNN


# Load and prepare data
prices = load_close_prices("AAPL")
X, y = create_windows(prices, window_size=5)
X, y, _, _ = scale_data(X, y)

X_train, X_test, y_train, y_test = train_test_split_time(X, y)

# Convert to torch tensors
X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32)

print("X_train tensor shape:", X_train_t.shape)
print("y_train tensor shape:", y_train_t.shape)

# Initialize model
model = StockNN(input_size=5)

# Forward pass test
sample_output = model(X_train_t[:2])
print("Model output shape:", sample_output.shape)