import torch
import torch.nn as nn
import torch.optim as optim
import joblib

from load_data import load_feature_matrix
from windowing import create_windows
from preprocess import scale_data
from split import train_test_split_time
from model import StockNN


# ---------------- CONFIG ----------------
TICKER = "SAATVIKGL.NS"
WINDOW_SIZE = 5
NUM_FEATURES = 3        # [return, ma_distance, volatility]
EPOCHS = 1000
LEARNING_RATE = 0.001
PERIOD = "60d"
# --------------------------------------


# 1. Load feature matrix
features = load_feature_matrix(TICKER, period=PERIOD)
# features shape: (N, 3)

# 2. Create sliding windows
X, y = create_windows(features, WINDOW_SIZE)
# X shape: (samples, WINDOW_SIZE, NUM_FEATURES)
# y shape: (samples, 1)

# 3. Flatten X for ANN
X = X.reshape(X.shape[0], -1)
# X shape: (samples, WINDOW_SIZE * NUM_FEATURES)

# 4. Time-based train-test split (VERY IMPORTANT)
X_train, X_test, y_train, y_test = train_test_split_time(X, y)

# 5. Scale using TRAIN data only
X_train, y_train, scaler_X, scaler_y = scale_data(X_train, y_train)
X_test = scaler_X.transform(X_test)
y_test = scaler_y.transform(y_test)

# 6. Convert to torch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)

X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# 7. Define model
model = StockNN(WINDOW_SIZE * NUM_FEATURES)

# 8. Loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# 9. Training loop
for epoch in range(EPOCHS):
    model.train()

    optimizer.zero_grad()
    predictions = model(X_train)
    loss = criterion(predictions, y_train)

    loss.backward()
    optimizer.step()

    if epoch % 5 == 0:
        print(f"Epoch {epoch} | Train MSE: {loss.item():.6f}")

# 10. Evaluation
model.eval()
with torch.no_grad():
    test_preds = model(X_test)
    test_loss = criterion(test_preds, y_test)

print("Final Test MSE:", test_loss.item())

# 11. Save model + scalers
torch.save(model.state_dict(), "stock_nn_weights.pth")
joblib.dump(scaler_X, "scaler_X.pkl")
joblib.dump(scaler_y, "scaler_y.pkl")

print("Model and scalers saved")