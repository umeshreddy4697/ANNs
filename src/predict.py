import torch
import joblib
import numpy as np

from model import StockNN
from load_data import load_feature_matrix, load_close_prices


# ---------------- CONFIG ----------------
TICKER = "SAATVIKGL.NS"
WINDOW_SIZE = 5
NUM_FEATURES = 3   # [return, MA_distance, volatility]
MODEL_PATH = "stock_nn_weights.pth"
PERIOD = "60d"
# --------------------------------------


# 1. Load model
model = StockNN(WINDOW_SIZE * NUM_FEATURES)
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

# 2. Load scalers
scaler_X = joblib.load("scaler_X.pkl")
scaler_y = joblib.load("scaler_y.pkl")

# 3. Load latest FEATURE MATRIX
features = load_feature_matrix(TICKER, period=PERIOD)

# Take last WINDOW_SIZE rows
last_window = features[-WINDOW_SIZE:]

print("Last window (features):")
print(last_window)

# 4. Prepare input (flatten → scale → tensor)
last_window = last_window.reshape(1, -1)
last_window_scaled = scaler_X.transform(last_window)
last_window_tensor = torch.tensor(last_window_scaled, dtype=torch.float32)

# 5. Predict return
with torch.no_grad():
    pred_scaled = model(last_window_tensor)

pred_return = scaler_y.inverse_transform(pred_scaled.numpy())[0, 0]

# 6. Convert return → price
last_price = load_close_prices(TICKER, period="2d")[-1]
predicted_price = last_price * (1 + pred_return)

print("Last close price:", last_price)
print("Predicted return:", pred_return)
print("Predicted next-day close:", predicted_price)

# ---------- FEATURE SENSITIVITY TEST ----------
print("\n--- Feature sensitivity test ---")

# Copy original window
test_window = last_window.copy()

# 1) Zero out MA distance (feature index 1)
test_window[:, 1] = 0.0

# Flatten + scale
test_flat = test_window.reshape(1, -1)
test_scaled = scaler_X.transform(test_flat)
test_tensor = torch.tensor(test_scaled, dtype=torch.float32)

with torch.no_grad():
    perturbed_pred = model(test_tensor)

perturbed_return = scaler_y.inverse_transform(
    perturbed_pred.numpy()
)[0, 0]

print("Original predicted return:", pred_return)
print("After zeroing MA_distance:", perturbed_return)

# ---------- VOLATILITY SENSITIVITY TEST ----------
print("\n--- Volatility sensitivity test ---")

# Copy original window again (important)
test_window_vol = last_window.copy()

# Zero out volatility (feature index = 2)
test_window_vol[:, 2] = 0.0

# Flatten + scale
test_flat_vol = test_window_vol.reshape(1, -1)
test_scaled_vol = scaler_X.transform(test_flat_vol)
test_tensor_vol = torch.tensor(test_scaled_vol, dtype=torch.float32)

with torch.no_grad():
    vol_pred = model(test_tensor_vol)

vol_pred_return = scaler_y.inverse_transform(
    vol_pred.numpy()
)[0, 0]

print("Original predicted return:", pred_return)
print("After zeroing volatility:", vol_pred_return)