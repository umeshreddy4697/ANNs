import yfinance as yf
import numpy as np


def load_close_prices(ticker, period="60d"):
    """
    Fetch historical closing prices.
    Returns a 1D numpy array of floats.
    """
    df = yf.download(ticker, period=period)
    return df["Close"].values.astype(float).flatten()


def load_returns(ticker, period="60d"):
    """
    Daily returns:
    return[t] = (price[t] - price[t-1]) / price[t-1]
    """
    prices = load_close_prices(ticker, period)
    returns = (prices[1:] - prices[:-1]) / prices[:-1]
    return returns.astype(float)


def load_feature_matrix(ticker, period="60d", ma_window=5, vol_window=5):
    """
    Feature matrix per day:
    [ return, MA_distance, rolling_volatility ]

    Shape:
    (num_samples, 3)
    """
    prices = load_close_prices(ticker, period)

    # returns aligned to prices[1:]
    returns = (prices[1:] - prices[:-1]) / prices[:-1]
    prices = prices[1:]

    features = []
    start = max(ma_window, vol_window)

    for i in range(start, len(prices)):
        ma = prices[i - ma_window:i].mean()
        ma_dist = (prices[i] - ma) / ma
        vol = np.std(returns[i - vol_window:i])

        features.append([
            float(returns[i]),
            float(ma_dist),
            float(vol)
        ])

    return np.array(features, dtype=np.float32)


# ---------------- SANITY TEST ----------------
if __name__ == "__main__":
    ticker = "SAATVIKGL.NS"

    prices = load_close_prices(ticker)
    returns = load_returns(ticker)
    features = load_feature_matrix(ticker)

    print("Ticker:", ticker)
    print("Last 5 prices:", prices[-5:])
    print("Last 5 returns:", returns[-5:])
    print("Feature matrix shape:", features.shape)
    print("Last feature row [return, ma_dist, vol]:", features[-1])