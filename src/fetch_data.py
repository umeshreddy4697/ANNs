import yfinance as yf

ticker = "APPLE"
df = yf.download(ticker, period="1y")

print(df.head())
