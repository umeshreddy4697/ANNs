import yfinance as yf

ticker = "AAPL"
df = yf.download(ticker, period="1y")

print(df.head())
