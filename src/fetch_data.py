import yfinance as yf

ticker = "APPL"
df = yf.download(ticker, period="1y")

print(df.head())
