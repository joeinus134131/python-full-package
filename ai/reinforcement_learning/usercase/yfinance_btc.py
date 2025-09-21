import yfinance as yf

btc_df = yf.download("BTC-USD", start="2020-01-01", end="2025-01-01", progress=True)
print(btc_df.head())
print(btc_df.tail())