import yfinance as yf

data = yf.Ticker("AMGN").history(start="2024-11-27", end="2024-11-30", interval="1d")
print(data)
