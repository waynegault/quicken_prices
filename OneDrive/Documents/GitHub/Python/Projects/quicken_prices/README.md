# quicken_prices
A python app that fetches and processes stock prices and exchange rates, converts currencies where necessary, logs detailed activity, and generates a Quicken-compatible CSV file. It also automates importing the CSV into Quicken and provides structured, human-readable terminal outputs summarizing its operations.

Wayne Gault
3/12/2024



pyinstaller QuickenPrices.py --clean --onefile  --name QuickenPrices --splash splash.png -i q.ico