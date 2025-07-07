import pandas as pd
import os

url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
df = pd.read_html(url, header=0)[0]
symbols = df["Symbol"].drop_duplicates()
os.makedirs("data", exist_ok=True)
with open("data/sp500_symbols.txt", "w") as f:
    for s in symbols:
        f.write(f"{s}\n")
print("SP500 symbols saved to data/sp500_symbols.txt")
