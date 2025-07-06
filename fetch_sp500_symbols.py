import pandas as pd

df = pd.read_csv("data/sp500_symbols.csv")
with open("data/sp500_symbols.txt", "w") as f:
    for symbol in df.iloc[:, 0]:
        f.write(f"{symbol}\n")
print("sp500_symbols.txt created with one symbol per line.")
