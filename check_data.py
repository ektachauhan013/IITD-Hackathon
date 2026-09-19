import pandas as pd
import os

pd.set_option('display.max_columns', 200)
pd.set_option('display.width', 250)

files = ['train.csv', 'dev.csv', 'dev_winners.csv', 'test.csv']

for f in files:
    if not os.path.exists(f):
        print(f"❌ Missing: {f}")
        continue

    df = pd.read_csv(f)

    print("=" * 120)
    print(f"FILE: {f}")
    print(f"SHAPE: {df.shape}")
    print(f"COLUMNS: {df.columns.tolist()}")

    print("\nHEAD 3:")
    print(df.head(3).to_string())

    print("\nDTYPES:")
    print(df.dtypes)

    print("\nMISSING (top 25):")
    print(df.isna().sum().sort_values(ascending=False).head(25))

    print("\nUNIQUE COUNTS (object/category columns):")
    for c in df.select_dtypes(include=['object', 'category']).columns:
        print(f"  {c}: {df[c].nunique()} unique")

    print("=" * 120)