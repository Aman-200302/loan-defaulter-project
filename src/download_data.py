"""
Downloads the dataset on CI where Kaggle credentials aren't available.
Uses a direct URL fallback.
"""
import pandas as pd
import os

os.makedirs('data/raw', exist_ok=True)

# Use the dataset directly from a public mirror
url = "https://raw.githubusercontent.com/dsrscientist/dataset1/master/credit_risk_dataset.csv"

print("Downloading dataset...")
df = pd.read_csv(url)
df.to_csv('data/raw/credit_risk_dataset.csv', index=False)
print(f"Downloaded {len(df)} rows → data/raw/credit_risk_dataset.csv")