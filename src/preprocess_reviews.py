# src/preprocess_reviews.py

import os
import pandas as pd
from glob import glob

RAW_DATA_DIR = "data/raw"
OUTPUT_FILE = "data/clean/cleaned_reviews.csv"
os.makedirs("data/clean", exist_ok=True)


def load_and_combine_raw_csvs(path_pattern="data/raw/*.csv"):
    """Load all raw review CSVs and combine into one DataFrame."""
    csv_files = glob(path_pattern)
    df_list = []
    for file in csv_files:
        try:
            df = pd.read_csv(file)
            df_list.append(df)
        except Exception as e:
            print(f"❌ Could not read {file}: {e}")
    combined_df = pd.concat(df_list, ignore_index=True)
    print(f"✅ Combined {len(csv_files)} files with total {len(combined_df)} reviews")
    return combined_df


def clean_reviews(df):
    """Clean and normalize the reviews DataFrame."""
    # Drop duplicates
    initial_len = len(df)
    df = df.drop_duplicates()
    print(f"🧹 Removed {initial_len - len(df)} duplicate rows")

    # Drop rows with missing critical values
    df = df.dropna(subset=["review_text", "rating", "date"])
    print(f"📉 Dropped rows with missing review, rating or date. Remaining: {len(df)}")

    # Normalize date column
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date'])
    df['date'] = df['date'].dt.strftime('%Y-%m-%d')

    # Keep required columns only
    df = df[['review_text', 'rating', 'date', 'bank_name', 'source']]
    df.columns = ['review', 'rating', 'date', 'bank', 'source']
    print("📁 Final cleaned dataset has:")
    print(df.dtypes)
    print(df.head())
    print(len(df))
    return df


def run_preprocessing():
    df = load_and_combine_raw_csvs()
    cleaned = clean_reviews(df)
    cleaned.to_csv(OUTPUT_FILE, index=False)
    print(f"✅ Saved cleaned dataset to {OUTPUT_FILE}")


if __name__ == "__main__":
    run_preprocessing()
