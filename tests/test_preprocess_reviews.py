import os
import pandas as pd
import pytest
from src import preprocess_reviews

# Test data directory
TEST_DATA_DIR = "tests/test_data"
os.makedirs(TEST_DATA_DIR, exist_ok=True)

@pytest.fixture
def sample_csv_files(tmp_path):
    """Create temporary CSV files for testing."""
    # Sample review data
    data1 = pd.DataFrame({
        "review_text": ["Good app", "Bad service"],
        "rating": [5, 1],
        "date": ["2023-01-01", "2023-01-02"],
        "bank_name": ["CBE", "CBE"],
        "source": ["Google Play", "Google Play"]
    })

    data2 = pd.DataFrame({
        "review_text": ["Nice UI", "Bad UI", None],
        "rating": [4, 2, 3],
        "date": ["2023-01-03", "2023-01-04", "2023-01-05"],
        "bank_name": ["BOA", "BOA", "BOA"],
        "source": ["Google Play", "Google Play", "Google Play"]
    })

    file1 = tmp_path / "test1.csv"
    file2 = tmp_path / "test2.csv"
    data1.to_csv(file1, index=False)
    data2.to_csv(file2, index=False)

    return str(tmp_path)

def test_load_and_combine_raw_csvs(sample_csv_files):
    """Test loading and combining multiple CSVs."""
    df = preprocess_reviews.load_and_combine_raw_csvs(f"{sample_csv_files}/*.csv")
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 5  # 2 from file1, 3 from file2

def test_clean_reviews_removes_duplicates_and_nulls():
    """Test cleaning logic: drops duplicates and nulls, formats date."""
    raw_data = pd.DataFrame({
        "review_text": ["Good", "Good", None],
        "rating": [5, 5, 3],
        "date": ["2023-01-01", "2023-01-01", "2023-01-03"],
        "bank_name": ["Dashen", "Dashen", "Dashen"],
        "source": ["Google", "Google", "Google"]
    })

    cleaned = preprocess_reviews.clean_reviews(raw_data)
    
    # Check duplicates removed
    assert len(cleaned) == 1

    # Check required columns
    assert all(col in cleaned.columns for col in ['review', 'rating', 'date', 'bank', 'source'])

    # Check date format
    assert cleaned['date'].iloc[0] == "2023-01-01"

def test_clean_reviews_drops_invalid_dates():
    """Test if invalid dates are coerced and dropped."""
    df = pd.DataFrame({
        "review_text": ["Valid", "Invalid"],
        "rating": [4, 2],
        "date": ["2023-01-01", "not-a-date"],
        "bank_name": ["BOA", "BOA"],
        "source": ["Google", "Google"]
    })

    cleaned = preprocess_reviews.clean_reviews(df)
    assert len(cleaned) == 1
    assert cleaned['date'].iloc[0] == "2023-01-01"
