# src/scraper.py

import csv
import logging
import os
from datetime import datetime
from google_play_scraper import Sort, reviews

# Setup logging
logging.basicConfig(
    filename='logs/scraper.log', 
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)

BANK_APPS = {
    'CBE': 'com.combanketh.mobilebanking',
    'BOA': 'com.boa.boaMobileBanking',
    'Dashen': 'com.dashen.dashensuperapp'
}

DATA_DIR = "data/raw"
os.makedirs(DATA_DIR, exist_ok=True)


def fetch_reviews(app_id, bank_name, count=500):
    """
    Fetch reviews from the Google Play Store for a given bank.
    
    Args:
        app_id (str): Google Play App ID
        bank_name (str): Human-readable name for the bank
        count (int): Number of reviews to fetch
    
    Returns:
        List[dict]: List of cleaned reviews
    """
    try:
        results, _ = reviews(
            app_id,
            lang='en',
            country='us',
            sort=Sort.NEWEST,
            count=count,
            filter_score_with=None
        )
        logging.info(f"✅ Successfully fetched {len(results)} reviews for {bank_name}")
        return [
            {
                'review_text': entry['content'],
                'rating': entry['score'],
                'date': entry['at'].strftime('%Y-%m-%d'),
                'bank_name': bank_name,
                'source': 'Google Play'
            }
            for entry in results
        ]
    except Exception as e:
        logging.error(f"❌ Error fetching reviews for {bank_name}: {e}")
        return []


def save_reviews_to_csv(reviews_data, bank_name):
    """
    Save reviews to CSV file.

    Args:
        reviews_data (list): List of review dictionaries
        bank_name (str): Bank name to use in the filename
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"{DATA_DIR}/{bank_name.lower()}_reviews_{timestamp}.csv"

    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=['review_text', 'rating', 'date', 'bank_name', 'source'])
        writer.writeheader()
        for row in reviews_data:
            writer.writerow(row)
    logging.info(f"💾 Saved {len(reviews_data)} reviews to {filename}")


def scrape_all_banks():
    """Scrape reviews for all target banks."""
    for bank, app_id in BANK_APPS.items():
        data = fetch_reviews(app_id, bank_name=bank, count=500)
        if data:
            save_reviews_to_csv(data, bank_name=bank)


if __name__ == "__main__":
    scrape_all_banks()
