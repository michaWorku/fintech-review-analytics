
# Setup & connect to Oracle
from src.oracle_db_handler import OracleDBHandler
import pandas as pd


# Initialize connection (update with your credentials)
db = OracleDBHandler(user="system", password="YourSecurePassword", dsn="localhost:1521/XEPDB1")                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 
db.connect()

# Create tables (only run once)
db.create_schema()


# Insert sample reviews
# Load cleaned review dataset
df = pd.read_csv("../data/processed/review_sentiment_theme.csv")

# Insert reviews into Oracle
db.insert_reviews(df)

# Fetch Reviews
# Fetch top 5 reviews from all banks
db.fetch_reviews(limit=5)

# Fetch Reviews by Bank
# Get reviews for BOA
db.get_reviews_by_bank("BOA", limit=5)


# Update Sentiment by Review ID
# Change sentiment of review ID 10
db.update_sentiment_by_review_id(review_id=10, sentiment_label="neutral", sentiment_score=0.0)


# Update Sentiment by Bank
# Set all Dashen reviews to 'negative' for test
db.update_sentiment_by_bank_name(bank_name="Dashen", sentiment_label="negative", sentiment_score=-0.5)

# Update Theme
# Set all Dashen reviews to 'negative' for test
db.update_sentiment_by_bank_name(bank_name="Dashen", sentiment_label="negative", sentiment_score=-0.5)

# Delete Review
# Delete a single review (use with caution)
db.delete_review_by_id(review_id=10)

# Close connection
db.close()