import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List, Dict
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

class ThematicAnalyzer:
    """
    A class for extracting keywords and assigning themes to reviews using TF-IDF and rule-based mapping.
    """

    def __init__(self, df: pd.DataFrame, text_col: str = 'cleaned_review', bank_col: str = 'bank'):
        self.df = df.copy()
        self.text_col = text_col
        self.bank_col = bank_col
        self.tfidf = None
        self.tfidf_matrix = None
        self.tfidf_df = None

        # Rule-based keyword to theme map
        self.keyword_theme_map = {
            'login': 'Account Access',
            'password': 'Account Access',
            'transfer': 'Transaction Issues',
            'delay': 'Transaction Issues',
            'update': 'App Functionality',
            'crash': 'App Functionality',
            'support': 'Customer Service',
            'slow': 'Performance',
            'balance': 'Account Info',
            'ui': 'User Experience',
            'design': 'User Experience'
        }

    def clean_data(self):
        """
        Drop rows with missing or empty text values.
        """
        self.df = self.df[self.df[self.text_col].notna()]
        self.df = self.df[self.df[self.text_col].str.strip() != ""]

    def compute_tfidf(self, max_features: int = 200):
        """
        Compute TF-IDF matrix for the review text.
        """
        self.tfidf = TfidfVectorizer(ngram_range=(1, 2), stop_words='english', max_features=max_features)
        self.tfidf_matrix = self.tfidf.fit_transform(self.df[self.text_col])
        self.tfidf_df = pd.DataFrame(self.tfidf_matrix.toarray(), columns=self.tfidf.get_feature_names_out())
        self.tfidf_df[self.bank_col] = self.df[self.bank_col].values

    def extract_top_keywords_per_bank(self, top_n: int = 10) -> Dict[str, List[str]]:
        """
        Extract top N TF-IDF keywords per bank.

        Returns:
            dict: Bank → list of top keywords
        """
        bank_keywords = {}
        for bank in self.df[self.bank_col].unique():
            subset = self.tfidf_df[self.tfidf_df[self.bank_col] == bank].drop(self.bank_col, axis=1)
            top_keywords = subset.mean().sort_values(ascending=False).head(top_n)
            bank_keywords[bank] = top_keywords.index.tolist()

        ## Print Top keywords per banck    
        for bank, keywords in bank_keywords.items():
            print(f"\n🔑 Top Keywords for {bank}:")
            print(", ".join(keywords))
        return bank_keywords

    def assign_themes(self):
        """
        Assign themes to reviews based on keyword matches.
        """
        def map_theme(text):
            for keyword, theme in self.keyword_theme_map.items():
                if keyword in text.lower():
                    return theme
            return 'Other'

        self.df['theme'] = self.df[self.text_col].apply(map_theme)

    def get_theme_distribution(self) -> pd.DataFrame:
        """
        Return counts of each theme per bank.
        """
        return self.df.groupby([self.bank_col, 'theme']).size().reset_index(name='count')

    def save_results(self, output_path: str = '../data/processed/review_sentiments_themes.csv'):
        """
        Save the DataFrame with assigned themes.
        """
        self.df.to_csv(output_path, index=False)
        
    def extract_keywords_by_sentiment(self, sentiment_col: str, label: str, top_n: int = 30) -> List[str]:
        """
        Extract top TF-IDF keywords for a specific sentiment label (e.g. positive or negative).
        
        Args:
            sentiment_col (str): Column name containing sentiment labels.
            label (str): Sentiment label to filter on.
            top_n (int): Number of keywords to return.

        Returns:
            List[str]: Top keywords
        """
        filtered = self.df[self.df[sentiment_col] == label]
        if filtered.empty:
            return []

        # Fit new TF-IDF on filtered reviews
        tfidf = TfidfVectorizer(ngram_range=(1, 2), stop_words='english', max_features=200)
        matrix = tfidf.fit_transform(filtered[self.text_col])
        tfidf_df = pd.DataFrame(matrix.toarray(), columns=tfidf.get_feature_names_out())
        return tfidf_df.mean().sort_values(ascending=False).head(top_n).index.tolist()


    def plot_sentiment_distribution(self, sentiment_col: str):
        """
        Plot bar chart showing sentiment distribution.
        """
        plt.figure(figsize=(6, 4))
        sns.countplot(x=self.df[sentiment_col], palette='coolwarm')
        plt.title(f"Sentiment Distribution ({sentiment_col})")
        plt.xlabel("Sentiment")
        plt.ylabel("Review Count")
        plt.grid(True)
        plt.tight_layout()
        plt.show()


    def plot_wordcloud(self, words: List[str], title: str):
        """
        Plot a word cloud for a list of words.

        Args:
            words (List[str]): List of keywords
            title (str): Plot title
        """
        text = ' '.join(words)
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(title)
        plt.tight_layout()
        plt.show()