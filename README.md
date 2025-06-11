# 🔍 AI-Powered Mobile Banking Review Insights

> End-to-end data pipeline to extract strategic insights from Ethiopian mobile banking app reviews.


## 🧭 Project Overview

This project explores the use of **AI and NLP techniques** to extract meaningful **user insights** from **Google Play reviews** of three major Ethiopian banks:

* 💳 Commercial Bank of Ethiopia (CBE)
* 🏛️ Bank of Abyssinia (BOA)
* 💰 Dashen Bank

The goal is to deliver actionable recommendations for **mobile banking app improvement** using real customer sentiment data.


## 🎯 Business Objective

Omega Consultancy, a fintech advisory firm, needs:

* 🔍 Analysis of app store reviews for major banks
* 📊 Strategic insights based on sentiment and user themes
* 🗃 Enterprise-compliant storage for structured data
* 🛠 Clear, data-driven UX improvement recommendations


## ✅ Tasks & Deliverables

| Task   | Description                                                 | Status |
| ------ | ----------------------------------------------------------- | ------ |
| Task 1 | Scrape, clean, and preprocess reviews                       | ✅ Done |
| Task 2 | Perform sentiment & thematic analysis                       | ✅ Done |
| Task 3 | Store cleaned data in Oracle DB                             | ✅ Done |
| Task 4 | Extract insights, create visualizations, write final report | ✅ Done |


## 🛠 Tools & Technologies

* `Python`, `pandas`, `nltk`, `spacy`, `transformers`
* `TextBlob`, `VADER`, `DistilBERT`
* `TF-IDF`, `Seaborn`, `Matplotlib`, `WordCloud`
* `Oracle XE` (via Docker) for enterprise data storage
* `oracledb` Python client for DB operations
* Git/GitHub: version control and PR-based workflow


## 🧠 Project Structure

```
📁 project-root/
│
├── data/                  # Raw and processed review data
│   ├── raw/               # Scraped CSVs
|   ├── clean/             # Cleaned reveiw
│   └── processed/         # Cleaned reviews with sentiment & themes
│
├── notebooks/             # Exploratory & production-ready notebooks
│   ├── exploratory_EDA.ipynb
│   ├── sentiment_thematic_analysis.ipynb
│   └── review_handler_demo.ipynb
│
|   ├── scripts/                   # Python scripts
│   ├── run_oracle_db_handler.py
│   ├── run_scraper.py
|   └── scheduler.py
|
├── src/                   # Modular Python scripts
│   ├── scraper.py
│   ├── preprocess_reviews.py
│   ├── sentiment_analysis.py
│   ├── oracle_db_handler.py
│   └── thematic_analysis.py
│
├── reports/
│   └── Transforming-User-Reviews-into-App-Strategy.pdf
│
├── .github/workflows/     # GitHub Actions for CI
│   └── ci.yml
│
|   tests/
│    └── __init__py
├── requirements.txt
├── .gitignore
└── README.md
```


## 🔍 Key Insights Summary

| Bank   | Strengths            | Pain Points                    |
| ------ | -------------------- | ------------------------------ |
| CBE    | Simpler UX           | Transaction failures           |
| BOA    | Clean layout         | Crashes, login/password errors |
| Dashen | Beautiful UI praised | App crashes, slowness          |

> Visualizations, bar charts, and word clouds are available in the `notebooks/sentiment_thematic_analysis.ipynb`.


## 📂 Oracle Database

* Docker container: Oracle XE 21c
* Tables: `banks`, `reviews`
* Inserted: >1,200 cleaned reviews
* CRUD features:

  * Insert, update, delete reviews
  * Update sentiment by review or bank
  * Get reviews by bank

> See: `src/oracle_db_handler.py`, `run_oracle_db_handler.py` and `review_handler_demo.ipynb`


## 🧪 How to Run

### 1. 📦 Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. 🐳 Start Oracle XE (Docker)

```bash
docker run -d --name oracle-xe \
  -p 1521:1521 -p 5500:5500 \
  -e ORACLE_PWD=YourSecurePassword \
  container-registry.oracle.com/database/express:21.3.0-xe
```

### 3. 🧠 Run Notebooks

* `notebooks/exploratory_EDA.ipynb`
* `notebooks/sentiment_thematic_analysis.ipynb`
* `notebooks/review_handler_demo_demo.ipynb`


## 📈 Final Deliverables

* ✅ Cleaned and enriched review dataset
* ✅ Oracle DB schema with 1,200+ reviews
* ✅ Sentiment + theme analysis with plots
* ✅ Strategic recommendations per bank
* ✅ Final Report: `Transforming-User-Reviews-into-App-Strategy.pdf`


## 📌 Authors & Attribution

**Developed by:**

> \[Mikias Worku] — Data Engineer, ML Specialist


## 📖 License

MIT License — open source, reproducible, and scalable for learning or extension.

