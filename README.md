# 👗 NLP Clothing Review Classifier and Recommendation App

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![NLP](https://img.shields.io/badge/NLP-GloVe%20%7C%20TF--IDF%20%7C%20BoW-blue?style=for-the-badge)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)

> End-to-end NLP pipeline classifying whether a clothing review recommends a product, comparing Bag-of-Words, GloVe unweighted embeddings, and TF-IDF weighted embeddings with Logistic Regression and 5-fold cross-validation. Best accuracy of 0.8915 using combined Title and Review text. Deployed as a Flask web application for real-time recommendation inference.

---

## 📌 Problem Statement

E-commerce platforms rely on customer reviews to drive product recommendations. This project builds an automated classification system to predict whether a clothing review represents a positive recommendation, and deploys the best model as a web application where new reviews are labelled in real time.

---

## 📂 Dataset

- **Source:** [Women's E-Commerce Clothing Reviews, Kaggle](https://www.kaggle.com/datasets/nicapotato/womens-ecommerce-clothing-reviews)
- **Size:** ~19,600 clothing reviews
- **Features used:** Review Title, Review Text, Recommended (binary: 0 = not recommended, 1 = recommended)

---

## 🗂️ Project Structure

```
clothing-nlp-classifier/
│
├── milestone1/
│   ├── task1.ipynb              # Text preprocessing pipeline and vocabulary building
│   └── task2_3.ipynb            # Feature representations and classification models
│
├── milestone2/
│   ├── app.py                   # Flask web application
│   ├── templates/
│   │   ├── index.html
│   │   ├── search_results.html
│   │   ├── clothing_details.html
│   │   ├── add_review.html
│   │   ├── confirm_review.html
│   │   └── category.html
│   └── static/
│       └── style.css
│
└── README.md
```

---

## 🔬 Methodology

### Part 1: Text Preprocessing Pipeline (task1.ipynb)

- Tokenisation using regex: `r"[a-zA-Z]+(?:[-'][a-zA-Z]+)?"`
- Lowercasing, removal of tokens shorter than 2 characters
- Stopword removal using custom stopword list
- Removed words appearing only once (term frequency filter)
- Removed top 20 most frequent words (document frequency filter)
- Built alphabetically sorted vocabulary saved as `vocab.txt` (format: `word:index`)
- Saved processed data as `processed.csv` and sparse count vectors as `count_vectors.txt`

### Part 2: Feature Representations (task2_3.ipynb)

Three feature representations generated for each review:

| Representation | Method |
|---|---|
| Bag of Words | Count vectors based on vocabulary from Part 1 |
| Unweighted Embeddings | GloVe pretrained vectors, averaged per review |
| TF-IDF Weighted Embeddings | GloVe vectors weighted by TF-IDF scores |

TruncatedSVD (300 components) applied for dimensionality reduction on count vectors.

### Part 3: Classification and Evaluation (task2_3.ipynb)

**Classifier:** Logistic Regression  
**Validation:** 5-fold stratified cross-validation  
**Metrics:** Accuracy, Precision, Recall, F1-Score, Confusion Matrix

Two experiments:

**Q1: Which feature representation performs best?**

| Feature Set | Mean Accuracy | Mean Precision | Mean Recall | Mean F1 |
|---|---|---|---|---|
| **Combined Title + Review** | **0.8915** | **0.8860** | **0.8860** | **0.8860** ✅ |
| Title Only | 0.8711 | 0.8644 | 0.8644 | 0.8644 |
| Review Only | 0.8668 | 0.8593 | 0.8593 | 0.8593 |
| Count Vector (BoW) | 0.8666 | 0.8592 | 0.8592 | 0.8592 |
| Unweighted Embedding | 0.8545 | 0.8438 | 0.8438 | 0.8438 |
| Weighted Embedding | 0.8487 | 0.8362 | 0.8362 | 0.8362 |

**Q2: Does adding more information improve accuracy?**

Yes. Combining Review Title and Review Text outperformed using either field alone, achieving the highest mean accuracy of 0.8915.

---

## 🌐 Flask Web Application (milestone2/)

A full shopping website built with Flask integrating the trained Logistic Regression classifier for real-time review labelling.

**Features:**
- Browse clothing items by category
- Keyword search with lemmatization (e.g. "dress" and "dresses" return same results)
- View full item details and all existing reviews
- Submit new reviews with title, description, and rating
- ML model predicts recommendation label (0/1) in real time on submission
- Reviewer can override the model prediction before confirming
- New reviews saved to `reviews.csv` and accessible via URL

**Routes:**

| Route | Method | Description |
|---|---|---|
| `/` | GET | Home page with search and categories |
| `/search` | POST | Keyword search with lemmatization |
| `/clothing/<item_id>` | GET | Item details and all reviews |
| `/add_review/<item_id>` | GET, POST | Submit a new review |
| `/confirm_review` | POST | Confirm and save review |
| `/category/<category_name>` | GET | Browse by category |

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| NLP | NLTK, GloVe, TF-IDF, CountVectorizer, Scikit-learn |
| Web Framework | Flask, Jinja2, HTML/CSS |
| Data Processing | Pandas, NumPy, Regex, TruncatedSVD |
| Visualisation | Matplotlib, Seaborn |

---

## ▶️ How to Run

### NLP Notebooks (Milestone 1)

```bash
# 1. Clone the repo
git clone https://github.com/tanishq19/clothing-nlp-classifier.git
cd clothing-nlp-classifier/milestone1

# 2. Download the dataset from Kaggle:
#    https://www.kaggle.com/datasets/nicapotato/womens-ecommerce-clothing-reviews
#    Place the CSV in the milestone1 directory

# 3. Install dependencies
pip install pandas numpy scikit-learn nltk gensim matplotlib seaborn jupyter

# 4. Run notebooks in order
jupyter notebook task1.ipynb
jupyter notebook task2_3.ipynb
```

### Flask Web App (Milestone 2)

```bash
cd milestone2

# Install dependencies
pip install flask pandas numpy scikit-learn nltk gensim

# Run the app
python app.py

# Visit http://localhost:5000 in your browser
```

> **Note:** The trained model files (`logistic_regression_model.pkl`, `tfidf_vectorizer.pkl`, `fasttext_model.bin`) and dataset CSV are not included. Run the Milestone 1 notebooks first to generate the model files.

---

## 🔗 Data Source

nicapotato. *Women's E-Commerce Clothing Reviews* [Data set]. Kaggle.  
https://www.kaggle.com/datasets/nicapotato/womens-ecommerce-clothing-reviews
