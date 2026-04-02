import os
from flask import Flask, render_template, request, redirect, url_for, flash
import pandas as pd
import pickle
from gensim.models import FastText
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)
app.secret_key = 'supersecretkey'  # for flash messages

# Load dataset
clothing_data = pd.read_csv('assignment3_II.csv')
# #drop duplicates
#clothing_data.drop_duplicates(subset=['Clothing ID'], keep='first', inplace=True)

# Load saved models
with open('logistic_regression_model.pkl', 'rb') as f:
    lr_model = pickle.load(f)
with open('tfidf_vectorizer.pkl', 'rb') as f:
    tfidf_vectorizer = pickle.load(f)
ft_model = FastText.load('fasttext_model.bin')

# Define the path to reviews.csv
reviews_file = 'reviews.csv'
if not os.path.exists(reviews_file):
    reviews_data = pd.DataFrame(columns=['Clothing ID', 'Review Title', 'Review Text', 'Rating', 'Recommendation'])
    reviews_data.to_csv(reviews_file, index=False)

# Helper function: Search algorithm with lemmatization
from nltk.stem import WordNetLemmatizer
import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')
lemmatizer = WordNetLemmatizer()

def search_items(keyword):
    keyword_lemmatized = lemmatizer.lemmatize(keyword.lower())
    results = clothing_data[clothing_data['Clothes Title'].str.contains(keyword_lemmatized, case=False, na=False)]
    return results

# Helper function: Predict recommendation using model
def generate_recommendation(review_text, review_title):
    try:
        combined_text = f"{review_title} {review_text}"
        vectorized_text = tfidf_vectorizer.transform([combined_text])
        prediction = lr_model.predict(vectorized_text)[0]
        return prediction
    except Exception as e:
        print(f"Error generating recommendation: {e}")
        return 0  # default to 0 if error occurs

@app.route('/')
def index():
    categories = clothing_data['Class Name'].unique()  # Extract unique categories
    return render_template('index.html', categories=categories)

@app.route('/search', methods=['GET', 'POST'])
def search():
    if request.method == 'POST':
        keyword = request.form['keyword']
        results = search_items(keyword)
       # results.drop_duplicates(subset=['Clothing ID'], keep='first', inplace=True)
        num_results = results.shape[0]
        return render_template('search_results.html', keyword=keyword, results=results, num_results=num_results)
    return redirect(url_for('index'))

@app.route('/clothing/<int:item_id>')
def clothing_details(item_id):
    # Get the item details
    item = clothing_data[clothing_data['Clothing ID'] == item_id].iloc[0]

    # Reload reviews_data to get the latest reviews
    reviews_data = pd.read_csv(reviews_file)

    # Get existing reviews from the original dataset
    existing_reviews = clothing_data[clothing_data['Clothing ID'] == item_id][['Title', 'Review Text', 'Rating', 'Recommended IND']]

    # Rename columns to match the new reviews DataFrame
    existing_reviews = existing_reviews.rename(columns={
        'Title': 'Review Title',
        'Recommended IND': 'Recommendation'
    })

    # Fill missing values if necessary
    existing_reviews.fillna({'Review Title': '', 'Review Text': '', 'Rating': 0, 'Recommendation': 0}, inplace=True)

    # Get new reviews from 'reviews.csv'
    new_reviews = reviews_data[reviews_data['Clothing ID'] == item_id][['Review Title', 'Review Text', 'Rating', 'Recommendation']]

    # Add a 'Source' column to indicate where the review came from
    existing_reviews['Source'] = 'Existing'
    new_reviews['Source'] = 'New'

    # Combine the reviews
    all_reviews = pd.concat([new_reviews, existing_reviews], ignore_index=True)

    # Sort reviews so that new reviews appear at the top
    all_reviews['Is New'] = all_reviews['Source'] == 'New'
    all_reviews = all_reviews.sort_values(by='Is New', ascending=False).drop('Is New', axis=1)

    return render_template('clothing_details.html', item=item, reviews=all_reviews)

@app.route('/add_review/<int:item_id>', methods=['GET', 'POST'])
def add_review(item_id):
    if request.method == 'POST':
        try:
            review_title = request.form['review_title']
            review_text = request.form['review_text']
            rating = int(request.form['rating'])

            # Generate recommendation using the model
            model_recommendation = generate_recommendation(review_text, review_title)

            # Store the review data
            review_data = {
                'Clothing ID': item_id,
                'Review Title': review_title,
                'Review Text': review_text,
                'Rating': rating,
                'Recommendation': model_recommendation
            }

            return render_template('confirm_review.html', review_data=review_data)

        except Exception as e:
            flash(f"Error submitting review: {e}")
            return redirect(url_for('clothing_details', item_id=item_id))

    item = clothing_data[clothing_data['Clothing ID'] == item_id].iloc[0]
    return render_template('add_review.html', item=item)

@app.route('/confirm_review', methods=['POST'])
def confirm_review():
    try:
        # Retrieve data from the form
        review_data = {
            'Clothing ID': int(request.form['Clothing ID']),
            'Review Title': request.form['Review Title'],
            'Review Text': request.form['Review Text'],
            'Rating': int(request.form['Rating']),
            'Recommendation': int(request.form['Recommendation'])
        }

        # Save the review to the CSV
        new_review = pd.DataFrame([review_data])
        new_review.to_csv(reviews_file, mode='a', header=False, index=False)

        return redirect(url_for('clothing_details', item_id=review_data['Clothing ID']))

    except Exception as e:
        print(f"Exception occurred: {e}")
        flash(f"Error saving review: {e}")
        return redirect(url_for('index'))

@app.route('/category/<category_name>')
def category_items(category_name):
    items = clothing_data[clothing_data['Class Name'] == category_name]
    return render_template('category.html', category=category_name, items=items)

if __name__ == '__main__':
    app.run(debug=True)
