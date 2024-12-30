from flask import Flask, render_template, request, session, redirect, url_for
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import uuid

app = Flask(__name__)
app.secret_key = 'your_secret_key'
app.config['SESSION_PERMANENT'] = False
app.config['SESSION_TYPE'] = 'filesystem'

# Load Pickled Data
with open('mmodel.pkl', 'rb') as model_file:
    model_data = pickle.load(model_file)

data = model_data['data']
tfidf_vectorizer = model_data['tfidf_vectorizer']
tfidf_matrix = model_data['tfidf_matrix']

# Temporary storage for recommendations
TEMP_RECOMMENDATION_STORE = {}


def get_top_books_by_theme(rank):
    top_books_by_theme = data[data['theme_rank'] == rank]
    return top_books_by_theme[['slno', 'title', 'author', 'rating', 'number of ratings',
                               'topic', 'synopsis_sentiment', 'synopsis']]


def recommend_books_cosine_similarity(selected_books, data):
    data['combined_features'] = data.apply(
        lambda row: f"{row['synopsis']} {row['rating']} {row['synopsis_sentiment']}", axis=1
    )
    selected_books['combined_features'] = selected_books.apply(
        lambda row: f"{row['synopsis']} {row['rating']} {row['synopsis_sentiment']}", axis=1
    )

    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(data['combined_features'])
    selected_matrix = tfidf.transform(selected_books['combined_features'])

    cosine_sim = cosine_similarity(selected_matrix, tfidf_matrix)
    avg_similarity = cosine_sim.mean(axis=0)

    data['similarity_score'] = avg_similarity
    recommended_books = data[~data['title'].isin(selected_books['title'])].copy()
    recommended_books = recommended_books.sort_values(by='similarity_score', ascending=False)
    return recommended_books[['title', 'author', 'rating', 'topic', 'synopsis', 'similarity_score']].to_dict(
        orient='records')


@app.route('/')
def index():
    if 'page_refresh_count' not in session:
        session['page_refresh_count'] = 1
    else:
        session['page_refresh_count'] = (session['page_refresh_count'] + 1) % 10 or 1

    page_refresh_count = session['page_refresh_count']
    top_books_for_page = get_top_books_by_theme(page_refresh_count)

    if top_books_for_page.empty:
        top_books_for_page = data.sample(n=5)

    initial_books = top_books_for_page.sample(n=min(5, len(top_books_for_page))).to_dict(orient='records')
    return render_template('index.html', books=initial_books)


@app.route('/recommend', methods=['POST', 'GET'])
def recommendation():
    if request.method == 'POST':
        # Handle initial recommendation request
        selected_book_ids = request.form.getlist('selected_books')

        if not selected_book_ids:
            return "<h1>No books selected. Please go back and select books.</h1>"

        selected_books = data[data['title'].astype(str).isin(selected_book_ids)]

        if selected_books.empty:
            return "<h1>No valid books found from selection.</h1>"

        recommendations = recommend_books_cosine_similarity(selected_books, data)

        # Store recommendations with a unique session ID
        session_id = str(uuid.uuid4())
        TEMP_RECOMMENDATION_STORE[session_id] = recommendations

        session['recommendation_session_id'] = session_id
        session['current_index'] = 0
        session.modified = True

        return redirect(url_for('get_next_recommendation'))

    # If it's a GET request, redirect to the next recommendation
    return redirect(url_for('get_next_recommendation'))


@app.route('/next-recommendation')
def get_next_recommendation():
    session_id = session.get('recommendation_session_id')
    if not session_id or session_id not in TEMP_RECOMMENDATION_STORE:
        return "<h1>No recommendations available. Please select books again.</h1>"

    recommendations = TEMP_RECOMMENDATION_STORE[session_id]
    current_index = session.get('current_index', 0)

    if current_index >= len(recommendations):
        return "<h1>No more recommendations available. Try selecting different books.</h1>"

    recommended_book = recommendations[current_index]

    # Increment the index for next time
    session['current_index'] = current_index + 1
    session.modified = True

    recommendation_details = {
        'title': recommended_book.get('title', 'Unknown Title'),
        'author': recommended_book.get('author', 'Unknown Author'),
        'rating': recommended_book.get('rating', 'No Rating'),
        'synopsis': recommended_book.get('synopsis', 'No Synopsis Available'),
        'genre': recommended_book.get('topic', 'Unknown Genre'),  # Changed from 'genre' to 'topic' to match your data
    }

    return render_template('recommendation.html', recommendation=recommendation_details)


if __name__ == '__main__':
    app.run(debug=True, port=3000)