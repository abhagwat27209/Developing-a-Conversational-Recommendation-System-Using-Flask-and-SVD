from flask import Flask, request, render_template, jsonify
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
import random

app = Flask(__name__)

# Load movies and ratings data
movies_data = pd.read_csv('movies.csv')
movies_data = movies_data.rename(columns={'movieId': 'movie_id', 'genre': 'genres'})  # Adjust based on actual column names
ratings_data = pd.read_csv('ratings.csv')
ratings_data.columns = ['user_id', 'movie_id', 'rating', 'timestamp']

# SVD model for recommendation
def build_svd_model():
    user_movie_matrix = ratings_data.pivot(index='user_id', columns='movie_id', values='rating').fillna(0)
    svd = TruncatedSVD(n_components=2)
    svd_matrix = svd.fit_transform(user_movie_matrix)
    return svd, svd_matrix

svd, svd_matrix = build_svd_model()

# Track recommendations and feedback
user_feedback = {}
recommended_movies = []

# Recommend one movie at a time with personalized feedback
def recommend_movie(genre, feedback=None):
    genre_movies = movies_data[(movies_data['genres'] == genre) & (~movies_data['movie_id'].isin(recommended_movies))]
    
    if feedback:
        # Adjust recommendations based on feedback
        if feedback == 'liked':
            genre_movies = genre_movies.sample(frac=1)
        elif feedback == 'disliked':
            genre_movies = genre_movies.sample(frac=1)

    if not genre_movies.empty:
        recommended_movie = genre_movies.sample(1)
        recommended_movies.append(recommended_movie.iloc[0]['movie_id'])
        return recommended_movie.iloc[0]['title']
    else:
        return "No more movies to recommend in this genre."

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_recommendation', methods=['POST'])
def get_recommendation():
    user_input = request.form['user_input']
    genre = user_input
    recommendation = recommend_movie(genre)
    
    # Add human-like responses
    if recommendation == "No more movies to recommend in this genre.":
        response = f"Looks like you've watched all the good ones in {genre}. Maybe explore a different genre?"
    else:
        response_phrases = [
            f"How about giving '{recommendation}' a try? It’s a great {genre} movie!",
            f"I think you'll enjoy '{recommendation}' in the {genre} category.",
            f"Here's a suggestion: '{recommendation}'. Let me know if it’s a good match!",
            f"How about this one? '{recommendation}' is quite popular in {genre}."
        ]
        response = random.choice(response_phrases)
    
    return jsonify(response)

@app.route('/submit_feedback', methods=['POST'])
def submit_feedback():
    feedback = request.form['feedback']
    genre = request.form['genre']
    
    # Save feedback in user_feedback dictionary
    user_feedback[genre] = feedback
    
    # Response to feedback
    if feedback.lower() == "liked":
        response_phrases = [
            "I'm glad you liked it! I'll find more like that.",
            "Fantastic! I’ll keep recommending similar movies.",
            "Awesome! Let’s keep up the good vibes with more like it!"
        ]
    elif feedback.lower() == "disliked":
        response_phrases = [
            "Noted. Let me find something different this time.",
            "Thanks for the feedback! I'll adjust my recommendations.",
            "Got it. Let's switch it up and try a different type of movie."
        ]
    else:
        response_phrases = ["Thank you for the feedback!"]

    recommendation = recommend_movie(genre, feedback)
    response = f"{random.choice(response_phrases)} How about '{recommendation}' next?"

    return jsonify({'status': response, 'next_recommendation': recommendation})

if __name__ == "__main__":
    app.run(debug=True)
