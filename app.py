from flask import Flask, render_template, request
import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load data
movies = pd.read_csv('movies.csv')
movies.fillna('', inplace=True)
movies['combined_features'] = (
    movies['title'] + ' ' + 
    movies['genres'] + ' ' +
    movies['overview'] + ' ' +
    movies['keywords']
)

# Load vectorizer and transform
with open('tfidf_vectorizer.pkl', 'rb') as f:
    tfidf_vectorizer = pickle.load(f)
tfidf_matrix = tfidf_vectorizer.transform(movies['combined_features'])

# Load KNN model
with open('movierecomendation.pkl', 'rb') as f:
    knn_model = pickle.load(f)

def get_recommendations(movie_title, n=5):
    try:
        index = movies[movies['title'].str.lower() == movie_title.lower()].index[0]
    except IndexError:
        return ["Movie not found in database."]
    
    vector = tfidf_matrix[index]
    distances, indices = knn_model.kneighbors(vector, n_neighbors=n + 1)
    
    rec_indices = indices.flatten()[1:]
    recommendations = movies.iloc[rec_indices]['title'].tolist()
    return recommendations

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend_movie():
    movie = request.form['movie']
    recommendations = get_recommendations(movie)
    return render_template('index.html', movie=movie, recommendations=recommendations)

if __name__ == '__main__':
    app.run(debug=True)
