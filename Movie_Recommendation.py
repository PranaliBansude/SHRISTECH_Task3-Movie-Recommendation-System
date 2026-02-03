# TASK 3: AI-Based Movie Recommendation System

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------------
# Load Dataset
# -------------------------------
movies = pd.read_csv("movies.csv")

# Keep only required columns
movies = movies[['title', 'genres']]
movies['genres'] = movies['genres'].fillna('')

print("\nDataset Loaded Successfully\n")
print(movies.head())

# -------------------------------
# Feature Extraction (TF-IDF)
# -------------------------------
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['genres'])

# -------------------------------
# Cosine Similarity
# -------------------------------
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# -------------------------------
# Recommendation Function
# -------------------------------
def recommend_movie(movie_title):
    if movie_title not in movies['title'].values:
        print("\nMovie not found in dataset!")
        return

    index = movies[movies['title'] == movie_title].index[0]
    similarity_scores = list(enumerate(cosine_sim[index]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

    print("\nRecommended Movies:")
    for i in similarity_scores[1:6]:
        print(movies.iloc[i[0]].title)

# -------------------------------
# User Input Interface
# -------------------------------
print("\nAI Movie Recommendation System")
movie_name = input("Enter movie name: ")

recommend_movie(movie_name)
