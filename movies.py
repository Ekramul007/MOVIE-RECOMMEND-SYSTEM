import pandas as pd
import numpy as np
from numpy.linalg import norm

# Load the data
movies = pd.read_csv('movies.csv')

# Display the first few rows of the dataframe
print(movies.head())

# Prepare the data
# Use the correct column name for genres
movies['genres'] = movies['genres'].apply(lambda x: ' '.join(x.split(',')))

# Create a function to compute cosine similarity between two genre vectors
def cosine_similarity(vec1, vec2):
    # Convert genres into vector form
    words1 = vec1.split()
    words2 = vec2.split()
    
    # Create a set of all unique words
    all_words = set(words1).union(set(words2))
    
    # Create vectors for each genre list
    vec1_counts = np.array([words1.count(word) for word in all_words])
    vec2_counts = np.array([words2.count(word) for word in all_words])
    
    # Compute cosine similarity
    dot_product = np.dot(vec1_counts, vec2_counts)
    norm1 = norm(vec1_counts)
    norm2 = norm(vec2_counts)
    
    return dot_product / (norm1 * norm2) if norm1 and norm2 else 0

# Function to get movie recommendations based on a given movie title
def recommend_movies(title, movies=movies):
    # Check if the movie exists in the dataset
    if title not in movies['title'].values:
        return "Movie not found in the dataset."

    # Get the genre vector of the input movie
    movie_genre = movies.loc[movies['title'] == title, 'genres'].values[0]
    
    # Calculate similarity for all movies
    similarities = []
    for index, row in movies.iterrows():
        if row['title'] != title:
            other_genre = row['genres']
            sim_score = cosine_similarity(movie_genre, other_genre)
            similarities.append((row['title'], sim_score))
    
    # Sort movies based on similarity scores
    similarities = sorted(similarities, key=lambda x: x[1], reverse=True)
    
    # Get the top 10 most similar movies
    top_similar = similarities[:10]
    
    return [movie[0] for movie in top_similar]

# Function to get user input and recommend movies based on genre preference
def get_user_preference(movies=movies):
    print("Available genres:")
    genres = movies['genres'].unique()
    for i, genre in enumerate(genres):
        print(f"{i+1}. {genre}")
    
    choice = int(input("Enter the number corresponding to the genre you prefer: ")) - 1
    if choice < 0 or choice >= len(genres):
        print("Invalid choice. Please select a valid genre.")
        return

    preferred_genre = genres[choice]
    print(f"You chose: {preferred_genre}")

    # Filter movies by the selected genre
    filtered_movies = movies[movies['genres'].str.contains(preferred_genre, case=False)]
    
    if filtered_movies.empty:
        print("No movies found for the selected genre.")
        return

    # Ask for a movie title to base recommendations on
    print("Available movies in the selected genre:")
    for i, movie in enumerate(filtered_movies['title']):
        print(f"{i+1}. {movie}")
    
    movie_choice = int(input("Enter the number corresponding to the movie you like: ")) - 1
    if movie_choice < 0 or movie_choice >= len(filtered_movies):
        print("Invalid choice. Please select a valid movie.")
        return
    
    movie_title = filtered_movies.iloc[movie_choice]['title']
    recommendations = recommend_movies(movie_title)
    print(f"Recommendations based on '{movie_title}':")
    print(recommendations)

# Example usage
if __name__ == "__main__":
    get_user_preference()
