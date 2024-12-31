import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
import numpy as np

# Sample user ratings for the movies (user-item matrix)
ratings_data = {
    "User": ["User1", "User1", "User1", "User2", "User2", "User2", "User3", "User3", "User3", "User4", "User4", "User4"],
    "Movie": ["Armageddon", "Billy Elliot", "Eragon", "Armageddon", "Billy Elliot", "Eragon", "Armageddon", "Billy Elliot", "Eragon", "Armageddon", 
              "Billy Elliot", "Eragon"],
    "Rating": [4, 3, 5, 2, 4, 4, 3, 2, 4, 5, 3, 2],
}

# Create a DataFrame for user ratings
ratings_df = pd.DataFrame(ratings_data)

# Create user-item matrix (rows = users, columns = movies)
user_item_matrix = ratings_df.pivot_table(index='User', columns='Movie', values='Rating').fillna(0)

# Compute cosine similarity between movies (item-based collaborative filtering)
cosine_sim = cosine_similarity(user_item_matrix.T)  # Transpose to compare movies
cosine_sim_df = pd.DataFrame(cosine_sim, index=user_item_matrix.columns, columns=user_item_matrix.columns)

# Function to recommend movies based on user input
def recommend_movies_for_user(user_name, num_recommendations=5):
    if user_name not in ratings_df['User'].values:
        return f"User '{user_name}' not found in the dataset. Please check your input."
    
    # Get the movies rated by the selected user
    user_ratings = user_item_matrix.loc[user_name]
    
    # For movies rated by the user, calculate the predicted ratings for other movies
    predicted_ratings = {}
    
    for movie in user_item_matrix.columns:
        if user_ratings[movie] == 0:  # Movie not rated by the user
            # Get the similarity scores for this movie with all the other movies rated by the user
            similar_movies = cosine_sim_df[movie]
            
            # Calculate predicted rating as the weighted sum of rated movies' ratings
            weighted_ratings = sum(user_ratings * similar_movies) / sum(similar_movies) if sum(similar_movies) != 0 else 0
            predicted_ratings[movie] = weighted_ratings

    # Sort the predicted ratings in descending order and return top N recommendations
    recommended_movies = sorted(predicted_ratings.items(), key=lambda x: x[1], reverse=True)[:num_recommendations]
    
    return recommended_movies

# Display available users and movies
print("Available users in the dataset:")
for user in ratings_df["User"].unique():
    print(f"- {user}")

# Assign user name directly instead of using input()
selected_user = "User1"  # Change this value for different inputs

# Generate recommendations
recommendations = recommend_movies_for_user(selected_user)

# Display recommendations
if isinstance(recommendations, str):  # If an error occurred
    print(recommendations)
else:
    print(f"\nTop movie recommendations for '{selected_user}':")
    for movie, score in recommendations:
        print(f"- {movie} (Predicted Rating: {score:.2f})")