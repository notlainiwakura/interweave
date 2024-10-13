# vector_db.py

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from database import SessionLocal
from models import User

# Function to create an embedding from a user's interests
def create_user_embedding(user):
    """
    Create a vector embedding from a user's interest scores.

    Args:
    user (User): The User object from the database.

    Returns:
    numpy.ndarray: The vector embedding of the user.
    """
    # List of interest fields
    interest_fields = [
        'sci_fi_movies', 'cooking', 'hiking', 'travel', 'reading', 'sports',
        'music', 'photography', 'gardening', 'video_games', 'board_games',
        'diy_projects', 'volunteering', 'movies', 'podcasts', 'social_media',
        'pets', 'workout', 'meditation', 'travel_adventure',
        'music_instruments', 'arts_crafts'
    ]

    # Retrieve interest scores, defaulting to 0 if None
    interest_scores = [getattr(user, field) or 0 for field in interest_fields]

    # Convert to numpy array
    embedding = np.array(interest_scores, dtype=float)

    return embedding

# Function to get user vectors for clustering
def get_user_vectors():
    """
    Retrieve all users' interest vectors and related data.

    Returns:
    - user_vectors (np.ndarray): Array of user interest vectors.
    - user_ids (list): List of user IDs corresponding to the vectors.
    - user_data (dict): Mapping of user IDs to their data (e.g., username).
    - interest_fields (list): List of interest fields used.
    """
    db_session = SessionLocal()
    users = db_session.query(User).all()
    db_session.close()

    # List of interest fields
    interest_fields = [
        'sci_fi_movies', 'cooking', 'hiking', 'travel', 'reading', 'sports',
        'music', 'photography', 'gardening', 'video_games', 'board_games',
        'diy_projects', 'volunteering', 'movies', 'podcasts', 'social_media',
        'pets', 'workout', 'meditation', 'travel_adventure',
        'music_instruments', 'arts_crafts'
    ]

    user_vectors = []
    user_ids = []
    user_data = {}  # To map user_id to username and other data if needed

    for user in users:
        interest_scores = [getattr(user, field) or 0 for field in interest_fields]
        user_vectors.append(interest_scores)
        user_ids.append(user.id)
        user_data[user.id] = {'username': user.username}

    return np.array(user_vectors), user_ids, user_data, interest_fields

# Function to find similar users using K-Means clustering
def find_similar_users_clustering(target_user_id, user_clusters):
    """
    Finds similar users based on cluster assignments.

    Args:
    - target_user_id (int): The ID of the target user.
    - user_clusters (dict): Mapping of user IDs to cluster labels.

    Returns:
    - List of user IDs of similar users.
    """
    target_cluster = user_clusters.get(target_user_id)
    if target_cluster is None:
        return []

    # Find all users in the same cluster, excluding the target user
    similar_user_ids = [
        user_id for user_id, cluster in user_clusters.items()
        if cluster == target_cluster and user_id != target_user_id
    ]

    return similar_user_ids

# Function to build the vector database (for completeness)
def build_vector_db():
    """
    Build the vector database from all users in the database.
    """
    global vector_db
    vector_db = {}  # Reset the vector database

    db_session = SessionLocal()
    users = db_session.query(User).all()

    for user in users:
        embedding = create_user_embedding(user)
        vector_db[user.id] = {
            'embedding': embedding,
            'username': user.username
        }

    db_session.close()

# Optional function to find similar users using cosine similarity
def find_similar_users_cosine(target_user_id, top_n=5):
    """
    Finds the top N most similar users to the target user using cosine similarity.

    Args:
    - target_user_id (int): The ID of the target user.
    - top_n (int): Number of similar users to return.

    Returns:
    - List of dictionaries containing user IDs and usernames of the most similar users.
    """
    db_session = SessionLocal()
    users = db_session.query(User).all()
    db_session.close()

    # Create embeddings
    user_embeddings = {}
    interest_fields = [
        'sci_fi_movies', 'cooking', 'hiking', 'travel', 'reading', 'sports',
        'music', 'photography', 'gardening', 'video_games', 'board_games',
        'diy_projects', 'volunteering', 'movies', 'podcasts', 'social_media',
        'pets', 'workout', 'meditation', 'travel_adventure',
        'music_instruments', 'arts_crafts'
    ]

    target_embedding = None
    for user in users:
        embedding = np.array([getattr(user, field) or 0 for field in interest_fields], dtype=float)
        user_embeddings[user.id] = {'embedding': embedding, 'username': user.username}
        if user.id == target_user_id:
            target_embedding = embedding

    if target_embedding is None:
        return []

    similarities = []
    for user_id, data in user_embeddings.items():
        if user_id == target_user_id:
            continue  # Skip comparing with oneself

        similarity = cosine_similarity(
            target_embedding.reshape(1, -1),
            data['embedding'].reshape(1, -1)
        )[0][0]
        similarities.append({
            'user_id': user_id,
            'username': data['username'],
            'similarity': similarity
        })

    # Sort by similarity in descending order
    similarities.sort(key=lambda x: x['similarity'], reverse=True)

    # Return the top N similar users
    similar_users = similarities[:top_n]

    return similar_users
