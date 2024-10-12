import json

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from database import SessionLocal
from models import User

# Assuming we are using an in-memory dictionary to simulate the vector database
vector_db = {}


# Function to add/update a user's embedding in the vector database
def add_user_embedding(user_id, embedding, username):
    """
    Add or update a user's embedding in the vector database.

    Args:
    user_id (int): The ID of the user.
    embedding (numpy.ndarray): The vector embedding of the user.
    username (str): The username of the user.
    """
    vector_db[user_id] = {
        'embedding': embedding,
        'username': username
    }


# Function to retrieve similar users based on cosine similarity
def find_similar_users(user_embedding, top_n=5):
    """
    Finds the top N most similar users to the given embedding using cosine similarity.

    Args:
    user_embedding (numpy.ndarray): The embedding to compare.
    top_n (int): Number of similar users to return.

    Returns:
    List of user IDs of the most similar users.
    """
    if not vector_db:
        return []  # Return an empty list if no users exist in the database

    similarities = []

    # Calculate cosine similarity between the provided embedding and all stored embeddings
    for user_id, data in vector_db.items():
        stored_embedding = data['embedding']
        similarity = cosine_similarity(stored_embedding.reshape(1, -1), user_embedding.reshape(1, -1))[0][0]
        similarities.append((user_id, similarity))

    # Sort by similarity in descending order
    similarities.sort(key=lambda x: x[1], reverse=True)

    # Return the top N similar users (excluding the current user)
    similar_users = [user_id for user_id, similarity in similarities[:top_n]]

    return similar_users


# Function to retrieve metadata of a specific user
def user_metadata(user_id):
    """
        Retrieve the metadata (username, email, interests) of a specific user from the SQL database.

        Args:
        user_id (int): The ID of the user.

        Returns:
        Dictionary containing the user's metadata or None if the user is not found.
        """
    db_session = SessionLocal()
    user = db_session.query(User).get(user_id)

    if user:
        # Assuming 'interests' is stored as a JSON string in the database
        interests = json.loads(user.interests) if user.interests else {}

        user_info = {
            'username': user.username,
            'email': user.email,
            'interests': interests
        }
        db_session.close()
        return user_info
    else:
        db_session.close()
        return None


# Optional utility function to remove a user from the vector database
def remove_user_embedding(user_id):
    """
    Removes a user's embedding from the vector database.

    Args:
    user_id (int): The ID of the user.
    """
    if user_id in vector_db:
        del vector_db[user_id]


# Function to check if the vector database contains a user embedding
def has_embedding(user_id):
    """
    Check if the vector database contains a user's embedding.

    Args:
    user_id (int): The ID of the user.

    Returns:
    True if the user exists, False otherwise.
    """
    return user_id in vector_db
