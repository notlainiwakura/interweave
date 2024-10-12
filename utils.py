import numpy as np
import json
from sklearn.metrics.pairwise import cosine_similarity


# Function to compute user embedding from their interests or activity
def compute_user_embedding(user_profile):
    """
    This function computes a user embedding based on the user's interests.
    If the user profile contains interest-related data, it generates a
    vector embedding.
    """
    interests = user_profile.get('interests', {})
    # In a real case, you'd map these interests to a vector space
    if not interests:
        return np.zeros(128)  # Return a zero vector if no interests exist

    # Simple vectorization of interests based on interest score and relevance
    embedding = np.array([interest['interest'] + interest['relevance'] for interest in interests.values()])

    # Normalize the embedding (optional)
    norm = np.linalg.norm(embedding)
    if norm == 0:
        return embedding  # Avoid division by zero
    return embedding / norm


# Function to deduce user interest and relevance from a message
def deduce_interest_and_relevance(message):
    """
    Takes a user message as input and returns a tuple:
    (interest, interest_score, relevance_score).

    This is a simple example function. In a more advanced implementation,
    NLP techniques (such as a pre-trained model) can be used to extract
    topics of interest from the message.
    """
    # For this example, we'll check for predefined keywords (this could be replaced with NLP models)
    possible_interests = {
        'sports': ['football', 'soccer', 'basketball', 'tennis'],
        'music': ['guitar', 'piano', 'singing', 'rock', 'pop'],
        'movies': ['film', 'cinema', 'actor', 'director'],
        # Add more categories and keywords as needed
    }

    for interest, keywords in possible_interests.items():
        for keyword in keywords:
            if keyword.lower() in message.lower():
                # Assign arbitrary scores (this can be more sophisticated)
                interest_score = np.random.uniform(0.7, 1.0)  # Interest score (0-1)
                relevance_score = np.random.uniform(0.5, 1.0)  # Relevance score (0-1)
                return interest, interest_score, relevance_score

    # No interest found
    return None, 0.0, 0.0


# Function to calculate similarity between two embeddings
def calculate_similarity(embedding1, embedding2):
    """
    Computes the cosine similarity between two embeddings.
    """
    if embedding1 is None or embedding2 is None:
        return 0.0

    # Reshape embeddings if necessary
    embedding1 = embedding1.reshape(1, -1)
    embedding2 = embedding2.reshape(1, -1)

    return cosine_similarity(embedding1, embedding2)[0][0]


# Function to parse user metadata for display
def user_metadata(user):
    """
    Converts user data to a simple dict format for display or processing.
    """
    return {
        'username': user.username,
        'email': user.email,
        'interests': json.loads(user.interests) if user.interests else {}
    }


# Example of a utility to clean or preprocess text data
def preprocess_message(message):
    """
    Preprocesses the message by lowercasing, stripping extra whitespace,
    and removing any non-alphanumeric characters.
    """
    return ''.join(e for e in message.lower().strip() if e.isalnum() or e.isspace())


# Example of JSON-safe helper to manage serialization
def jsonify_embeddings(embedding):
    """
    Converts a numpy embedding into a JSON-serializable list.
    """
    if isinstance(embedding, np.ndarray):
        return embedding.tolist()
    return embedding
