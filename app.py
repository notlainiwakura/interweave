import logging
from datetime import timedelta
import os
from dotenv import load_dotenv
import time
import random

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from flask_session import Session
from flask import Flask, render_template, request, redirect, url_for, jsonify, session

from flask_login import (
    LoginManager,
    login_user,
    login_required,
    logout_user,
    current_user
)
from models import User
from database import SessionLocal, init_db
from utils import compute_user_embedding, deduce_interest_and_relevance
from vector_db import get_user_vectors, find_similar_users_clustering

import numpy as np
import json

# New imports for Hugging Face API and error handling
from huggingface_hub import InferenceClient
from huggingface_hub.utils import HfHubHTTPError
from functools import lru_cache

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY')
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_FILE_DIR'] = './.flask_session/'
app.config['SESSION_PERMANENT'] = False
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(minutes=5)
Session(app)

# Initialize Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Initialize Hugging Face Inference Client
hf_token = os.getenv('HUGGINGFACE_TOKEN')
llama_model = "tiiuae/falcon-7b-instruct"
client = InferenceClient(llama_model, token=hf_token)


# Helper function for retrying API calls
def retry_api_call(func, max_retries=3, delay=1):
    for i in range(max_retries):
        try:
            return func()
        except HfHubHTTPError as e:
            if i == max_retries - 1:
                raise e
            time.sleep(delay * (2 ** i))  # Exponential backoff


# Caching decorator for API calls
@lru_cache(maxsize=100)
def cached_text_generation(prompt):
    return client.text_generation(prompt, max_new_tokens=200)


# Fallback method for generating responses
def fallback_text_generation(prompt):
    # This is a simple rule-based fallback. You might want to implement a more sophisticated local model.
    if "greeting" in prompt.lower():
        return "Hello! I'm here to help create your user profile. Shall we begin?"
    elif "ready to proceed" in prompt.lower():
        return "Great! Let's start with your first interest. How much do you enjoy reading on a scale of 1 to 10?"
    elif "next question" in prompt.lower():
        interests = ["sports", "music", "cooking", "travel", "movies"]
        return f"How much do you enjoy {random.choice(interests)} on a scale of 1 to 10?"
    elif "profile complete" in prompt.lower():
        return "Your profile is now complete. Is there anything else you'd like to know?"
    else:
        return "I understand. Let's move on to the next question."


# User loader callback for Flask-Login
@login_manager.user_loader
def load_user(user_id):
    db_session = SessionLocal()
    user = db_session.query(User).get(int(user_id))
    db_session.close()
    return user


# Home route redirects to log in or chat
@app.route('/')
def index():
    if current_user.is_authenticated:
        return redirect(url_for('chat'))
    else:
        return redirect(url_for('login'))


# User registration route
@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('chat'))
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']

        db_session = SessionLocal()
        # Check if username or email already exists
        if db_session.query(User).filter_by(username=username).first():
            db_session.close()
            return render_template('register.html', message='Username already exists.')
        if db_session.query(User).filter_by(email=email).first():
            db_session.close()
            return render_template('register.html', message='Email already registered.')

        # Create new user
        new_user = User(username=username, email=email)
        new_user.set_password(password)
        db_session.add(new_user)
        db_session.commit()
        db_session.close()
        return redirect(url_for('login'))
    else:
        return render_template('register.html')


# User login route
@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('chat'))
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        db_session = SessionLocal()
        user = db_session.query(User).filter_by(username=username).first()
        db_session.close()

        if user and user.check_password(password):
            login_user(user)
            # Load user's existing interests and embedding into session
            user_profile = {
                'interests': json.loads(user.interests) if user.interests else {},
                'embedding': np.frombuffer(user.embedding) if user.embedding else None
            }
            session['user_profile'] = user_profile
            return redirect(url_for('chat'))
        else:
            return render_template('login.html', message='Invalid username or password.')
    else:
        return render_template('login.html')


# User logout route
@app.route('/logout', methods=['GET', 'POST'])
@login_required
def logout():
    logout_user()
    session.clear()  # Clear the session data
    return jsonify({"success": True, "redirect": url_for('login')}), 200


@app.route('/profile')
@login_required
def profile():
    return render_template('profile.html', user=current_user)


@app.route('/update_profile', methods=['POST'])
@login_required
def update_profile():
    data = request.get_json()
    db_session = SessionLocal()
    user = db_session.query(User).get(current_user.id)

    # Map frontend field names to database column names
    field_mapping = {
        'sci_fi_movies': 'sci_fi_movies',
        'cooking': 'cooking',
        'hiking': 'hiking',
        'travel': 'travel',
        'reading': 'reading',
        'sports': 'sports',
        'music': 'music',
        'photography': 'photography',
        'gardening': 'gardening',
        'video_games': 'video_games',
        'board_games': 'board_games',
        'diy_projects': 'diy_projects',
        'volunteering': 'volunteering',
        'movies': 'movies',
        'podcasts': 'podcasts',
        'social_media': 'social_media',
        'pets': 'pets',
        'workout': 'workout',
        'meditation': 'meditation',
        'travel_adventure': 'travel_adventure',
        'music_instruments': 'music_instruments',
        'arts_crafts': 'arts_crafts'
    }

    # Update user's interests based on mapping
    for frontend_field, db_field in field_mapping.items():
        setattr(user, db_field, data.get(frontend_field))

    try:
        db_session.commit()
    except Exception as e:
        print(f"Error committing changes: {e}")
        return jsonify({'status': 'error', 'message': 'Error updating profile'}), 500

    db_session.close()

    return jsonify({'status': 'success'})


# Chat page route
@app.route('/chat')
@login_required
def chat():
    return render_template('chat.html', username=current_user.username)


# Modified chat_api route
@app.route('/chat_api', methods=['POST'])
@login_required
def chat_api():
    data = request.get_json()
    message = data.get('message')
    conversation_state = data.get('conversation_state', 'start')

    interest_fields = [
        'sci_fi_movies', 'cooking', 'hiking', 'travel', 'reading', 'sports',
        'music', 'photography', 'gardening', 'video_games', 'board_games',
        'diy_projects', 'volunteering', 'movies', 'podcasts', 'social_media',
        'pets', 'workout', 'meditation', 'travel_adventure',
        'music_instruments', 'arts_crafts'
    ]

    def post_process(response):
        # Remove any leading/trailing whitespace
        response = response.strip()
        # Capitalize the first letter
        return response[0].upper() + response[1:] if response else ""

    try:
        if conversation_state == 'start':
            prompt = """You are an AI assistant helping to create a user profile. Generate a friendly greeting and casually ask if the user is ready to begin talking about their interests. Keep the conversation light, engaging, and casual, as if you're having a relaxed conversation with a friend. Avoid sounding too formal."""
            reply = post_process(retry_api_call(lambda: cached_text_generation(prompt)))
            conversation_state = 'ready_check'

        elif conversation_state == 'ready_check':
            prompt = f"""The user responded '{message}' to your greeting. Now, if they seem ready to proceed, casually ask them about one of their interests from this list: {', '.join(interest_fields)}. Feel free to mix in casual talk like 'By the way,' or 'Just curious,' to make the conversation flow naturally. If the user isn't ready, respond politely and offer to come back later. If you're unsure, ask for clarification, but keep it light and friendly."""
            response = post_process(retry_api_call(lambda: cached_text_generation(prompt)))

            if any(word in response.lower() for word in ["let's begin", "first interest", "tell me about"]):
                conversation_state = 'asking_questions'
            elif any(word in response.lower() for word in ["not ready", "come back later", "another time"]):
                conversation_state = 'end'

            reply = response

        elif conversation_state == 'asking_questions':
            prompt = f"""Based on the user's response '{message}', casually ask them another question about one of their interests from {interest_fields}. For example, if they've already mentioned one, ask how much they enjoy that on a scale of 1-10, or ask them about a new interest that hasn't been discussed. Keep the tone friendly and conversational, as if you're chatting casually. Avoid repeating their message back word-for-word, and be sure to add a touch of casual talk.

    After generating your response, on a new line, add:
    INTERNAL_NOTE: Interest: [interest_name], Value: [1-10]"""

            response = retry_api_call(lambda: cached_text_generation(prompt))

            # Split the response into the user-facing part and the internal note
            user_response, internal_note = response.split('INTERNAL_NOTE:', 1)
            user_response = post_process(user_response)

            # Parse the internal note to update user profile
            interest = None
            value = None
            if 'Interest:' in internal_note and 'Value:' in internal_note:
                interest_part, value_part = internal_note.split(',')
                interest = interest_part.split('Interest:')[1].strip().lower()
                try:
                    value = float(value_part.split('Value:')[1].strip())
                except ValueError:
                    value = None

            if interest and value is not None and interest in interest_fields:
                db_session = SessionLocal()
                user = db_session.query(User).get(current_user.id)
                setattr(user, interest, value)
                db_session.commit()
                db_session.close()

            if "profile complete" in user_response.lower() or "all interests covered" in user_response.lower():
                conversation_state = 'end'

            reply = user_response

        else:
            prompt = """Wrap up the conversation by informing the user that their profile is complete, but do so in a friendly and conversational tone. Feel free to ask if there's anything else they need help with, and ensure that the conversation stays light and relaxed."""
            reply = post_process(retry_api_call(lambda: cached_text_generation(prompt)))

    except HfHubHTTPError as e:
        logging.error(f"Hugging Face API error: {str(e)}")
        reply = fallback_text_generation(prompt)

    return jsonify({
        'reply': reply,
        'conversation_state': conversation_state
    })


@app.route('/connections')
@login_required
def connections():
    # Get user vectors and data
    user_vectors, user_ids, user_data, interest_fields = get_user_vectors()

    # Check if there are enough users to perform clustering
    if len(user_vectors) < 2:
        # Not enough users to cluster
        similar_users = []
    else:
        # Preprocess the data
        scaler = StandardScaler()
        user_vectors_scaled = scaler.fit_transform(user_vectors)

        # Apply K-Means clustering
        k = 5  # Adjust the number of clusters as needed
        if len(user_vectors_scaled) < k:
            k = len(user_vectors_scaled)  # Ensure k is not greater than the number of users
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(user_vectors_scaled)
        cluster_labels = kmeans.labels_

        # Assign users to clusters
        user_clusters = dict(zip(user_ids, cluster_labels))

        # Find similar users for the current user
        target_user_id = current_user.id
        similar_user_ids = find_similar_users_clustering(target_user_id, user_clusters)

        # Get data of similar users
        similar_users = [
            {'id': user_id, 'username': user_data[user_id]['username']}
            for user_id in similar_user_ids
        ]

    # Render the template with similar users
    return render_template('connections.html', similar_users=similar_users)


# Route to find similar users
@app.route('/find_similar_users')
@login_required
def find_similar_users_route():
    # Get the updated interests from the session
    user_profile = session.get('user_profile')
    if not user_profile:
        return jsonify({'similar_users': []})

    # Update the user's interests in the database
    db_session = SessionLocal()
    user = db_session.query(User).get(current_user.id)

    # List of interest fields
    interest_fields = [
        'sci_fi_movies', 'cooking', 'hiking', 'travel', 'reading', 'sports',
        'music', 'photography', 'gardening', 'video_games', 'board_games',
        'diy_projects', 'volunteering', 'movies', 'podcasts', 'social_media',
        'pets', 'workout', 'meditation', 'travel_adventure',
        'music_instruments', 'arts_crafts'
    ]

    # Update user's interests
    for field in interest_fields:
        setattr(user, field, user_profile.get(field))
    db_session.commit()
    db_session.close()

    # Get all user vectors and data
    user_vectors, user_ids, user_data, _ = get_user_vectors()

    # Check if there are enough users to perform clustering
    if len(user_vectors) < 2:
        similar_users_info = []
    else:
        # Preprocess the data
        scaler = StandardScaler()
        user_vectors_scaled = scaler.fit_transform(user_vectors)

        # Apply K-Means clustering
        k = 5  # Adjust as needed
        if len(user_vectors_scaled) < k:
            k = len(user_vectors_scaled)
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(user_vectors_scaled)
        cluster_labels = kmeans.labels_

        # Assign users to clusters
        user_clusters = dict(zip(user_ids, cluster_labels))

        # Find similar users
        target_user_id = current_user.id
        similar_user_ids = find_similar_users_clustering(target_user_id, user_clusters)

        # Retrieve usernames of similar users
        similar_users_info = [
            {'username': user_data[user_id]['username']}
            for user_id in similar_user_ids
        ]

    # Return similar users as JSON
    return jsonify({'similar_users': similar_users_info})


# Initialize the database before the app starts
if __name__ == '__main__':
    init_db()  # Initialize the database (create tables if they don't exist)
    app.run(debug=True)