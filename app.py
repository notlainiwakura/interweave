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
from database import SessionLocal, init_db  # Import the database session and init_db
from utils import compute_user_embedding, deduce_interest_and_relevance
from vector_db import add_user_embedding, find_similar_users, user_metadata
import numpy as np
import json

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'  # Replace with a secure secret key
app.config['SESSION_TYPE'] = 'filesystem'
Session(app)

# Initialize Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'  # Redirect unauthorized users to the login page

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
@app.route('/logout')
@login_required
def logout():
    logout_user()
    session.pop('user_profile', None)
    return redirect(url_for('login'))

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
    user.sci_fi_movies = data.get('sci_fi_movies')
    user.cooking = data.get('cooking')
    user.hiking = data.get('hiking')
    db_session.commit()
    db_session.close()
    return jsonify({'status': 'success'})

# Chat page route
@app.route('/chat')
@login_required
def chat():
    return render_template('chat.html', username=current_user.username)

# Chat API route for handling chat messages
@app.route('/chat_api', methods=['POST'])
@login_required
def chat_api():
    data = request.get_json()
    message = data.get('message')
    current_question = data.get('current_question', '')

    if not current_question:
        if message.lower() in ['sci-fi movies', 'scifi movies', 'sci fi movies']:
            current_question = 'sci_fi_movies'
        elif message.lower() == 'cooking':
            current_question = 'cooking'
        elif message.lower() == 'hiking':
            current_question = 'hiking'

        if current_question:
            reply = f'From 1 to 10 how much do you {"are interested in" if current_question == "sci_fi_movies" else "like"} {current_question.replace("_", " ")}?'
        else:
            reply = "I can ask you about your interests in sci-fi movies, cooking, and hiking. Which one would you like to rate?"
    else:
        try:
            value = float(message)
            if 1 <= value <= 10:
                db_session = SessionLocal()
                user = db_session.query(User).get(current_user.id)
                setattr(user, current_question, value)
                db_session.commit()
                db_session.close()
                reply = f'Great! Your {current_question.replace("_", " ")} interest has been updated to {value}. Would you like to rate another interest?'
                current_question = ''  # Reset question for the next input
            else:
                reply = 'Please provide a number between 1 and 10.'
        except ValueError:
            reply = 'Please provide a valid number between 1 and 10.'

    return jsonify({'reply': reply, 'current_question': current_question})



# Route to find similar users
@app.route('/find_similar_users', methods=['POST'])
@login_required
def find_similar_users_route():
    user_profile = session.get('user_profile')
    if not user_profile:
        return jsonify({'similar_users': []})
    # Compute embedding with current data
    compute_user_embedding(user_profile)
    # Add or update the user's embedding in the vector database
    add_user_embedding(current_user.id, user_profile['embedding'], current_user.username)
    # Update user's interests and embedding in the database
    db_session = SessionLocal()
    user = db_session.query(User).get(current_user.id)
    user.interests = json.dumps(user_profile['interests'])
    user.embedding = user_profile['embedding'].tobytes()
    db_session.commit()
    db_session.close()
    # Find similar users
    similar_user_ids = find_similar_users(user_profile['embedding'])
    # Exclude the current user
    similar_user_ids = [uid for uid in similar_user_ids if uid != current_user.id]
    # Retrieve usernames from the database
    similar_users_info = []
    db_session = SessionLocal()
    for uid in similar_user_ids:
        similar_user = db_session.query(User).get(uid)
        if similar_user:
            similar_users_info.append({'username': similar_user.username})
    db_session.close()
    return jsonify({'similar_users': similar_users_info})

# Initialize the database before the app starts
if __name__ == '__main__':
    init_db()  # Initialize the database (create tables if they don't exist)
    app.run(debug=True)
