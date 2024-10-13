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

    # Update fields based on the new columns added
    user.sci_fi_movies = data.get('sci_fi_movies')
    user.cooking = data.get('cooking')
    user.hiking = data.get('hiking')
    user.travel = data.get('travel')
    user.reading = data.get('reading')
    user.sports = data.get('sports')
    user.music = data.get('music')
    user.photography = data.get('photography')
    user.gardening = data.get('gardening')
    user.video_games = data.get('video_games')
    user.board_games = data.get('board_games')
    user.diy_projects = data.get('diy_projects')
    user.volunteering = data.get('volunteering')
    user.movies = data.get('movies')
    user.podcasts = data.get('podcasts')
    user.social_media = data.get('social_media')
    user.pets = data.get('pets')
    user.workout = data.get('workout')
    user.meditation = data.get('meditation')
    user.travel_adventure = data.get('travel_adventure')
    user.music_instruments = data.get('music_instruments')
    user.arts_crafts = data.get('arts_crafts')

    # Commit changes and close session
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
    conversation_state = data.get('conversation_state', 'start')
    current_question_index = data.get('current_question_index', None)

    # Template for the questions
    question_template = "From 1 to 10, how much are you interested in {}?"

    # Define the list of questions
    questions = [
        {'field': 'sci_fi_movies', 'label': 'sci-fi movies'},
        {'field': 'cooking', 'label': 'cooking'},
        {'field': 'hiking', 'label': 'hiking'},
        {'field': 'travel', 'label': 'travel'},
        {'field': 'reading', 'label': 'reading'},
        {'field': 'sports', 'label': 'sports'},
        {'field': 'music', 'label': 'music'},
        {'field': 'photography', 'label': 'photography'},
        {'field': 'gardening', 'label': 'gardening'},
        {'field': 'video_games', 'label': 'video games'},
        {'field': 'board_games', 'label': 'board games'},
        {'field': 'diy_projects', 'label': 'DIY projects'},
        {'field': 'volunteering', 'label': 'volunteering'},
        {'field': 'movies', 'label': 'movies'},
        {'field': 'podcasts', 'label': 'podcasts'},
        {'field': 'social_media', 'label': 'social media'},
        {'field': 'pets', 'label': 'pets'},
        {'field': 'workout', 'label': 'working out'},
        {'field': 'meditation', 'label': 'meditation'},
        {'field': 'travel_adventure', 'label': 'adventure travel'},
        {'field': 'bucket_list', 'label': 'bucket list'},
        {'field': 'music_instruments', 'label': 'playing musical instruments'},
        {'field': 'arts_crafts', 'label': 'arts and crafts'}
    ]

    if conversation_state == 'start':
        reply = ("Let's create your profile now. I will ask you a series of questions, "
                 "please rank your level of interest from 1 to 10. If you like you can click on the Profile button and manually adjust the values at any time. Would you like to start? y/n")
        conversation_state = 'ready_check'

    elif conversation_state == 'ready_check':
        if message.lower() == 'y':
            # Start with the first question
            current_question_index = 0
            reply = question_template.format(questions[current_question_index]['label'])
            conversation_state = 'asking_questions'
        elif message.lower() == 'n':
            reply = "No problem, please come back whenever you are ready, I'll be right here!"
            conversation_state = 'end'
        else:
            reply = "I'm sorry, I didn't understand that. Please answer with 'y' for yes or 'n' for no."

    elif conversation_state == 'asking_questions':
        try:
            value = float(message)
            if 1 <= value <= 10:
                # Ensure current_question_index is valid
                if current_question_index is None or current_question_index >= len(questions):
                    reply = 'An error occurred. Please start over.'
                    conversation_state = 'start'
                    current_question_index = None
                else:
                    current_question = questions[current_question_index]['field']

                    # Update the user's interest in the database
                    db_session = SessionLocal()
                    user = db_session.query(User).get(current_user.id)
                    setattr(user, current_question, value)
                    db_session.commit()
                    db_session.close()

                    # Prepare the reply
                    field_label = questions[current_question_index]['label']
                    reply = f'Great! Your interest in {field_label} has been updated to {value}. '

                    # Move to the next question
                    current_question_index += 1
                    if current_question_index < len(questions):
                        next_label = questions[current_question_index]['label']
                        reply += question_template.format(next_label)
                    else:
                        # No more questions
                        reply += ('Your profile is now complete. Is there anything else you would like to know?')
                        conversation_state = 'end'
                        current_question_index = None
            else:
                reply = 'Please provide a number between 1 and 10.'
        except ValueError:
            reply = 'Please provide a valid number between 1 and 10.'
    else:
        reply = ("Your profile is complete. If you'd like to update your interests, just let me know!")

    return jsonify({
        'reply': reply,
        'conversation_state': conversation_state,
        'current_question_index': current_question_index
    })

@app.route('/connections')
def connections():
    return render_template('connections.html')


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
