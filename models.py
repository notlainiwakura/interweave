from sqlalchemy import Column, Integer, String, LargeBinary, Text, Float
from sqlalchemy.orm import relationship
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import UserMixin
from database import Base


class User(Base, UserMixin):
    __tablename__ = 'users'

    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String(50), unique=True, nullable=False)
    email = Column(String(120), unique=True, nullable=False)
    password_hash = Column(String(128), nullable=False)
    interests = Column(Text, nullable=True)
    embedding = Column(LargeBinary, nullable=True)

    # New columns
    sci_fi_movies = Column(Float, nullable=True)
    cooking = Column(Float, nullable=True)
    hiking = Column(Float, nullable=True)
    travel = Column(Float, nullable=True)
    reading = Column(Float, nullable=True)
    sports = Column(Float, nullable=True)
    music = Column(Float, nullable=True)
    photography = Column(Float, nullable=True)
    gardening = Column(Float, nullable=True)
    video_games = Column(Float, nullable=True)
    board_games = Column(Float, nullable=True)
    diy_projects = Column(Float, nullable=True)
    volunteering = Column(Float, nullable=True)
    movies = Column(Float, nullable=True)
    podcasts = Column(Float, nullable=True)
    social_media = Column(Float, nullable=True)
    pets = Column(Float, nullable=True)
    workout = Column(Float, nullable=True)
    meditation = Column(Float, nullable=True)
    travel_adventure = Column(Float, nullable=True)
    music_instruments = Column(Float, nullable=True)
    arts_crafts = Column(Float, nullable=True)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

    # Flask-Login requires this method to check if the user is active.
    @property
    def is_active(self):
        return True  # You can customize this to check if the user is banned or inactive.

    # Flask-Login requires this method to check if the user is authenticated.
    @property
    def is_authenticated(self):
        return True  # Since this is a logged-in user, return True.

    # Flask-Login requires this method to check if the user is anonymous.
    @property
    def is_anonymous(self):
        return False  # Regular users are not anonymous.

    # Flask-Login requires this method to return the user ID.
    def get_id(self):
        return str(self.id)
