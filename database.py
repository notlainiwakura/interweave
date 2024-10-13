from sqlalchemy import create_engine, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

SQLALCHEMY_DATABASE_URL = "sqlite:///./test.db"

engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


def init_db():
    import models  # Import here to avoid circular imports
    Base.metadata.create_all(bind=engine)

    # Check if the new columns exist, if not, add them
    from sqlalchemy import inspect
    from sqlalchemy.sql import text
    inspector = inspect(engine)
    existing_columns = [c['name'] for c in inspector.get_columns('users')]

    with engine.connect() as connection:
        if 'sci_fi_movies' not in existing_columns:
            connection.execute(text('ALTER TABLE users ADD COLUMN sci_fi_movies FLOAT'))
        if 'cooking' not in existing_columns:
            connection.execute(text('ALTER TABLE users ADD COLUMN cooking FLOAT'))
        if 'hiking' not in existing_columns:
            connection.execute(text('ALTER TABLE users ADD COLUMN hiking FLOAT'))
        if 'travel' not in existing_columns:
            connection.execute(text('ALTER TABLE users ADD COLUMN travel FLOAT'))
        if 'reading' not in existing_columns:
            connection.execute(text('ALTER TABLE users ADD COLUMN reading FLOAT'))
        if 'sports' not in existing_columns:
            connection.execute(text('ALTER TABLE users ADD COLUMN sports FLOAT'))
        if 'music' not in existing_columns:
            connection.execute(text('ALTER TABLE users ADD COLUMN music FLOAT'))
        if 'photography' not in existing_columns:
            connection.execute(text('ALTER TABLE users ADD COLUMN photography FLOAT'))
        if 'gardening' not in existing_columns:
            connection.execute(text('ALTER TABLE users ADD COLUMN gardening FLOAT'))
        if 'video_games' not in existing_columns:
            connection.execute(text('ALTER TABLE users ADD COLUMN video_games FLOAT'))
        if 'board_games' not in existing_columns:
            connection.execute(text('ALTER TABLE users ADD COLUMN board_games FLOAT'))
        if 'diy_projects' not in existing_columns:
            connection.execute(text('ALTER TABLE users ADD COLUMN diy_projects FLOAT'))
        if 'volunteering' not in existing_columns:
            connection.execute(text('ALTER TABLE users ADD COLUMN volunteering FLOAT'))
        if 'movies' not in existing_columns:
            connection.execute(text('ALTER TABLE users ADD COLUMN movies FLOAT'))
        if 'podcasts' not in existing_columns:
            connection.execute(text('ALTER TABLE users ADD COLUMN podcasts FLOAT'))
        if 'social_media' not in existing_columns:
            connection.execute(text('ALTER TABLE users ADD COLUMN social_media FLOAT'))
        if 'pets' not in existing_columns:
            connection.execute(text('ALTER TABLE users ADD COLUMN pets FLOAT'))
        if 'workout' not in existing_columns:
            connection.execute(text('ALTER TABLE users ADD COLUMN workout FLOAT'))
        if 'meditation' not in existing_columns:
            connection.execute(text('ALTER TABLE users ADD COLUMN meditation FLOAT'))
        if 'travel_adventure' not in existing_columns:
            connection.execute(text('ALTER TABLE users ADD COLUMN travel_adventure FLOAT'))
        if 'music_instruments' not in existing_columns:
            connection.execute(text('ALTER TABLE users ADD COLUMN music_instruments FLOAT'))
        if 'arts_crafts' not in existing_columns:
            connection.execute(text('ALTER TABLE users ADD COLUMN arts_crafts FLOAT'))
        connection.commit()