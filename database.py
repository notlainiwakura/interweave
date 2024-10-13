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
    inspector = inspect(engine)
    existing_columns = [c['name'] for c in inspector.get_columns('users')]

    with engine.connect() as connection:
        if 'sci_fi_movies' not in existing_columns:
            connection.execute(text('ALTER TABLE users ADD COLUMN sci_fi_movies FLOAT'))
        if 'cooking' not in existing_columns:
            connection.execute(text('ALTER TABLE users ADD COLUMN cooking FLOAT'))
        if 'hiking' not in existing_columns:
            connection.execute(text('ALTER TABLE users ADD COLUMN hiking FLOAT'))
        connection.commit()