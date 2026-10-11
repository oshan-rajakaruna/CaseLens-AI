"""Database dependency; does not claim connectivity until a request uses it."""
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from backend.config import get_settings

def _database_url() -> str:
    url = get_settings().database_url
    if not url:
        raise RuntimeError("DATABASE_URL is not configured")
    return url

def get_engine():
    return create_engine(_database_url(), pool_pre_ping=True)

def get_session():
    factory = sessionmaker(bind=get_engine(), expire_on_commit=False, class_=Session)
    with factory() as session:
        yield session
