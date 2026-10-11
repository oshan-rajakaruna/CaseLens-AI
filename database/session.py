"""Explicit engine/session construction; schema creation is never automatic."""

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.orm import Session, sessionmaker


def build_engine(database_url: str, **engine_options: object) -> Engine:
    if not database_url.strip():
        raise ValueError("DATABASE_URL is required")
    engine = create_engine(database_url, pool_pre_ping=True, **engine_options)
    if engine.dialect.name == "sqlite":
        event.listen(engine, "connect", _enable_sqlite_foreign_keys)
    return engine


def build_session_factory(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(bind=engine, expire_on_commit=False)


def _enable_sqlite_foreign_keys(dbapi_connection, connection_record) -> None:
    del connection_record
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()
