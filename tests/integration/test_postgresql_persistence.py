"""Opt-in PostgreSQL integration test for an already-migrated isolated test DB."""

from os import getenv
from uuid import uuid4

import pytest
from sqlalchemy import select
from sqlalchemy.engine import make_url

from database import TrustedPrincipal, build_engine, build_session_factory
from database.models import CaseModel
from database.repository import CaseLensRepository


def _isolated_postgres_url() -> str:
    enabled = getenv("RUN_POSTGRES_PERSISTENCE_TESTS", "").casefold() == "true"
    database_url = getenv("CASELENS_TEST_DATABASE_URL", "").strip()
    if not enabled or not database_url:
        pytest.skip("isolated PostgreSQL persistence test is not enabled")
    parsed = make_url(database_url)
    if not parsed.drivername.startswith("postgresql"):
        pytest.fail("CASELENS_TEST_DATABASE_URL must use PostgreSQL")
    if "test" not in (parsed.database or "").casefold():
        pytest.fail("refusing to use a database whose name does not contain 'test'")
    return database_url


def test_postgresql_insert_retrieve_and_transaction_rollback() -> None:
    engine = build_engine(_isolated_postgres_url())
    sessions = build_session_factory(engine)
    identity = uuid4().hex
    case_id = f"case-pg-{identity}"
    principal = TrustedPrincipal(subject=f"test|{identity}")

    connection = engine.connect()
    transaction = connection.begin()
    session = sessions(bind=connection)
    try:
        repository = CaseLensRepository(session)
        created = repository.create_case(
            principal,
            case_id=case_id,
            title="Isolated PostgreSQL rollback check",
        )
        loaded = repository.get_case(principal, case_id)
        assert loaded == created
        assert loaded.created_at.utcoffset() is not None
        transaction.rollback()
    finally:
        session.close()
        if transaction.is_active:
            transaction.rollback()
        connection.close()

    with sessions() as session:
        assert session.scalar(select(CaseModel).where(CaseModel.id == case_id)) is None
    engine.dispose()
