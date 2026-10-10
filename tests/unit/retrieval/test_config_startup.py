"""Tests for one-time application dotenv startup loading."""

import os

import pytest

from retrieval.config import load_retrieval_environment


def test_dotenv_loads_configuration_without_overriding_os_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EMBEDDING_PROVIDER", "gemini")
    monkeypatch.delenv("OPENAI_EMBEDDING_MODEL", raising=False)
    calls: list[tuple[str, bool]] = []

    def fake_load_dotenv(*, dotenv_path: str, override: bool) -> bool:
        calls.append((dotenv_path, override))
        values = {
            "EMBEDDING_PROVIDER": "openai",
            "OPENAI_EMBEDDING_MODEL": "text-embedding-3-small",
        }
        for key, value in values.items():
            if override or key not in os.environ:
                os.environ[key] = value
        return True

    monkeypatch.setattr("retrieval.config.load_dotenv", fake_load_dotenv)

    loaded = load_retrieval_environment("test.env")

    assert loaded is True
    assert calls == [("test.env", False)]
    assert os.environ["EMBEDDING_PROVIDER"] == "gemini"
    assert os.environ["OPENAI_EMBEDDING_MODEL"] == "text-embedding-3-small"
