"""Application configuration bootstrap helpers."""

from __future__ import annotations

from os import PathLike

from dotenv import find_dotenv, load_dotenv


def load_retrieval_environment(
    dotenv_path: str | PathLike[str] | None = None,
) -> bool:
    """Load local configuration without overriding existing OS variables.

    Application entry points call this once during startup. Embedding providers
    intentionally do not load dotenv files themselves.
    """

    resolved_path = str(dotenv_path) if dotenv_path is not None else find_dotenv(usecwd=True)
    if not resolved_path:
        return False
    return load_dotenv(dotenv_path=resolved_path, override=False)
