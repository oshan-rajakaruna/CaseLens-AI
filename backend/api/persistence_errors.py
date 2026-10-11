"""Sanitized translation of internal persistence errors to HTTP responses."""

from fastapi import HTTPException, status

from database.errors import (
    DuplicateRecordError,
    InvalidRecordError,
    MissingRecordError,
    OwnershipError,
    PersistenceError,
)


def persistence_http_error(error: Exception) -> HTTPException:
    if isinstance(error, (MissingRecordError, OwnershipError)):
        return HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Record not found")
    if isinstance(error, DuplicateRecordError):
        return HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Record already exists")
    if isinstance(error, InvalidRecordError):
        return HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid record")
    if isinstance(error, PersistenceError):
        return HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Database is unavailable")
    return HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Request failed")
