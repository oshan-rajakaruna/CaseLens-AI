"""Controlled persistence failures that contain no private evidence."""


class PersistenceError(RuntimeError):
    """Base class for safe persistence-layer errors."""


class DuplicateRecordError(PersistenceError):
    """A caller supplied an identity that already exists."""


class MissingRecordError(PersistenceError):
    """A requested or related record does not exist."""


class OwnershipError(PersistenceError):
    """The trusted principal does not own the requested record."""


class InvalidStateTransitionError(PersistenceError):
    """A task lifecycle transition is not permitted."""


class ImmutableRecordError(PersistenceError):
    """A completed or otherwise immutable record cannot be replaced."""


class InvalidRecordError(PersistenceError):
    """Metadata failed a persistence-boundary safety check."""
