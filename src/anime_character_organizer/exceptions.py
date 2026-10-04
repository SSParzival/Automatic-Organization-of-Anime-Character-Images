"""
Custom exception hierarchy for anime_character_organizer.
"""


class PipelineError(Exception):
    """Base exception for all pipeline errors."""

    pass


class ConfigurationError(PipelineError):
    """Raised when pipeline configuration is invalid or missing."""

    pass


class RunNotFoundError(PipelineError):
    """Raised when an expected prior pipeline run directory cannot be located."""

    pass


class InvalidImageError(PipelineError):
    """Raised when an image fails decoding, verification, or format checks."""

    pass


class EmbeddingError(PipelineError):
    """Raised when embedding extraction fails."""

    pass


class ClusteringError(PipelineError):
    """Raised when clustering execution or validation fails."""

    pass


class MaterializationError(PipelineError):
    """Raised when file linking or copying fails."""

    pass


class TaggingError(PipelineError):
    """Raised when anime character tagging fails."""

    pass
