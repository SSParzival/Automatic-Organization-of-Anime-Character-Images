"""
Automatic Organization of Anime Character Images.

A modular pipeline for unsupervised organization of anime character images into
visual identity clusters and optional semantic tagging.
"""

from .config import (
    AuditConfig,
    ClusteringConfig,
    CropConfig,
    EmbeddingConfig,
    MaterializationConfig,
    TaggingConfig,
)
from .exceptions import (
    ClusteringError,
    ConfigurationError,
    EmbeddingError,
    InvalidImageError,
    MaterializationError,
    PipelineError,
    RunNotFoundError,
    TaggingError,
)
from .workflows.audit import run_dataset_audit
from .workflows.cluster import run_clustering
from .workflows.crop import run_crop_preparation
from .workflows.embed import run_embedding_extraction
from .workflows.materialize import run_folder_materialization
from .workflows.naming import run_cluster_naming
from .workflows.pipeline import run_full_pipeline

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "AuditConfig",
    "CropConfig",
    "EmbeddingConfig",
    "ClusteringConfig",
    "MaterializationConfig",
    "TaggingConfig",
    "PipelineError",
    "ConfigurationError",
    "RunNotFoundError",
    "InvalidImageError",
    "EmbeddingError",
    "ClusteringError",
    "MaterializationError",
    "TaggingError",
    "run_dataset_audit",
    "run_crop_preparation",
    "run_embedding_extraction",
    "run_clustering",
    "run_folder_materialization",
    "run_cluster_naming",
    "run_full_pipeline",
]
