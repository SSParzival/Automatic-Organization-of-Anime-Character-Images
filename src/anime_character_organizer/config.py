"""
Centralized configuration defaults and dataclasses for all pipeline stages.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Set, Tuple

DEFAULT_VALID_EXTENSIONS: Set[str] = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff", ".avif"}


@dataclass
class AuditConfig:
    """Configuration parameters for dataset audit stage."""

    input_dir: Path
    project_dir: Path = Path("./anime_character_pipeline")
    valid_extensions: Set[str] = field(default_factory=lambda: set(DEFAULT_VALID_EXTENSIONS))
    phash_threshold: int = 6
    max_workers: Optional[int] = None


@dataclass
class CropConfig:
    """Configuration parameters for crop preparation stage."""

    project_dir: Path = Path("./anime_character_pipeline")
    previous_run_dir: Optional[Path] = None
    use_perceptual_representatives: bool = False  # False avoids prematurely dropping images before clustering
    head_conf_threshold: float = 0.35
    head_iou_threshold: float = 0.50
    person_conf_threshold: float = 0.30
    person_iou_threshold: float = 0.50
    head_padding_ratio: float = 0.25  # More context around hair and head accessories
    person_padding_ratio: float = 0.10
    crop_format: str = "JPEG"
    crop_quality: int = 95
    crop_output_size: Tuple[int, int] = (512, 512)
    min_crop_box_area_ratio: float = 0.010
    max_workers: Optional[int] = None


@dataclass
class EmbeddingConfig:
    """Configuration parameters for CCIP embedding extraction stage."""

    project_dir: Path = Path("./anime_character_pipeline")
    previous_run_dir: Optional[Path] = None
    ccip_model: str = "ccip-caformer-24-randaug-pruned"
    ccip_image_size: int = 384
    batch_size: int = 16
    use_review_crops: bool = True
    require_crop_file_exists: bool = True
    normalize_embeddings: bool = True


@dataclass
class ClusteringConfig:
    """Configuration parameters for HDBSCAN clustering stage."""

    project_dir: Path = Path("./anime_character_pipeline")
    previous_run_dir: Optional[Path] = None
    embedding_file_name: str = "ccip_embeddings_l2.npy"
    manifest_file_name: str = "successful_embedding_manifest.csv"
    min_cluster_size: int = 2  # Allows pairs and triplets of characters to form clusters
    min_samples: int = 1  # Reduces reachability penalty in sparse/small clusters
    cluster_selection_epsilon: float = 0.50  # Merges points within close visual distance
    cluster_selection_method: str = "eom"
    metric: str = "euclidean"
    low_probability_threshold: float = 0.35
    high_outlier_quantile: float = 0.95
    reassign_noise: bool = True  # Reassigns borderline noise points to closest cluster centroid
    max_reassign_distance: float = 0.55  # Max Euclidean distance to centroid for noise reassignment
    separate_review_folders: bool = False  # Keep character folders unified instead of splitting
    random_seed: int = 42


@dataclass
class MaterializationConfig:
    """Configuration parameters for folder materialization stage."""

    project_dir: Path = Path("./anime_character_pipeline")
    previous_run_dir: Optional[Path] = None
    final_output_dir: Optional[Path] = None
    materialization_mode: str = "hardlink"  # hardlink, copy, symlink
    allow_hardlink_fallback_to_copy: bool = True
    on_existing: str = "error"  # error, skip, overwrite
    source_column: str = "source_path"
    create_cluster_metadata_files: bool = True
    create_folder_manifest_files: bool = True
    create_output_readme: bool = True
    create_output_contact_sheet_index: bool = True
    copy_cluster_contact_sheets_to_output: bool = True


@dataclass
class TaggingConfig:
    """Configuration parameters for cluster naming with anime tagger stage."""

    project_dir: Path = Path("./anime_character_pipeline")
    previous_clustering_run_dir: Optional[Path] = None
    previous_materialization_run_dir: Optional[Path] = None
    primary_tagger: str = "pixai"
    pixai_model_name: str = "v0.9"
    wd14_model_name: str = "SwinV2_v3"
    use_wd14_fallback: bool = True
    min_cluster_size_to_name: int = 2
    max_images_per_cluster_to_tag: int = 12
    min_character_score: float = 0.65
    min_name_share: float = 0.30
    min_weighted_score: float = 0.50
    min_top_margin: float = 0.05
    ignore_review_folders_for_naming: bool = False
    tag_only_non_noise_clusters: bool = True
    create_rename_plan: bool = True
    create_named_output: bool = True
    named_output_mode: str = "hardlink"
    allow_hardlink_fallback_to_copy: bool = True
    on_existing: str = "error"
    named_output_dir: Optional[Path] = None
