"""
End-to-end pipeline orchestrator chaining stages 01 through 06; character naming is always part of it.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Union

from ..config import validate_parameters
from ..utils.paths import checked_path
from .audit import run_dataset_audit
from .cluster import run_clustering
from .crop import run_crop_preparation
from .embed import run_embedding_extraction
from .materialize import run_folder_materialization
from .naming import run_cluster_naming


def release_imgutils_models() -> None:
    """Drop imgutils' cached detector and CCIP sessions (about 2 GB) before the tagger loads."""
    import gc

    from imgutils.generic import yolo
    from imgutils.metrics import ccip

    for cached in (yolo._open_models_for_repo_id, ccip._open_feat_model, ccip._open_metric_model):
        cached.cache_clear()
    gc.collect()


def run_full_pipeline(
    input_dir: Union[str, Path],
    project_dir: Union[str, Path] = "./anime_character_pipeline",
    min_cluster_size: int = 2,
    min_samples: int = 1,
    cluster_selection_epsilon: float = 0.50,
    reassign_noise: bool = True,
    max_reassign_distance: float = 0.55,
    materialization_mode: str = "hardlink",
    max_images: Optional[int] = None,
    sample_strategy: str = "stratified",
    online_lookup: bool = True,
    show_progress: bool = True,
) -> Dict[str, Any]:
    """
    Execute full pipeline from raw input folder to one folder per named character.

    Args:
        input_dir: Source folder with images to organize.
        project_dir: Base directory for run artifacts and organized output.
        min_cluster_size: Minimum cluster size for HDBSCAN (default: 2).
        min_samples: Minimum samples for HDBSCAN reachability (default: 1).
        cluster_selection_epsilon: Distance threshold for cluster merging (default: 0.50).
        reassign_noise: Whether to reassign borderline noise to nearest centroid (default: True).
        max_reassign_distance: Max Euclidean distance for noise reassignment (default: 0.55).
        materialization_mode: 'hardlink', 'copy', or 'symlink'.
        max_images: Optional cap on audited images (representative sample of the input).
        sample_strategy: 'stratified', 'first' or 'random' selection when max_images is set.
        online_lookup: Identify characters online (Danbooru, SauceNAO) before using the local tagger.
        show_progress: Whether to display tqdm progress bars.

    Returns:
        Dict mapping stage names to their respective result dictionaries.
    """
    validate_parameters(**locals())
    input_dir = checked_path(input_dir)
    project_dir = checked_path(project_dir)

    results: Dict[str, Any] = {}

    print(">>> Stage 1: Dataset Audit")
    results["audit"] = run_dataset_audit(
        input_dir=input_dir,
        project_dir=project_dir,
        max_images=max_images,
        sample_strategy=sample_strategy,
        show_progress=show_progress,
    )

    print(">>> Stage 2: Crop Preparation")
    results["crop"] = run_crop_preparation(
        project_dir=project_dir,
        previous_run_dir=results["audit"]["run_dir"],
        show_progress=show_progress,
    )

    print(">>> Stage 3: CCIP Embedding Extraction")
    results["embed"] = run_embedding_extraction(
        project_dir=project_dir,
        previous_run_dir=results["crop"]["run_dir"],
        show_progress=show_progress,
    )

    print(">>> Stage 4: HDBSCAN Clustering")
    results["cluster"] = run_clustering(
        project_dir=project_dir,
        previous_run_dir=results["embed"]["run_dir"],
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_epsilon=cluster_selection_epsilon,
        reassign_noise=reassign_noise,
        max_reassign_distance=max_reassign_distance,
        show_progress=show_progress,
    )

    release_imgutils_models()
    print(">>> Stage 5: Character Naming")
    results["naming"] = run_cluster_naming(
        project_dir=project_dir,
        previous_run_dir=results["cluster"]["run_dir"],
        online_lookup=online_lookup,
        show_progress=show_progress,
    )

    print(">>> Stage 6: Folder Materialization")
    results["materialize"] = run_folder_materialization(
        project_dir=project_dir,
        previous_run_dir=results["naming"]["run_dir"],
        materialization_mode=materialization_mode,
        show_progress=show_progress,
    )

    return results
