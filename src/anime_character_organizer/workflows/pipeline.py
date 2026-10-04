"""
End-to-end pipeline orchestrator chaining stages 01 through 06.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Union

from .audit import run_dataset_audit
from .crop import run_crop_preparation
from .embed import run_embedding_extraction
from .cluster import run_clustering
from .materialize import run_folder_materialization
from .naming import run_cluster_naming


def run_full_pipeline(
    input_dir: Union[str, Path],
    project_dir: Union[str, Path] = "./anime_character_pipeline",
    min_cluster_size: int = 2,
    min_samples: int = 1,
    cluster_selection_epsilon: float = 0.50,
    reassign_noise: bool = True,
    max_reassign_distance: float = 0.55,
    separate_review_folders: bool = False,
    run_tagging: bool = False,
    materialization_mode: str = "hardlink",
    show_progress: bool = True,
) -> Dict[str, Any]:
    """
    Execute full pipeline from raw input folder to organized folders.
    
    Args:
        input_dir: Source folder with images to organize.
        project_dir: Base directory for run artifacts and organized output.
        min_cluster_size: Minimum cluster size for HDBSCAN (default: 2).
        min_samples: Minimum samples for HDBSCAN reachability (default: 1).
        cluster_selection_epsilon: Distance threshold for cluster merging (default: 0.50).
        reassign_noise: Whether to reassign borderline noise to nearest centroid (default: True).
        max_reassign_distance: Max Euclidean distance for noise reassignment (default: 0.55).
        separate_review_folders: Whether to split review items into separate folders (default: False).
        run_tagging: If True, executes stage 06 semantic naming after materialization.
        materialization_mode: 'hardlink', 'copy', or 'symlink'.
        show_progress: Whether to display tqdm progress bars.
        
    Returns:
        Dict mapping stage names to their respective result dictionaries.
    """
    input_dir = Path(input_dir).expanduser().resolve()
    project_dir = Path(project_dir).expanduser().resolve()

    results: Dict[str, Any] = {}

    print(">>> Stage 1: Dataset Audit")
    results["audit"] = run_dataset_audit(
        input_dir=input_dir,
        project_dir=project_dir,
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
        separate_review_folders=separate_review_folders,
        show_progress=show_progress,
    )

    print(">>> Stage 5: Folder Materialization")
    results["materialize"] = run_folder_materialization(
        project_dir=project_dir,
        previous_run_dir=results["cluster"]["run_dir"],
        materialization_mode=materialization_mode,
        show_progress=show_progress,
    )

    if run_tagging:
        print(">>> Stage 6: Semantic Cluster Naming")
        results["naming"] = run_cluster_naming(
            project_dir=project_dir,
            previous_clustering_run_dir=results["cluster"]["run_dir"],
            previous_materialization_run_dir=results["materialize"]["run_dir"],
            show_progress=show_progress,
        )

    return results
