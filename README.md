# Automatic Organization of Anime Character Images

> **Educational and experimental project.**  
> This repository is provided only for educational purposes. It is not intended as a production-grade system, a commercial tool, or a definitive character-recognition solution. All files in this repository were **vibe-coded**, meaning that the code was produced through iterative AI-assisted development, experimentation, correction, and refinement. The project should therefore be read as a learning-oriented prototype rather than as a formally engineered software package.

## Overview

This project explores a complete pipeline for automatically organizing a large collection of anime-style images into folders that approximately correspond to visual character identities.

The central problem is that the full list of characters is not known beforehand. Therefore, the task is not treated as ordinary supervised classification. Instead, the project follows an unsupervised or semi-supervised strategy:

1. audit and validate the image collection;
2. remove or mark duplicates;
3. detect and crop character-relevant regions;
4. extract anime-character similarity embeddings;
5. cluster images according to visual similarity;
6. materialize a non-destructive folder structure;
7. optionally suggest semantic names for the discovered clusters using anime taggers.

The main idea is that grouping images by visual identity should happen before trying to name the character. This is important because a tagger may fail on rare, obscure, fan-made, original, or heavily stylized characters, while an embedding-based clustering pipeline can still group visually similar images together.

## Educational Purpose and Disclaimer

This repository is intended for learning, experimentation, and documentation of a practical computer-vision workflow. It should be used as a reference for understanding how an image-clustering pipeline can be structured in notebooks.

The repository does **not** guarantee:

- perfect character recognition;
- perfect cluster purity;
- correct semantic character names;
- reliable performance on all anime styles;
- compatibility with every operating system or hardware setup;
- production-level robustness.

The outputs should always be manually reviewed. The system is designed to reduce manual work, not to eliminate human judgment.

## Vibe-Coded Nature of the Repository

All files in this repository were **vibe-coded**. In this context, this means that the project was developed through an exploratory AI-assisted workflow, where code was iteratively generated, tested, corrected, and improved.

As a result:

- the notebooks are intentionally verbose and explicit;
- the code prioritizes clarity, auditability, and reproducibility;
- many intermediate files are saved for inspection;
- the pipeline is modular rather than compressed into a single script;
- the project should be treated as an educational prototype.

Users are encouraged to inspect the notebooks carefully before running them on large image collections.

## Project Structure

The project is organized around six main Jupyter notebooks. Each notebook corresponds to one stage of the pipeline and produces artifacts consumed by the following stage.

```text
.
├── notebooks/
│   ├── 01_dataset_audit.ipynb
│   ├── 02_crop_preparation.ipynb
│   ├── 03_ccip_embedding_extraction.ipynb
│   ├── 04_hdbscan_clustering.ipynb
│   ├── 05_non_destructive_folder_materialization.ipynb
│   └── 06_cluster_naming_with_anime_tagger.ipynb
│
├── anime_character_pipeline/
│   ├── runs/
│   │   ├── 01_dataset_audit_*/
│   │   ├── 02_crop_preparation_*/
│   │   ├── 03_ccip_embeddings_*/
│   │   ├── 04_hdbscan_clustering_*/
│   │   ├── 05_folder_materialization_*/
│   │   └── 06_cluster_naming_*/
│   │
│   └── organized_output/
│       ├── anime_organized_*/
│       └── anime_named_*/
│
└── README.md
```

The exact notebook filenames may differ depending on how they were saved locally, but the logical structure is the one described below.

## Notebook 01: Dataset Audit, Validation, and Duplicate Index

The first notebook scans the input image directory recursively and builds a complete metadata index of the collection.

Its main functions are:

- discover candidate image files;
- validate image readability;
- detect corrupted or invalid files;
- compute basic image metadata;
- compute exact hashes using SHA-256;
- compute perceptual hashes for near-duplicate detection;
- generate duplicate candidate groups;
- save auditable CSV and JSON reports.

This notebook does **not** move, delete, or reorganize images. It only creates a reliable first inventory of the dataset.

Typical outputs include:

```text
image_metadata.csv
valid_images.csv
invalid_images.csv
exact_duplicate_groups.csv
perceptual_duplicate_candidate_pairs.csv
perceptual_duplicate_groups.csv
summary.json
```

## Notebook 02: Representative Set and Crop Preparation

The second notebook prepares the images for character-identity embedding extraction.

Its main functions are:

- load the valid images from notebook 01;
- exclude exact duplicate non-representatives;
- optionally exclude perceptual duplicate non-representatives;
- detect anime heads;
- detect anime bodies or persons;
- choose the best crop for each image;
- fall back to the full image when detection fails;
- mark images that require manual review;
- create crop files;
- create contact sheets for quick inspection.

The preferred crop hierarchy is:

```text
head crop → person crop → full image fallback
```

This is because character identity is usually encoded in the face, hair, eyes, accessories, and upper-body features.

Typical outputs include:

```text
selected_images_for_detection.csv
crop_manifest.csv
crop_manifest_with_review_flags.csv
selected_region_counts.csv
review_reason_counts.csv
contact_sheets.csv
summary.json
```

## Notebook 03: CCIP Embedding Extraction

The third notebook extracts anime-character similarity embeddings from the crop files created in notebook 02.

Its main functions are:

- load the crop manifest;
- validate crop files before inference;
- run CCIP feature extraction;
- create raw embedding matrices;
- create L2-normalized embedding matrices;
- save embedding manifests;
- run consistency checks;
- save diagnostics and reproducibility reports.

The embedding matrix is the numerical representation used for clustering. Each row corresponds to one image crop, and each vector is intended to encode character-level visual similarity.

Typical outputs include:

```text
ccip_embeddings_raw.npy
ccip_embeddings_l2.npy
ccip_embeddings_bundle.npz
embedding_manifest.csv
successful_embedding_manifest.csv
embedding_status.csv
embedding_diagnostics.json
consistency_checks.json
summary.json
```

## Notebook 04: HDBSCAN Clustering and Cluster Diagnostics

The fourth notebook performs unsupervised clustering over the normalized CCIP embeddings.

Its main functions are:

- load the embedding matrix and manifest from notebook 03;
- cluster images using HDBSCAN;
- assign noise labels to uncertain points;
- compute cluster membership probabilities;
- compute outlier scores;
- compute centroid distances;
- generate cluster summaries;
- flag low-confidence or suspicious images for manual review;
- generate representative contact sheets;
- create a conservative folder-assignment manifest.

The clustering stage intentionally favors conservative grouping. It is better to produce more small clusters than to merge different characters incorrectly.

Typical outputs include:

```text
cluster_labels.npy
cluster_probabilities.npy
outlier_scores.npy
cluster_centroids.npz
cluster_manifest.csv
cluster_summary.csv
manual_review_manifest.csv
noise_manifest.csv
folder_assignment_manifest.csv
cluster_contact_sheets.csv
special_contact_sheets.csv
summary.json
```

## Notebook 05: Non-Destructive Folder Materialization

The fifth notebook creates the organized folder structure from the clustering results.

Its main functions are:

- load the folder-assignment manifest from notebook 04;
- build a destination plan;
- create output folders;
- materialize images using one of the supported modes:
  - hard links;
  - copies;
  - symbolic links;
- preserve the original image directory untouched;
- create per-folder manifests;
- create per-folder metadata files;
- generate output contact sheets;
- write a global materialization index;
- validate that all materialized files exist.

By default, the pipeline is designed to be non-destructive. Original files are not moved or deleted.

Typical outputs include:

```text
anime_organized_*/
_global_materialization_index.csv
_folder_manifest.csv
_cluster_info.json
materialization_plan.csv
materialization_result.csv
materialization_validation.csv
final_folder_counts.csv
operation_counts.csv
summary.json
```

## Notebook 06: Cluster Naming with Anime Tagger

The sixth notebook optionally suggests semantic names for the discovered clusters.

Its main functions are:

- load the cluster results from notebook 04;
- optionally load the materialized folder structure from notebook 05;
- select representative images from each cluster;
- run an anime tagger on representative crops;
- aggregate character tags at the cluster level;
- decide whether a name is sufficiently reliable;
- create a folder rename plan;
- create naming contact sheets;
- optionally create a second non-destructive named output tree.

This notebook does not assume that tagger predictions are always correct. A name is accepted only if it passes configurable agreement thresholds, such as minimum tag frequency, weighted score, and margin over the second-best candidate.

Typical outputs include:

```text
tagging_input_manifest.csv
tagging_ready_manifest.csv
tagging_manifest.csv
cluster_name_suggestions.csv
folder_rename_plan.csv
folder_rename_preview.csv
naming_contact_sheets.csv
cluster_naming_metadata_index.csv
summary.json
```

If enabled, it may also create:

```text
anime_named_*/
```

## Pipeline Summary

The complete pipeline can be summarized as follows:

```text
Input image folder
    ↓
01 Dataset audit and duplicate detection
    ↓
02 Anime head/person detection and crop preparation
    ↓
03 CCIP embedding extraction
    ↓
04 HDBSCAN clustering and review diagnostics
    ↓
05 Non-destructive folder materialization
    ↓
06 Optional semantic naming with anime taggers
```

Each stage writes its own outputs under:

```text
anime_character_pipeline/runs/
```

This makes the pipeline auditable and reproducible. A later stage can always be traced back to the exact run directory that generated its input.

## Design Principles

### 1. Non-destructive processing

The original image collection should not be modified. All organized outputs are written to new directories.

### 2. Auditability

Every major step writes CSV, JSON, and diagnostic outputs. This makes it easier to understand what happened and debug bad results.

### 3. Modularity

Each notebook performs one logical stage. This makes the pipeline easier to test, rerun, and improve.

### 4. Conservative clustering

The pipeline prefers uncertain or smaller clusters over aggressive merging. Incorrectly merging two characters is usually worse than splitting one character into multiple groups.

### 5. Naming after clustering

The project separates visual grouping from semantic naming. The tagger is used only after clusters have been discovered.

### 6. Manual review support

Contact sheets, review manifests, outlier scores, and confidence metrics are generated to make human inspection easier.

## Expected Limitations

The pipeline may struggle with:

- images containing multiple characters;
- rare or obscure characters;
- original characters;
- heavy stylization;
- alternate outfits;
- chibi or deformed versions;
- extreme crops;
- occluded faces;
- characters with very similar designs;
- low-resolution or corrupted images;
- images where the detector chooses the wrong region.

These limitations are expected. The goal is not perfect automation, but a strong first-pass organization that reduces manual sorting effort.

## Recommended Usage

A typical workflow is:

1. Place the input images in a separate source directory.
2. Run notebook 01.
3. Inspect invalid files and duplicate reports.
4. Run notebook 02.
5. Inspect crop contact sheets.
6. Run notebook 03.
7. Run notebook 04.
8. Inspect cluster contact sheets and review summaries.
9. Run notebook 05 to create the organized folder tree.
10. Run notebook 06 if semantic folder-name suggestions are desired.
11. Manually review uncertain clusters and rename folders if necessary.

## Important Safety Notes

Do not run the materialization notebooks on a unique image collection without backups. Although the pipeline is designed to be non-destructive, mistakes in paths, permissions, or configuration can still produce unwanted results.

Before running notebooks 05 or 06 with output creation enabled, verify:

- the selected source directory;
- the selected output directory;
- the materialization mode;
- available disk space;
- whether hard links, copies, or symbolic links are appropriate.

## Repository Status

This repository is an educational prototype. It is suitable for experimentation, learning, and adapting the workflow to personal collections. It is not maintained as a polished Python package and should not be assumed to follow production software engineering standards.

## License and Responsibility

This repository does not grant rights over any images processed with it. Users are responsible for ensuring that they have the right to store, process, organize, or redistribute any images they use with this pipeline.

The code and notebooks are provided as-is, without warranty. Use them at your own risk.

## Final Note

This project demonstrates how a practical unsupervised image-organization workflow can be built from modular notebook stages. Its main value is educational: it shows how validation, duplicate detection, cropping, embedding extraction, clustering, materialization, and optional semantic naming can be combined into a coherent pipeline for anime-style character images.
