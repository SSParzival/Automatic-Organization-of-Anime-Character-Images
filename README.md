# Automatic Organization of Anime Character Images

> **Important:**
> This repository is designed for educational exploration, research, and practical automated sorting of unlabelled anime character collections. All code in this repository was originally developed through exploratory AI-assisted programming ("vibe-coding") and subsequently professionalized into a modular Python package, robust command-line interface, pedagogical Jupyter notebooks, and a comprehensive test suite. The system is designed to accelerate dataset preparation and exploratory analysis; outputs should always be reviewed by humans before downstream use.

---

## 1. Overview and Purpose

When managing large, unlabelled collections of anime-style illustrations, illustrations frequently lack character tags or contain unreliable metadata. Supervised classification cannot be applied directly because the full character inventory is unknown upfront.

This project implements an end-to-end unsupervised and semi-supervised computer vision workflow to group unlabelled images by visual character identity. Rather than attempting semantic recognition immediately, the pipeline enforces a fundamental architectural invariant: **visual identity clustering precedes semantic tagging**.

Grouping images visually first ensures that characters with rare designs, original characters (OCs), or illustrations missing from online tag databases are safely grouped into visual clusters before optional tagging tools (such as Danbooru-trained taggers) attempt to suggest names.

---

## 2. Principal Functionality

The pipeline executes through six modular, auditable stages:

1. **Dataset Integrity Audit and Duplicate Detection**:
   - Recursively scans candidate image files (`.jpg`, `.jpeg`, `.png`, `.webp`, `.bmp`).
   - Verifies file integrity and dimensions using PIL.
   - Computes SHA-256 cryptographic hashes for exact duplicate grouping.
   - Computes perceptual hashes (pHash) and indexes them in a discrete BK-tree with Union-Find disjoint sets for near-duplicate discovery.
2. **Crop Preparation and Identity Region Selection**:
   - Applies deep learning detectors (`dghs-imgutils`) to locate anime heads and bodies.
   - Employs a deterministic fallback ladder: `head crop -> person crop -> full image fallback`.
   - Pads and squares bounding boxes with clamp protections against out-of-boundary coordinates.
   - Flags low-confidence, extreme-aspect, or tiny crops for manual review.
3. **CCIP Visual Identity Embedding Extraction**:
   - Extracts character identity feature vectors using CCIP (Character Classification and Identification Pre-training with CAFormer backbones).
   - Normalizes feature vectors using L2 row normalization, mapping Euclidean distance directly to cosine similarity.
   - Operates in memory-safe minibatches with fallback mechanisms for damaged crops.
4. **HDBSCAN Density-Based Clustering and Diagnostics**:
   - Clusters normalized embeddings in metric space using HDBSCAN.
   - Employs conservative parameters (`min_cluster_size=2`, `min_samples=1`, `cluster_selection_epsilon=0.50`) to avoid conflating distinct characters.
   - Computes cluster centroids, distance distributions, and outlier scores.
   - Softly reassigns borderline noise points to nearest cluster centroids within a strict distance threshold.
5. **Non-Destructive Folder Materialization**:
   - Generates collision-free, deterministic target paths (`{row:07d}__{stem}{ext}`).
   - Materializes organized folders using hard links, file copies, or symbolic links.
   - Preserves source datasets completely untouched.
   - Verifies file existence and byte-level size matches post-materialization.
6. **Semantic Cluster Naming with Anime Taggers (Optional)**:
   - Samples representative images closest to each cluster's centroid.
   - Evaluates images using anime taggers (PixAI or DeepDanbooru/WD14).
   - Aggregates predicted character tags using weighted frequency voting and margin testing.
   - Generates an auditable rename plan and optional secondary named folder tree.

---

## 3. Architecture and Repository Structure

The project follows a hybrid architecture balancing reusable library code, headless scripts, and step-by-step pedagogical notebooks:

```text
.
├── src/anime_character_organizer/   # Core reusable Python package
│   ├── config.py                    # Strongly typed dataclass configurations
│   ├── exceptions.py                # Typed domain exceptions hierarchy
│   ├── utils/                       # Common utilities (paths, hashing, BK-tree, runs, etc.)
│   ├── data/                        # Image auditing, validation, duplicate detection
│   ├── preprocessing/               # Geometry, anime face/body detection, cropping
│   ├── embeddings/                  # CCIP feature extraction and normalization
│   ├── clustering/                  # HDBSCAN clustering, centroids, review diagnostics
│   ├── materialization/             # Non-destructive file linking/copying and validation
│   ├── tagging/                     # Anime tagger inference, name aggregation & voting
│   ├── visualization/               # Contact sheets and diagnostic plots
│   └── workflows/                   # High-level pipeline stage orchestrators
│
├── scripts/                         # Standalone headless CLI executables
│   ├── run_audit.py                 # Stage 1: Dataset audit & duplicate detection
│   ├── run_crop_preparation.py      # Stage 2: Face/body detection & crop extraction
│   ├── run_embeddings.py            # Stage 3: CCIP embedding extraction
│   ├── run_clustering.py            # Stage 4: HDBSCAN clustering & diagnostics
│   ├── run_materialization.py       # Stage 5: Non-destructive folder materialization
│   ├── run_naming.py                # Stage 6: Character tag inference & naming
│   └── run_pipeline.py              # End-to-end master pipeline runner
│
├── notebooks/                       # Pedagogical step-by-step Jupyter notebooks
│   ├── pipeline_master.ipynb        # Unified end-to-end master pipeline notebook
│   ├── 01_dataset_audit.ipynb       # Stage 1 exploration and duplicate analysis
│   ├── 02_crop_preparation.ipynb   # Stage 2 detection visualization and crop validation
│   ├── 03_ccip_embedding_extraction.ipynb # Stage 3 embedding extraction and L2 normalization
│   ├── 04_hdbscan_clustering.ipynb # Stage 4 density clustering, centroids, and diagnostics
│   ├── 05_non_destructive_folder_materialization.ipynb # Stage 5 link/copy operations & verification
│   └── 06_cluster_naming_with_anime_tagger.ipynb # Stage 6 tagger inference, voting, and naming
│
├── tests/                           # Automated test suite
│   ├── unit/                        # Tests for all domain modules (30+ unit tests)
│   └── integration/                 # End-to-end multi-stage integration tests
│
├── pyproject.toml                   # Build configuration, metadata, pytest, and ruff settings
└── README.md                        # Authoritative documentation
```

---

## 4. Prerequisites and Environment Setup

### Supported Python Versions
- **Python 3.10** or **Python 3.11** (tested on 3.11.14 on Linux x86_64).

### System Prerequisites
- `git`
- `python3` (>= 3.10)
- `python3-venv`
- Hardware: CPU execution is supported across all stages. For large image collections (> 500 images), an NVIDIA GPU with CUDA support is recommended for faster ONNX/PyTorch model inference.

### Environment Setup

```bash
# 1. Clone repository
git clone https://github.com/ssparzival/Automatic-Organization-of-Anime-Character-Images.git
cd Automatic-Organization-of-Anime-Character-Images

# 2. Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Upgrade pip and packaging tools
pip install --upgrade pip setuptools wheel

# 4. Install package in editable mode
pip install -e .

# 5. Optionally install development and testing tools
pip install -e ".[dev]"
```

---

## 5. Configuration and Environment Variables

### Configuration Dataclasses
All pipeline stages use strongly typed dataclasses defined in `anime_character_organizer.config`:
- `AuditConfig`: Controls candidate extensions, worker counts, and perceptual hash thresholds.
- `CropConfig`: Controls crop dimensions, format, detection score thresholds, and padding margins.
- `EmbeddingConfig`: Controls CCIP model selection, batch size, and L2 normalization.
- `ClusteringConfig`: Controls HDBSCAN `min_cluster_size`, `min_samples`, `cluster_selection_epsilon`, noise reassignment thresholds, and review separation.
- `MaterializationConfig`: Controls link modes (`hardlink`, `copy`, `symlink`), fallback behavior, and overwrite policies.
- `TaggingConfig`: Controls primary tagger selection, confidence thresholds, and voting margins.

### Environment Variables
- `ANIME_PIPELINE_INPUT_DIR`: Overrides the default source directory (`./input_images`) in all notebooks and scripts.
- `HF_HOME`: Sets the cache directory for Hugging Face and ONNX models downloaded by `dghs-imgutils`.

---

## 6. Input and Output Expectations

### Inputs
- A local directory containing anime illustrations in supported formats: `.png`, `.jpg`, `.jpeg`, `.webp`, `.bmp`.
- Directory structure can be flat or arbitrarily nested; candidate file discovery is fully recursive.

### Outputs
All pipeline runs write timestamped, self-contained run artifacts under `anime_character_pipeline/runs/`:
- `01_dataset_audit_<timestamp>/`: Metadata tables (`image_metadata.csv`), valid/invalid file manifests, duplicate candidate tables (`exact_duplicate_groups.csv`, `perceptual_duplicate_groups.csv`), and execution summary.
- `02_crop_preparation_<timestamp>/`: Bounding box crops, manifest (`crop_manifest.csv`), review flag manifests, and visual contact sheets.
- `03_ccip_embeddings_<timestamp>/`: Embedding matrices (`ccip_embeddings_raw.npy`, `ccip_embeddings_l2.npy`, bundle `.npz`), and embedding manifest.
- `04_hdbscan_clustering_<timestamp>/`: Cluster assignments (`cluster_labels.npy`), probabilities, outlier scores, centroids, summary tables, and distribution plots.
- `05_folder_materialization_<timestamp>/`: Materialization plan, verification report, and organized folder tree in `organized_output/anime_organized_<timestamp>/`.
- `06_cluster_naming_<timestamp>/`: Tagger outputs, voting scores, rename plan, and optional renamed folder tree in `organized_output/anime_named_<timestamp>/`.

---

## 7. Command-Line Interface (CLI Scripts)

Every stage can be executed directly from the terminal without opening Jupyter:

### Complete End-to-End Pipeline
```bash
python scripts/run_pipeline.py --input-dir /path/to/images --mode hardlink
```

### Stage-by-Stage Script Invocations

```bash
# Stage 1: Audit dataset and identify duplicates
python scripts/run_audit.py --input-dir /path/to/images --phash-threshold 6

# Stage 2: Create character crops (auto-discovers latest audit run)
python scripts/run_crop_preparation.py --crop-size 512 --crop-format JPEG

# Stage 3: Extract CCIP visual identity embeddings
python scripts/run_embeddings.py --batch-size 16

# Stage 4: Run HDBSCAN density clustering
python scripts/run_clustering.py --min-cluster-size 2 --min-samples 1 --cluster-selection-epsilon 0.50

# Stage 5: Non-destructively materialize organized folders
python scripts/run_materialization.py --mode hardlink

# Stage 6: Predict character tags and plan folder renaming
python scripts/run_naming.py --apply-rename --min-score 0.70 --min-share 0.35
```

> **Note:**
> For backward compatibility, `--link-mode` is accepted as an alias for `--mode` in `run_materialization.py` and `run_pipeline.py`, and `--apply-rename` is accepted as an alias for `--create-named-output` in `run_naming.py`.

---

## 8. Python Package API Usage

You can embed the organizer into custom Python workflows:

```python
from pathlib import Path
from anime_character_organizer.config import (
    AuditConfig,
    CropConfig,
    EmbeddingConfig,
    ClusteringConfig,
    MaterializationConfig,
)
from anime_character_organizer.workflows.pipeline import run_pipeline

results = run_pipeline(
    input_dir=Path("./my_anime_images"),
    project_dir=Path("./anime_character_pipeline"),
    audit_config=AuditConfig(phash_threshold=6),
    crop_config=CropConfig(crop_size=512),
    embedding_config=EmbeddingConfig(batch_size=16),
    clustering_config=ClusteringConfig(min_cluster_size=2, cluster_selection_epsilon=0.50),
    materialization_config=MaterializationConfig(mode="hardlink"),
    with_tagging=False,
)

print(f"Materialized run created at: {results['stage_05_materialize']['outputs']['materialized_root']}")
```

---

## 9. Jupyter Notebooks Workflow

All notebooks are designed to be pedagogical, step-by-step learning and verification interfaces. They import reusable functionality from `anime_character_organizer`, display intermediate statistics, render visual contact sheets, and explain algorithmic principles.

### Notebook Execution Order
1. [`notebooks/01_dataset_audit.ipynb`](notebooks/01_dataset_audit.ipynb): Audits files, verifies PIL readability, extracts SHA-256 and pHash, and detects duplicates.
2. [`notebooks/02_crop_preparation.ipynb`](notebooks/02_crop_preparation.ipynb): Runs anime face/person detection, applies crop padding, and flags review items.
3. [`notebooks/03_ccip_embedding_extraction.ipynb`](notebooks/03_ccip_embedding_extraction.ipynb): Extracts CCIP feature representations and applies mathematical L2 normalization.
4. [`notebooks/04_hdbscan_clustering.ipynb`](notebooks/04_hdbscan_clustering.ipynb): Performs HDBSCAN density clustering, centroid calculations, and outlier diagnostics.
5. [`notebooks/05_non_destructive_folder_materialization.ipynb`](notebooks/05_non_destructive_folder_materialization.ipynb): Plans and executes collision-free hardlinking/copying and verifies file integrity.
6. [`notebooks/06_cluster_naming_with_anime_tagger.ipynb`](notebooks/06_cluster_naming_with_anime_tagger.ipynb): Evaluates representative images with taggers and generates rename proposals.

### Unified Master Notebook
- [`notebooks/pipeline_master.ipynb`](notebooks/pipeline_master.ipynb): An all-in-one interactive control room orchestrating the entire six-stage pipeline with centralized parameter configuration in the first code cell.

---

## 10. Stage-by-Stage Details and Outputs

### Notebook 01: Dataset Audit and Duplicate Detection
- **Purpose**: Creates a reliable baseline inventory without moving or modifying files.
- **Methodology**: Parallel PIL inspection, SHA-256 collision checks, BK-tree discrete metric index for perceptual hash Hamming distance exploration.
- **Outputs**: `image_metadata.csv`, `valid_images.csv`, `invalid_images.csv`, `exact_duplicate_groups.csv`, `perceptual_duplicate_candidate_pairs.csv`, `perceptual_duplicate_groups.csv`, `summary.json`.

### Notebook 02: Representative Set and Crop Preparation
- **Purpose**: Prepares identity-focused crops to prevent background clutter from distorting similarity.
- **Methodology**: Detection ladder (`head -> person -> full image fallback`), margin expansion, square padding, and visual contact sheet generation.
- **Outputs**: `selected_images_for_detection.csv`, `crop_manifest.csv`, `crop_manifest_with_review_flags.csv`, `selected_region_counts.csv`, `review_reason_counts.csv`, `contact_sheets.csv`, `summary.json`.

### Notebook 03: CCIP Embedding Extraction
- **Purpose**: Generates high-dimensional vector representations capturing anime character identity.
- **Methodology**: Inference with CCIP CAFormer model, row-wise L2 vector normalization, array serialization.
- **Outputs**: `ccip_embeddings_raw.npy`, `ccip_embeddings_l2.npy`, `ccip_embeddings_bundle.npz`, `embedding_manifest.csv`, `embedding_diagnostics.json`, `summary.json`.

### Notebook 04: HDBSCAN Clustering and Cluster Diagnostics
- **Purpose**: Unsupervised character grouping without requiring a predefined cluster count $K$.
- **Methodology**: HDBSCAN with excess-of-mass clustering, cosine distance on L2-normalized embeddings, cluster centroid calculation, soft centroid noise reassignment.
- **Outputs**: `cluster_labels.npy`, `cluster_probabilities.npy`, `outlier_scores.npy`, `cluster_centroids.npz`, `cluster_manifest.csv`, `cluster_summary.csv`, `noise_manifest.csv`, `folder_assignment_manifest.csv`, `cluster_contact_sheets.csv`, `summary.json`.

### Notebook 05: Non-Destructive Folder Materialization
- **Purpose**: Physically arranges images into character folders while guaranteeing the safety of the source collection.
- **Methodology**: Generates unique target filenames (`{row:07d}__{stem}{ext}`), links or copies files, and validates existence and file sizes.
- **Outputs**: `anime_organized_<timestamp>/`, `materialization_plan.csv`, `materialization_result.csv`, `materialization_validation.csv`, `_global_materialization_index.csv`, `summary.json`.

### Notebook 06: Cluster Naming with Anime Tagger
- **Purpose**: Suggests semantic Danbooru character names for clusters using deep anime taggers.
- **Methodology**: Representative sample selection, confidence-thresholded tag extraction, weighted cluster voting, and margin checks against runner-up names.
- **Outputs**: `tagging_input_manifest.csv`, `tagging_manifest.csv`, `cluster_name_suggestions.csv`, `folder_rename_plan.csv`, `naming_contact_sheets.csv`, `summary.json`.

---

## 11. Testing and Code Quality

The repository includes a comprehensive automated test suite and adheres to modern Python code quality standards.

### Running Tests
Execute the full test suite (unit and integration tests):

```bash
# Using pytest (recommended):
pytest tests

# Using Python's standard unittest runner:
python -m unittest discover tests
```

### Formatting and Linting
The codebase is formatted and linted using `ruff`:

```bash
# Format Python source files, scripts, and tests (excluding sandbox/):
ruff format src/ scripts/ tests/

# Check for linting violations and code issues:
ruff check src/ scripts/ tests/

# Automatically fix fixable lint issues:
ruff check --fix src/ scripts/ tests/
```

---

## 12. Design Principles and Invariants

1. **Non-Destructive Guarantee**: Source images are strictly treated as read-only. No script, notebook, or module moves, modifies, or deletes files in the input directory.
2. **Auditability and Traceability**: Every stage writes self-contained run directories with tabular manifests, JSON execution summaries, and visual contact sheets.
3. **Conservative Identity Grouping**: Over-merging two distinct characters into one folder is treated as a severe error. The default parameters favor smaller, purer clusters over broad mixtures.
4. **Visual Grouping Precedes Semantic Naming**: Taggers are prone to hallucinations or silence on unrepresented characters. Grouping purely by visual features ensures robust sorting regardless of tagger availability.
5. **Deterministic Fallbacks**: Robust fallback ladders prevent hard execution failures when processing imperfect real-world art datasets (e.g., cross-device hardlink failures automatically fall back to copies).

---

## 13. Known Limitations and Edge Cases

- **Multiple Characters in One Image**: If an image contains multiple characters, the detector selects the most prominent face or body. Images with multiple distinct characters may end up assigned to a single character's cluster.
- **Extreme Stylization and Chibi Art**: Heavily distorted, chibi, or monochromatic manga pages may produce embeddings distant from standard full-color illustrations.
- **Outfit and Hairstyle Changes**: Characters with drastically different outfits, hair colors, or disguises across seasons may be split into separate clusters.
- **Cross-Filesystem Hardlinks**: Hard links cannot cross filesystem or mount boundaries (`EXDEV`). If the source dataset and project directory reside on different drives, the materialization stage automatically falls back to file copying.
- **Semantic Tagger Bias**: Anime taggers are trained on specific web datasets (e.g., Danbooru). Characters absent from those training corpora will not receive correct name predictions, but will remain safely grouped in their visual cluster.

---

## 14. Troubleshooting

- **Error: Cross-device link (`EXDEV`) during materialization**:
  *Cause*: Source images and output folder are on different filesystems or physical disks.
  *Solution*: The pipeline automatically falls back to copy mode by default. You can explicitly pass `--mode copy` to avoid the warning.
- **Model Download Failures / Timeouts**:
  *Cause*: Network connectivity issues when downloading ONNX models for head detection or CCIP embeddings.
  *Solution*: Set `HF_HOME=/path/to/cache` in your environment and retry with an active internet connection. Downloaded models are cached permanently.
- **Out of Memory during Embedding Extraction**:
  *Cause*: Batch size too large for available GPU/system RAM.
  *Solution*: Pass `--batch-size 8` or `--batch-size 4` to `scripts/run_embeddings.py`.

---

## 15. Security and Privacy Considerations

- **Local Offline Processing**: All processing occurs locally on your machine. Image files, embeddings, and metadata are never uploaded to external servers or cloud services.
- **Path Traversal Protection**: Relative destination paths are strictly validated through `safe_relative_path` to prevent path traversal outside designated workspace directories.
- **Untrusted File Safety**: Image files are inspected using PIL image verification routines to safely detect corrupted or malformed headers before downstream deep learning inference.

---

## 16. Contributing and Development

Contributions that improve stability, test coverage, or pedagogical clarity are welcome!

1. Fork the repository and create a feature branch.
2. Follow PEP 8 guidelines and format code using `ruff format src/ scripts/ tests/`.
3. Verify that all tests pass (`pytest tests`).
4. Maintain the strict non-destructive invariant for user data.
5. **Strict Sandbox Invariant**: Never inspect, modify, stage, or commit files located in any `sandbox/` directory.

---

## 17. License and Disclaimer

This project is released for educational and research purposes.

Users are solely responsible for ensuring they possess the necessary rights and permissions to store, process, organize, or distribute any images processed by this software. The authors provide the software "as-is", without warranty of any kind, express or implied.
