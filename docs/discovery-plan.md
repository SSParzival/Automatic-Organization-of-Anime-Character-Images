# Discovery and correction plan

Recorded before implementation on 2026-10-05. Sandbox contents were neither read nor inventoried.

## Git and baseline

- Root: `/home/ssparzival/Proyectos/GitHub/Automatic-Organization-of-Anime-Character-Images`; branch: `main`; HEAD: `7d954307e27da38582d44f8972b3955a05a8a806`.
- Pre-existing user work: master notebook cell 2 execution output/count and kernel display name. Preserve and leave that file uncommitted.
- Baseline: 38 tests passed (17.08 seconds); Ruff lint passed; 65 Python files formatted. Tests called real detectors and wrote temporary files outside the repository; isolate later validation under `.audit/` and mock model boundaries.
- No AGENTS.md, lockfile, CI, shell scripts, schema files, prompt templates, environment examples, contribution guide, or standalone license file found in the inspected tree.

## Contract and architecture

Research-oriented local image sorting: scan and hash → crop representative originals → CCIP features → normalized Euclidean HDBSCAN → organize original images → optional tagger voting and additional named tree. Users are dataset curators and researchers. Inputs are image folders and prior CSV/NumPy run artifacts; outputs are timestamped manifests, summaries, crops, arrays, plots and linked/copied originals. Python >=3.10 declared, 3.11.14 available; setuptools, src layout, dataclasses, argparse scripts, unittest tests collected by pytest, Ruff.

Retain existing domain packages and stage workflows. Existing extraction already covers inference, geometry, hashing, metrics and persistence; no mechanical cell export is justified. Add only shared run allocation, explicit notebook workspace configuration, safe filesystem boundaries and validation where demonstrated gaps require them. Scripts depend on workflows, workflows on domain modules, domain modules on narrow utilities/configuration. No additional layers.

## Prioritized findings and correction plan

| ID | Severity | Evidence / root cause | Planned correction and validation |
|---|---|---|---|
| F01 | High | materialize_one_file deletes source when source=destination with overwrite; broad hardlink fallback may overwrite concurrent destination; relative symlink target breaks | Validate policies before mutations; forbid aliases; exclusive copy; restrict fallback; regression tests |
| F02 | High | scan_candidate_files traverses sandbox and accepts file symlinks; destination folder symlinks escape output root | Prune sandbox and all symlinks; guard explicit paths and destination ancestors; synthetic traversal tests without creating sandbox contents |
| F03 | High | All six stages reuse second-resolution run directories with exist_ok=True | Exclusive unique run allocation; completion-aware prior-run selection; collision tests |
| F04 | High | Notebook 01/master silently scan cwd when input missing; all notebooks derive workspace from launch cwd; master uses undefined plt | Explicit absolute workspace environment variable; informative prerequisite errors; normal imports; fresh kernel smoke execution |
| F05 | Medium | Detector wrappers catch all failures as empty detections; tests download/infer real models | Warn on detector failure; record/test fallback visibility; mock external inference in automated tests |
| F06 | High | Cluster manifest row ordering/shape/finite vectors never checked before assigning labels | Validate and align embedding_row mapping; reject malformed arrays/manifests before output; regression tests |
| F07 | Medium | Reassignment overwrites HDBSCAN probabilities with heuristic values and stale diagnostic flags; workflow changes global RNG; figures leak | Retain original diagnostics, label heuristics explicitly; refresh flags; preserve caller RNG; close saved figures; tests |
| F08 | Medium | Config thresholds differ from workflow, invalid settings reach I/O; min crop field unused | Align authoritative defaults, validate settings at boundaries; document compatibility fields; config tests |
| F09 | Medium | Naming concat fails on no crop samples; create_named_output without plan silently does nothing; named planner duplicates Stage 5 and lacks validation | Actionable errors, shared destination plan and validation, truthful summaries; integration tests |
| F10 | Medium | Empty perceptual frames lack schemas; parallel audit ordering nondeterministic; EXIF applied after conversion loses orientation | Stable schemas/order and EXIF before RGB; tests |
| F11 | Medium | README imports nonexistent run_pipeline and wrong dataclass arguments, asserts calibrated thresholds/cosine metric/strict traversal protection | Synchronize APIs, limitations, distances, hardlink semantics, installation, notebook setup; validate examples/links |
| F12 | Medium | Notebook/dev installation lacks Jupyter execution tools; pytest ignores only root sandbox; no CI | notebook/dev optional groups, nested exclusion, isolated offline CI, package build/install checks |
| F13 | Low | Ruff clean Python but notebook imports/code not linted; inaccurate Markdown including white alpha compositing/BK-tree guarantees | Cell-aware Ruff lint/format; preserve narrative and correct unsupported claims |

## Notebook inventory and migration matrix

All seven notebooks use nbformat 4.5, python3 kernel, declared Python 3.11.14. No magics, shell installs, sys.path edits, classes, local function definitions, credentials or external API calls in notebook cells. External model inference is delegated to package modules; parameters and stage results are notebook globals. Original counts below.

| Path under notebooks/ | Role; original Markdown/code/output counts | Keep and improve | Extract/reuse | Validation and risk |
|---|---|---|---|---|
| 01_dataset_audit.ipynb | audit and duplicate inspection; 10/8/0 | Preserve methodology, parameters, intermediate summaries and limitations; add prerequisites, explicit paths, checks and interpretation | Shared workspace resolution only; reuse existing domain modules and workflows | Structure, Ruff, normal imports, isolated fresh kernels with synthetic model substitutes; live model quality unavailable |
| 02_crop_preparation.ipynb | crop preprocessing and visual review; 9/7/0 | Preserve methodology, parameters, intermediate summaries and limitations; add prerequisites, explicit paths, checks and interpretation | Shared workspace resolution only; reuse existing domain modules and workflows | Structure, Ruff, normal imports, isolated fresh kernels with synthetic model substitutes; live model quality unavailable |
| 03_ccip_embedding_extraction.ipynb | embedding generation and norm evaluation; 7/5/0 | Preserve methodology, parameters, intermediate summaries and limitations; add prerequisites, explicit paths, checks and interpretation | Shared workspace resolution only; reuse existing domain modules and workflows | Structure, Ruff, normal imports, isolated fresh kernels with synthetic model substitutes; live model quality unavailable |
| 04_hdbscan_clustering.ipynb | clustering experimentation and diagnostics; 8/6/0 | Preserve methodology, parameters, intermediate summaries and limitations; add prerequisites, explicit paths, checks and interpretation | Shared workspace resolution only; reuse existing domain modules and workflows | Structure, Ruff, normal imports, isolated fresh kernels with synthetic model substitutes; live model quality unavailable |
| 05_non_destructive_folder_materialization.ipynb | operational materialization and validation; 7/5/0 | Preserve methodology, parameters, intermediate summaries and limitations; add prerequisites, explicit paths, checks and interpretation | Shared workspace resolution only; reuse existing domain modules and workflows | Structure, Ruff, normal imports, isolated fresh kernels with synthetic model substitutes; live model quality unavailable |
| 06_cluster_naming_with_anime_tagger.ipynb | tagging evaluation and naming; 8/6/0 | Preserve methodology, parameters, intermediate summaries and limitations; add prerequisites, explicit paths, checks and interpretation | Shared workspace resolution only; reuse existing domain modules and workflows | Structure, Ruff, normal imports, isolated fresh kernels with synthetic model substitutes; live model quality unavailable |
| pipeline_master.ipynb | complete staged demonstration; 9/8/1 | Preserve methodology, parameters, intermediate summaries and limitations; add prerequisites, explicit paths, checks and interpretation | Shared workspace resolution only; reuse existing domain modules and workflows | Structure, Ruff, normal imports, isolated fresh kernels with synthetic model substitutes; live model quality unavailable |

Only master cell 2 has execution count 1 and one stdout output containing user-local paths. Preserve it as user work; document that it is a historical configuration snapshot. All other outputs are absent. Stage notebooks 02–06 auto-discover previous runs, with a risk of mixing datasets; expose explicit run selection and document precedence. Notebook 03 has unconditional norm assertions despite a normalization switch; notebook 04 equates heuristic epsilon to a model threshold; notebook 05 understates hardlink mutation coupling; notebook 06 thresholds are experimental and differ from library defaults.

## README, formatting, emoji and security inventory

- Only source README: root README.md, for users and contributors. It remains authoritative; docs will contain audit evidence, notebook migration detail and validation report.
- Python baseline Ruff clean; notebooks have unused imports/order and code formatting debt. JSON/TOML parse; Markdown factual inaccuracies and command gaps require corrections.
- No decorative emoji found in the 106 inspected files, including every notebook cell and generated CSV/JSON. Preserve accented text and meaningful arrows/math.
- Credential pattern scan found no token/private-key pattern; manual source review found no embedded credentials or shell construction. Generated ignored audit CSVs contain personal absolute paths: preserve locally, never commit or execute referenced paths. Image decoding is not a security sandbox. Destination aliasing/symlinks are confirmed filesystem risks.
- Generated egg-info is stale ignored metadata; refresh through packaging. Three ignored historical audit run directories contain 23,864 metadata rows each; preserve all. No file qualifies for deletion with strong evidence.

## Execution and delivery strategy

Characterize safety defects with regressions; implement filesystem/run correctness first, then config and workflow diagnostics, then notebooks and tooling, then accurate documentation. Validate each area and stage exact paths into focused commits. Leave the user-owned master notebook unstaged. Use `.audit/` for transient files, execution copies, logs and build outputs. Unit/integration and notebook smoke tests must use synthetic images and patched inference; they do not establish real model performance. Attempt actual local notebook execution up to unavailable inputs/models, report exact blocker. Review all changes independently at the end and compare protected Git index/status metadata without reading protected contents.

## Complete non-protected file inventory

- `.gitattributes`
- `.gitignore`
- `README.md`
- `pyproject.toml`
- `.vscode/settings.json`
- `notebooks/01_dataset_audit.ipynb`
- `notebooks/02_crop_preparation.ipynb`
- `notebooks/03_ccip_embedding_extraction.ipynb`
- `notebooks/04_hdbscan_clustering.ipynb`
- `notebooks/05_non_destructive_folder_materialization.ipynb`
- `notebooks/06_cluster_naming_with_anime_tagger.ipynb`
- `notebooks/pipeline_master.ipynb`
- `notebooks/anime_character_pipeline/runs/01_dataset_audit_20261004_134630/reports/scan_report.json`
- `notebooks/anime_character_pipeline/runs/01_dataset_audit_20261004_134630/reports/summary.json`
- `notebooks/anime_character_pipeline/runs/01_dataset_audit_20261004_134630/tables/exact_duplicate_groups.csv`
- `notebooks/anime_character_pipeline/runs/01_dataset_audit_20261004_134630/tables/image_metadata.csv`
- `notebooks/anime_character_pipeline/runs/01_dataset_audit_20261004_134630/tables/invalid_images.csv`
- `notebooks/anime_character_pipeline/runs/01_dataset_audit_20261004_134630/tables/perceptual_duplicate_candidate_pairs.csv`
- `notebooks/anime_character_pipeline/runs/01_dataset_audit_20261004_134630/tables/perceptual_duplicate_groups.csv`
- `notebooks/anime_character_pipeline/runs/01_dataset_audit_20261004_134630/tables/valid_images.csv`
- `notebooks/anime_character_pipeline/runs/01_dataset_audit_20261004_141109/reports/scan_report.json`
- `notebooks/anime_character_pipeline/runs/01_dataset_audit_20261004_141109/reports/summary.json`
- `notebooks/anime_character_pipeline/runs/01_dataset_audit_20261004_141109/tables/exact_duplicate_groups.csv`
- `notebooks/anime_character_pipeline/runs/01_dataset_audit_20261004_141109/tables/image_metadata.csv`
- `notebooks/anime_character_pipeline/runs/01_dataset_audit_20261004_141109/tables/invalid_images.csv`
- `notebooks/anime_character_pipeline/runs/01_dataset_audit_20261004_141109/tables/perceptual_duplicate_candidate_pairs.csv`
- `notebooks/anime_character_pipeline/runs/01_dataset_audit_20261004_141109/tables/perceptual_duplicate_groups.csv`
- `notebooks/anime_character_pipeline/runs/01_dataset_audit_20261004_141109/tables/valid_images.csv`
- `notebooks/anime_character_pipeline/runs/01_dataset_audit_20261004_183615/reports/scan_report.json`
- `notebooks/anime_character_pipeline/runs/01_dataset_audit_20261004_183615/reports/summary.json`
- `notebooks/anime_character_pipeline/runs/01_dataset_audit_20261004_183615/tables/exact_duplicate_groups.csv`
- `notebooks/anime_character_pipeline/runs/01_dataset_audit_20261004_183615/tables/image_metadata.csv`
- `notebooks/anime_character_pipeline/runs/01_dataset_audit_20261004_183615/tables/invalid_images.csv`
- `notebooks/anime_character_pipeline/runs/01_dataset_audit_20261004_183615/tables/perceptual_duplicate_candidate_pairs.csv`
- `notebooks/anime_character_pipeline/runs/01_dataset_audit_20261004_183615/tables/perceptual_duplicate_groups.csv`
- `notebooks/anime_character_pipeline/runs/01_dataset_audit_20261004_183615/tables/valid_images.csv`
- `scripts/run_audit.py`
- `scripts/run_clustering.py`
- `scripts/run_crop_preparation.py`
- `scripts/run_embeddings.py`
- `scripts/run_materialization.py`
- `scripts/run_naming.py`
- `scripts/run_pipeline.py`
- `src/anime_character_organizer/__init__.py`
- `src/anime_character_organizer/config.py`
- `src/anime_character_organizer/exceptions.py`
- `src/anime_character_organizer/clustering/__init__.py`
- `src/anime_character_organizer/clustering/algorithm.py`
- `src/anime_character_organizer/clustering/centroids.py`
- `src/anime_character_organizer/clustering/diagnostics.py`
- `src/anime_character_organizer/data/__init__.py`
- `src/anime_character_organizer/data/audit.py`
- `src/anime_character_organizer/data/duplicates.py`
- `src/anime_character_organizer/embeddings/__init__.py`
- `src/anime_character_organizer/embeddings/extraction.py`
- `src/anime_character_organizer/embeddings/normalization.py`
- `src/anime_character_organizer/materialization/__init__.py`
- `src/anime_character_organizer/materialization/operations.py`
- `src/anime_character_organizer/materialization/planner.py`
- `src/anime_character_organizer/materialization/validation.py`
- `src/anime_character_organizer/preprocessing/__init__.py`
- `src/anime_character_organizer/preprocessing/cropping.py`
- `src/anime_character_organizer/preprocessing/detection.py`
- `src/anime_character_organizer/preprocessing/geometry.py`
- `src/anime_character_organizer/tagging/__init__.py`
- `src/anime_character_organizer/tagging/aggregation.py`
- `src/anime_character_organizer/tagging/extraction.py`
- `src/anime_character_organizer/tagging/normalization.py`
- `src/anime_character_organizer/tagging/renaming.py`
- `src/anime_character_organizer/tagging/sampling.py`
- `src/anime_character_organizer/utils/__init__.py`
- `src/anime_character_organizer/utils/hashing.py`
- `src/anime_character_organizer/utils/naming.py`
- `src/anime_character_organizer/utils/paths.py`
- `src/anime_character_organizer/utils/runs.py`
- `src/anime_character_organizer/utils/serialization.py`
- `src/anime_character_organizer/utils/structures.py`
- `src/anime_character_organizer/utils/time.py`
- `src/anime_character_organizer/utils/validation.py`
- `src/anime_character_organizer/visualization/__init__.py`
- `src/anime_character_organizer/visualization/contact_sheet.py`
- `src/anime_character_organizer/visualization/plots.py`
- `src/anime_character_organizer/workflows/__init__.py`
- `src/anime_character_organizer/workflows/audit.py`
- `src/anime_character_organizer/workflows/cluster.py`
- `src/anime_character_organizer/workflows/crop.py`
- `src/anime_character_organizer/workflows/embed.py`
- `src/anime_character_organizer/workflows/materialize.py`
- `src/anime_character_organizer/workflows/naming.py`
- `src/anime_character_organizer/workflows/pipeline.py`
- `src/anime_character_organizer.egg-info/PKG-INFO`
- `src/anime_character_organizer.egg-info/SOURCES.txt`
- `src/anime_character_organizer.egg-info/dependency_links.txt`
- `src/anime_character_organizer.egg-info/requires.txt`
- `src/anime_character_organizer.egg-info/top_level.txt`
- `tests/__init__.py`
- `tests/integration/__init__.py`
- `tests/integration/test_pipeline_workflows.py`
- `tests/unit/__init__.py`
- `tests/unit/test_clustering.py`
- `tests/unit/test_data.py`
- `tests/unit/test_embeddings.py`
- `tests/unit/test_materialization.py`
- `tests/unit/test_preprocessing.py`
- `tests/unit/test_tagging.py`
- `tests/unit/test_utils.py`
