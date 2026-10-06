# Repository professionalization audit

## A. Executive summary

The initial repository already had a coherent `src` package, seven explanatory notebooks, seven argparse scripts and 38 passing tests. Passing checks concealed source-deletion/overwrite risks, run-directory collisions, unsafe discovery, ambiguous detector fallback, invalid documentation examples, fragile notebook paths and weak manifest/diagnostic contracts.

The existing architecture was retained. Corrections strengthened filesystem boundaries, run allocation, configuration, tabular schemas, embedding row identity, original versus reassigned clustering diagnostics, strict JSON reporting, and failure visibility. Notebook code and Markdown were upgraded without reducing their narrative: each retains all its original code-cell stages and gains prerequisites and interpretation. Shared notebook paths replace duplicated cwd-based resolution; naming now reuses the destination planner and validation.

Final local validation: **109 tests passed**, all seven notebooks passed fresh kernels with model substitutes, Ruff checks passed, MyPy passed all 47 package files, editable installation/dependency checks passed, and both source archive and wheel built. Wheel import was verified from a separate target installation using the validated runtime dependencies. All seven notebooks also passed bounded actual-model execution on synthetic images; detection produced no failure warnings, CCIP generated ten 768-dimensional vectors, real HDBSCAN ran, and tagging processed ten samples without errors. No semantic names were accepted. These are interface/execution observations, not identity-accuracy results. There is no labelled identity-quality benchmark or cross-platform validation.

Verdict: **Professionally organized with documented limitations**. Confidence is high for tested local behavior and notebook orchestration, moderate for research use, and insufficient to claim calibrated identity accuracy or universal platform support. The master notebook is intentionally uncommitted because its edits overlap pre-existing user execution state.

## B. Scope and sandbox exclusion

The discovery plan lists all 106 original relevant files: package source, scripts, tests, configuration, root README, seven notebooks, ignored editor settings, generated egg-info and three ignored historical audit runs. Each notebook cell was inspected before modification/execution. Generated CSV/JSON files were inspected for schema, counts and sensitive-content risks; their referenced external datasets were not accessed.

No AGENTS.md, API schemas, prompts, shell scripts, templates, lockfile, contribution guide, CI, environment example or standalone license text existed in the inspected tree. CI and an environment example were added because missing execution tooling and unclear settings were confirmed gaps. No speculative schema/model/template layers were created.

Every directory named `sandbox` was pruned before traversal. Symlinks were not followed. Tests use simulated directory names to prove pruning without creating, opening or accessing protected contents. Final safety uses Git index/status/diff path metadata, not content hashes. Installed third-party environments, Git internals and transient caches are not project source audit targets. No conclusions about protected implementation are possible or claimed.

## C. Initial Git state

- Root: `/home/ssparzival/Proyectos/GitHub/Automatic-Organization-of-Anime-Character-Images`.
- Branch: `main`; original HEAD: `7d954307e27da38582d44f8972b3955a05a8a806`.
- Initial staged changes: none. Initial unstaged work: `notebooks/pipeline_master.ipynb` only. No reported untracked user files.
- User change: configuration code cell execution count became 1, one stdout output containing local paths was added, and kernel display name changed to `.venv`.
- Those outputs/count/kernel metadata were preserved exactly, verified against `.audit/baseline.json`. All master enhancements remain unstaged. `.audit/master-upgrade.diff` separates this task's changes from the user's original baseline.
- Historical generated audit runs and uncertain artifacts remain intact. Only exact reviewed paths were staged. No branches were switched and no history was rewritten.

## D. Project contract

Purpose: propose visual character groups and optional semantic names for local anime image collections. Users: dataset curators, researchers and learners. Workflow: audit → representative cropping → CCIP features → normalized Euclidean HDBSCAN → organize original images → optional tag voting/named output.

Inputs: images, explicit stage settings and preceding run CSV/NumPy artifacts. Outputs: unique run directories, metadata/duplicate/review tables, crops, float32 feature arrays, cluster labels/centroids, contact sheets/plots, organized original-image trees, name votes/rename plans and JSON summaries. Exact duplicate non-representatives and failed crops/embeddings do not reach final output.

Interfaces: six `run_*` stage functions and `run_full_pipeline`; validated dataclasses describe settings; scripts call workflows with keyword arguments. Full pipeline result keys are `audit`, `crop`, `embed`, `cluster`, `materialize`, optional `naming`. Notebook globals are explicit experiment parameters and stage results, not reusable implementation.

Environment: setuptools package `anime-character-organizer`, import `anime_character_organizer`, version 0.1.0, Python >=3.10 declared, Python 3.11.14 Linux locally validated. Dependencies include Pillow, pandas, NumPy, imagehash, dghs-imgutils, hdbscan/scikit-learn and matplotlib. Model weights require caching/downloads. No application credentials are required. Configuration precedence is explicit function/CLI settings; notebooks require an absolute workspace environment variable and accept an absolute input override. `.env` is not auto-loaded.

Constraints: protected directories and input symlinks excluded; workspace outside source dataset; custom output new/empty; source images not moved or intentionally overwritten; hardlink bytes remain shared; size validation is not cryptographic equality; tagger labels do not repair mixed clusters. Thresholds are uncalibrated heuristics.

## E. Material findings

Each correction was validated by focused checks and the final suite unless a limitation is explicitly retained.

| ID | Severity/category | Affected path and evidence/root cause | Correction and evidence | Final status |
|---|---|---|---|---|
| F01 | High / data safety | materialization/operations.py: overwrite removed a source when source=destination; broad hardlink fallback copied over concurrent files; relative symlink target pointed into output | Prevalidate mode/policy, reject path/inode aliases, exclusive copy, restrict fallback errno, absolute symlink source; regressions reproduce then protect bytes | Corrected |
| F02 | High / protected paths | data/audit.py traversed protected directories and file symlinks; destination directory aliases escaped output | Pruned directory names/symlinks, checked_path guards explicit I/O paths and ancestors, safe leaf symlink targets validated before rendering; no protected fixture created | Corrected within documented non-adversarial boundary |
| F03 | High / persistence | all workflows reused second-resolution directories with exist_ok=True | Atomic unique reservation with microseconds/random suffix, completion-aware automatic prior-run selection; allocation sentinel test | Corrected |
| F04 | High / notebooks | 01/master substituted cwd when input missing; all paths depended on launch cwd; master used undefined plt | Shared absolute notebook_paths, no fallback dataset scan, explicit prerequisites/normal imports; seven fresh-kernel runs | Corrected; master unstaged |
| F05 | Medium / inference honesty | detection.py swallowed any import/download/inference error as no detection; tests exercised real models | Runtime warnings preserve reviewable fallback; unit/integration tests substitute external inference; visible failure regression | Corrected; crop completion still requires warning review |
| F06 | High / row identity | cluster.py assigned matrix labels to CSV row order without validating embedding_row/shape/finiteness | Validate complete unique row mapping and align rows; finite nonempty matrices; shuffled/duplicate mapping regressions | Corrected |
| F07 | Medium / diagnostic semantics | reassignment replaced probabilities/outliers, left stale flags, reseeded global RNG and leaked figures | Preserve hdbscan_label/probability/outlier_score, mark reassignment, refresh diagnostics and review reason, remove global RNG mutation, close persisted figures | Corrected; final reassigned scores remain explicitly heuristic |
| F08 | Medium / configuration | dataclass naming defaults differed from workflow; invalid sizes/policies reached I/O | Central parameter validation in configs/workflow boundaries; align supported defaults; preserve/document two compatibility fields | Corrected; compatibility fields intentionally retained |
| F09 | Medium / naming | empty sample concatenation gave obscure error; no-plan/output mismatch silently did nothing; duplicate planner lacked output validation | Actionable preflight, plan requirement, shared planner with retained 140-character naming stem limit, validation and typed failures; all-six-stage/error tests | Corrected |
| F10 | Medium / audit contracts | empty perceptual results had no columns; parallel results unordered; EXIF conversion order obscured orientation | Stable schemas/path order; orientation before RGB conversion; empty/ordering/EXIF regressions | Corrected |
| F11 | Medium / documentation | README imported nonexistent run_pipeline, wrong dataclass kwargs/result keys, misstated cosine metric/security/accuracy and CLI environment behavior | Authoritative README synchronized with APIs, outputs, CLI flags, notebook setup and research limitations; CLI/link/API checks | Corrected |
| F12 | Medium / packaging/automation | notebook execution tools undeclared; pytest excluded only root protected path; no CI | dev/notebook extras, nested exclusions, explicit package list, modern license metadata, local CI-equivalent checks and build/install tests | Corrected; remote CI/platform matrix not executed |
| F13 | Low / formatting/narrative | 52 notebook Ruff issues; Markdown claimed white alpha compositing, guaranteed BK-tree logarithmic search and calibrated thresholds | Cell-aware Ruff fixes; accurate methodology, provider requirements, interpretation; no blanket output removal or cell export | Corrected |
| F14 | Medium / iterable correctness | UnionFind consumed one-shot iterables twice; boolean Series ignored whitespace; optional CSV reader swallowed malformed CSVs | Rank from existing parent keys; scalar/vector boolean consistency; only empty/missing CSVs optional; regressions | Corrected |
| F15 | Medium / JSON/schema | json.dump emitted NaN/Infinity; stage table errors used raw KeyError | Recursive strict JSON conversion to null with allow_nan=False; required-column checks before processing; regression and artifact parsing | Corrected |
| F16 | Low / test isolation | fixed /tmp planner destination could collide with host files | Temporary fixture under configured TMPDIR; four focused tests pass | Corrected |
| F17 | Medium / backend equivalence | sklearn fallback used contrib min_samples without its self-count convention | Adjust/clamp sklearn parameter; mocked fallback contract and real primary tests | Corrected; outlier diagnostics still differ by backend |
| F18 | Medium / confidence thresholds | upstream WD14 defaults filtered character tags at 0.85 before project thresholds such as 0.65/0.70 could apply | Request full character scores from wrappers; apply operational threshold downstream | Corrected; model quality not inferred |
| F19 | Medium / publication metadata | MIT declared without standalone LICENSE text | Preserve declared license as SPDX metadata; disclose missing text rather than invent ownership/grant | Maintainer decision remains |
| F20 | Medium / scientific validation | no representative labelled identity benchmark or multi-platform evidence | Distinguish real operations, synthetic inference, bounded model execution and accuracy validation | Documented limitation |

The backend convention correction follows the [primary sklearn HDBSCAN documentation](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.HDBSCAN.html). Tag threshold controls were checked against the [upstream PixAI](https://dghs-imgutils.deepghs.org/main/api_doc/tagging/pixai.html) and [WD14 interfaces](https://dghs-imgutils.deepghs.org/main/api_doc/tagging/wd14.html).

## F. Architecture

Original organization already separated data, preprocessing, embeddings, clustering, materialization, tagging, visualization, utilities and workflows. The problems were contracts between those responsibilities, not the directory tree. No source files were moved and no ceremonial layers introduced.

Final direction: scripts/notebooks → workflows → domain modules → narrow utilities/configuration. `utils/paths.py` owns path guards/notebook workspace resolution; `utils/runs.py` allocates/discovers runs; `utils/serialization.py` owns CSV/JSON boundary handling. Domain modules retain algorithms and inference. Workflow modules own orchestration and artifact production. Naming reuses `materialization/planner.py` and `validation.py` rather than duplicating file planning. The public stage APIs and primary clustering defaults remain recognizable.

Intentional behavior changes: explicit protected/symlink path rejection; no missing-input fallback to cwd; unique run names; incomplete runs excluded from automatic selection; malformed manifests fail clearly; custom output reuse refused; failed materialization/all tagging fails instead of reporting success; strict JSON missing metrics become null; naming dataclass defaults aligned; sklearn fallback sample convention corrected. Existing output filenames/index ordering/default primary backend remain intact; named stems preserve their original maximum length.

Remaining limits: keyword-based workflow settings still coexist with descriptive dataclasses; two old compatibility fields are non-operational. Third-party table/array APIs are not fully typed. The system is not hardened against adversarial concurrent filesystem mutation.

## G. Notebook inventory and outcomes

All notebooks are nbformat 4.5, python3 kernel, declared Python 3.11.14. All original code-cell counts are retained. No functions/classes, shell installs, magics, sys.path hacks or credentials were found in source cells. Inputs/outputs and external services are delegated to package stages. Visible mutable state consists of experiment parameters, DataFrames, figures and returned stage results.

| Path under notebooks/ | Purpose, audience and classification | Original → final Markdown/code counts | Main corrections and final role | Synthetic / actual-model execution |
|---|---|---|---|---|
| 01_dataset_audit.ipynb | curators; data inspection/audit | 10/8 → 12/8 | Fail on missing input, shared extensions/paths, accurate pHash/alpha explanation, inspect invalid/exact/perceptual results; audit narrative | Passed / passed |
| 02_crop_preparation.ipynb | curators; preprocessing/visual review | 9/7 → 11/7 | Explicit previous-run contract/model prerequisites, correct JPEG sheets, fallback interpretation, display generated sheet; crop-quality narrative | Passed / passed |
| 03_ccip_embedding_extraction.ipynb | researchers; feature generation/evaluation | 7/5 → 9/5 | Correct CCIP name/model/provider claims, conditional normalization assertions, finite/row checks, failures and privacy interpretation | Passed / passed |
| 04_hdbscan_clustering.ipynb | researchers; experimentation/visualization | 8/6 → 10/6 | Heuristic distance claims corrected, original/reassigned diagnostics distinguished, plots and contact-sheet review, naming does not split clusters | Passed / passed |
| 05_non_destructive_folder_materialization.ipynb | operators; operational demonstration | 7/5 → 9/5 | New/empty output preflight, hardlink coupling/metadata costs, explicit validation assertion, size-versus-byte-equality explanation | Passed / passed |
| 06_cluster_naming_with_anime_tagger.ipynb | curators; naming evaluation/reporting | 8/6 → 10/6 | Inputs independent of Stage 5, sampling/voting denominator and explicit experimental thresholds, uncertain names/contact-sheet interpretation | Passed / passed |
| pipeline_master.ipynb | curators/researchers; complete staged demonstration | 9/8 → 11/8 | Absolute workspace, no cwd input fallback, missing plt import, honest heuristic claims, stage count/limitations and retained historical stdout | Passed / passed; left unstaged |

Only master has embedded source output: one original user stdout and execution count 1. All existing cell/kernel metadata was preserved. No useful outputs were stripped; the other six notebooks originally had no outputs. New synthetic executed copies and plots are retained privately under `.audit/`, not embedded as fabricated model results. All seven live executions also passed; the earlier offline missing-cache attempt remains recorded in U as a distinct unavailable-model check.

## H. Notebook migration map

| Source | Responsibility | Target/interface | Extraction rationale and behavioral validation |
|---|---|---|---|
| All seven configuration cells | Resolve input/workspace paths | utils/paths.py: notebook_paths() | Repeated cwd-dependent path expressions replaced by explicit absolute settings; cwd-change, missing/relative settings tests and fresh kernels |
| All seven stage cells | Orchestrate and inspect stages | Existing workflows.run_* interfaces | Retained concise calls and stage parameters; algorithms were already extracted before this audit, so no redundant export |
| Naming workflow called by notebook 06/master | Destination filename/collision planning | materialization/planner.py: build_materialization_plan() | Shared Stage 5 logic, preserves named output 140-character stem limit; long-stem and all-six-stage tests |
| Naming workflow called by notebook 06/master | Named output verification | materialization/validation.py: validate_materialized_files() | Reuse actual existence/size checks and fail on incomplete output; mode/error integration tests |

Methodology, parameters, exploratory calculations, plots, result inspection and conclusions remain in notebooks. No notebook was reduced to an unexplained wrapper or deleted.

## I. Notebook explanatory quality

Every notebook now identifies purpose, scope, prerequisites, inputs, outputs, configuration, method, intermediate checks, interpretation, limitations and conclusions/next steps. Stage introductions explain why each operation matters. Explanations target methodology and decision points rather than trivial syntax. The added prerequisites and conclusion cells supplement existing detailed sections.

01 interprets integrity versus identity and transitive pHash candidates. 02 interprets crop selection/fallback counts and displays a sheet. 03 verifies array/manifest invariants and explains normalized versus raw geometry. 04 retains plots, centroids/review summaries, shows a sheet and separates fitted scores from reassignment heuristics. 05 checks actual operations and validates outputs before use. 06 inspects accepted/uncertain votes, previews names and displays a sheet. Master explicitly chains runs and compares stage counts. Fresh-kernel execution removes evidence of hidden state; version/provider/model/data effects remain limitations.

## J. Source-code corrections

Regression coverage protects source aliases, invalid policies, relative symlinks, concurrent hardlink collisions, cross-device fallback, directory aliases, protected pruning, file-symlink exclusion, stable audit order, empty schemas, one-shot UnionFind, whitespace booleans, batch sizes, EXIF orientation, safe run allocation, workspace placement, row mapping, RNG preservation, figure closure, strict JSON and malformed manifests. Integration tests protect actual embedding/naming orchestration, failed model/file operations and all three materialization modes. See E for affected paths, causes and statuses.

## K. Configuration, models and schemas

No new domain model or API schema was invented. Existing dataclasses validate finite/ranged numeric settings, positive integer dimensions/batches/workers, modes/policies and required naming plan/output relationships. Naming defaults align with functions; notebook experimental overrides remain explicit. Workspace variables are documented in `.env.example`; no secrets/default credentials were added. Required CSV columns and complete embedding-row identity are checked. JSON metrics are portable null values rather than non-standard NaN/Infinity.

## L. Operational scripts and commands

All seven scripts retain minimal argparse orchestration and package-based imports. Each has --help and zero/one success/error exits. No script was added for a trivial notebook wrapper. Every script passed help and missing-input/upstream smoke tests from an unrelated cwd; audit also wrote real fixture reports successfully.

| Script | Important arguments | Input → output and outcome |
|---|---|---|
| run_audit.py | --input-dir, --project-dir, --phash-threshold, --max-workers | image folder → metadata/duplicate CSVs and summary; fixture execution passed |
| run_crop_preparation.py | --project-dir, --previous-run-dir, --filter-perceptual-duplicates, --crop-size, --crop-format | audit artifacts → crops/review manifests; workflow/missing-run checks passed |
| run_embeddings.py | --project-dir, --previous-run-dir, --model, --batch-size | crops → raw/normalized arrays and row manifest; orchestration tested |
| run_clustering.py | --project-dir, --previous-run-dir, --min-cluster-size, --min-samples, --cluster-selection-epsilon, --cluster-selection-method, --no-reassign-noise, --max-reassign-distance, --separate-review-folders | arrays/manifest → clustering diagnostics/folder plan; real HDBSCAN exercised |
| run_materialization.py | --project-dir, --previous-run-dir, --output-dir, --mode/--link-mode, --no-copy-fallback | folder plan → validated output tree; all modes tested |
| run_naming.py | --project-dir, --previous-clustering-run, --primary-tagger, --create-named-output/--apply-rename, --min-score, --min-share | cluster crops → votes/rename plan/optional additional tree; orchestration/error coverage |
| run_pipeline.py | --input-dir, --project-dir, --mode/--link-mode, clustering controls, --with-tagging | complete workflow → stage result dictionaries; synthetic full integration and missing-input exit passed |

`python -m tests.notebook_smoke` is a test utility, not an additional business CLI. It accepts --output-dir, --live and --timeout, validates exact notebook paths and writes separate execution evidence.

## M. Tests

Initial suite: 38 passed, 17.08 seconds. Fourteen added regressions initially failed and confirmed defects. Final broad suite: **109 passed, zero failed**, 12.87 seconds in the final recorded broad run before the last test-fixture-only adjustment; that affected file then passed all four focused tests. Final delivery rerun passed all 109 tests in 13.75 seconds, with zero failures.

Coverage is behavior-focused: unittest characterization; filesystem/configuration/JSON/schema regression tests; actual multi-stage integrations with model substitutes; 15 subprocess CLI cases; seven notebook structure contracts; seven fresh-kernel execution runs. No tests were weakened to conceal source failures. One CLI message-prefix assertion was corrected because both error formats were already actionable and returned status one. A synthetic clustering assertion was strengthened from at least one group to exactly two expected groups.

No coverage percentage, production credentials, copied production image dataset or fabricated accuracy result is used.

## N. README inventory

Only original source README: `README.md`, for users, researchers and contributors. Its APIs/parameter names/result keys, supported extensions, model/cache assumptions, metric claims, CLI environment behavior, tests, output semantics, notebook order and license status were inconsistent or overstated. It is now the authoritative entry point with links to this report, discovery plan and validation guide.

CLI examples were checked through parser help, audit fixture execution and error cases. Python imports/signatures/results were exercised in the full-pipeline integrations. Notebook links and local documentation links were checked. External technical references were verified using primary documentation. Interactive JupyterLab version was checked; no browser UI automation was performed. No redundant directory README copies were created. Generated output READMEs remain stage artifacts.

## O. Formatting and quality

Ruff performs Python/notebook formatting, linting and import ordering. The original 65 Python files were clean; 52 notebook lint violations were corrected. Final check covered 81 code/notebook files and passed. Nbformat validates every notebook; AST parses every code cell. MyPy covers all 47 package source files with permissive third-party import handling. TOML and CI YAML parse. Markdown local targets/headings/fences and JSON artifacts were checked using an explicit-path validation script. No shell source exists; CI shell commands are simple argument-safe operations. No overlapping formatter was introduced.

## P. Emoji audit

All original and new technical source, notebook cells, configuration, documentation and historical generated reports were searched without protected-directory traversal. No decorative emoji was found initially and none was introduced; zero files/cells required emoji deletion. No intentional source emoji exceptions exist. Accents, mathematical notation, arrows and multiplication symbols remain. Installed third-party packages/cached model artifacts are externally controlled content, not rewritten as project technical source.

## Q. Security and privacy

Manual source review and token/private-key pattern checks found no embedded credential. No secret value was used or reproduced. No unsafe shell construction or pickle loading was introduced; NumPy loads disallow pickle. Protected paths, input symlinks, output directory aliases, source inode aliases and concurrent destination collisions have boundary checks/regressions.

Generated manifests and retained notebook stdout reveal user-local paths; preserve privately and review before sharing. Historical ignored reports were not committed or used to access their referenced external datasets. Image decoding remains an untrusted-format risk; Pillow verification is not a sandbox. No comprehensive dependency CVE service scan or adversarial race-resistance test was performed. Linked images remain coupled to originals; use copy for independently editable results.

## R. Dependencies and packaging

Runtime dependency requirements were retained; no runtime dependency was removed or intentionally upgraded. Added developer requirements: nbclient, ipykernel, build, mypy, PyYAML; existing nbformat retained. New notebook extra includes JupyterLab, ipykernel, nbclient and nbformat. Ruff's minimum was raised to 0.6 for the configured notebook/quality workflow; setuptools minimum 77 supports SPDX license metadata. MIT identifier was retained, standalone license text remains missing.

Package names/version/Python constraint remain unchanged. Ten explicit packages replace recursive package discovery to exclude accidental/protected packages. No console entry point or lockfile was invented. Editable install, pip check, JupyterLab version, sdist/wheel build, archive-member safety and target wheel import passed. The build used a 49-file approved source snapshot containing no protected directories. No clean-machine/cross-platform installation is claimed.

Recorded versions: Python 3.11.14, NumPy 1.26.4, pandas 3.0.3, Pillow 12.2.0, dghs-imgutils 0.19.0, hdbscan 0.8.43, scikit-learn 1.8.0, pytest 9.1.1, Ruff 0.16.10, nbclient 0.11.0, nbformat 5.11.1, MyPy 2.4.0, setuptools 79.0.1 and JupyterLab 4.6.4.

## S. Files moved, renamed or deleted

None. Existing notebooks, source modules and historical artifacts remain in place. No uncertain artifact was deleted. Generated egg-info was refreshed by installation; caches/build/logs/execution copies reside under ignored `.audit/` and the environment under ignored `.venv/`. The uncertain standalone license issue requires an owner decision rather than file deletion. No empty source layer/module/directory was added.

## T. Git history

All commits below are new local commits, never amendments. Each staged only reviewed relevant exact non-protected paths. Their validation appears in U.

| Hash | Message | Purpose/validation |
|---|---|---|
| `b7c19f0` | test: document audit baseline and characterize safety defects | Baseline/migration plan and 14 intentionally failing characterization regressions. |
| `5c97c08` | fix: preserve sources and enforce safe deterministic file operations | Filesystem/order/schema corrections; 35 focused tests passed. |
| `20e8c5e` | fix: validate staged workflows and preserve auditable model diagnostics | Run/configuration/diagnostic/naming corrections; 77 tests and scoped type/lint checks. |
| `208bb43` | fix: validate manifest schemas and preserve strict diagnostic contracts | Manifest/JSON/backend/mode contracts; 108 broad and 27 focused tests plus full typing. |
| `74d0efd` | build: configure notebook dependencies and protected quality checks | Dependency groups/package/CI configuration; editable install, YAML/TOML, build and CLI smoke checks. |
| `29063e0` | test: isolate destination collision fixtures from host files | Host-independent fixture; four focused tests passed. |
| `107a9a2` | refactor: preserve notebook narratives and validate fresh kernel workflows | Six stage notebooks plus repeatable fresh-kernel test utility; 109 tests and 7 notebook executions passed. |

The final documentation commit is identified by final HEAD in the delivery response; its own hash cannot be embedded inside its committed file without changing that hash. The master notebook remains unstaged to preserve the boundary around pre-existing work.

## U. Validation matrix and command record

Use `.venv/bin/python` for every `python` command below. Repeated invocations are grouped by command; results identify the failing characterization/intermediate runs and final state. Shell writing/reading and exact commit staging are listed separately. Full tool-call transcript contains the literal heredocs and path lists.

| Command / operation | Purpose | Result and limitation |
|---|---|---|
| pwd; git rev-parse --show-toplevel; git rev-parse HEAD; git branch --show-current; git status --short --branch; git log --oneline --decorate -15 | Required Git baseline | Passed; branch main, initial HEAD and sole master user change recorded |
| rg -n -i 'anime\|Automatic-Organization' MEMORY.md | Quick prior-context lookup | No relevant hit; no memory-derived findings used |
| find with -name sandbox/-name .git/-name .venv/-name cache -prune and no symlink following | File/instruction inventory | 106 relevant original files; no project AGENTS.md found; protected content excluded |
| git ls-files with :!sandbox/** and :!**/sandbox/** | Tracked inventory and approved build input list | Passed; used only Git path metadata |
| cat/sed exact known source/config/test files; notebook JSON cell printer | Source/configuration and exhaustive notebook review | Completed outside protected paths; truncated output was reread in smaller groups |
| Python os.walk with protected/env/cache pruning | CSV/JSON schemas, emoji and credential pattern inventory | Completed; historical paths not followed; no credential/decorative emoji found |
| pytest tests --noconftest --ignore-glob='**/sandbox/**' -q | Original baseline | 38 passed; 17.08s; original tests attempted model inference and used system temporary files |
| pytest tests/unit/test_safety_regressions.py --ignore-glob='**/sandbox/**' -q | Characterize 14 defects | 14 failures before correction; then all passed |
| pytest focused data/materialization/embedding/utils/safety paths | Validate first correction area | 35 passed |
| pytest tests --ignore-glob='**/sandbox/**' -q | Broad incremental/full validation | 52, 69, 77, 103, 108 and 109 passing checkpoints; final 109 passed in 13.75s; one intermediate CLI prefix assertion failure corrected |
| pytest tests/unit/test_workflow_contracts.py tests/unit/test_clustering.py --ignore-glob='**/sandbox/**' -q | Schema/backend/diagnostic focused check | 27 passed |
| pytest tests/unit/test_materialization.py --ignore-glob='**/sandbox/**' -q | Final fixture isolation check | 4 passed |
| ruff check [src scripts tests or exact notebooks] --force-exclude [--exclude sandbox]; ruff check --fix exact reviewed scopes | Lint/import validation and correction | Initial Python pass; 52 initial notebook fixes; intermediate syntax/undefined import issues repaired; final pass |
| ruff format [scopes] --force-exclude; ruff format --check src scripts tests notebooks --force-exclude --exclude sandbox | Formatting | Baseline 65 files; final 81 files pass; existing notebook metadata preserved |
| mypy --cache-dir .audit/mypy-cache; mypy src/anime_character_organizer --exclude '(^\|/)sandbox/' --cache-dir .audit/mypy-full | Type validation | Intermediate 4/scoped and 10/full errors corrected; final 47 source files pass; third-party types permissive |
| pip install -e '.[dev]'; pip install -e '.[dev,notebook]' with repository-local TMPDIR/PIP_CACHE_DIR | Editable/dev/notebook installation | Passed; generated metadata/environment updated within repository |
| pip check; jupyter lab --version | Dependency and interactive tool availability | Passed; no broken requirements; JupyterLab 4.6.4 |
| tests.notebook_smoke --output-dir .audit/notebook-smoke / notebook-final / notebook-release | Fresh kernel notebook validation | 7/7 passed on each synthetic checkpoint; kernels shut down explicitly; no real prediction claim |
| HF_HUB_OFFLINE=1 tests.notebook_smoke --live --timeout 30 --output-dir .audit/notebook-live | Safe uncached offline behavior | Audit/crop cells completed; crop fallback warnings; CCIP missing weights, downstream RunNotFoundError and master EmbeddingError; unavailable inference not called passing |
| HF_HUB_OFFLINE=0 tests.notebook_smoke --live --timeout 45 --output-dir .audit/notebook-network | Bounded actual model attempt | 7/7 passed with actual downloaded models; 10 CCIP vectors (768D), real HDBSCAN, 10 tagged samples/zero errors/zero accepted names; no accuracy claim |
| Python approved-file build snapshot; build --no-isolation .audit/build-source --outdir .audit/dist | Protected-input-safe build | Passed: sdist and wheel, no protected archive members |
| pip install --no-deps --target .audit/wheel-install <exact wheel>; python -I target-import check | Wheel smoke | Passed from target package location; dependencies reused, no clean-machine claim |
| tomllib.loads, yaml.BaseLoader, nbformat.validate, ast.parse, strict JSON loader | Configuration/schema/code structure | Passed: 90 tracked/new technical files, 56 strict JSON artifact reports, local Markdown targets/fences, master metadata/output preservation; no issues |
| rg TODO/FIXME/HACK/XXX/sys.path/shell/eval/pickle/absolute user path patterns with explicit sandbox exclusions | Static/safety review | No actionable placeholder/unsafe execution found; matches were tests, safe NumPy flag, documentation or cluster name templates |
| .audit/fix_safety.py, fix_workflows.py, finish_workflow_fixes.py, review_safety.py, upgrade_notebooks.py; exact Python/heredoc writes | Incremental implementation | Applied reviewed transformations to explicit paths; intermediate syntax/lint issues corrected before final validation |
| git diff --check/stat/name-status/diff; git diff --cached --check/stat/diff; git add -- <exact paths>; git commit -m <listed messages> | Incremental review and local history | Passed final checks; master excluded; no protected or unrelated user paths staged |
| Git protected-path index/status/name-status comparisons against initial metadata | Sandbox verification without content reads | Passed: initial protected index identical, no protected worktree/staged/history path changes; no content hashing/opening used |

## V. Remaining issues

- **Critical:** none confirmed after correction.
- **High:** no calibrated identity-quality evidence for representative real anime collections; do not infer purity from successful execution or reduced noise.
- **Medium:** standalone license text/owner confirmation before publication; Python 3.10 and other platforms not locally exercised; no dependency CVE database audit; no adversarial filesystem race hardening.
- **Low:** two non-operational dataclass compatibility fields; no lockfile/exact reproducibility pin; optional pandas/NumPy typing precision.
- **Optional:** labelled benchmark, reviewed multi-character/chibi/recolor examples, clean-machine distribution install, GPU-provider performance checks and remote CI run.
- **Blocked/unavailable during specific checks:** uncached offline model inference failed as expected. The subsequent bounded network attempt passed all seven notebooks with downloaded models and no detector-failure warnings. That resolves local model availability, but does not validate identity quality. No production credentials or private datasets were used.

## W. Final verdict

**Professionally organized with documented limitations.** High confidence in the executed local package, filesystem, reporting, CLI and notebook orchestration checks; moderate confidence in research workflow readiness. Scientific usefulness, calibrated accuracy and broad environment support require additional data and validation. No perfection claim is made.

## X. Safety confirmations

- No remote push, pull request, history rewrite, branch switch, amend, rebase or destructive Git cleanup occurred.
- No pre-existing user work was discarded or staged; the master's original stdout/count/kernel metadata remain intact and its file is uncommitted.
- No result was fabricated, missing model check called passing, or secret value reproduced.
- No project source/document/artifact outside the repository was edited. **Exception to the requested absolute filesystem confirmation:** the initial baseline's existing tests used default temporary files outside the repository and may have exercised upstream default model caches before isolation was established. Their temporary files were cleaned by their own fixtures. Later validation sets repository-local TMPDIR/cache/runtime directories. Consequently an unconditional “no file outside the repository was modified” statement would be inaccurate and is not asserted.
- No decorative source/document/notebook emoji remains; no intentional source exceptions. Mathematical/meaningful Unicode and externally controlled dependency/cache content are preserved.
- No file inside any protected directory was opened, read, inspected, modified, deleted, renamed, moved, created, formatted, generated, executed, extracted or staged. Verification uses Git path/index/status metadata only. No protected-content hashes or fixtures were created.

Final independent review checked all changed source responsibilities, notebook cells, test assertions, README claims, package/CI configuration, local links, source output metadata and the complete non-protected diff. Its additional fixes covered strict JSON, schema errors, fallback parameter convention, symlink-output sheet rendering and host-independent fixtures. Final Git metadata comparison found no protected path changes; only the master notebook is intentionally left uncommitted after the documentation commit.

The report is intentionally evidence-bounded: protected contents, dependency internals, real identity ground truth and unavailable environments are outside its assurance.
