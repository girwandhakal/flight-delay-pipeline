---
goal: Comprehensive Code Audit, Simplification, and Documentation of Pipeline Scripts
version: 1.0
date_created: 2026-11-18
last_updated: 2026-11-18
owner: Antigravity
status: 'Planned'
tags:
  - refactor
  - code-quality
  - simplification
  - documentation
---

# Introduction

![Status: Planned](https://img.shields.io/badge/status-Planned-blue)

This plan defines the step-by-step refactoring, simplification, cleanup, and documentation for the entire data and model pipeline codebase under the `scripts/` directory. The goal is to eliminate dead code, simplify overly dense or convoluted expressions, add type annotations and PEP 257 docstrings, format logic with clean spacing and clear variable names, and provide natural one-line comments explaining non-obvious logic while preserving all pipeline contracts, behavior, and test guarantees.

## 1. Requirements & Constraints

- **REQ-001**: Audit all Python files under `scripts/` including `scripts/progress.py`, `scripts/results.py`, `scripts/weather_cache.py`, `scripts/bts/`, `scripts/origin_weather/`, `scripts/destination_weather/`, and `scripts/aircraft_rotation/`.
- **REQ-002**: Remove all dead code, unused imports, and redundant logic without breaking public interfaces or test contracts.
- **REQ-003**: Replace cryptic, overly dense, or multi-nested expressions with clear, well-named, readable Python constructs.
- **REQ-004**: Add complete PEP 257 docstrings to all functions, methods, and modules detailing arguments, return values, and exceptions.
- **REQ-005**: Add Python type hints across all function signatures to enhance static analysis and IDE readability.
- **REQ-006**: Add concise, natural single-line comments for non-obvious domain rules, algorithms, and PySpark transformations.
- **CON-001**: Preserve exact functional parity, numerical precision, output file structures, parquet schemas, and JSON metadata schemas.
- **CON-002**: Do not alter external interfaces consumed by `pipeline/main.py` or existing tests under `tests/`.
- **CON-003**: No code changes shall be executed until the user explicitly reviews and approves this plan.
- **GUD-001**: Follow PEP 8 style conventions, consistent 4-space indentation, and clean vertical whitespace.
- **PAT-001**: Prefer explicit readable variables over dense nested generator expressions or inline lambdas.

## 2. Implementation Steps

### Implementation Phase 1: Shared Utilities Refactoring

- GOAL-001: Clean, simplify, and document shared pipeline utilities in `scripts/progress.py`, `scripts/results.py`, and `scripts/weather_cache.py`.

| Task | Description | Completed | Date |
|------|-------------|-----------|------|
| TASK-001 | In `scripts/progress.py`, add type annotations, PEP 257 docstrings, and clean format strings for stage timing context manager. | | |
| TASK-002 | In `scripts/results.py`, add type hints and docstrings for `replace_file`, `records`, and `update_results`, with comments explaining atomic write and history snapshotting. | | |
| TASK-003 | In `scripts/weather_cache.py`, break down the dense 6-clause compound condition in `find_gold` into descriptive variables, rename single-letter variables (`e` to `station_entry`), and add module/function docstrings. | | |

### Implementation Phase 2: BTS Pipeline Modules Refactoring

- GOAL-002: Streamline, simplify, and document the core BTS download, clean, feature engineering, training, and evaluation scripts.

| Task | Description | Completed | Date |
|------|-------------|-----------|------|
| TASK-004 | In `scripts/bts/download.py`, convert top-of-function comment blocks into standard docstrings, add type annotations, and clarify retry and atomic rename logic. | | |
| TASK-005 | In `scripts/bts/clean.py`, remove unused `import os`, document `clean_bts_data` and `get_spark_session`, clean up aggregation metrics unpacking, and add comments on quarantine and filtering criteria. | | |
| TASK-006 | In `scripts/bts/recent_delays.py`, fix `type(window) is not int` to `not isinstance(window, int)`, break down dense nested comprehensions in `add_recent_delays`, add step-by-step comments for the timeline-merge logic, and add docstrings to all functions. | | |
| TASK-007 | In `scripts/bts/build_features.py`, eliminate duplicate `config.json` file reads in `build_features`, remove redundant runtime config calls, document functions, and add comments on cyclical feature encoding. | | |
| TASK-008 | In `scripts/bts/train.py`, replace obscure ternary dict unpacking in `algorithms[name]` with explicit conditional dictionary construction, safely initialize `vector_dimension` and `imputation_means` before loops, simplify `build_preprocessor` SQL assembly, and add docstrings. | | |
| TASK-009 | In `scripts/bts/evaluate.py`, remove unused `import os`, extract a small helper for POSIX relative path conversion, add type hints and docstrings, and simplify subgroup metrics unpacking. | | |

### Implementation Phase 3: Origin & Destination Weather Refactoring

- GOAL-003: Refactor, clean, and document weather acquisition, decoding, cleaning, and matching modules.

| Task | Description | Completed | Date |
|------|-------------|-----------|------|
| TASK-010 | In `scripts/origin_weather/download.py`, extract a clean `haversine_km` helper for airport-to-station distance calculation, add docstrings and type hints, and comment the station matching rules. | | |
| TASK-011 | In `scripts/origin_weather/measurements.py`, decouple numeric missing flags from the 12-item `MISSING_FLAGS` list to avoid silent truncation in `zip`, document regex capture groups in `cloud_conditions`, and add docstrings explaining weather decoding. | | |
| TASK-012 | In `scripts/origin_weather/clean.py`, unpack the dense cache-check condition in `clean_weather` into named booleans, add comments on FM-15 vs FM-16 priority and SHA-256 tie-breaking, and add complete docstrings. | | |
| TASK-013 | In `scripts/origin_weather/build_features.py`, clarify hourly bucket matching logic (`match_hour` vs `weather_hour`) with inline comments, clean up column renaming expressions, and add comprehensive docstrings and type annotations. | | |
| TASK-014 | In `scripts/destination_weather/build_features.py`, remove unused dead function `validate_parent_data`, add explicit `enriched.unpersist()` in a finally block, add docstrings, and document the civil-date envelope lookup. | | |

### Implementation Phase 4: Aircraft Rotation & Package Initialization

- GOAL-004: Simplify and document aircraft rotation feature construction, experiment runner, and package metadata.

| Task | Description | Completed | Date |
|------|-------------|-----------|------|
| TASK-015 | In `scripts/aircraft_rotation/build_features.py`, format long chained `when` expressions cleanly, add explanatory comments for ambiguous history boundaries and turnaround calculations, and add docstrings to all functions. | | |
| TASK-016 | In `scripts/aircraft_rotation/run.py`, simplify compound validation assertions, add docstrings and type hints to all helper functions, and comment the W2 vs H1 paired comparison flow. | | |
| TASK-017 | In `scripts/bts/__init__.py`, `scripts/origin_weather/__init__.py`, `scripts/destination_weather/__init__.py`, and `scripts/aircraft_rotation/__init__.py`, add package-level docstrings describing each subpackage's role. | | |

### Implementation Phase 5: Verification and Test Suite Validation

- GOAL-005: Validate that all refactored scripts pass full unit, synthetic, and integration test suites.

| Task | Description | Completed | Date |
|------|-------------|-----------|------|
| TASK-018 | Run the synthetic regression test suite using `cs523_project_env` python and verify 100% test pass rate across all test files. | | |
| TASK-019 | Verify code style, formatting, and lack of syntax errors using static inspection and compilation checks. | | |

## 3. Alternatives

- **ALT-001**: Perform a major architectural rewrite merging origin and destination weather into a single generic module. Rejected because it would introduce unnecessary churn, violate scope boundaries, and risk breaking established contracts.
- **ALT-002**: Leave code as-is without documentation. Rejected because the codebase has several complex domain rules (e.g. METAR clock verification, timeline-merge delays, ambiguous rotation bounding) that benefit immensely from concise documentation and clean formatting.

## 4. Dependencies

- **DEP-001**: Python 3.11 runtime under `cs523_project_env`.
- **DEP-002**: PySpark 3.5.7 and Java 17 with Windows Hadoop binaries in `envs/hadoop/bin`.
- **DEP-003**: Existing regression test fixtures under `tests/`.

## 5. Files

- **FILE-001**: `scripts/progress.py` - shared stage timing context manager.
- **FILE-002**: `scripts/results.py` - atomic results publishing and history snapshotting.
- **FILE-003**: `scripts/weather_cache.py` - independent weather Gold cache discovery.
- **FILE-004**: `scripts/bts/download.py` - BTS raw zip download and extraction.
- **FILE-005**: `scripts/bts/clean.py` - BTS CSV cleaning into Silver Parquet.
- **FILE-006**: `scripts/bts/recent_delays.py` - rolling delay and holiday calendar feature builder.
- **FILE-007**: `scripts/bts/build_features.py` - BTS schedule features and chronological split assignment.
- **FILE-008**: `scripts/bts/train.py` - Logistic regression training and threshold selection.
- **FILE-009**: `scripts/bts/evaluate.py` - Test evaluation and subgroup analysis.
- **FILE-010**: `scripts/origin_weather/download.py` - Airport references and NOAA LCDv2 download.
- **FILE-011**: `scripts/origin_weather/measurements.py` - LCDv2 measurement and token decoding.
- **FILE-012**: `scripts/origin_weather/clean.py` - NOAA LCDv2 observation cleaning and quality reporting.
- **FILE-013**: `scripts/origin_weather/build_features.py` - Origin weather temporal feature matching.
- **FILE-014**: `scripts/destination_weather/build_features.py` - Destination weather feature matching.
- **FILE-015**: `scripts/aircraft_rotation/build_features.py` - Predecessor leg linkage and rotation features.
- **FILE-016**: `scripts/aircraft_rotation/run.py` - Paired validation experiment runner for rotation.
- **FILE-017**: `scripts/bts/__init__.py`, `scripts/origin_weather/__init__.py`, `scripts/destination_weather/__init__.py`, `scripts/aircraft_rotation/__init__.py` - package inits.

## 6. Testing

- **TEST-001**: Run `tests/test_progress.py` to verify stage timing formatting.
- **TEST-002**: Run `tests/test_download_bts.py` and `tests/test_bts_cleaning.py` to verify BTS download and cleaning logic.
- **TEST-003**: Run `tests/test_build_features.py` and `tests/test_recent_delays.py` to verify BTS features, rolling windows, and holiday calendar features.
- **TEST-004**: Run `tests/test_bts_train.py` and `tests/test_bts_evaluate.py` to verify model training, threshold tuning, and evaluation calculations.
- **TEST-005**: Run `tests/test_origin_weather.py`, `tests/test_origin_weather_measurements.py`, `tests/test_weather_silver_cache.py`, and `tests/test_weather_gold_cache.py`.
- **TEST-006**: Run `tests/test_destination_weather.py` and `tests/test_aircraft_rotation.py`.
- **TEST-007**: Run full pytest test suite across all 14 test modules to confirm zero regressions.

## 7. Risks & Assumptions

- **RISK-001**: Subtle behavioral changes during PySpark expression simplification. Mitigation: Preserve exact column names, expressions, types, and verify using existing synthetic test fixtures.
- **ASSUMPTION-001**: No real-data model retraining or full downloads are required for this code quality and refactoring task; all verification is conducted via synthetic fixtures.

## 8. Related Specifications / Further Reading

- [Repository Root Instructions (AGENTS.md)](file:///c:/Users/g_dha/OneDrive%20-%20The%20University%20of%20Alabama/Uni/Fall%202026/CS%20523/Project/AGENTS.md)
- [Project Readme](file:///c:/Users/g_dha/OneDrive%20-%20The%20University%20of%20Alabama/Uni/Fall%202026/CS%20523/Project/README.md)
- [BTS EDA and Pipeline Documentation](file:///c:/Users/g_dha/OneDrive%20-%20The%20University%20of%20Alabama/Uni/Fall%202026/CS%20523/Project/docs/specs/)
