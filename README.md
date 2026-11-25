# Scalable Flight Delay Prediction Pipeline

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![Apache Spark](https://img.shields.io/badge/Apache%20Spark-3.5-E25A1C?style=flat-square&logo=apachespark&logoColor=white)](https://spark.apache.org/)
[![PySpark ML](https://img.shields.io/badge/PySpark-MLlib-blue?style=flat-square)](https://spark.apache.org/mllib/)
[![Architecture](https://img.shields.io/badge/Data%20Architecture-Medallion%20(Bronze%E2%86%92Silver%E2%86%92Gold)-lightgrey?style=flat-square)](#data-architecture-and-pipeline-flow)

An end-to-end, reproducible **Big Data Machine Learning Pipeline** engineered in **PySpark** to predict commercial flight arrival delays ($\ge 15$ minutes, BTS `ArrDel15`) at scheduled departure.

Designed for scalability on large tabular time-series datasets, the pipeline processes over **6 million flight records** from the Bureau of Transportation Statistics (BTS) paired with millions of hourly surface meteorological observations from NOAA Local Climatological Data (LCDv2) and physical aircraft tail-number rotation sequences.

### Reusing prepared data

Each pipeline run checks its data dependencies before starting the corresponding Spark transformations. BTS Silver and Gold generations live under `data/{silver,gold}/bts_generations`. Model run manifests pin the full-feature Gold generation and record its source provenance. Keep older generations while a frozen run references them. The first run after migrating builds generation manifests and can require substantial additional disk space.

Origin and destination weather use their requested NOAA station-years. Rotation and en-route weather enrich all three splits. Rotation uses the original BTS archive window for training and validation and extends that window for test. TAF station selection, TAF Silver, and SIGMET Silver have separate caches. Replacing a verified TAF archive rebuilds its Silver table and en-route Gold while allowing rotation and SIGMET Silver to be reused.

Turnaround shortfall is the positive difference between typical ground time and the time from an observed prior arrival to the next scheduled departure. For each carrier and origin airport, the benchmark is the median of at least 20 historical turns lasting up to three hours. Training flights use the preceding UTC month. Validation and test use the same reference frozen at the validation boundary. The feature is marked missing when the arrival or benchmark is unavailable.

Outbound schedule slack is the scheduled gate-to-gate duration minus the historical 20th percentile for the reporting carrier and directional route. The fallback is the same directional route across carriers. Both references require 100 flights. The lookup admits months available before the training snapshot and freezes at the validation boundary for validation and test. Recovery-adjusted shortfall subtracts slack from turnaround shortfall. Both features retain signed values and distinguish missing from zero.

The default pipeline reuses the monthly CSVs in `data/bronze/bts/`. It uses the official BTS archive dates recorded in `config.json`, waits until the listed date has ended in all time zones, and pins the local CSV checksums. This is a retrospective reconstruction because the exact historical file vintages remain unverified. An optional schema-version 1 manifest at `data/bronze/bts_block_history/manifest.json` can instead supply independently verified vintages. Its `months` entries use `source_month`, `csv_path`, `sha256`, `available_utc`, `source_url`, `evidence_url` and `evidence_note`. BTS Gold's `_block_time.json` records the history mode and coverage. Strict operational replay also requires timestamped schedules, aircraft assignments and event reporting.

Cache messages report hits, misses and elapsed lookup or preparation time. A completed `_stage.json` records the dependency key, output fingerprints, required sidecars and cold-build duration. A missing marker or a changed output triggers a fresh generation. Interrupted generations may remain on disk. The first run with this workflow builds new rotation and en-route Gold generations. Earlier runs and cached data stay at their original paths.

---

## Problem and Motivation

Flight delays cost the global aviation industry tens of billions of dollars annually in operational disruption, passenger compensation, and gate congestion. 

Accurate pre-departure flight delay forecasting poses significant data science challenges:
1. **Severe Temporal Data Leakage**: Naïve machine learning pipelines frequently incorporate outcome-derived features (e.g., actual departure delay or taxi time) or future weather observations that would not be available in real-world dispatch.
2. **Cascading Network Disruptions**: Delays do not occur in isolation. An incoming aircraft arriving late at a gate propagates downstream delays across subsequent flights on that physical airframe (aircraft rotation).
3. **Data Engineering Scale**: Joining multi-gigabyte continuous flight schedules with hourly weather readings across hundreds of US airports exceeds standard in-memory Pandas workflows, demanding distributed memory management, custom temporal bucket joins, and out-of-core execution.

The pipeline predicts arrival delays using one feature dataset and chronological validation.

---

## Model Inputs

- **BTS** supplies schedules, carrier and route, calendar effects, recent airport delays, taxi-out conditions, and outbound schedule slack.
- **Surface weather** matches NOAA observations at both airports using the scheduled departure cutoff.
- **Aircraft rotation** uses prior gate arrivals, scheduled turnaround, and recovery-adjusted shortfall.
- **Aviation weather** uses destination TAFs issued by departure and valid at scheduled arrival, plus convective SIGMETs active at departure along the route corridor.

* **Core Estimator**: Binomial Logistic Regression via L-BFGS optimization (`regParam=0.01`, $L_2$ Ridge penalty, `maxIter=100`, `tol=1e-6`).
* **Decision Optimization**: Dynamically searches 19 candidate thresholds ($\tau \in [0.05, 0.95]$ in increments of $0.05$) on validation data to maximize $F_1$-score on imbalanced delays (~20% prevalence).
* **Strict Temporal Splitting**:
  * **Train Cohort**: `2024-01-01` to `2024-09-30` (~4.2M flights)
  * **Validation Cohort**: `2024-10-01` to `2024-11-30` (~1.2M flights)
  * **Test Period**: `2024-12-01` to `2025-01-31` (potentially inspected, scored with frozen models)

The BTS stage computes taxi-out surge as the mean taxi time of other flights in the two-hour window ending at scheduled departure, minus the origin airport's 15th-percentile taxi time. It uses verified wheels-off clocks and caps individual observations at 120 minutes. Training uses earlier training months for the baseline. Validation and test share a frozen training baseline. When an airport has no baseline, the feature uses 12 minutes. Gold keeps missing window measurements null. The model fills missing rolling inputs with zero.

A normal pipeline run rebuilds BTS Gold and dependent stages when feature provenance changes.


---

## Tech Stack

| Layer | Technology | Purpose |
| :--- | :--- | :--- |
| **Distributed Engine** | **Apache Spark 3.5 / PySpark** | Large-scale parallel transformations, temporal bucket joins, and ML pipelines |
| **Runtime & Language** | **Python 3.11 / Java 17** | Core pipeline logic, typing, and JVM Spark backend execution |
| **Storage Architecture** | **Apache Parquet (Snappy)** | Partitioned, columnar Bronze $\to$ Silver $\to$ Gold Medallion storage |
| **Machine Learning** | **Spark MLlib** | StringIndexer, OneHotEncoder, Imputer, VectorAssembler, LogisticRegression |
| **Data Sources** | **BTS TranStats / NOAA LCDv2** | US DOT on-time flight tables and hourly meteorological surface reports |
| **Quality & Testing** | **Pytest / PySpark Fixtures** | Unit and integration tests verifying temporal causality, schemas, and math contracts |
| **Environment** | **Conda / Hadoop 3.3 (winutils)** | Deterministic dependency isolation and cross-platform native I/O |

---

## Repository Structure

```text
pipeline/main.py        Sequential stage orchestration (Stages 1-6)
scripts/
  common/               Shared config, context, I/O, Spark, cache, progress, and results
  bts/                  Stage 1: BTS acquisition, cleaning, caching, and features
  origin_weather/       Stage 2: NOAA acquisition, quality control, and origin matching
  destination_weather/  Stage 3: Destination observations at scheduled departure
  aircraft_rotation/    Stage 4: Aircraft linkage and turnaround features
  enroute_weather/      Stage 5: TAF and SIGMET acquisition, cleaning, and enrichment
  modeling/             Stage 6: Training, validation publication, and frozen test scoring
    run.py              Stage 6 entry point, run manifest, and classification report
    train.py            Feature assembly, model fitting, validation, and thresholds
    evaluate.py         Frozen test scoring and subgroup metrics
config.json             Sources, study dates, and weather policy
data/                   Raw files, intermediate data, and model-ready splits
models/flight_delay/    Model artifacts and run reports
results/results.json    Current model metrics under flight_delay
tests/                  Synthetic fixtures
```

---

## Getting Started

### 1. Prerequisites
* [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/)
* Git

### 2. Environment Setup
Clone the repository and build the isolated conda environment (includes Python 3.11, Java 17, and PySpark):

```bash
git clone https://github.com/girwandhakal/flight-delay-pipeline.git
cd flight-delay-pipeline

# Create and activate the conda environment
conda env create -f envs/environment.yml
conda activate cs523_project_env
```

*(Windows users: Native Hadoop binaries and `winutils.exe` are pre-configured in `envs/hadoop/bin/`).*

### 3. Run the Verification Suite
The test suite is partitioned into fast pure-Python unit tests and Spark integration tests to provide both instant local feedback and rigorous contract verification:

* **Fast Unit Tests** — Verifies NOAA LCD decoding, METAR/TAF parsers, evaluation metrics, manifest validation, and date math without starting the Spark JVM:
  ```bash
  python -m pytest -m unit
  ```

* **End-to-End Pipeline Smoke Test** — Checks stage order and runs real fitting and frozen scoring on synthetic full-feature Gold:
  ```bash
  python -m pytest tests/test_pipeline_smoke.py
  ```

* **Targeted Spark Integration Tests** — Runs temporal joins, leakage guards, and ML pipeline tests requiring PySpark:
  ```bash
  python -m pytest -m integration
  ```

* **Complete Test Suite**:
  ```bash
  python -m pytest
  ```

---

## Execution Commands

### Prepare, fit, and validate

```bash
python pipeline/main.py
```

The command prepares BTS and weather inputs, builds all three Gold splits, and selects each fitted model's threshold on validation. The unweighted model is the primary result. Each run saves `validation.json`, `run.json`, models, thresholds, and validation predictions under `models/flight_delay/RUN_ID`. The class-weighted model uses the same features and balances the total training weight of the two classes.

### Score the completed run on test

```bash
python -m scripts.modeling.evaluate RUN_ID
```

Use the current run ID reported by the default workflow. The scorer verifies the run manifest, saved models and thresholds, and the pinned test Gold. A successful attempt writes per-model predictions under `models/flight_delay/RUN_ID/test/ATTEMPT_ID` and publishes test metrics together in `results/results.json`. Earlier runs keep their existing artifacts. Generate a new run to use this command.

### Prepare data separately

The BTS and origin-weather packages each support `download`, `clean`, and `build_features` as Python modules. Run them in source order when working on those stages directly.

```bash
python -m scripts.bts.download
python -m scripts.bts.clean
python -m scripts.bts.build_features
python -m scripts.origin_weather.download
python -m scripts.origin_weather.clean
python -m scripts.origin_weather.build_features
```

---

## Methodological and Engineering Rigor

1. **Point-in-time feature rules**:
   * All predictions occur at **scheduled departure cutoff**.
   * Surface weather uses observations taken strictly **15 to 120 minutes prior to departure**.
   * Airport delay histories track strictly **completed events** prior to the flight's cutoff.
   * Aircraft rotation uses the prior **gate arrival** if it occurred by **scheduled departure**. Final aircraft assignments and archived reporting times are retrospective research assumptions.
2. **Cohort Identity Assertion**:
   * Enrichment checks preserve flight IDs, dates, and labels. Rotation checks training label availability before fitting. Both models score the same validation and test cohorts.
3. **Smart Cryptographic Caching**:
   * The pipeline records SHA-256 hashes of input files, configuration parameters, and transformation source code in `_metadata.json` and `_reuse.json`. Re-running `pipeline/main.py` detects matching Gold tables and bypasses redundant ingestion/cleaning stages.
4. **Train-Serve Parity**:
   * Feature preprocessing (StringIndexers, OneHotEncoders, Mean Imputers) and classifiers are packaged together into serialized Spark `PipelineModel` objects. Inference calls the frozen `.transform()` method directly, eliminating train-serve skew.

---

## Author
**Girwan Dhakal**  
*The University of Alabama — Department of Computer Science (CS 523)*  
* GitHub: [@girwandhakal](https://github.com/girwandhakal)
