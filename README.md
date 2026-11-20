# Scalable Flight Delay Prediction Pipeline

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![Apache Spark](https://img.shields.io/badge/Apache%20Spark-3.5-E25A1C?style=flat-square&logo=apachespark&logoColor=white)](https://spark.apache.org/)
[![PySpark ML](https://img.shields.io/badge/PySpark-MLlib-blue?style=flat-square)](https://spark.apache.org/mllib/)
[![Testing](https://img.shields.io/badge/Tests-83%20passed-success?style=flat-square&logo=pytest&logoColor=white)](https://docs.pytest.org/)
[![Architecture](https://img.shields.io/badge/Data%20Architecture-Medallion%20(Bronze%E2%86%92Silver%E2%86%92Gold)-lightgrey?style=flat-square)](#data-architecture-and-pipeline-flow)

An end-to-end, reproducible **Big Data Machine Learning Pipeline** engineered in **PySpark** to predict commercial flight arrival delays ($\ge 15$ minutes, BTS `ArrDel15`) strictly before takeoff. 

Designed for scalability on large tabular time-series datasets, the pipeline processes over **6 million flight records** from the Bureau of Transportation Statistics (BTS) paired with millions of hourly surface meteorological observations from NOAA Local Climatological Data (LCDv2) and physical aircraft tail-number rotation sequences.

### Reusing prepared data

Each pipeline run checks its data dependencies before starting the corresponding Spark transformations. BTS Silver and Gold generations live under `data/{silver,gold}/bts_generations`. New comparison manifests pin the BTS generation they trained against. Keep older generations while a frozen run references them. The first run after migrating builds generation manifests and can require substantial additional disk space.

Origin and destination weather use their requested NOAA station-years. Rotation uses destination train and validation, while en-route preparation also reads destination test schedules. TAF station selection, TAF Silver, and SIGMET Silver have separate caches. Replacing a verified TAF archive rebuilds its Silver table and en-route Gold while allowing rotation and SIGMET Silver to be reused.

Cache messages report hits, misses and elapsed lookup or preparation time. A completed `_stage.json` records the dependency key, output fingerprints, required sidecars and cold-build duration. A missing marker or a changed output triggers a fresh generation. Interrupted generations may remain on disk. Model fitting and explicit scoring retain their existing CLI behavior.

---

## Problem and Motivation

Flight delays cost the global aviation industry tens of billions of dollars annually in operational disruption, passenger compensation, and gate congestion. 

Accurate pre-departure flight delay forecasting poses significant data science challenges:
1. **Severe Temporal Data Leakage**: Naïve machine learning pipelines frequently incorporate outcome-derived features (e.g., actual departure delay or taxi time) or future weather observations that would not be available in real-world dispatch.
2. **Cascading Network Disruptions**: Delays do not occur in isolation. An incoming aircraft arriving late at a gate propagates downstream delays across subsequent flights on that physical airframe (aircraft rotation).
3. **Data Engineering Scale**: Joining multi-gigabyte continuous flight schedules with hourly weather readings across hundreds of US airports exceeds standard in-memory Pandas workflows, demanding distributed memory management, custom temporal bucket joins, and out-of-core execution.

**Our Objective**: Measure the incremental predictive value of BTS flight features, origin weather, destination weather, aircraft rotation, and en-route aviation weather using chronological validation.

---

## Modeling and Feature Ablation Tiers

The pipeline evaluates five progressive model tiers on identical validation cohorts:

```
[ Tier 1: BTS Baseline ]
       │  • Carrier, Origin, Dest, Scheduled Departure & Arrival Hours, Day of Week, Distance
       │  • Cyclical Month Encoding (MonthSin, MonthCos) & US Federal Holiday Proximity
       │  • 2-Hour Rolling Airport Delay Rates (Completed Dep/Arr Congestion Histories)
       │  • Origin taxi-out mean, observation count, and surge above the airport's training baseline
       │  • Upstream delay pressure, scheduled feeder arrivals, and observed coverage at both airports
       ▼
[ Tier 2: + Origin Weather ]
       │  • Surface Temperature, Dew Point, Relative Humidity, Pressure, Wind Speed & Gusts
       │  • Circular Wind Direction (Sin/Cos), Visibility, Precipitation, Cloud Cover & Ceiling
       │  • 13 METAR Weather Phenomenon Flags (Thunderstorm, Snow, Fog, Freezing, Rain, etc.)
       ▼
[ Tier 3: + Destination Weather ]
       │  • Full parallel suite of meteorological features at arrival airport at scheduled departure
       ▼
[ Tier 4: + Aircraft Rotation Mechanics ]
       │  • Prior Flight Arrival Delay of the same physical aircraft (Tail Number tracking)
       │  • Scheduled Ground Turnaround Buffer & Operational Availability Flags
       ▼
[ Tier 5: + TAF and Convective SIGMET ]
          • Destination forecasts valid at scheduled arrival
          • Active advisory overlap along the route corridor
```

* **Core Estimator**: Binomial Logistic Regression via L-BFGS optimization (`regParam=0.01`, $L_2$ Ridge penalty, `maxIter=100`, `tol=1e-6`).
* **Challenger Estimator**: Gradient-Boosted Trees (`GBTClassifier`, `maxDepth=3`, `maxIter=50`, `stepSize=0.05`).
* **Decision Optimization**: Dynamically searches 19 candidate thresholds ($\tau \in [0.05, 0.95]$ in increments of $0.05$) on validation data to maximize $F_1$-score on imbalanced delays (~20% prevalence).
* **Strict Temporal Splitting**:
  * **Train Cohort**: `2024-01-01` to `2024-09-30` (~4.2M flights)
  * **Validation Cohort**: `2024-10-01` to `2024-11-30` (~1.2M flights)
  * **Test Holdout**: `2024-12-01` to `2025-01-31` (Frozen out-of-time evaluation)

The BTS stage computes taxi-out surge as the mean taxi time of other flights in the two-hour window ending 15 minutes before scheduled departure, minus the origin airport's 15th-percentile taxi time. It uses verified wheels-off clocks and caps individual observations at 120 minutes. Training uses earlier training months for the baseline. Validation and test share a frozen training baseline. When an airport has no baseline, the feature uses 12 minutes. Gold keeps missing window measurements null and records the observed taxi count. The model fills missing rolling inputs with zero.


Upstream delay pressure weights each feeder airport's recent departure delay rate by its scheduled incoming flights. The origin window starts at scheduled departure. The destination window extends 30 minutes on either side of scheduled arrival. Both use only departure events available before the target's scheduled departure, rounded down to a 15-minute snapshot with a modeled 15-minute reporting lag. The score reconstructs historical knowledge from BTS schedules and actual events. Missing feeder observations are tracked separately from observed on-time departures.

A normal pipeline run rebuilds older BTS Gold and its weather descendants when their feature provenance changes. All five tiers inherit the six upstream inputs.

---

## Tech Stack

| Layer | Technology | Purpose |
| :--- | :--- | :--- |
| **Distributed Engine** | **Apache Spark 3.5 / PySpark** | Large-scale parallel transformations, temporal bucket joins, and ML pipelines |
| **Runtime & Language** | **Python 3.11 / Java 17** | Core pipeline logic, typing, and JVM Spark backend execution |
| **Storage Architecture** | **Apache Parquet (Snappy)** | Partitioned, columnar Bronze $\to$ Silver $\to$ Gold Medallion storage |
| **Machine Learning** | **Spark MLlib** | StringIndexer, OneHotEncoder, Imputer, VectorAssembler, LogisticRegression, GBT |
| **Data Sources** | **BTS TranStats / NOAA LCDv2** | US DOT on-time flight tables and hourly meteorological surface reports |
| **Quality & Testing** | **Pytest / PySpark Fixtures** | 83 unit/integration tests verifying temporal causality, schemas, and math contracts |
| **Environment** | **Conda / Hadoop 3.3 (winutils)** | Deterministic dependency isolation and cross-platform native I/O |

---

## Repository Structure

```text
flight-delay-pipeline/
├── pipeline/
│   └── main.py                     # Main orchestrator (workflow manager & 4-tier model evaluation)
├── scripts/
│   ├── bts/
│   │   ├── download.py             # BTS TranStats data acquisition & SHA-256 verification
│   │   ├── clean.py                # Schema standardization, type casting & quarantine filtering
│   │   ├── build_features.py       # BTS Gold feature engineering & cyclical calendar encoding
│   │   ├── recent_delays.py        # 2-hour completed airport delay history aggregations
│   │   ├── train.py                # PySpark ML preprocessing, classifier fitting & threshold search
│   │   └── evaluate.py             # Standalone model inference & PR-AUC / ROC-AUC scoring
│   ├── origin_weather/
│   │   ├── download.py             # NOAA LCDv2 surface weather acquisition & station lookup
│   │   ├── clean.py                # Quality control, station normalization & UTC synchronization
│   │   ├── measurements.py         # METAR regex token decoding, SI unit conversion & QC flags
│   │   └── build_features.py       # Point-in-time temporal bucket join (15-120 min pre-departure)
│   ├── destination_weather/
│   │   └── build_features.py       # Destination airport weather matching & parent cohort merge
│   ├── aircraft_rotation/
│   │   ├── build_features.py       # Tail-number sequence tracking & turnaround buffer calculation
│   │   └── run.py                  # Standalone rotation feature generation runner
│   ├── progress.py                 # Structured console timing & stage banner formatting
│   ├── results.py                  # Thread-safe atomic JSON results publication & history
│   └── weather_cache.py            # Cryptographic input fingerprinting & Gold cache validation
├── data/
│   ├── bronze/                     # Raw ingested CSVs (BTS & NOAA weather)
│   ├── silver/                     # Cleaned, validated, and typed Parquet datasets
│   ├── gold/                       # Model-ready partitioned train/val/test splits
│   └── reference/                  # Station mappings, IATA airport timezones & checksums
├── models/                         # Serialized PipelineModels, threshold configs, and predictions
├── results/
│   └── results.json                # Published benchmark metrics, PR-AUC, F1, and ablation runs
├── tests/                          # Fast synthetic PySpark test suite (83 tests + 20 subtests)
├── config.json                     # Central pipeline configuration (dates, columns, windows)
└── README.md
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
Run the synthetic test suite to verify Spark session initialization, schema contracts, and leakage-prevention logic without downloading large datasets:

```bash
python -m pytest
```

---

## Execution Commands

### Run Complete End-to-End Pipeline
Verifies inputs, downloads the public IEM archives, and fits and validates all five tiers. TAF + SIGMET uses the Tier 4 predictions from the same invocation.

```bash
python pipeline/main.py
```

### Run Sub-Workflows & Specific Configurations

The explicit `--include-enroute` command also runs the full five-tier comparison:

```bash
python pipeline/main.py --include-enroute
```

This mode downloads public IEM raw TAF bulletins for the top 50 training destinations and domestic convective SIGMET polygons. It caches checksummed archives under `data/bronze/taf/` and `data/bronze/sigmets/`, and writes processed Silver and cohort-preserving `data/gold/enroute/GENERATION/`. Train and validation use rotation Gold. Test enrichment uses the existing weather-only test Gold because the rotation experiment has no test cohort. The command publishes validation metrics only, including Tier 4 versus en-route PR-AUC, F1, precision, recall and false alarms at both operating points. The direct TAF and nearby-station coverage and the route overlap are reported as modeled features based on public archive coverage.

TAF forecasts must be issued by departure and valid at scheduled arrival. The deterioration feature subtracts the destination ceiling observed by departure. SIGMET features use advisories active at scheduled departure and a buffered great-circle corridor approximation. The report is saved under `models/enroute/RUN_ID/validation_comparison.json`.

* **BTS Baseline Only** (Skip weather ingestion):
  ```bash
  python pipeline/main.py --bts-only
  ```
* **Origin Weather Comparison**:
  ```bash
  python pipeline/main.py --origin-weather
  ```
* **Three-Tier Weather Comparison**:
  ```bash
  python pipeline/main.py --destination-weather
  ```
* **Include Gradient-Boosted Trees (GBT) Challenger**:
  ```bash
  python pipeline/main.py --include-gbt
  ```
* **Evaluate Frozen Models on Test Holdout**:
  ```bash
  python pipeline/main.py --evaluate-destination-weather RUN_ID
  ```

---

## Methodological and Engineering Rigor

1. **Zero Future Data Leakage**:
   * All predictions occur at **scheduled departure cutoff**.
   * Surface weather uses observations taken strictly **15 to 120 minutes prior to departure**.
   * Airport delay histories track strictly **completed events** prior to the flight's cutoff.
   * Incoming aircraft rotation links require arrival touchdown at least **15 minutes before departure**.
2. **Cohort Identity Assertion**:
   * Comparative ablation between models is strictly enforced through bidirectional PySpark `exceptAll` assertions, guaranteeing that every model is benchmarked on the exact same set of flight IDs.
3. **Smart Cryptographic Caching**:
   * The pipeline records SHA-256 hashes of input files, configuration parameters, and transformation source code in `_metadata.json` and `_reuse.json`. Re-running `pipeline/main.py` detects matching Gold tables and bypasses redundant ingestion/cleaning stages.
4. **Train-Serve Parity**:
   * Feature preprocessing (StringIndexers, OneHotEncoders, Mean Imputers) and classifiers are packaged together into serialized Spark `PipelineModel` objects. Inference calls the frozen `.transform()` method directly, eliminating train-serve skew.

---

## Author
**Girwan Dhakal**  
*The University of Alabama — Department of Computer Science (CS 523)*  
* GitHub: [@girwandhakal](https://github.com/girwandhakal)
