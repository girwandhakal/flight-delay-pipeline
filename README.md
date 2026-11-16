# Flight Delay Pipeline

- Predicts arrival delays of at least 15 minutes using information available by scheduled departure. Compares flight-only, origin-weather, and origin-plus-destination-weather models.
- **Models:** Each feature set uses cyclical month predictors and fits logistic regression (`regParam=0.01`, L2, up to 100 iterations) plus a gradient-boosted tree challenger (depth 3, 50 iterations, step size 0.05, seed 42). Hyperparameters are fixed: no candidate search or internal cross-validation. Each algorithm is fitted once on January through September training data and receives its own F1 threshold on October through November validation; logistic regression remains the reference for weather comparisons and optional test scoring. The complete workflow performs six classifier fits rather than 150. These fixed settings are not claimed to be optimal; tree training still adds runtime.
- **Data:** BTS flight records (January 2024–January 2025), NOAA LCDv2 weather, and airport/station references.
- **Setup:** Install [Conda](https://docs.conda.io/projects/conda/en/stable/user-guide/install/index.html), then create and activate the environment from the repository root:

  ```bash
  conda env create -f envs/environment.yml
  conda activate cs523_project_env
  ```

  The YAML installs Python 3.11, Java 17, and the required packages. Windows Hadoop helpers are included in `envs/hadoop/bin/`.

- **Run the complete pipeline:** Downloads missing data, prepares datasets, and trains and compares all three models on validation data.

  ```bash
  python pipeline/main.py
  ```

- **Requirements:** Internet access for downloads, substantial free disk space, and RAM beyond Spark's 6 GB Java heap. If Spark runs out of disk space, free space or set `SPARK_LOCAL_DIRS` to a writable directory on a roomier local drive before retrying.
- **Configuration:** Dataset dates and weather settings are in `config.json`.
- **Recent airport delays:** Every model includes the origin's departure-delay rate, mean departure delay (early flights contribute zero) and departure count, plus the destination's corresponding arrival measurements, over the two hours ending at scheduled departure. The interval excludes its lower endpoint and includes its upper endpoint; the predicted flight is excluded. Events are assumed available immediately when they occur. Histories use actual gate events from raw BTS records, include all reporting airlines and cross date/split boundaries. Missing, inconsistent and ambiguous clocks are excluded; flights without usable history stay in the cohort. Empty windows retain null rates/means and a zero count; saved preprocessing uses zero fillers alongside counts. These features describe completed events, not the outstanding backlog.
- **Federal holidays:** Three flags identify actual or observed US federal holidays and the day immediately before/after, using the origin-local service `FlightDate`. Actual weekend dates and their weekday observances are both included; school breaks and special events are not included.
- **Applying these features:** Run `python pipeline/main.py`. The normal workflow checks `data/gold/bts/_features.json` against the feature policy, raw/Silver file fingerprints and reference checksums, rebuilds stale BTS datasets and weather parents, then trains all three feature sets on validation. This adds CSV parsing, rolling-window sorting and disk use. Do not reuse the old Gold inputs for new training; direct stage users must run `python -m scripts.bts.build_features` before rebuilding weather features and retraining. Existing fitted models retain their saved preprocessing, but rebuilding shared Gold can invalidate historical runs' frozen data fingerprints. No predictive improvement is claimed without a new measured run.
- **Outputs:** Datasets in `data/`, models and predictions in `models/`, and reports in `results/results.json`.
- **Weather Silver reuse:** Origin and destination processing reuse completed cleaned weather when raw-file paths, sizes and modification times, station time zones, cleaning code and Spark version match. Reuse also checks the saved Parquet file inventory and carries forward quality reports. Older Silver without this provenance is rebuilt once; changed or incomplete inputs/outputs trigger cleaning again. Files modified while preserving both size and modification time are outside this check.
- **Algorithm comparison:** Each model section's `training.algorithm_comparison` records both algorithms' fixed settings, validation metrics and thresholds. Tree artifacts and optional validation predictions are in the feature set's model directory under `gbt/`; they are validation-only challengers. Month sine/cosine transformations are saved inside both pipelines. Previously completed runs retain their original encoding and results.
- **Optional test scoring:** After validation, use the printed run ID: `python pipeline/main.py --evaluate-destination-weather RUN_ID`.
- **Tests:** Run `python -m pytest`. The synthetic suite shares one Spark session and keeps training/persistence, chronological splits, leakage prevention, weather quality/cohort checks, and safe result publication. Overlapping model-fit and reload tests were removed; no real datasets or downloads are needed.
