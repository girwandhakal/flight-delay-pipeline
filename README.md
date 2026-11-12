# Flight Delay Pipeline

A data science project exploring flight delays and building a pipeline to predict whether a flight will arrive **15 minutes or more late**.

The initial dataset is the [BTS Reporting Carrier On-Time Performance data](https://www.transtats.bts.gov/Fields.asp?gnoyr_VQ=FGJ). The prediction target is `ArrDel15`, using scheduled flight information such as airline, airports, departure time, distance, and scheduled duration.

## Approach

The project starts with a BTS-only baseline, then adds weather and other relevant information to evaluate whether additional features improve predictions.

The workflow covers data collection, exploratory data analysis, cleaning, feature engineering, modeling, and evaluation. Predictions use information available by scheduled departure, with chronological data splits to evaluate performance on later flights:
- **Train:** 2024-01-01 to 2024-09-30 (5,199,967 flights)
- **Validation:** 2024-10-01 to 2024-11-30 (1,180,447 flights)
- **Test / Holdout:** 2024-12-01 to 2025-01-31 (1,107,101 flights)

## Pipeline Commands

Run the end-to-end data pipeline:
```bash
python pipeline/main.py
```

Run individual pipeline stages:
```bash
python -m scripts.bts.download
python -m scripts.bts.clean
python -m scripts.bts.build_features
python -m scripts.bts.train
```

Run tests from the project root:
```bash
conda activate cs523_project_env
python -m pip install -r envs/requirements.txt
python -m pytest
```

Pytest shows compact pass/fail progress and a final summary. Passing-test output
is captured; failures show tracebacks and captured logs. To see individual test
names, use `python -m pytest -v`. To include the warnings summary, add
`-o addopts="-q"`.

## Origin weather

Run the origin-weather experiment from the project root:

```powershell
python pipeline/main.py --origin-weather
```

This reuses completed BTS Gold splits (or runs the first three BTS stages if missing), prepares origin airport/station/time-zone references, downloads required NOAA LCDv2 station-year files, cleans observations and builds origin features. It then trains BTS-only and BTS + origin-weather models using the same chronological search, saves both models and paired validation predictions, and writes a comparison. This command stops after validation. The no-flag pipeline retains the existing BTS-only workflow, including its test evaluation.

```text
scripts/
  bts/
    download.py
    clean.py
    build_features.py
    train.py
    evaluate.py
  origin_weather/
    download.py
    clean.py
    build_features.py

data/bronze/
  bts/<year>/<month>/
  weather/
    manifest.json
    <year>/LCD_<station_id>_<year>.csv
```

The shared `scripts/bts/train.py` trains either feature set. For weather, it uses temperature, wind speed, visibility, observation age and four missingness flags. Numeric means and category mappings are fitted on each fold's training rows and saved with the model. Flights without usable weather remain present and receive predictions.

Small lookup tables live under `data/reference/weather/`: airport time zones and exact ICAO/WBAN station links. Reference construction cross-checks NOAA's LCD inventory and station history, accepts only a unique match within 5 km of the named airport and leaves ambiguous/unmapped origins without weather. This is an identifier linkage check, not nearest-station selection. OpenFlights airport metadata is current, so historical metadata availability is not established. The retained source files and hashes document the evidence and OpenFlights' ODbL attribution.

Weather CSV clocks can use UTC or local standard time. Each hourly observation is normalized only when one of those interpretations matches its embedded METAR/SPECI UTC clock; inconsistent or missing clocks are quarantined. Scheduled flights use historical airport civil time, including DST; ambiguous/nonexistent departures remain unmatched. Default matching uses the latest report at or before scheduled departure, up to 120 minutes old. The zero-minute publication lag assumes reports are available at their observation timestamps; historical publication availability is not established. Flagged numeric measurements become missing rather than zero. Existing generated datasets, models and results retain their original matching policy until the weather experiment is rebuilt and retrained.

Outputs:

- `data/silver/weather/<dataset_id>/`: normalized UTC observations.
- `data/gold/bts_origin_weather/<dataset_id>/{train,val,test}/`: unchanged BTS flights plus weather/audit columns.
- `data/gold/bts_origin_weather/<dataset_id>/_metadata.json`: source contract and dataset provenance.
- `models/origin_weather_comparison/<run_id>/{bts,bts_origin_weather}/`: fitted pipelines and thresholds.
- `models/origin_weather_comparison/<run_id>/`: prediction Parquet, internal run manifest and completion markers beside the fitted models.
- `results/results.json`: latest BTS and BTS + origin-weather results, each under its own named section.

To explicitly evaluate a completed comparison on test, replace `RUN_ID` with its saved directory name:

```powershell
python pipeline/main.py --evaluate-origin-weather RUN_ID
```

This loads both frozen models and thresholds, performs no tuning and stores paired test predictions beside the models and updates each model's `test` section in `results/results.json`. Only the current comparison run can update these latest results. Existing test results may already have been inspected; treat that period as a development benchmark. Failed runs preserve previous completed outputs. Existing weather CSVs are reused without checksum or content scans. The download stage handles transfer completeness; weather schema, timestamps and measurements are checked during cleaning. Small reference-file checksum checks remain in place. Full acquisition requires network access and can download many annual station files.

Standalone weather stages use `python -m scripts.origin_weather.download`, `python -m scripts.origin_weather.clean` and `python -m scripts.origin_weather.build_features`; the last command coordinates weather acquisition/cleaning/integration against existing BTS Gold. Run `conda activate cs523_project_env` before these commands. The environment needs Java 17 and the existing Windows Hadoop binaries. See [origin weather specification](docs/specs/origin_weather_spec.md).

## Results

All reported results live in `results/results.json`, keyed by feature set (`bts`,
`bts_origin_weather`). Cleaning counts, split counts, training/search settings,
validation scores, test scores and subgroup metrics are sections of that file.
Weather quality and coverage are included as JSON tables;
no result CSVs, Markdown reports or figures are written.
Before each update, the existing JSON is archived in the root-level
`results_history/results_<UTC timestamp>_<unique ID>.json`. A pipeline run can
produce several snapshots because stages publish separately; standalone stages
archive results too. The first write has nothing to archive. Archives preserve
the exact previous file and are excluded from Git. If archiving fails, the
latest file is left unchanged. History is retained until manually removed.
Each update preserves other models and unrelated stages. Retraining clears that
model's previous test scores; test results appear only after explicit evaluation
(or the existing BTS end-to-end pipeline's evaluation stage). Failed JSON writes
preserve the previous file. Run only one pipeline writer at a time.

Datasets stay under `data/`; fitted models, thresholds, saved predictions and the
internal manifest used to reload a frozen comparison stay under `models/`.

## Current Progress

- BTS download script and downloader tests.
- Environment specifications for the project.
- Exploratory data analysis (`notebooks/eda_bts.ipynb`) examining data quality, arrival-delay frequency, schedule features, and diurnal delay patterns.
- Origin-weather EDA (`notebooks/eda_origin_weather.ipynb`) showing the original LCDv2 schema, retained source columns, basic counts, missing values, summary statistics, and weather distributions.
- Data cleaning (`scripts/bts/clean.py`) producing 7,487,515 cleaned silver records across 13 months.
- Feature engineering and chronological splitting (`scripts/bts/build_features.py`) writing partitioned gold datasets.
- Baseline logistic regression modeling, threshold selection, and validation evaluation (`scripts/bts/train.py`).

Training searches the existing 12 regularization and L1/L2 combinations with three expanding chronological folds inside the training period: January–June → July, January–July → August, and January–August → September 2024. Each fold fits preprocessing only on its earlier training flights. Candidates are selected by mean fold PR-AUC, and the winner is refitted on all January–September flights. October–November remains separate for validation and F1-based threshold selection. The training summary records fold boundaries, each candidate's fold PR-AUC values, and their mean (`internal_pr_auc`). Training does not read the test split; the full pipeline currently runs test evaluation afterward.

Existing saved models and results predate this change. Run `python -m scripts.bts.train` to regenerate training artifacts with chronological tuning.
