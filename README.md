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
python scripts/bts_download.py
python scripts/bts_clean.py
python scripts/bts_transform.py
python scripts/bts_train.py
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

## Current Progress

- BTS download script and downloader tests.
- Environment specifications for the project.
- Exploratory data analysis (`notebooks/eda_bts.ipynb`) examining data quality, arrival-delay frequency, schedule features, and diurnal delay patterns.
- Data cleaning (`scripts/bts_clean.py`) producing 7,487,515 cleaned silver records across 13 months.
- Feature engineering and chronological splitting (`scripts/bts_transform.py`) writing partitioned gold datasets.
- Baseline logistic regression modeling, threshold selection, and validation evaluation (`scripts/bts_train.py`).

Training searches the existing 12 regularization and L1/L2 combinations with three expanding chronological folds inside the training period: January–June → July, January–July → August, and January–August → September 2024. Each fold fits preprocessing only on its earlier training flights. Candidates are selected by mean fold PR-AUC, and the winner is refitted on all January–September flights. October–November remains separate for validation and F1-based threshold selection. The training summary records fold boundaries, each candidate's fold PR-AUC values, and their mean (`internal_pr_auc`). Training does not read the test split; the full pipeline currently runs test evaluation afterward.

Existing saved models and results predate this change. Run `python scripts/bts_train.py` to regenerate training artifacts with chronological tuning.
