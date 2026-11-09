# Flight Delay Pipeline

A data science project exploring flight delays and building a pipeline to predict whether a flight will arrive **15 minutes or more late**.

The initial dataset is the [BTS Reporting Carrier On-Time Performance data](https://www.transtats.bts.gov/Fields.asp?gnoyr_VQ=FGJ). The prediction target is `ArrDel15`, using scheduled flight information such as airline, airports, departure time, distance, and scheduled duration.

## Approach

The project starts with a BTS-only baseline, then adds weather and other relevant information to evaluate whether additional features improve predictions.

The workflow covers data collection, exploratory data analysis, cleaning, feature engineering, modeling, and evaluation. Predictions will use information available by scheduled departure, with chronological data splits to evaluate performance on later flights.

## Current progress

- BTS download script and downloader tests.
- Environment specifications for the project.
- An EDA notebook examining data quality, arrival-delay frequency, schedule features, and airline/airport patterns.

Data cleaning and model training are the next steps. Raw datasets and local documentation are excluded from version control.
