# Flight Delay Pipeline

A PySpark research pipeline that predicts whether a flight will arrive at least 15 minutes late, using information available by scheduled departure. We compare models using flight data alone, flight data with origin weather, and flight data with both origin and destination weather.

## Datasets

- **BTS on-time performance:** flight schedules, airlines, airports, distances and arrival-delay labels from January 2024 through January 2025.
- **NOAA Local Climatological Data (LCDv2):** airport weather observations matched at or before scheduled departure for both origin and destination airports.
- **Airport and station references:** OpenFlights airport metadata and NOAA station records for airport time zones and weather-station links.

## Setup and run

Install [Conda](https://docs.conda.io/projects/conda/en/stable/user-guide/install/index.html), then run these commands from the repository root:

```bash
conda env create -f envs/environment.yml
conda activate cs523_project_env
python pipeline/main.py
```

The environment file installs Python 3.11, Java 17 and the required Python packages. On Windows, the pipeline automatically uses the Hadoop helpers in `envs/hadoop/bin/`.

The pipeline downloads missing data, prepares the datasets, and trains and compares all three models on validation data. Internet access and substantial free disk space are required for the first run. Dataset dates and weather settings are in `config.json`.

Generated datasets go in `data/`, trained models and predictions in `models/`, and the latest reports in `results/results.json`.

Test scoring is a separate step. Use the run ID printed after a successful comparison:

```bash
python pipeline/main.py --evaluate-destination-weather RUN_ID
```
