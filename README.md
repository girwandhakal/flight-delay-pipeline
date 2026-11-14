# Flight Delay Pipeline

- Predicts arrival delays of at least 15 minutes using information available by scheduled departure. Compares flight-only, origin-weather, and origin-plus-destination-weather models.
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
- **Outputs:** Datasets in `data/`, models and predictions in `models/`, and reports in `results/results.json`.
- **Optional test scoring:** After validation, use the printed run ID: `python pipeline/main.py --evaluate-destination-weather RUN_ID`.
