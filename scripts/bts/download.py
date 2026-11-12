"""Download and extract an inclusive range of monthly BTS flight files."""

from datetime import datetime
import json
from pathlib import Path
import shutil
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
import zipfile

from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]

BTS_URL = "https://transtats.bts.gov/PREZIP/On_Time_Reporting_Carrier_On_Time_Performance_1987_present_{year}_{month}.zip"


# Return every year and month in the requested range, including both endpoints.
def iter_months(start_month, end_month):
    start = datetime.strptime(start_month, "%Y-%m")
    end = datetime.strptime(end_month, "%Y-%m")
    # Parsing accepts unpadded months, so check the spelling too.
    if start.strftime("%Y-%m") != start_month or end.strftime("%Y-%m") != end_month:
        raise ValueError("Months must use YYYY-MM format.")
    if start.year < 1987 or start > end:
        raise ValueError(
            "BTS range must start in 1987 or later and end on or after its start."
        )
    year, month = start.year, start.month
    while (year, month) <= (end.year, end.month):
        yield year, month
        # December rolls into January of the following year.
        year, month = (year + 1, 1) if month == 12 else (year, month + 1)


# Read the BTS download settings and check the dates and output folder.
def load_config(root=ROOT):
    settings = json.loads((root / "config.json").read_text(encoding="utf-8"))[
        "datasets"
    ]["bts"]
    # Run the generator now so invalid dates fail before any files are written.
    list(iter_months(settings["start_month"], settings["end_month"]))
    directory = Path(settings.setdefault("output_directory", "data/bronze/bts"))
    if directory.is_absolute() or not (root / directory).resolve().is_relative_to(
        root.resolve()
    ):
        raise ValueError(
            "output_directory must be relative and stay inside the project."
        )
    return settings


# Build the official BTS ZIP address for one month.
def build_download_url(year, month):
    return BTS_URL.format(year=year, month=month)


# Check that the ZIP is readable and contains exactly one CSV.
def validate_archive(path):
    with zipfile.ZipFile(path) as archive:
        bad_member = archive.testzip()
        if bad_member is not None:
            raise ValueError(f"Corrupt ZIP member: {bad_member}")
        members = [
            item
            for item in archive.infolist()
            if not item.is_dir() and item.filename.lower().endswith(".csv")
        ]
        if len(members) != 1:
            raise ValueError(
                f"Expected exactly one flight CSV in {path.name}, found {len(members)}"
            )
        return members[0]


# Download a missing ZIP, retry temporary failures, and keep incomplete files separate.
def download_file(url, destination):
    if destination.exists():
        return destination
    temporary = destination.with_suffix(".zip.part")
    for attempt in range(3):
        try:
            request = Request(url, headers={"User-Agent": "CS523-BTS-Research/1.0"})
            with (
                urlopen(request, timeout=90) as response,
                temporary.open("wb") as output,
            ):
                expected_size = response.headers.get("Content-Length")
                with tqdm.wrapattr(
                    output,
                    "write",
                    total=int(expected_size) if expected_size else None,
                    desc=destination.name,
                    unit="B",
                    unit_scale=True,
                ) as progress:
                    shutil.copyfileobj(response, progress, length=1024 * 1024)
            if expected_size and temporary.stat().st_size != int(expected_size):
                raise ValueError("Download size differs from Content-Length")
            validate_archive(temporary)
            # Give the file its final name only after the download passes ZIP checks.
            temporary.replace(destination)
            return destination
        except (OSError, URLError, ValueError, zipfile.BadZipFile) as error:
            temporary.unlink(missing_ok=True)
            # Retry rate limits and temporary server errors, but stop on errors such as 404.
            if isinstance(error, HTTPError) and error.code not in (
                429,
                500,
                502,
                503,
                504,
            ):
                raise
            if attempt == 2:
                raise
            print(f"Retrying after {type(error).__name__}: {error}", flush=True)
            time.sleep(2 ** (attempt + 1))


# Extract the flight CSV and delete its ZIP only after extraction succeeds.
def extract_csv(zip_path, destination):
    if destination.exists():
        return destination
    member = validate_archive(zip_path)
    temporary = destination.with_suffix(".csv.part")
    try:
        with (
            zipfile.ZipFile(zip_path) as archive,
            archive.open(member) as source,
            temporary.open("wb") as output,
        ):
            # Fixed destination prevents archive paths from escaping the month folder.
            shutil.copyfileobj(source, output)
        temporary.replace(destination)
        # Keep the ZIP if extraction fails so the original data is still available.
        zip_path.unlink()
    finally:
        temporary.unlink(missing_ok=True)
    return destination


# Process each requested month and report which months completed, skipped, or failed.
def main():
    settings = load_config()
    for layer in ("bronze", "silver", "gold", "quarantine"):
        (ROOT / "data" / layer / "bts").mkdir(parents=True, exist_ok=True)
    output_directory = ROOT / settings["output_directory"]
    output_directory.mkdir(parents=True, exist_ok=True)
    completed, skipped, failed = 0, 0, []

    for year, month in iter_months(settings["start_month"], settings["end_month"]):
        period = f"{year}-{month:02d}"
        try:
            folder = output_directory / str(year) / f"{month:02d}"
            folder.mkdir(parents=True, exist_ok=True)
            url = build_download_url(year, month)
            zip_path = folder / url.rsplit("/", 1)[1]
            csv_path = zip_path.with_suffix(".csv")

            if csv_path.exists():
                skipped += 1
                print(f"  Skipped {period}: CSV already exists.", flush=True)
                continue

            download_file(url, zip_path)
            extract_csv(zip_path, csv_path)
            completed += 1
            print(f"  Completed {period}", flush=True)

        except (
            OSError,
            ValueError,
            zipfile.BadZipFile,
            RuntimeError,
            EOFError,
        ) as error:
            failed.append(period)
            print(f"  Failed {period}: {error}", flush=True)

    print("-" * 80)
    print(
        f"BTS Acquisition Summary: Completed: {completed}; Skipped: {skipped}; Failed: {len(failed)}"
    )
    if failed:
        print("  Failed months: " + ", ".join(failed))
    print("-" * 80)
    return 1 if failed else 0


if __name__ == "__main__":
    main()
