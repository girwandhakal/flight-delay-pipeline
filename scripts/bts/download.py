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


"""Download and extract an inclusive range of monthly BTS flight files."""

from collections.abc import Iterator
from datetime import datetime
import json
from pathlib import Path
import shutil
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
import zipfile

from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]

BTS_URL = (
    "https://transtats.bts.gov/PREZIP/"
    "On_Time_Reporting_Carrier_On_Time_Performance_1987_present_{year}_{month}.zip"
)


def iter_months(start_month: str, end_month: str) -> Iterator[tuple[int, int]]:
    """Yield every (year, month) tuple in the requested inclusive range.

    Args:
        start_month: Start month string in 'YYYY-MM' format.
        end_month: End month string in 'YYYY-MM' format.

    Yields:
        Tuple of (year, month) integers.

    Raises:
        ValueError: If dates are not in YYYY-MM format, start year is before 1987,
            or start date is after end date.
    """
    start = datetime.strptime(start_month, "%Y-%m")
    end = datetime.strptime(end_month, "%Y-%m")

    # Parsing accepts unpadded months, so check exact string representation
    if start.strftime("%Y-%m") != start_month or end.strftime("%Y-%m") != end_month:
        raise ValueError("Months must use YYYY-MM format.")
    if start.year < 1987 or start > end:
        raise ValueError(
            "BTS range must start in 1987 or later and end on or after its start."
        )

    year, month = start.year, start.month
    while (year, month) <= (end.year, end.month):
        yield year, month
        # December rolls into January of the subsequent year
        year, month = (year + 1, 1) if month == 12 else (year, month + 1)


def load_config(root: Path = ROOT) -> dict[str, Any]:
    """Read BTS download settings and validate configured date range and directory.

    Args:
        root: Project root path. Defaults to ROOT.

    Returns:
        BTS configuration dictionary.

    Raises:
        ValueError: If output directory is absolute or escapes the project root.
    """
    settings = json.loads((root / "config.json").read_text(encoding="utf-8"))[
        "datasets"
    ]["bts"]

    # Validate the full month sequence before creating any directories
    list(iter_months(settings["start_month"], settings["end_month"]))

    directory = Path(settings.setdefault("output_directory", "data/bronze/bts"))
    if directory.is_absolute() or not (root / directory).resolve().is_relative_to(
        root.resolve()
    ):
        raise ValueError(
            "output_directory must be relative and stay inside the project."
        )
    return settings


def build_download_url(year: int, month: int) -> str:
    """Build the official BTS ZIP download URL for a given year and month."""
    return BTS_URL.format(year=year, month=month)


def validate_archive(path: Path) -> zipfile.ZipInfo:
    """Verify ZIP file integrity and ensure it contains exactly one CSV member.

    Args:
        path: Path to the ZIP archive.

    Returns:
        ZipInfo member for the flight CSV.

    Raises:
        ValueError: If ZIP is corrupted or contains other than one CSV file.
    """
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


def download_file(url: str, destination: Path) -> Path:
    """Download a remote ZIP file with retries and atomic staging.

    Args:
        url: Remote URL to download.
        destination: Final destination Path for the archive.

    Returns:
        Destination Path once verified and renamed.
    """
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

            # Validate ZIP structure prior to committing the downloaded file
            validate_archive(temporary)
            temporary.replace(destination)
            return destination

        except (OSError, URLError, ValueError, zipfile.BadZipFile) as error:
            temporary.unlink(missing_ok=True)
            # Retry transient rate limits or 5xx server errors; re-raise permanent 4xx errors
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


def extract_csv(zip_path: Path, destination: Path) -> Path:
    """Extract the flight CSV from a ZIP archive and delete the ZIP on success.

    Args:
        zip_path: Source ZIP archive path.
        destination: Target CSV output path.

    Returns:
        Destination Path of the extracted CSV.
    """
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
            shutil.copyfileobj(source, output)

        temporary.replace(destination)
        # Delete the ZIP only after extraction completes successfully to save disk space
        zip_path.unlink()
    finally:
        temporary.unlink(missing_ok=True)

    return destination


def main() -> int:
    """Execute the full BTS monthly acquisition process based on config.json."""
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
                print(f"  Reusing BTS file {csv_path} because it already exists.", flush=True)
                continue

            started = time.monotonic()
            if zip_path.exists():
                print(f"  Reusing BTS archive {zip_path} because it already exists.", flush=True)
            else:
                print(f"  Downloading BTS archive {zip_path} from {url}", flush=True)

            download_file(url, zip_path)
            print(f"  Extracting BTS file {csv_path}", flush=True)
            extract_csv(zip_path, csv_path)
            completed += 1
            print(f"  Prepared BTS file {csv_path} in {time.monotonic() - started:.1f}s.", flush=True)

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
        f"BTS Acquisition Summary: Prepared: {completed}; Reused: {skipped}; Failed: {len(failed)}"
    )
    if failed:
        print("  Failed months: " + ", ".join(failed))
    print("-" * 80)
    return 1 if failed else 0



if __name__ == "__main__":
    main()
