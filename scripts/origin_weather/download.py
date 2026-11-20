"""Acquire airport references and NOAA LCDv2 station-year files."""

import csv
from datetime import date, datetime, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import time
from typing import Any, Union
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from tqdm import tqdm

from scripts.results import replace_file
from .measurements import MEASUREMENTS, SOURCE_FIELDS, UNITS

ROOT = Path(__file__).resolve().parents[2]
BASE_URL = "https://www.ncei.noaa.gov/oa/local-climatological-data/v2"
SOURCES = {
    "airports.dat": "https://raw.githubusercontent.com/jpatokal/openflights/master/data/airports.dat",
    "isd_history.csv": "https://www.ncei.noaa.gov/pub/data/noaa/isd-history.csv",
    "lcd_stations.txt": BASE_URL + "/doc/lcdv2-station-list.txt",
}
REPORTS = ["FM-15", "FM-16"]


def haversine_distance_km(
    lat1_deg: float,
    lon1_deg: float,
    lat2_deg: float,
    lon2_deg: float,
) -> float:
    """Calculate the great-circle distance between two coordinate pairs in kilometers."""
    lat1, lat2 = math.radians(lat1_deg), math.radians(lat2_deg)
    dlat = lat2 - lat1
    dlon = math.radians(lon2_deg - lon1_deg)
    arc = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    )
    return 12742 * math.asin(min(1.0, math.sqrt(arc)))


def write_json(path: Union[str, Path], value: Any) -> None:
    """Write an object to a formatted JSON file atomically using a temporary file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    replace_file(temporary, path)


def sha256(path: Union[str, Path]) -> str:
    """Calculate the hex-encoded SHA-256 digest of a file on disk."""
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def reference_matches(path: Union[str, Path], checksum: str) -> bool:
    """Verify that file content matches checksum, tolerating CRLF vs LF differences."""
    content = Path(path).read_bytes()
    lf = content.replace(b"\r\n", b"\n")
    return any(
        hashlib.sha256(value).hexdigest() == checksum
        for value in (content, lf, lf.replace(b"\n", b"\r\n"))
    )


def project_path(root: Path, name: Union[str, Path]) -> Path:
    """Resolve a path relative to the project root, ensuring it stays within root."""
    path = Path(name)
    resolved = (root / path).resolve()
    if path.is_absolute() or not resolved.is_relative_to(root.resolve()):
        raise ValueError(f"Path must stay inside the project: {name}")
    return resolved


def load_config(
    root: Path = ROOT,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Load and validate weather dataset and policy configuration from config.json."""
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    settings = config["datasets"]["weather"].copy()
    if settings["product"] != "lcdv2":
        raise ValueError("Only NOAA LCDv2 is supported.")
    for key in ("output_directory", "reference_directory"):
        settings[key] = project_path(root, settings[key])
    policy = config["origin_weather"]
    age, lag = policy["max_age_minutes"], policy["publication_lag_minutes"]
    if not 0 <= lag <= age or age <= 0:
        raise ValueError(
            "Weather age must be positive and at least the nonnegative publication lag."
        )
    return settings, policy, config


def fetch(url: str, path: Union[str, Path]) -> Path:
    """Download a URL to a temporary .part file with retries and progress tracking."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    for attempt in range(3):
        try:
            request = Request(url, headers={"User-Agent": "CS523-Origin-Weather/1.0"})
            with (
                urlopen(request, timeout=90) as response,
                temporary.open("wb") as stream,
            ):
                expected = response.headers.get("Content-Length")
                with tqdm.wrapattr(
                    stream,
                    "write",
                    total=int(expected) if expected else None,
                    desc=path.name,
                    unit="B",
                    unit_scale=True,
                    miniters=1,
                    leave=False,
                ) as progress:
                    while block := response.read(1024 * 1024):
                        progress.write(block)
            if not temporary.stat().st_size or (
                expected and temporary.stat().st_size != int(expected)
            ):
                raise ValueError(f"Incomplete download: {url}")
            return temporary
        except (OSError, URLError, ValueError) as error:
            temporary.unlink(missing_ok=True)
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
            time.sleep(2**attempt)


def prepare_references(
    origins: list[str],
    directory: Union[str, Path],
    start_date: str,
    end_date: str,
    prefix: str = "origin",
) -> None:
    """Download OpenFlights and NOAA metadata to construct airport-to-station mappings."""
    if prefix not in ("origin", "destination"):
        raise ValueError("Weather role must be origin or destination.")
    directory = Path(directory)
    if (directory / "sources.json").exists():
        evidence = json.loads((directory / "sources.json").read_text())
        if evidence["origins"] != sorted(origins) or evidence["scope"] != [
            start_date,
            end_date,
        ]:
            raise ValueError("Reference scope changed; use a new reference_directory.")
        for name, checksum in evidence["files"].items():
            if not reference_matches(directory / name, checksum):
                raise ValueError(f"Reference changed: {name}")
        return
    directory.mkdir(parents=True, exist_ok=True)
    evidence = {}
    for name, url in SOURCES.items():
        target = directory / name
        temporary = fetch(url, target)
        temporary.replace(target)
        evidence[name] = {"url": url, "sha256": sha256(target)}
    with (directory / "airports.dat").open(encoding="utf-8", newline="") as stream:
        airports = list(csv.reader(stream))
    with (directory / "isd_history.csv").open(encoding="utf-8", newline="") as stream:
        history = list(csv.DictReader(stream))
    stations = {
        line[:11]: line
        for line in (directory / "lcd_stations.txt").read_text().splitlines()
    }
    end_exclusive = (date.fromisoformat(end_date) + timedelta(days=1)).isoformat()
    airport_rows, mapping_rows = [], []
    for origin in sorted(origins):
        # OpenFlights airports.dat fields:
        # [1]=Name, [4]=IATA/Origin, [5]=ICAO, [6]=Latitude, [7]=Longitude, [11]=Timezone
        candidates = [row for row in airports if row[4] == origin and row[11] != "\\N"]
        if len(candidates) != 1:
            continue
        airport = candidates[0]
        airport_name, airport_icao = airport[1], airport[5]
        airport_lat, airport_lon = float(airport[6]), float(airport[7])
        airport_tz = airport[11]
        try:
            ZoneInfo(airport_tz)
        except (ZoneInfoNotFoundError, ValueError):
            if prefix == "destination":
                continue
            raise
        airport_rows.append(
            [
                origin,
                airport_name,
                airport_icao,
                airport[6],
                airport[7],
                airport_tz,
                start_date,
                end_exclusive,
            ]
        )
        matches = {}
        for station in history:
            if station["ICAO"] != airport_icao or station["WBAN"] == "99999":
                continue
            station_id = "USW000" + station["WBAN"]
            if (
                station_id not in stations
                or station["BEGIN"] > start_date.replace("-", "")
                or station["END"] < end_date.replace("-", "")
            ):
                continue
            # NOAA LCD station inventory fixed-width format:
            # line[12:20] = latitude, line[21:30] = longitude
            line = stations[station_id]
            st_lat, st_lon = float(line[12:20]), float(line[21:30])
            distance = haversine_distance_km(airport_lat, airport_lon, st_lat, st_lon)
            # Exact ICAO linkage and a <= 5 km distance check avoid a nearest-station fallback.
            if distance <= 5:
                matches[station_id] = round(distance, 3)
        if len(matches) == 1:
            station_id, distance = next(iter(matches.items()))
            mapping_rows.append(
                [
                    origin,
                    station_id,
                    start_date,
                    end_exclusive,
                    "exact_icao_wban",
                    distance,
                ]
            )
    for name, fields, rows in [
        (
            "airports.csv",
            [
                "Origin",
                "name",
                "icao",
                "latitude",
                "longitude",
                "time_zone",
                "valid_from",
                "valid_to",
            ],
            airport_rows,
        ),
        (
            "airport_station.csv",
            ["Origin", "station_id", "valid_from", "valid_to", "method", "distance_km"],
            mapping_rows,
        ),
    ]:
        with (directory / name).open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(fields)
            writer.writerows(rows)
    write_json(
        directory / "sources.json",
        {
            "retrieved_utc": datetime.now(timezone.utc).isoformat(),
            "sources": evidence,
            "origins": sorted(origins),
            "scope": [start_date, end_date],
            "files": {
                name: sha256(directory / name)
                for name in ("airports.csv", "airport_station.csv")
            },
            "mapping_policy": "Unique exact ICAO/WBAN linkage, LCD inventory location within 5 km, ISD history covering study dates; ambiguous mappings stay absent.",
            "limitations": "Current OpenFlights airport/time-zone metadata does not establish historical metadata publication availability. OpenFlights data is ODbL; NOAA data is public domain.",
        },
    )


def standard_offset(zone: str, year: int) -> int:
    """Return local standard time offset (seconds) for Northern Hemisphere airports.

    Compares Jan 1 (winter) and Jul 1 (summer) offsets; standard time is the minimum.
    """
    return int(
        min(
            datetime(year, month, 1, tzinfo=ZoneInfo(zone)).utcoffset().total_seconds()
            for month in (1, 7)
        )
    )


def download_weather(
    requests: list[tuple[str, int, str]],
    directory: Union[str, Path],
) -> dict[str, Any]:
    """Download NOAA LCDv2 CSV files for the requested station-years and write manifest."""
    directory = Path(directory)
    manifest_path = directory / "manifest.json"
    manifest = (
        json.loads(manifest_path.read_text())
        if manifest_path.exists()
        else {
            "provider": "NOAA",
            "product": "LCDv2",
            "timestamp_basis": "local_standard",
            "units": {"temperature": "C", "wind_speed": "m/s", "visibility": "km"},
            "files": {},
        }
    )
    requests = sorted(set(requests))
    zones = {(entry["station_id"], entry["year"]): entry["time_zone"] for entry in manifest["files"].values()}
    for station, year, zone in requests:
        if not re.fullmatch(r"[A-Z0-9]{11}", station) or not 1900 <= year <= 2100:
            raise ValueError("Invalid station/year request.")
        ZoneInfo(zone)
        previous = zones.setdefault((station, year), zone)
        if previous != zone:
            raise ValueError(f"Conflicting station time zones: {station}/{year}")
    manifest["timestamp_basis"] = "unvalidated"
    manifest["units"] = UNITS.copy()
    manifest["weather_fields"] = SOURCE_FIELDS
    manifest["complete"] = False
    write_json(manifest_path, manifest)
    for station_id, year, time_zone in tqdm(
        requests, desc="Weather files", unit="file"
    ):
        name = f"{year}/LCD_{station_id}_{year}.csv"
        target = directory / name
        url = f"{BASE_URL}/access/{year}/LCD_{station_id}_{year}.csv"
        entry = {
            "station_id": station_id,
            "year": year,
            "time_zone": time_zone,
            "url": url,
        }
        if target.exists():
            entry.update(status="downloaded", bytes=target.stat().st_size)
            manifest["files"][name] = entry
            tqdm.write(f"Reusing weather file {target} because it already exists.")
            continue
        temporary = None
        started = time.monotonic()
        tqdm.write(f"Downloading weather file {target} from {url}")
        try:
            temporary = fetch(url, target)
            entry.update(
                status="downloaded",
                bytes=temporary.stat().st_size,
            )
            temporary.replace(target)
        except HTTPError as error:
            if error.code != 404:
                raise
            entry["status"] = "unavailable"
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        entry["retrieved_utc"] = datetime.now(timezone.utc).isoformat()
        manifest["files"][name] = entry
        write_json(manifest_path, manifest)
        elapsed = time.monotonic() - started
        if entry["status"] == "unavailable":
            tqdm.write(f"Weather file {target} is unavailable from NOAA (HTTP 404; {elapsed:.1f}s).")
        else:
            tqdm.write(f"Downloaded weather file {target} in {elapsed:.1f}s.")
    manifest["complete"] = True
    write_json(manifest_path, manifest)
    return manifest


def main() -> int:
    """Download weather reference tables and NOAA LCDv2 station files for all flight cohorts."""
    from scripts.bts.build_features import get_spark_session
    from .build_features import (
        build_flight_cutoffs,
        load_references,
        required_station_years,
    )

    settings, policy, config = load_config()
    from scripts.bts.cache import active
    spark = get_spark_session("weather_download")
    try:
        flights = spark.read.parquet(
            *[str(active(ROOT, "gold") / name) for name in config["splits"]]
        )
        origins = [row.Origin for row in flights.select("Origin").distinct().collect()]
        prepare_references(
            origins,
            settings["reference_directory"],
            config["splits"]["train"]["start_date"],
            config["splits"]["test"]["end_date"],
        )
        references = load_references(settings["reference_directory"])
        cutoffs = build_flight_cutoffs(spark, flights, references)
        download_weather(
            required_station_years(cutoffs, policy["max_age_minutes"]),
            settings["output_directory"],
        )
        return 0
    finally:
        spark.stop()


if __name__ == "__main__":
    raise SystemExit(main())
