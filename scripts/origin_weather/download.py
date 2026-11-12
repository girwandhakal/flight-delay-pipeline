"""Acquire airport references and verified NOAA LCDv2 station-year files."""

import csv
from datetime import date, datetime, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

from scripts.results import replace_file

ROOT = Path(__file__).resolve().parents[2]
BASE_URL = "https://www.ncei.noaa.gov/oa/local-climatological-data/v2"
SOURCES = {
    "airports.dat": "https://raw.githubusercontent.com/jpatokal/openflights/master/data/airports.dat",
    "isd_history.csv": "https://www.ncei.noaa.gov/pub/data/noaa/isd-history.csv",
    "lcd_stations.txt": BASE_URL + "/doc/lcdv2-station-list.txt",
}
MEASUREMENTS = {
    "HourlyDryBulbTemperature": "origin_temperature_c",
    "HourlyWindSpeed": "origin_wind_speed_ms",
    "HourlyVisibility": "origin_visibility_km",
}
REPORTS = ["FM-15", "FM-16"]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    replace_file(temporary, path)


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def project_path(root, name):
    path = Path(name)
    resolved = (root / path).resolve()
    if path.is_absolute() or not resolved.is_relative_to(root.resolve()):
        raise ValueError(f"Path must stay inside the project: {name}")
    return resolved


def load_config(root=ROOT):
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


def fetch(url, path):
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
                while block := response.read(1024 * 1024):
                    stream.write(block)
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


def prepare_references(origins, directory, start_date, end_date):
    directory = Path(directory)
    if (directory / "sources.json").exists():
        evidence = json.loads((directory / "sources.json").read_text())
        if evidence["origins"] != sorted(origins) or evidence["scope"] != [
            start_date,
            end_date,
        ]:
            raise ValueError("Reference scope changed; use a new reference_directory.")
        for name, checksum in evidence["files"].items():
            if sha256(directory / name) != checksum:
                raise ValueError(f"Reference changed: {name}")
        return
    directory.mkdir(parents=True, exist_ok=True)
    evidence = {}
    for name, url in SOURCES.items():
        target = directory / name
        temporary = fetch(url, target)
        temporary.replace(target)
        evidence[name] = {"url": url, "sha256": sha256(target)}
    airports = list(
        csv.reader((directory / "airports.dat").open(encoding="utf-8", newline=""))
    )
    history = list(
        csv.DictReader(
            (directory / "isd_history.csv").open(encoding="utf-8", newline="")
        )
    )
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
        ZoneInfo(airport_tz)
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
            lat1, lat2 = math.radians(airport_lat), math.radians(st_lat)
            dlat, dlon = lat2 - lat1, math.radians(st_lon - airport_lon)
            distance = 12742 * math.asin(
                min(
                    1,
                    math.sqrt(
                        math.sin(dlat / 2) ** 2
                        + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
                    ),
                )
            )
            # Exact ICAO linkage and a location check avoid a nearest-station fallback.
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


def standard_offset(zone, year):
    """Return local standard time offset (seconds) for Northern Hemisphere airports.

    Compares Jan 1 (winter) and Jul 1 (summer) offsets; standard time is the minimum.
    """
    return int(
        min(
            datetime(year, month, 1, tzinfo=ZoneInfo(zone)).utcoffset().total_seconds()
            for month in (1, 7)
        )
    )


def validate_file(path, station_id, year, time_zone):
    count, observations, checked_times, time_mismatches, utc_times = 0, 0, 0, 0, 0
    offset = timedelta(seconds=standard_offset(time_zone, year))
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"STATION", "DATE", "REPORT_TYPE", "REM", *MEASUREMENTS}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Invalid LCDv2 header: {path}")
        for row in reader:
            count += 1
            if row["STATION"] != station_id or not row["DATE"].startswith(
                str(year) + "-"
            ):
                raise ValueError(f"Unexpected station/year: {path}")
            clock = datetime.fromisoformat(row["DATE"])
            if clock.tzinfo is not None:
                raise ValueError("Expected a timezone-naive CSV clock.")
            if row["REPORT_TYPE"] not in REPORTS:
                continue
            observations += 1
            metar = re.search(
                r"(?:METAR|SPECI)\s+\w+\s+(\d{2})(\d{2})(\d{2})Z", row["REM"]
            )
            if metar:
                reported_utc = "".join(metar.groups())
                if clock.strftime("%d%H%M") == reported_utc:
                    checked_times += 1
                    utc_times += 1
                elif (clock - offset).strftime("%d%H%M") == reported_utc:
                    checked_times += 1
                else:
                    time_mismatches += 1
    if not count or not observations or not checked_times:
        raise ValueError(f"No verifiable hourly weather in {path}")
    return {
        "rows": count,
        "observations": observations,
        "metar_time_checks": checked_times,
        "utc_time_checks": utc_times,
        "local_standard_time_checks": checked_times - utc_times,
        "quarantine_time_mismatches": time_mismatches,
    }


def download_weather(requests, directory):
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
    manifest["timestamp_basis"] = "metar_verified_utc_or_local_standard"
    manifest["complete"] = False
    write_json(manifest_path, manifest)
    for station_id, year, time_zone in sorted(set(requests)):
        if not re.fullmatch(r"[A-Z0-9]{11}", station_id) or not 1900 <= year <= 2100:
            raise ValueError("Invalid station/year request.")
        name = f"{year}/LCD_{station_id}_{year}.csv"
        target = directory / name
        previous = manifest["files"].get(name)
        if target.exists():
            if not previous or previous["sha256"] != sha256(target):
                raise ValueError(f"Raw file changed or lacks provenance: {target}")
            previous.update(validate_file(target, station_id, year, time_zone))
            print(f"Weather {station_id} {year}: reused verified file", flush=True)
            continue
        url = f"{BASE_URL}/access/{year}/LCD_{station_id}_{year}.csv"
        temporary = None
        entry = {
            "station_id": station_id,
            "year": year,
            "time_zone": time_zone,
            "url": url,
        }
        try:
            temporary = fetch(url, target)
            entry.update(validate_file(temporary, station_id, year, time_zone))
            entry.update(
                status="verified",
                sha256=sha256(temporary),
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
        print(f"Weather {station_id} {year}: {entry['status']}", flush=True)
    manifest["complete"] = True
    write_json(manifest_path, manifest)
    return manifest


def main():
    from scripts.bts.build_features import get_spark_session
    from .build_features import (
        build_flight_cutoffs,
        load_references,
        required_station_years,
    )

    settings, policy, config = load_config()
    spark = get_spark_session("weather_download")
    try:
        flights = spark.read.parquet(
            *[str(ROOT / "data/gold/bts" / name) for name in config["splits"]]
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
