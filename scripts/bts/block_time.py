"""Build publication-aware historical gate-to-gate references."""

import json
import re
from datetime import date, datetime, timedelta, timezone
from hashlib import sha256 as digest_bytes
from pathlib import Path
from urllib.parse import urlsplit

from pyspark.sql import functions as F

from scripts.common.config import project_path
from scripts.common.io import sha256

HISTORY_ROOT = "data/bronze/bts_block_history"
DEFAULT_MANIFEST = f"{HISTORY_ROOT}/manifest.json"
BTS_ARCHIVE_INDEX = "https://transtats.bts.gov/PREZIP/"
BLOCK_TIME_SIDECAR = "_block_time.json"
SLACK_FEATURES = ("outbound_schedule_slack_minutes", "outbound_schedule_slack_missing")
PERCENTILE = 0.20
ACCURACY = 10_000
MIN_SAMPLES = 100
MIN_DURATION = 10
MAX_DURATION = 1440
ROUTE = ["Origin", "Dest"]
CARRIER_ROUTE = ["Reporting_Airline", *ROUTE]
KEY = ["FlightDate", "Reporting_Airline", "Flight_Number_Reporting_Airline", *ROUTE, "CRSDepTime"]
POLICY = {
    "version": 2,
    "percentile": PERCENTILE,
    "accuracy": ACCURACY,
    "min_samples": MIN_SAMPLES,
    "duration_bounds_minutes": [MIN_DURATION, MAX_DURATION],
    "schedule_scope": "Retrospective BTS schedule. Strict replay requires as-of schedule evidence.",
    "history_scope": "Local BTS CSVs use official archive dates as a release proxy. Historical CSV versions are unverified.",
}


def _trusted_url(value):
    if not isinstance(value, str) or re.search(r"[\s\\]", value):
        return False
    url = urlsplit(value)
    host = url.hostname or ""
    return (
        url.scheme == "https" and url.username is None and url.password is None
        and url.port in (None, 443)
        and (host == "web.archive.org" or any(
            host == domain or host.endswith("." + domain) for domain in ("bts.gov", "transportation.gov")
        ))
    )


def load_block_history_manifest(root, config):
    """Verify local CSV bytes against curated historical release evidence."""
    base = project_path(root, HISTORY_ROOT)
    if base != Path(root).resolve() / HISTORY_ROOT:
        raise ValueError("Block history directory must not be a symlink.")
    name = config["bts_features"].get("outbound_schedule_slack", {}).get("release_manifest", DEFAULT_MANIFEST)
    manifest = project_path(root, name)
    if not manifest.is_relative_to(base) or manifest.suffix.lower() != ".json" or Path(name).drive:
        raise ValueError("Block history manifest must be inside its configured Bronze directory.")
    if not manifest.exists():
        return {"status": "absent", "manifest_sha256": None, "months": []}
    payload = manifest.read_bytes()
    document = json.loads(payload)
    fields = {"source_month", "csv_path", "sha256", "available_utc", "source_url", "evidence_url", "evidence_note"}
    if (not isinstance(document, dict) or set(document) != {"schema_version", "months"}
            or type(document["schema_version"]) is not int or document["schema_version"] != 1
            or not isinstance(document["months"], list)):
        raise ValueError("Invalid block history manifest schema.")
    months, paths, hashes = set(), set(), set()
    for entry in document["months"]:
        if (not isinstance(entry, dict) or set(entry) != fields
                or not all(isinstance(value, str) and value.strip() for value in entry.values())):
            raise ValueError("Invalid block history source declaration.")
        month = entry["source_month"]
        if not re.fullmatch(r"\d{4}-\d{2}", month):
            raise ValueError(f"Invalid block history source month: {month}")
        month_end = (date.fromisoformat(month + "-01") + timedelta(days=32)).replace(day=1)
        available = entry["available_utc"]
        earliest = datetime.combine(month_end, datetime.min.time(), timezone(timedelta(hours=-12)))
        if (not re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?Z", available)
                or datetime.fromisoformat(available.replace("Z", "+00:00")) < earliest):
            raise ValueError(f"Invalid release time for block history {month}.")
        relative = Path(entry["csv_path"])
        if relative.is_absolute() or relative.drive:
            raise ValueError("Block history CSV paths must be relative.")
        path = project_path(manifest.parent, relative)
        digest = entry["sha256"]
        if (month in months or path in paths or digest in hashes or path.suffix.lower() != ".csv"
                or not re.fullmatch(r"[a-f0-9]{64}", digest)
                or not all(_trusted_url(entry[key]) for key in ("source_url", "evidence_url"))):
            raise ValueError(f"Invalid or duplicate block history source: {month}")
        if sha256(path) != digest:
            raise ValueError(f"Block history CSV checksum changed: {month}")
        months.add(month)
        paths.add(path)
        hashes.add(digest)
        entry["csv_path"] = path.relative_to(Path(root).resolve()).as_posix()
    return {"status": "verified", "manifest_sha256": digest_bytes(payload).hexdigest(),
            "months": sorted(document["months"], key=lambda entry: entry["source_month"])}


def resolve_block_history(root, config, bronze_dir=None):
    """Reuse the normal BTS download when a separately verified vintage is unavailable."""
    from .download import build_download_url

    history = load_block_history_manifest(root, config)
    dates = config["bts_features"].get("outbound_schedule_slack", {}).get("bts_archive_dates", {})
    if history["status"] != "absent" or not dates:
        return history
    root = Path(root).resolve()
    bronze = Path(bronze_dir).resolve() if bronze_dir else project_path(
        root, config["datasets"]["bts"]["output_directory"],
    )
    train = config["splits"]["train"]
    freeze = validation_boundary(config["splits"])
    releases = []
    for month, listed_date in sorted(dates.items()):
        if not train["start_date"][:7] <= month <= train["end_date"][:7]:
            continue
        service_month = date.fromisoformat(f"{month}-01")
        listed_day = date.fromisoformat(listed_date)
        if listed_day < (service_month + timedelta(days=32)).replace(day=1):
            raise ValueError(f"BTS archive date precedes the end of service month {month}.")
        # The index does not state a time zone. Wait until the listed day has ended worldwide.
        available = datetime.combine(listed_day + timedelta(days=1), datetime.min.time(),
                                     timezone(timedelta(hours=-12)))
        if available >= freeze:
            continue
        year, month_number = service_month.year, service_month.month
        url = build_download_url(year, month_number)
        name = Path(url.rsplit("/", 1)[1]).with_suffix(".csv")
        path = project_path(bronze, Path(str(year)) / f"{month_number:02d}" / name)
        if not path.is_file():
            continue
        if not path.is_relative_to(root):
            raise ValueError("BTS block-time inputs must stay inside the project.")
        releases.append({
            "source_month": month,
            "csv_path": path.relative_to(root).as_posix(),
            "sha256": sha256(path),
            "available_utc": available.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
            "source_url": url,
            "evidence_url": BTS_ARCHIVE_INDEX,
            "evidence_note": f"BTS lists {listed_date}. The local CSV is pinned for this run. "
                             "Its exact historical version is unverified.",
        })
    return {**history, "status": "retrospective_bts_archive" if releases else "absent", "months": releases}


def validation_boundary(splits):
    return datetime.combine(
        date.fromisoformat(splits["val"]["start_date"]), datetime.min.time(),
        timezone(timedelta(hours=14)),
    )


def read_block_history(spark, root, releases, airports, train_dates):
    from .recent_delays import EVENT_COLUMNS

    raw = None
    fields = [*EVENT_COLUMNS, "CRSArrTime", "Cancelled"]
    for source in releases:
        path = project_path(root, source["csv_path"])
        frame = spark.read.option("header", True).csv(str(path))
        if not set(fields).issubset(frame.columns):
            raise ValueError(f"Block history CSV lacks required fields: {path.name}")
        frame = frame.select(*fields, F.lit(source["source_month"]).alias("_source_month"),
                             F.to_timestamp(F.lit(source["available_utc"])).alias("_release"))
        raw = frame if raw is None else raw.unionByName(frame)
    return build_block_duration_history(raw, airports, train_dates)


def build_block_duration_history(raw, airports, train_dates):
    """Resolve historical gate clocks, discarding conflicting source rows."""
    from .recent_delays import EVENT_COLUMNS, event_clocks, scheduled_clocks

    fields = [*EVENT_COLUMNS, "CRSArrTime", "Cancelled"]
    rows = raw.select(*[F.trim(name).alias(name) for name in fields], "_source_month", "_release")
    valid_key = F.col("CRSDepTime").rlike(r"^\d{1,4}$")
    for name in KEY[1:-1]:
        valid_key = valid_key & (F.length(name) > 0)
    rows = rows.withColumn("FlightDate", F.expr("try_cast(FlightDate as date)"))
    valid_date = (
        (F.date_format("FlightDate", "yyyy-MM") == F.col("_source_month"))
        & F.col("FlightDate").between(train_dates["start_date"], train_dates["end_date"])
    )
    rows = rows.withColumn("_eligible", F.coalesce(valid_key & valid_date, F.lit(False)))
    ineligible = (rows.where("not _eligible").groupBy("_source_month").count()
                  .withColumn("reason", F.lit("invalid_identity_or_outside_training")))
    rows = (rows.where("_eligible").withColumn("CRSDepTime", F.lpad("CRSDepTime", 4, "0"))
            .withColumn("flight_id", F.concat_ws("_", *KEY)))
    payload = F.struct(*[name for name in fields if name not in KEY])
    grouped = (rows.groupBy("_source_month", "_release", "flight_id", *KEY)
               .agg(F.min(payload).alias("_first"), F.max(payload).alias("_last"))
               .select("*", "_first.*"))
    def gate_clocks(row):
        if not scheduled_clocks(row, airports)[1]:
            return None, None
        return event_clocks(row, airports)

    clocks = F.udf(gate_clocks, "departure string, arrival string")
    verified = F.when(
        (F.col("_first") == F.col("_last")) & (F.expr("try_cast(Cancelled as double)") == 0)
        & (F.expr("try_cast(Diverted as double)") == 0), clocks(F.struct(*EVENT_COLUMNS, "CRSArrTime")),
    )
    grouped = (grouped.withColumn("_clocks", verified)
               .withColumn("_arrived", F.to_timestamp("_clocks.arrival"))
               .withColumn("_duration", (F.col("_arrived").cast("long")
                                         - F.to_timestamp("_clocks.departure").cast("long")) / 60))
    reason = (F.when(F.col("_first") != F.col("_last"), "conflicting_identity")
              .when(F.col("_duration").isNull(), "invalid_or_ineligible_operation")
              .when(~F.col("_duration").between(MIN_DURATION, MAX_DURATION), "invalid_duration")
              .when(F.col("_arrived") >= F.col("_release"), "completion_not_before_release")
              .otherwise("usable"))
    return (grouped.select("flight_id", *CARRIER_ROUTE, "_source_month", "_release", "_duration",
                           reason.alias("reason")), ineligible.select("_source_month", "reason", "count"))


def build_block_time_reference(history, snapshots):
    eligible = history.where("reason = 'usable'").join(F.broadcast(snapshots), F.col("_release") < F.col("_snapshot"))
    references = [
        eligible.groupBy("_snapshot", *keys).agg(
            F.countDistinct("flight_id").alias(f"{prefix}_samples"),
            F.percentile_approx("_duration", PERCENTILE, ACCURACY).alias(prefix),
        ).where(F.col(f"{prefix}_samples") >= MIN_SAMPLES)
        for keys, prefix in ((CARRIER_ROUTE, "_carrier_block"), (ROUTE, "_route_block"))
    ]
    return eligible, references


def add_outbound_schedule_slack(flights, history, splits, releases):
    """Left-join past route references and retain a reason for coverage accounting."""
    cutoff, bound = F.col("_cutoff"), F.lit(validation_boundary(splits))
    snapshot = (F.when(F.col("split") == "train", F.least(F.date_trunc("month", cutoff), bound))
                .when(cutoff >= bound, bound))
    rows = flights.withColumn("_snapshot", F.when(cutoff.isNotNull(), snapshot))
    snapshots = rows.select("_snapshot").where("_snapshot is not null").distinct()
    if releases:
        eligible, references = build_block_time_reference(history, snapshots)
        if eligible.select("flight_id", "_snapshot").join(
            rows.select("flight_id", "_snapshot"), ["flight_id", "_snapshot"], "left_semi",
        ).limit(1).count():
            raise ValueError("A flight contributed to its own block-time reference.")
        for frame, keys in zip(references, (CARRIER_ROUTE, ROUTE)):
            rows = rows.join(frame, ["_snapshot", *keys], "left")
    else:
        rows = rows.withColumns({
            f"{prefix}{suffix}": F.lit(None).cast(kind)
            for prefix in ("_carrier_block", "_route_block")
            for suffix, kind in (("", "double"), ("_samples", "long"))
        })
    release_times = [datetime.fromisoformat(source["available_utc"].replace("Z", "+00:00")) for source in releases]
    released = F.col("_snapshot") > F.lit(min(release_times)) if release_times else F.lit(False)
    scheduled = F.expr("try_cast(CRSElapsedTime as double)")
    reason = (F.when(F.col("_snapshot").isNull(), "invalid_cutoff")
              .when(~F.coalesce(scheduled.between(MIN_DURATION, MAX_DURATION), F.lit(False)), "invalid_scheduled_duration")
              .when(~released, "no_released_history")
              .when(F.col("_carrier_block").isNotNull(), "carrier_route")
              .when(F.col("_route_block").isNotNull(), "route")
              .otherwise("insufficient_history"))
    value, missing = SLACK_FEATURES
    return (rows.withColumn("_block_reason", reason)
            .withColumn(value, F.when(F.col("_block_reason").isin("carrier_route", "route"),
                                      scheduled - F.coalesce("_carrier_block", "_route_block")))
            .withColumn(missing, F.col(value).isNull().cast("int"))
            .withColumn("_block_samples", F.when(F.col(value).isNotNull(),
                        F.coalesce("_carrier_block_samples", "_route_block_samples")))
            .drop("_carrier_block", "_route_block", "_carrier_block_samples", "_route_block_samples"))
