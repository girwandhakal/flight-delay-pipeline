"""Match flight clocks to terminal forecast segments and convective route corridors."""

from math import acos, atan2, ceil, cos, degrees, hypot, radians, sin

from pyspark.sql import DataFrame, Window, functions as F

from scripts.aircraft_rotation.build_features import scheduled_clocks


TAF_FLAGS = ("dest_taf_is_ifr", "dest_taf_thunderstorm", "dest_taf_freezing_precip", "dest_taf_wind_shear")
ENROUTE_NUMERIC = ("dest_taf_ceiling_m", "dest_taf_visibility_km", "dest_weather_deterioration_delta_m",
                   "enroute_sigmet_max_top_fl")
ENROUTE_FLAGS = (*TAF_FLAGS, "enroute_sigmet_convective_count", "enroute_sigmet_corridor_blocked", "taf_sigmet_missing")
FEATURES = (*ENROUTE_FLAGS, *ENROUTE_NUMERIC)
IDENTITY = ("flight_id", "FlightDate", "ArrDel15")


def effective_airport(airports: dict, code: str, day) -> dict:
    if day is None:
        return {}
    stamp = day.isoformat()
    return next((row for row in airports.get(code, []) if row["valid_from"] <= stamp < row["valid_to"]), {})


def coordinate(row: dict) -> tuple | None:
    try:
        lat, lon = float(row["latitude"]), float(row["longitude"])
        return (lat, lon) if -90 <= lat <= 90 and -180 <= lon <= 180 else None
    except (KeyError, TypeError, ValueError):
        return None


def buffered_route(start: tuple, finish: tuple) -> list[tuple]:
    """Bound short great-circle segments with a 50 NM latitude/longitude envelope."""
    def vector(point):
        lat, lon = map(radians, point)
        return cos(lat) * cos(lon), cos(lat) * sin(lon), sin(lat)

    u, v = vector(start), vector(finish)
    angle = acos(max(-1.0, min(1.0, sum(a * b for a, b in zip(u, v)))))
    steps = max(1, ceil(angle * 3440.065 / 25))
    points = [start]
    for step in range(1, steps):
        a, b = sin((1 - step / steps) * angle) / sin(angle), sin(step / steps * angle) / sin(angle)
        x, y, z = (a * p + b * q for p, q in zip(u, v))
        points.append((degrees(atan2(z, hypot(x, y))), degrees(atan2(y, x))))
    points.append(finish)
    boxes = []
    for (lat, lon), (next_lat, next_lon) in zip(points, points[1:]):
        pad_lat = 50 / 60
        pad_lon = pad_lat / cos(radians(max(abs(lat), abs(next_lat)) + pad_lat))
        boxes.append((min(lon, next_lon) - pad_lon, min(lat, next_lat) - pad_lat,
                      max(lon, next_lon) + pad_lon, max(lat, next_lat) + pad_lat))
    return boxes


def attach_schedule(parent: DataFrame, schedule: DataFrame, airports: dict) -> DataFrame:
    """Resolve airport locations and scheduled departure and arrival clocks from frozen Gold."""
    fields = ("FlightDate", "Origin", "Dest", "CRSDepTime", "CRSArrTime", "CRSElapsedTime")

    def resolve(row):
        origin = effective_airport(airports, row.Origin, row.FlightDate)
        dest = effective_airport(airports, row.Dest, row.FlightDate)
        dep, arr = scheduled_clocks(row, airports)
        start, finish = coordinate(origin), coordinate(dest)
        conus = bool(start and finish and all(24 <= lat <= 50 and -125 <= lon <= -66
                                              for lat, lon in (start, finish)))
        station = dest.get("icao")
        return (dep, arr, station, *(start or (None, None)), *(finish or (None, None)), conus)

    clock = F.udf(resolve, "dep string, arr string, station string, lat1 double, lon1 double, "
                  "lat2 double, lon2 double, conus boolean")
    keys = parent.select("flight_id", "FlightDate", "Origin", "Dest", "CRSElapsedTime").join(
        schedule.select("flight_id", "CRSDepTime", "CRSArrTime"), "flight_id", "left"
    )
    return keys.withColumn("clock", clock(F.struct(*fields))).select("flight_id", "clock.*").select(
        "flight_id", "lat1", "lon1", "lat2", "lon2", "conus",
        F.to_timestamp("dep").cast("long").alias("cutoff"),
        F.to_timestamp("arr").cast("long").alias("arrival"), "station",
    )


def hour(name: str):
    return F.floor(F.col(name) / 3600).cast("long")


def match_destination_taf(flights: DataFrame, taf: DataFrame, stations: list[str]) -> DataFrame:
    """Select the latest arrival-valid bulletin known at departure, with a nearby fallback."""
    available_stations = taf.select("station").distinct()
    locations = (flights.select(F.col("station").alias("candidate"), F.col("lat2").alias("clat"),
                                F.col("lon2").alias("clon")).where(F.col("candidate").isin(stations))
                 .groupBy("candidate").agg(F.min(F.struct("clat", "clon")).alias("position"))
                 .select("candidate", "position.*")
                 .join(available_stations, F.col("candidate") == F.col("station")).drop("station"))
    destinations = flights.select("station", "lat2", "lon2").where("station is not null").distinct()
    delta = (F.pow(F.sin(F.radians(F.col("lat2") - F.col("clat")) / 2), 2)
             + F.cos(F.radians("lat2")) * F.cos(F.radians("clat"))
             * F.pow(F.sin(F.radians(F.col("lon2") - F.col("clon")) / 2), 2))
    nearby = (destinations.crossJoin(F.broadcast(locations)).withColumn(
        "distance", 2 * 3440.065 * F.asin(F.sqrt(F.least(F.lit(1.0), delta)))
    ).where("distance <= 50 and station != candidate").select("station", "lat2", "lon2", "candidate", "distance"))
    primary = destinations.select("*", F.col("station").alias("candidate"), F.lit(0.0).alias("distance"))
    sites = primary.unionByName(nearby)
    forecast = taf.withColumn("arrival_hour", F.explode(F.sequence(hour("segment_start"), hour("segment_end"))))
    candidates = (flights.join(sites, ["station", "lat2", "lon2"])
                  .withColumn("arrival_hour", hour("arrival"))
                  .join(forecast.withColumnRenamed("station", "candidate"), ["candidate", "arrival_hour"])
                  .where("issue <= cutoff and segment_start <= arrival and arrival < segment_end"))
    rank = Window.partitionBy("flight_id").orderBy("distance", F.desc("issue"), "candidate", F.desc("product"))
    return candidates.withColumn("rank", F.row_number().over(rank)).where("rank = 1").select(
        "flight_id", (F.col("candidate") != F.col("station")).alias("taf_fallback"),
        F.col("issue").alias("taf_issue_utc_seconds"),
        *[F.col(source).alias(target) for source, target in zip(
            ("ifr", "thunderstorm", "freezing_precip", "wind_shear", "ceiling_m", "visibility_km"),
            (*TAF_FLAGS, *ENROUTE_NUMERIC[:2]),
        )],
    )


def match_enroute_sigmets(flights: DataFrame, sigmets: DataFrame) -> DataFrame:
    """Count distinct active advisories overlapping the buffered great-circle boxes."""
    points = ["lat1", "lon1", "lat2", "lon2"]
    route = F.udf(lambda lat1, lon1, lat2, lon2: buffered_route((lat1, lon1), (lat2, lon2)),
                  "array<struct<west:double,south:double,east:double,north:double>>")
    eligible = flights.where("conus and cutoff is not null").withColumn("cutoff_hour", hour("cutoff"))
    routes = (eligible.select(*points).distinct()
              .withColumn("box", F.explode(route(*points))).select(*points, "box.*"))

    def cells(frame):
        for key, lo, hi in (("x", "west", "east"), ("y", "south", "north")):
            frame = frame.withColumn(key, F.explode(F.sequence(
                F.floor(F.col(lo) / 5).cast("int"), F.floor(F.col(hi) / 5).cast("int"),
            )))
        return frame

    active = (sigmets.withColumn("cutoff_hour", F.explode(F.sequence(hour("issue"), hour("expire"))))
              .select("sigmet_id", "issue", "expire", "top_fl", "cutoff_hour",
                      "west", "south", "east", "north"))
    route_cells = cells(routes).groupBy(*points, "x", "y").agg(
        F.collect_set(F.struct("west", "south", "east", "north")).alias("boxes")
    )
    route_hours = eligible.select(*points, "cutoff_hour").distinct()
    spatial = (route_hours.join(route_cells, points).join(cells(active), ["x", "y", "cutoff_hour"])
               .where("exists(boxes, b -> b.west <= east and b.east >= west and b.south <= north and b.north >= south)")
               .select(*points, "cutoff_hour", "sigmet_id", "issue", "expire", "top_fl").distinct())
    hits = (eligible.join(spatial, [*points, "cutoff_hour"])
            .where("issue <= cutoff and cutoff <= expire"))
    return hits.groupBy("flight_id").agg(
        F.countDistinct("sigmet_id").cast("int").alias("enroute_sigmet_convective_count"),
        F.max("top_fl").alias("enroute_sigmet_max_top_fl"),
    )


def enrich(parent: DataFrame, flights: DataFrame, taf: DataFrame, sigmets: DataFrame, stations: list[str]) -> DataFrame:
    """Preserve each flight while distinguishing covered clear routes from missing inputs."""
    matched = (flights.select("flight_id", "cutoff", "conus")
               .join(match_destination_taf(flights, taf, stations), "flight_id", "left")
               .join(match_enroute_sigmets(flights, sigmets), "flight_id", "left"))
    covered = F.col("conus") & F.col("cutoff").isNotNull()
    return (parent.join(matched, "flight_id", "left")
            .withColumn("taf_sigmet_missing", ((~F.coalesce(covered, F.lit(False)))
                        | F.col(TAF_FLAGS[0]).isNull() | F.coalesce("taf_fallback", F.lit(False))).cast("int"))
            .withColumn("enroute_sigmet_convective_count",
                        F.when(covered, F.coalesce("enroute_sigmet_convective_count", F.lit(0))))
            .withColumn("enroute_sigmet_corridor_blocked", (F.col("enroute_sigmet_convective_count") > 0).cast("int"))
            .withColumn("dest_weather_deterioration_delta_m",
                        F.col("dest_taf_ceiling_m") - F.col("destination_cloud_ceiling_m"))
            .drop("cutoff", "conus"))
