"""LCDv2 SI measurements and conservative decoding of weather/cloud tokens.

Source: https://www.ncei.noaa.gov/oa/local-climatological-data/v2/doc/lcdv2_DOCUMENTATION.pdf
"""

import math
import re


# Keep the original EDA's three-field contract while expanding pipeline inputs.
MEASUREMENTS = {
    "HourlyDryBulbTemperature": "origin_temperature_c",
    "HourlyWindSpeed": "origin_wind_speed_ms",
    "HourlyVisibility": "origin_visibility_km",
}
NUMERIC_MEASUREMENTS = {
    **MEASUREMENTS,
    "HourlyPrecipitation": "origin_precipitation_mm",
    "HourlyWindGustSpeed": "origin_wind_gust_speed_ms",
    "HourlyWindDirection": "origin_wind_direction_degrees",
    "HourlyRelativeHumidity": "origin_relative_humidity_pct",
    "HourlyDewPointTemperature": "origin_dew_point_c",
    "HourlySeaLevelPressure": "origin_sea_level_pressure_hpa",
}
SOURCE_FIELDS = [*NUMERIC_MEASUREMENTS, "HourlyPresentWeatherType", "HourlySkyConditions"]
CONDITIONS = {
    "rain": {"RA"}, "drizzle": {"DZ"}, "snow": {"SN", "SG"},
    "ice": {"IC", "PL"}, "hail": {"GR", "GS", "HAIL"},
    "fog": {"FG"}, "mist": {"BR"}, "thunderstorm": {"TS"},
    "freezing": {"FZ"}, "haze": {"HZ"}, "smoke": {"FU"},
    "dust_sand": {"DU", "SA", "DS", "SS", "PO"},
    "unknown_precipitation": {"UP"},
    "other": {"VA", "PY", "SQ", "FC", "LIGHTNING", "RECENT"},
}
# Numeric AW and MW codes have different meanings; never decode them across groups.
AW_CODES = {
    code: set(phenomena.split())
    for codes, phenomena in [
        ([4], "HZ"), ([5], "FU"), ([7], "DU"), ([10], "BR"), ([11], "IC"),
        ([12], "LIGHTNING"), ([18], "SQ"), (range(20, 30), "RECENT"),
        (range(30, 35), "FG"), ([35], "FZ FG"), (range(40, 47), "UP"),
        ([47, 48], "FZ UP"), (range(50, 54), "DZ"), (range(54, 57), "FZ DZ"),
        ([57, 58], "DZ RA"), (range(60, 64), "RA"), (range(64, 67), "FZ RA"),
        ([67, 68], "RA SN"), (range(70, 74), "SN"), (range(74, 77), "PL"),
        ([77], "SG"), ([78], "IC"), ([80], "SH UP"), (range(81, 85), "SH RA"),
        (range(85, 88), "SH SN"), ([89], "GR"), ([90, 91, 94], "TS"),
        ([92, 95], "TS UP"), ([93, 96], "TS GR"), ([99], "FC"),
    ] for code in codes
}
MW_CODES = {
    code: set(phenomena.split())
    for codes, phenomena in [
        ([4], "FU"), ([5], "HZ"), (range(6, 10), "DU"), ([10], "BR"),
        ([11, 12], "FG"), ([13], "LIGHTNING"), (range(14, 17), "VC UP"),
        ([17], "TS"), ([18], "SQ"), ([19], "FC"), (range(20, 30), "RECENT"),
        (range(30, 36), "DU"), (range(36, 40), "SN"), (range(40, 50), "FG"),
        (range(50, 56), "DZ"), ([56, 57], "FZ DZ"), ([58, 59], "DZ RA"),
        (range(60, 66), "RA"), ([66, 67], "FZ RA"), ([68, 69], "RA SN"),
        (range(70, 76), "SN"), ([76], "IC"), ([77], "SG"), ([78], "SN"),
        ([79], "PL"), (range(80, 83), "SH RA"), ([83, 84], "SH RA SN"),
        ([85, 86], "SH SN"), (range(87, 91), "SH GR"), ([91, 92], "RA"),
        ([93, 94], "UP"), ([95, 97], "TS UP"), ([96, 99], "TS GR"), ([98], "TS DU"),
    ] for code in codes
}
NUMERIC_FEATURES = [
    value for source, value in NUMERIC_MEASUREMENTS.items()
    if source != "HourlyWindDirection"
] + [
    "origin_wind_direction_sin", "origin_wind_direction_cos",
    *["origin_weather_" + name for name in CONDITIONS],
    "origin_cloud_cover_oktas", "origin_cloud_ceiling_m",
]
MISSING_FLAGS = [
    "origin_temperature_missing", "origin_wind_speed_missing", "origin_visibility_missing",
    "origin_precipitation_missing", "origin_wind_gust_speed_missing",
    "origin_wind_direction_missing", "origin_relative_humidity_missing",
    "origin_dew_point_missing", "origin_sea_level_pressure_missing",
    "origin_weather_type_missing", "origin_sky_conditions_missing", "origin_cloud_ceiling_missing",
]
INDICATOR_FLAGS = [
    "origin_precipitation_trace", "origin_wind_calm", "origin_wind_variable",
    "origin_cloud_no_ceiling", "origin_cloud_obscured",
]
FEATURE_COLUMNS = [*NUMERIC_FEATURES, *MISSING_FLAGS, *INDICATOR_FLAGS]
UNITS = {
    "temperature": "C", "wind_speed": "m/s", "visibility": "km",
    "precipitation": "mm", "wind_gust_speed": "m/s", "wind_direction": "degrees true",
    "relative_humidity": "%", "dew_point": "C", "sea_level_pressure": "hPa",
    "cloud_ceiling": "m (CSV cloud heights multiplied by 100)",
}


def number(token):
    text = (token or "").strip()
    if not re.fullmatch(r"[+-]?(\d+(\.\d*)?|\.\d+)", text):
        return None
    value = float(text)
    return value if math.isfinite(value) else None


def weather_conditions(token):
    """Decode AU contractions and source-specific AW/MW codes, preserving uncertainty."""
    found, unknown = set(), False
    allowed = set().union(*CONDITIONS.values(), {"VC", "MI", "PR", "BC", "DR", "BL", "SH", "VA", "PY", "SQ", "FC"})
    for group, tokens in enumerate((token or "").strip().split("|")):
        for item in tokens.split():
            match = re.fullmatch(r"[+-]?([A-Z]+(?:\+[A-Z]+)*)(?::(\d{1,2}))?|(\d{1,2})", item)
            if not match or group > 2:
                unknown = True
                continue
            if match[1]:
                spelling = match[1].replace("HAIL", "GR").replace("+", "")
                parts = set(re.findall(r"..", spelling))
                if len(spelling) % 2 or not parts <= allowed:
                    unknown = True
                    continue
                found.update(parts)
            code = match[2] or match[3]
            if code is not None and group in (1, 2):
                decoded = (AW_CODES if group == 1 else MW_CODES).get(int(code))
                if decoded is None:
                    unknown = True
                else:
                    found.update(decoded)
            elif match[3]:
                unknown = True
    known = bool(found) and not unknown
    return {
        "origin_weather_" + name: 1.0 if codes & found else (0.0 if known else None)
        for name, codes in CONDITIONS.items()
    }, int(not known)


def cloud_conditions(token):
    text = (token or "").strip()
    pattern = r"(CLR|FEW|SCT|BKN|OVC|VV|10):(\d{2})(?:\s+([+]?(?:\d+(?:\.\d*)?|\.\d+)))?"
    layers = list(re.finditer(pattern, text))
    valid = bool(layers) and not re.sub(pattern, "", text).strip()
    ranges = {"CLR": {0}, "FEW": {1, 2}, "SCT": {3, 4}, "BKN": {5, 6, 7}, "OVC": {8}, "VV": {9}, "10": {10}}
    valid = valid and all(int(layer[2]) in ranges[layer[1]] for layer in layers)
    if not valid:
        return None, None, 1, 1, 0, 0
    obscured = any(layer[1] in {"VV", "10"} for layer in layers)
    cover = None if obscured else float(max(int(layer[2]) for layer in layers))
    ceiling_layers = [layer for layer in layers if layer[1] in {"BKN", "OVC", "VV"}]
    heights = [number(layer[3]) for layer in ceiling_layers]
    ceiling_known = bool(heights) and all(height is not None for height in heights)
    ceiling = min(heights) * 100 if ceiling_known else None
    if ceiling is not None and not math.isfinite(ceiling):
        ceiling, ceiling_known = None, False
    no_ceiling = not ceiling_layers and not obscured
    return cover, ceiling, 0, int(not ceiling_known and not no_ceiling), int(no_ceiling), int(obscured)


def decode_weather(values):
    """Return training values and flags from one report, never mixing timestamps."""
    result = {}
    for (source, output), flag in zip(NUMERIC_MEASUREMENTS.items(), MISSING_FLAGS):
        value = number(values.get(source))
        if source not in {"HourlyDryBulbTemperature", "HourlyDewPointTemperature"}:
            value = value if value is not None and value >= 0 else None
        if source == "HourlyRelativeHumidity" and value is not None and value > 100:
            value = None
        if source == "HourlySeaLevelPressure" and value == 0:
            value = None
        result[output] = value
        result[flag] = int(value is None)

    trace = (values.get("HourlyPrecipitation") or "").strip() == "T"
    result["origin_precipitation_trace"] = int(trace)
    if trace:
        # Zero is the numeric lower bound; the trace flag preserves nonzero precipitation.
        result["origin_precipitation_mm"] = 0.0
        result["origin_precipitation_missing"] = 0

    direction = result.pop("origin_wind_direction_degrees")
    variable = (values.get("HourlyWindDirection") or "").strip() == "VRB"
    calm = result["origin_wind_speed_ms"] == 0
    direction_known = direction is not None and 0 < direction <= 360
    result["origin_wind_calm"] = int(calm)
    result["origin_wind_variable"] = int(variable and not calm)
    result["origin_wind_direction_missing"] = int(not (direction_known or calm or variable))
    for name, function in [("sin", math.sin), ("cos", math.cos)]:
        result["origin_wind_direction_" + name] = (
            0.0 if calm or variable else function(math.radians(direction)) if direction_known else None
        )

    conditions, missing = weather_conditions(values.get("HourlyPresentWeatherType"))
    result.update(conditions)
    result["origin_weather_type_missing"] = missing
    (result["origin_cloud_cover_oktas"], result["origin_cloud_ceiling_m"],
     result["origin_sky_conditions_missing"], result["origin_cloud_ceiling_missing"],
     result["origin_cloud_no_ceiling"], result["origin_cloud_obscured"]) = cloud_conditions(values.get("HourlySkyConditions"))
    return result
