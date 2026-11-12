"""Check the expanded origin-weather contract without needing a Spark JVM."""

import math

import pytest

from scripts.origin_weather.measurements import (
    SOURCE_FIELDS, NUMERIC_FEATURES, MISSING_FLAGS, INDICATOR_FLAGS,
    FEATURE_COLUMNS, decode_weather,
)


def report(**changes):
    values = dict(zip(SOURCE_FIELDS, [
        "20", "4", "16.093", "0.3", "8", "360", "70", "-2", "1013.2",
        "-RA:02 BR:1 |RA BR |RA", "FEW:02 15.24 BKN:07 6.1 OVC:08 12.19",
    ]))
    return decode_weather({**values, **changes})


def test_all_eleven_fields_produce_finite_training_inputs():
    result = report()
    assert len(SOURCE_FIELDS) == 11
    assert set(result) == set(FEATURE_COLUMNS)
    assert all(result[name] is not None and math.isfinite(result[name]) for name in NUMERIC_FEATURES)
    assert all(result[name] == 0 for name in MISSING_FLAGS)
    assert all(result[name] in (0, 1) for name in INDICATOR_FLAGS)
    assert result["origin_dew_point_c"] == -2
    assert result["origin_weather_rain"] == result["origin_weather_mist"] == 1
    assert result["origin_weather_snow"] == 0
    assert result["origin_cloud_cover_oktas"] == 8
    assert result["origin_cloud_ceiling_m"] == 610


@pytest.mark.parametrize("token", [None, "", "12.3s", "*", "NaN", "inf", "1e2", "-1"])
def test_bad_numeric_tokens_stay_missing(token):
    result = report(HourlyWindGustSpeed=token)
    assert result["origin_wind_gust_speed_ms"] is None
    assert result["origin_wind_gust_speed_missing"] == 1


def test_trace_precipitation_is_distinct_from_zero_and_missing():
    trace = report(HourlyPrecipitation="T")
    zero = report(HourlyPrecipitation="0.0")
    missing = report(HourlyPrecipitation="")
    assert trace["origin_precipitation_mm"] == zero["origin_precipitation_mm"] == 0
    assert trace["origin_precipitation_trace"] == 1
    assert trace["origin_precipitation_missing"] == zero["origin_precipitation_missing"] == 0
    assert zero["origin_precipitation_trace"] == 0
    assert missing["origin_precipitation_mm"] is None
    assert missing["origin_precipitation_missing"] == 1


def test_wind_direction_distinguishes_north_calm_variable_and_unknown():
    north = report(HourlyWindDirection="360")
    east = report(HourlyWindDirection="090")
    calm = report(HourlyWindDirection="000", HourlyWindSpeed="0")
    variable = report(HourlyWindDirection="VRB")
    assert north["origin_wind_direction_sin"] == pytest.approx(0, abs=1e-12)
    assert north["origin_wind_direction_cos"] == 1
    assert east["origin_wind_direction_sin"] == 1
    for value, flag in [(calm, "origin_wind_calm"), (variable, "origin_wind_variable")]:
        assert value[flag] == 1 and value["origin_wind_direction_missing"] == 0
        assert value["origin_wind_direction_sin"] == value["origin_wind_direction_cos"] == 0
    for token in ["", "000", "361", "-1", "90s"]:
        result = report(HourlyWindDirection=token)
        assert result["origin_wind_direction_missing"] == 1
        assert result["origin_wind_direction_cos"] is None


def test_simultaneous_conditions_and_unknown_codes():
    result = report(HourlyPresentWeatherType="-TSRASN:03 FZFG:2 |TS RA SN FG |")
    for name in ["rain", "snow", "thunderstorm", "freezing", "fog"]:
        assert result["origin_weather_" + name] == 1
    for token in ["", None, "garbage", "|03|"]:
        result = report(HourlyPresentWeatherType=token)
        assert result["origin_weather_type_missing"] == 1
        assert result["origin_weather_rain"] is None
    partial = report(HourlyPresentWeatherType="RA |03|")
    assert partial["origin_weather_rain"] == 1
    assert partial["origin_weather_type_missing"] == 1
    assert partial["origin_weather_snow"] is None


def test_numeric_codes_use_their_source_group_and_expand_mixed_weather():
    automated = report(HourlyPresentWeatherType="|41|")
    manual = report(HourlyPresentWeatherType="||41")
    assert automated["origin_weather_unknown_precipitation"] == 1
    assert automated["origin_weather_fog"] == 0
    assert manual["origin_weather_fog"] == 1
    assert manual["origin_weather_unknown_precipitation"] == 0
    mixed = report(HourlyPresentWeatherType="||RA:68")
    assert mixed["origin_weather_rain"] == mixed["origin_weather_snow"] == 1
    previous = report(HourlyPresentWeatherType="||29")
    assert previous["origin_weather_thunderstorm"] == 0
    assert previous["origin_weather_other"] == 1
    hail = report(HourlyPresentWeatherType="|TS+HAIL:96|")
    assert hail["origin_weather_thunderstorm"] == hail["origin_weather_hail"] == 1


def test_no_ceiling_is_not_missing_ceiling():
    clear = report(HourlySkyConditions="CLR:00")
    few = report(HourlySkyConditions="FEW:02 76.2 SCT:04 91.44")
    for result in [clear, few]:
        assert result["origin_cloud_no_ceiling"] == 1
        assert result["origin_cloud_ceiling_m"] is None
        assert result["origin_sky_conditions_missing"] == result["origin_cloud_ceiling_missing"] == 0
    assert clear["origin_cloud_cover_oktas"] == 0
    assert few["origin_cloud_cover_oktas"] == 4
    obscured = report(HourlySkyConditions="VV:09 0.91")
    assert obscured["origin_cloud_obscured"] == 1
    assert obscured["origin_cloud_ceiling_m"] == 91
    assert obscured["origin_cloud_cover_oktas"] is None
    assert report(HourlySkyConditions="BKN:07")["origin_cloud_ceiling_missing"] == 1


@pytest.mark.parametrize("token", ["", "BKN:07 1.0s", "OVC:02 1.0", "garbage"])
def test_bad_cloud_tokens_do_not_become_clear_sky(token):
    result = report(HourlySkyConditions=token)
    assert result["origin_sky_conditions_missing"] == 1
    assert result["origin_cloud_cover_oktas"] is None
    assert result["origin_cloud_no_ceiling"] == 0


def test_missing_report_has_no_numeric_zeros_invented():
    result = decode_weather({})
    assert all(result[name] is None for name in NUMERIC_FEATURES)
    assert all(result[name] == 1 for name in MISSING_FLAGS)
    assert all(result[name] == 0 for name in INDICATOR_FLAGS)


def test_field_specific_ranges():
    result = report(HourlyDryBulbTemperature="-10", HourlyDewPointTemperature="-20",
                    HourlyRelativeHumidity="101", HourlySeaLevelPressure="0")
    assert result["origin_temperature_c"] == -10
    assert result["origin_dew_point_c"] == -20
    assert result["origin_relative_humidity_missing"] == 1
    assert result["origin_sea_level_pressure_missing"] == 1
