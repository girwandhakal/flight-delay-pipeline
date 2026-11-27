"""Test automatic study split dates without starting Spark."""

import json

import pytest

from scripts.common.config import ROOT, read_config


@pytest.mark.parametrize(
    ("start_month", "end_month", "expected"),
    [
        ("2024-01", "2025-01", ("2024-09-30", "2024-11-30", "2025-01-31")),
        ("2024-01", "2026-01", ("2025-05-31", "2025-09-30", "2026-01-31")),
        ("2024-01", "2024-03", ("2024-01-31", "2024-02-29", "2024-03-31")),
    ],
)
def test_automatic_splits(tmp_path, start_month, end_month, expected):
    from datetime import date, timedelta

    config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    config["datasets"]["bts"].update(start_month=start_month, end_month=end_month)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")

    splits = read_config(path)["splits"]
    assert tuple(splits) == ("train", "val", "test")
    assert tuple(split["end_date"] for split in splits.values()) == expected
    assert splits["train"]["start_date"] == f"{start_month}-01"
    for previous, following in (("train", "val"), ("val", "test")):
        day_after = date.fromisoformat(splits[previous]["end_date"]) + timedelta(days=1)
        assert splits[following]["start_date"] == day_after.isoformat()


def test_automatic_splits_require_three_months(tmp_path):
    config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    config["datasets"]["bts"].update(start_month="2024-01", end_month="2024-02")
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ValueError, match="at least three BTS months"):
        read_config(path)
