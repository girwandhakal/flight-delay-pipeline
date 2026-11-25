"""Shared local civil-time to UTC cutoff and scheduled clock conversion helpers."""

from datetime import date, datetime, timedelta, timezone
import math
from typing import Any, Optional, Union
from zoneinfo import ZoneInfo

from .references import airport_zone


def valid_delay(value: Any) -> Optional[float]:
    """Safely convert numeric delay tokens to float, returning None for non-finite values."""
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def local_cutoff(
    flight_date: Optional[date],
    clock: Optional[Union[int, str]],
    zone: Optional[str],
) -> tuple[Optional[datetime], str]:
    """Convert local scheduled departure time to UTC, rejecting DST ambiguity or gaps."""
    if flight_date is None or clock is None or not zone:
        return None, "invalid_cutoff"
    spelling = str(clock).strip()
    if not spelling.isdigit() or len(spelling) > 4:
        return None, "invalid_cutoff"
    hours, minutes = divmod(int(spelling), 100)
    if not (0 <= hours < 24 and 0 <= minutes < 60) and (hours, minutes) != (24, 0):
        return None, "invalid_cutoff"

    local = datetime.combine(flight_date, datetime.min.time()) + timedelta(
        hours=hours, minutes=minutes
    )
    zone_info = ZoneInfo(zone)
    candidates = set()
    for fold in (0, 1):
        utc = local.replace(tzinfo=zone_info, fold=fold).astimezone(timezone.utc)
        if utc.astimezone(zone_info).replace(tzinfo=None) == local:
            candidates.add(utc)

    # A DST gap or repeated hour has no uniquely defensible departure instant.
    if len(candidates) != 1:
        return None, "invalid_cutoff"
    return candidates.pop().replace(tzinfo=None), "valid"


def clock_matches(instant: datetime, clock: Any, zone: Optional[str]) -> bool:
    """Verify that an instant converted to a local time zone matches an HHMM clock string."""
    spelling = str(clock).strip()
    if not spelling.isdigit() or len(spelling) > 4 or not zone:
        return False

    hours, minutes = divmod(int(spelling), 100)
    if (hours, minutes) != (24, 0) and not (0 <= hours < 24 and 0 <= minutes < 60):
        return False

    local = instant.replace(tzinfo=timezone.utc).astimezone(ZoneInfo(zone))
    return (local.hour, local.minute) == (hours % 24, minutes)


def event_zone(
    airports: dict[str, list[dict[str, Any]]],
    airport: str,
    instant: datetime,
) -> Optional[str]:
    """Identify the unique airport time zone covering an event instant."""
    zones = {
        r["time_zone"]
        for r in airports.get(airport, [])
        if r["valid_from"]
        <= instant.replace(tzinfo=timezone.utc).astimezone(ZoneInfo(r["time_zone"])).date().isoformat()
        < r["valid_to"]
    }
    return next(iter(zones)) if len(zones) == 1 else None


def scheduled_clocks(
    row: Any,
    airports: dict[str, list[dict[str, Any]]],
) -> tuple[Optional[str], Optional[str]]:
    """Resolve scheduled departure and arrival UTC clocks using only schedule fields."""
    scheduled, _ = local_cutoff(
        row.FlightDate, row.CRSDepTime, airport_zone(airports, row.Origin, row.FlightDate)
    )
    if scheduled is None:
        return None, None
    elapsed = valid_delay(row.CRSElapsedTime)
    scheduled_arrival = None
    if elapsed is not None and elapsed > 0:
        try:
            instant = scheduled + timedelta(minutes=elapsed)
            if clock_matches(instant, row.CRSArrTime, event_zone(airports, row.Dest, instant)):
                scheduled_arrival = instant.isoformat() + "Z"
        except (OverflowError, ValueError):
            pass
    return scheduled.isoformat() + "Z", scheduled_arrival
