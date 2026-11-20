"""Decode archived TAF bulletins into nonoverlapping UTC forecast segments."""

from calendar import monthrange
from datetime import datetime, timedelta, timezone
import re
from zipfile import ZipFile


HEADER = re.compile(r"\b([A-Z][A-Z0-9]{3})\s+(\d{6})Z\s+(?:(\d{4})/(\d{4})\s*)?")
CHANGE = re.compile(r"\b(?:FM(\d{6})|((?:PROB(?:30|40)(?:\s+TEMPO)?|TEMPO|BECMG))\s+(\d{4})/(\d{4}))\b")
SCHEMA = ("station string, product string, issue long, segment_start long, segment_end long, "
          "ifr int, thunderstorm int, freezing_precip int, wind_shear int, ceiling_m double, visibility_km double")


def coded_time(code: str, reference: datetime) -> datetime:
    """Resolve a day/hour code across month boundaries, including hour 24."""
    day, hour = int(code[:2]), int(code[2:4])
    minute = int(code[4:] or "0")
    if hour > 24 or minute > 59 or (hour == 24 and minute):
        raise ValueError(f"Invalid TAF day/hour: {code}")
    first = reference.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    neighboring = (first - timedelta(days=1), first, first + timedelta(days=32))
    candidates = [
        month.replace(day=day) + timedelta(hours=hour, minutes=minute)
        for month in neighboring if 1 <= day <= monthrange(month.year, month.month)[1]
    ]
    return min(candidates, key=lambda candidate: abs(candidate - reference))


def conditions(text: str) -> dict:
    """Decode explicit weather elements so temporary groups can inherit the rest."""
    result = {}
    layers = re.findall(r"\b(BKN|OVC|VV)(\d{3}|///)(?:CB|TCU)?(?!\S)", text)
    if layers:
        heights = [int(value) * 100 for _, value in layers if value != "///"]
        result["ceiling_ft"] = min(heights, default=None)
        result["clear_ceiling"] = False
    elif re.search(r"\b(?:SKC|CLR|NSC|NCD|CAVOK|(?:FEW|SCT)\d{3}(?:CB|TCU)?)\b", text):
        result.update(ceiling_ft=None, clear_ceiling=True)
    vis = re.search(r"(?<!\S)([PM]?)(?:(\d+)\s+)?(\d+(?:/\d+)?)SM\b", text)
    if vis:
        mark, whole, amount = vis.groups()
        numerator, _, denominator = amount.partition("/")
        if denominator != "0":
            value = int(whole or 0) + int(numerator) / int(denominator or 1)
            result["visibility_mi"] = max(0.0, value - 0.01) if mark == "M" else value
    elif "CAVOK" in text.split():
        result["visibility_mi"] = 10 / 1.609344
    else:
        metric = re.search(r"(?<!\S)(\d{4})(?:NDV)?(?!\S)", text)
        if metric:
            result["visibility_mi"] = int(metric[1]) / 1609.344
    tokens = text.split()
    weather = [token for token in tokens if re.fullmatch(
        r"[-+]?(?:VC)?(?:MI|PR|BC|DR|BL|SH|TS|FZ)?(?:DZ|RA|SN|SG|IC|PL|GR|GS|UP|BR|FG|FU|VA|DU|SA|HZ|PY|PO|SQ|FC|SS|DS)+|VCTS|TS|NSW",
        token,
    )]
    if weather or "CAVOK" in tokens:
        result["thunderstorm"] = any("TS" in token for token in weather)
        result["freezing_precip"] = any("FZRA" in token or "PL" in token or
                                       (token.startswith("+") and "SN" in token) for token in weather)
    if any(re.fullmatch(r"(?:FEW|SCT|BKN|OVC)\d{3}CB", token) for token in tokens):
        result["thunderstorm"] = True
    if any(re.fullmatch(r"WS\d{3}/(?:\d{3}|VRB)\d{2,3}KT", token) for token in tokens):
        result["wind_shear"] = True
    return result


def measurements(states: list[dict]) -> tuple:
    """Use the more adverse values when conditional forecasts overlap."""
    ceiling = min((state["ceiling_ft"] for state in states if state.get("ceiling_ft") is not None), default=None)
    visibility = min((state["visibility_mi"] for state in states if state.get("visibility_mi") is not None), default=None)
    if (ceiling is not None and ceiling < 1000) or (visibility is not None and visibility < 3):
        ifr = 1
    elif visibility is not None and all(state.get("ceiling_ft") is not None or state.get("clear_ceiling") for state in states):
        ifr = 0
    else:
        ifr = None
    flags = tuple(int(any(state.get(key, False) for state in states))
                  for key in ("thunderstorm", "freezing_precip", "wind_shear"))
    return (ifr, *flags, ceiling * 0.3048 if ceiling is not None else None,
            visibility * 1.609344 if visibility is not None else None)


def parse_taf(text: str, published: datetime, product: str) -> list[tuple]:
    """Decode prevailing and conditional groups without extending bulletin validity."""
    text = " ".join(text.upper().rsplit("=", 1)[0].split())
    header = HEADER.search(text)
    if not header:
        raise ValueError(f"Missing TAF header in {product}")
    station, issue_code, start_code, end_code = header.groups()
    issued = max(published, coded_time(issue_code, published))
    body = text[header.end():]
    if re.match(r"(?:CNL|NIL)\b", body):
        return [(station, product, int(issued.timestamp()), int(issued.timestamp()),
                 int((issued + timedelta(hours=30)).timestamp()), None, None, None, None, None, None)]
    if not start_code or not end_code:
        raise ValueError(f"Missing TAF validity in {product}")
    start = coded_time(start_code, issued)
    end = coded_time(end_code, start + timedelta(hours=12))
    if not timedelta(0) < end - start <= timedelta(hours=36):
        raise ValueError(f"Invalid TAF validity in {product}")
    matches = list(CHANGE.finditer(body))
    prevailing = {start: conditions(body[:matches[0].start()] if matches else body)}
    conditional = []
    for index, change in enumerate(matches):
        fm, kind, lower, upper = change.groups()
        state = conditions(body[change.end():matches[index + 1].start() if index + 1 < len(matches) else len(body)])
        begin = coded_time(fm or lower, start)
        if fm and start <= begin < end:
            prevailing[begin] = state
        elif not fm:
            finish = coded_time(upper, begin + timedelta(hours=2))
            if start <= begin < finish <= end:
                conditional.append((begin, finish, state))
                if kind == "BECMG":
                    previous = prevailing[max(key for key in prevailing if key <= begin)]
                    prevailing[finish] = {**previous, **state}
    edges = sorted({start, end, *prevailing, *(edge for a, b, _ in conditional for edge in (a, b))})
    rows = []
    for left, right in zip(edges, edges[1:]):
        base = prevailing[max(key for key in prevailing if key <= left)]
        states = [base, *[{**base, **state} for a, b, state in conditional if a <= left < b]]
        rows.append((station, product, int(issued.timestamp()), int(left.timestamp()), int(right.timestamp()),
                     *measurements(states)))
    return rows


def read_archive(path: str) -> list[tuple]:
    """Read IEM AFOS ZIP members without extracting files."""
    with ZipFile(path) as archive:
        if len(archive.infolist()) >= 9999:
            raise ValueError(f"TAF export reached IEM's product limit: {path}")
        rows = []
        for member in archive.infolist():
            if member.is_dir():
                continue
            match = re.search(r"(?<!\d)(\d{12})(?!\d)", member.filename)
            if not match or member.file_size > 65536:
                raise ValueError(f"Unexpected IEM TAF member: {member.filename}")
            published = datetime.strptime(match[1], "%Y%m%d%H%M").replace(tzinfo=timezone.utc)
            try:
                rows.extend(parse_taf(archive.read(member).decode("utf-8", errors="replace"), published, member.filename))
            except ValueError:
                # A malformed bulletin provides no trustworthy forecast segments.
                continue
        if archive.infolist() and not rows:
            raise ValueError(f"No interpretable TAF bulletins in {path}")
        return rows
