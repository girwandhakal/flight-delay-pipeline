"""Normalize domestic convective SIGMET polygons and cloud tops."""

from datetime import datetime, timezone
import math
import re
from xml.etree import ElementTree


SCHEMA = ("sigmet_id string, issue long, expire long, west double, south double, "
          "east double, north double, top_fl double")


def polygon_bounds(wkt: str) -> tuple[float, float, float, float]:
    """Extract a geographic bounding box from a polygon or multipolygon WKT."""
    if not re.fullmatch(r"(?:MULTI)?POLYGON\s*\([\dEe+.,()\s-]+\)", wkt.strip(), re.IGNORECASE):
        raise ValueError("Expected polygon WKT")
    points = []
    for ring in re.findall(r"\(([^()]+)\)", wkt):
        vertices = [tuple(map(float, point.split())) for point in ring.split(",")]
        if len(vertices) < 4 or vertices[0] != vertices[-1] or any(
            len(point) != 2 or not all(map(math.isfinite, point))
            or not -180 <= point[0] <= 180 or not -90 <= point[1] <= 90 for point in vertices
        ):
            raise ValueError("Invalid SIGMET polygon ring")
        points.extend(vertices)
    if not points:
        raise ValueError("SIGMET polygon is empty")
    lon, lat = zip(*points)
    return min(lon), min(lat), max(lon), max(lat)


def stamp(value: str) -> int:
    utc = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if utc.utcoffset() != timezone.utc.utcoffset(None):
        raise ValueError(f"SIGMET time is not UTC: {value}")
    return int(utc.timestamp())


def read_archive(path: str) -> list[tuple]:
    """Read IEM KML, which carries geometries omitted from its CSV export."""
    root = ElementTree.parse(path).getroot()
    if not root.tag.endswith("}kml"):
        raise ValueError(f"Expected IEM SIGMET KML: {path}")
    rows = []
    for item in root.findall(".//{*}Placemark"):
        attrs = {node.attrib.get("name"): node.text or "" for node in item.findall(".//{*}SimpleData")}
        if attrs.get("TYPE") != "C" or not re.fullmatch(r"\d+[ECW]", attrs.get("LABEL", "")):
            continue
        issue, expire = stamp(attrs["ISSUE"]), stamp(attrs["EXPIRE"])
        if not 0 < expire - issue <= 6 * 3600:
            raise ValueError(f"Invalid active interval for {attrs['PROD_ID']}")
        tops = re.findall(r"\bTOPS?\s+(?:(?:TO|ABV|ABOVE|BLW|BELOW)\s+)?FL(\d{2,3})\b", attrs.get("TEXT", ""))
        identity = f"{attrs['PROD_ID']}:{attrs['LABEL']}"
        polygons = item.findall(".//{*}Polygon")
        if not polygons:
            raise ValueError(f"Missing convective SIGMET geometry: {identity}")
        for polygon in polygons:
            rings = []
            for coords in polygon.findall(".//{*}coordinates"):
                rings.append("(" + ",".join(" ".join(point.split(",")[:2])
                                             for point in (coords.text or "").split()) + ")")
            bounds = polygon_bounds("POLYGON (" + ",".join(rings) + ")")
            rows.append((identity, issue, expire, *bounds, float(max(map(int, tops))) if tops else None))
    return rows
