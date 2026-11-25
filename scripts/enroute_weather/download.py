"""Download and checksum the public IEM bulletin and polygon archives."""

import json
import os
from pathlib import Path
import re
import subprocess
from urllib.parse import urlencode

from scripts.common.io import fetch, sha256, write_json
from scripts.common.results import replace_file

from . import clean_sigmets, clean_taf


IEM = "https://mesonet.agron.iastate.edu/cgi-bin/"
TAF_URL = IEM + "afos/retrieve.py"
SIGMET_URL = IEM + "request/gis/sigmets.py"


def acquire(directory: Path, name: str, url: str, decoder) -> tuple[Path, str]:
    """Reuse a checksum-verified archive or atomically stage a validated download."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    manifest = path.with_suffix(path.suffix + ".json")
    if path.is_file() and manifest.is_file():
        record = json.loads(manifest.read_text(encoding="utf-8"))
        if record == {"url": url, "sha256": sha256(path)}:
            return path, record["sha256"]
        raise ValueError(f"Cached archive failed verification: {path}")
    staged = path.with_suffix(path.suffix + ".download")
    try:
        if os.name == "nt":
            # Windows curl uses the system certificate store for this public HTTPS endpoint.
            subprocess.run(["curl.exe", "--fail", "--silent", "--show-error", "--location",
                            "--proto", "=https", "--proto-redir", "=https", "--retry", "2",
                            "--max-time", "180", "--output", str(staged), url], check=True)
        else:
            fetch(url, staged)
        decoder(str(staged))
        digest = sha256(staged)
        replace_file(staged, path)
        write_json(manifest, {"url": url, "sha256": digest})
        return path, digest
    finally:
        staged.unlink(missing_ok=True)


def download(root: Path, start, end, stations: list[str], *, taf: dict, sigmets: dict) -> tuple[dict, dict]:
    """Fetch bounded monthly archives with small TAF batches to stay below IEM's cap."""
    from datetime import timedelta
    from hashlib import sha256 as hash_text

    from scripts.common.config import project_path

    if not stations or any(not re.fullmatch(r"[A-Z][A-Z0-9]{3}", station) for station in stations):
        raise ValueError("TAF requests require safe four-character ICAO station codes.")
    taf_outputs, sigmet_outputs = {}, {}
    cursor = start
    while cursor < end:
        following = min(end, (cursor.replace(day=28) + timedelta(days=4)).replace(day=1))
        lower, upper = cursor.isoformat() + "T00:00Z", following.isoformat() + "T00:00Z"
        period = f"{cursor:%Y%m%d}-{following:%Y%m%d}"
        for index in range(0, len(stations), 20):
            batch = stations[index:index + 20]
            digest = hash_text(",".join(batch).encode()).hexdigest()[:12]
            params = {"pil": ",".join("TAF" + site[1:] for site in batch), "fmt": "zip",
                      "sdate": lower, "edate": upper, "limit": 9999}
            path, checksum = acquire(project_path(root, taf["bronze_directory"]), f"{period}-{digest}.zip",
                                     TAF_URL + "?" + urlencode(params), clean_taf.read_archive)
            taf_outputs[path] = checksum
        params = {"sts": lower, "ets": upper, "format": "kml"}
        path, checksum = acquire(project_path(root, sigmets["bronze_directory"]), f"{period}.kml",
                                 SIGMET_URL + "?" + urlencode(params), clean_sigmets.read_archive)
        sigmet_outputs[path] = checksum
        cursor = following
    return taf_outputs, sigmet_outputs
