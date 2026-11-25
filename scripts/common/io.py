"""Shared file hashing, atomic JSON writing, fingerprinting, and HTTP downloads."""

from collections.abc import Iterable
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Union
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from tqdm import tqdm

from .results import replace_file


def sha256(path: Union[str, Path]) -> str:
    """Calculate the hex-encoded SHA-256 digest of a file on disk."""
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def reference_matches(path: Union[str, Path], checksum: str) -> bool:
    """Verify file content against checksum, tolerating CRLF vs LF differences."""
    content = Path(path).read_bytes()
    lf = content.replace(b"\r\n", b"\n")
    return any(
        hashlib.sha256(value).hexdigest() == checksum
        for value in (content, lf, lf.replace(b"\n", b"\r\n"))
    )


def write_json(path: Union[str, Path], value: Any) -> None:
    """Write a JSON-serializable object atomically using a temporary .part file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    replace_file(temporary, path)


def file_fingerprints(paths: Iterable[Path]) -> list[dict[str, Any]]:
    """Capture file sizes and modification timestamps without reading full contents."""
    return [
        {
            "path": str(path.resolve()),
            "bytes": path.stat().st_size,
            "modified_ns": path.stat().st_mtime_ns,
        }
        for path in sorted(paths)
    ]


def fetch(
    url: str,
    path: Union[str, Path],
    user_agent: str = "CS523-Flight-Delay/1.0",
) -> Path:
    """Download a URL to a staged .part file with retries and progress tracking."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    for attempt in range(3):
        try:
            request = Request(url, headers={"User-Agent": user_agent})
            with (
                urlopen(request, timeout=90) as response,
                temporary.open("wb") as stream,
            ):
                expected = response.headers.get("Content-Length")
                with tqdm.wrapattr(
                    stream,
                    "write",
                    total=int(expected) if expected else None,
                    desc=path.name,
                    unit="B",
                    unit_scale=True,
                    miniters=1,
                    leave=False,
                ) as progress:
                    while block := response.read(1024 * 1024):
                        progress.write(block)
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
            time.sleep(2 ** attempt)
    raise RuntimeError(f"Download failed unexpectedly: {url}")
