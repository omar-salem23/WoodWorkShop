#!/usr/bin/env python3
"""Resumably download a public Google Drive folder into a dataset directory."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

_local = threading.local()


def session() -> requests.Session:
    if not hasattr(_local, "session"):
        _local.session = requests.Session()
    return _local.session


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder_url", help="Public Google Drive folder URL")
    parser.add_argument("--manifest", type=Path, help="Use a saved gdown --json listing")
    parser.add_argument("--output", type=Path, required=True, help="Output dataset directory")
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    if args.workers < 1 or args.workers > 16:
        parser.error("--workers must be between 1 and 16")

    if args.manifest:
        entries = json.loads(args.manifest.read_text(encoding="utf-8"))
    else:
        try:
            import gdown  # noqa: F401
        except ImportError:
            parser.error("Install project requirements first; gdown is required to list Drive folders")
        listing = subprocess.run(
            [sys.executable, "-m", "gdown", "--json", "--folder", args.folder_url],
            check=True,
            capture_output=True,
            text=True,
        )
        entries = json.loads(listing.stdout)
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    jobs: list[tuple[str, Path]] = []

    for entry in entries:
        relative = Path(entry["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Unsafe path in Drive listing: {relative}")
        if relative.suffix.lower() == ".cache":
            continue
        destination = (root / relative).resolve()
        if root not in destination.parents:
            raise ValueError(f"Drive path escapes destination: {relative}")
        if not destination.is_file() or destination.stat().st_size == 0:
            jobs.append((entry["url"], destination))

    print(f"Files to download: {len(jobs)}; existing files are kept.", flush=True)
    failures: list[tuple[str, str]] = []
    completed = 0
    lock = threading.Lock()

    def fetch(job: tuple[str, Path]) -> tuple[str, str] | None:
        url, destination = job
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + ".partial")
        download_url = url + ("&" if "?" in url else "?") + "export=download"
        for attempt in range(4):
            retry_delay = None
            try:
                with session().get(download_url, stream=True, timeout=(20, 90)) as response:
                    if response.status_code in (429, 503):
                        try:
                            retry_delay = min(120, int(response.headers.get("Retry-After", "0")))
                        except ValueError:
                            retry_delay = None
                    response.raise_for_status()
                    content_type = response.headers.get("Content-Type", "").lower()
                    if "text/html" in content_type:
                        raise ValueError("Google Drive returned an HTML page instead of the file")
                    with temporary.open("wb") as output:
                        for chunk in response.iter_content(1024 * 1024):
                            if chunk:
                                output.write(chunk)
                if not temporary.is_file() or temporary.stat().st_size == 0:
                    raise ValueError("empty response")
                temporary.replace(destination)
                return None
            except Exception as exc:
                temporary.unlink(missing_ok=True)
                if attempt == 3:
                    return (str(destination), f"{type(exc).__name__}: {exc}")
                time.sleep(retry_delay or min(60, 5 * 2**attempt))
        return None

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(fetch, job) for job in jobs]
        for future in as_completed(futures):
            error = future.result()
            with lock:
                completed += 1
                if error:
                    failures.append(error)
                if completed % 250 == 0 or completed == len(jobs):
                    print(f"Completed {completed}/{len(jobs)}; failures={len(failures)}", flush=True)

    if failures:
        report = root / "download_failures.json"
        report.write_text(json.dumps(failures, indent=2), encoding="utf-8")
        print(f"{len(failures)} files failed; details saved to {report}")
        return 1
    print(f"Download complete: {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
