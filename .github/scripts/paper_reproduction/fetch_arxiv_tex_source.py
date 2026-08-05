"""Fetch an arXiv paper's TeX source tarball and extract it to REPRO_DIR/tex_src/.

Table numbers and macro definitions are read more accurately from the TeX source than
from PDF text extraction, so the generation agent is told to prefer tex_src/ when present
(see run_paper_reproduction.md). This is a no-op for non-arXiv URLs, or when arXiv has no
TeX source for this paper (PDF-only submission) — in both cases the agent falls back to
fetching the paper itself.

Usage:
  python3 fetch_arxiv_tex_source.py --paper-url <url> --repro-dir .reproduction/<repro_id>
"""

from __future__ import annotations

import argparse
import re
import tarfile
import urllib.request
from io import BytesIO
from pathlib import Path

_ARXIV_ID_RE = re.compile(r"arxiv\.org/(?:abs|pdf|src)/([0-9v.]+)")
_REQUEST_TIMEOUT_SECONDS = 120


def parse_arxiv_id(url: str) -> str | None:
    m = _ARXIV_ID_RE.search(url)
    return m.group(1) if m else None


def _extract_tex_files(tar: tarfile.TarFile, tex_dir: Path) -> int:
    extracted = 0
    for member in tar.getmembers():
        if not member.isfile() or not member.name.lower().endswith(".tex"):
            continue
        f = tar.extractfile(member)
        if f is None:
            continue
        # arXiv tarballs sometimes wrap everything in a single top-level directory;
        # strip it so tex_src/ holds the sources directly.
        parts = Path(member.name).parts
        rel = Path(*parts[1:]) if len(parts) > 1 else Path(parts[0])
        dest = tex_dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(f.read())
        extracted += 1
    return extracted


def fetch_tex_source(arxiv_id: str, repro_dir: Path) -> None:
    src_url = f"https://arxiv.org/src/{arxiv_id}"
    with urllib.request.urlopen(src_url, timeout=_REQUEST_TIMEOUT_SECONDS) as resp:
        data = resp.read()

    if data.startswith(b"%PDF"):
        print(f"arXiv {arxiv_id} has no TeX source (PDF-only submission); skipping.")
        return

    tex_dir = repro_dir / "tex_src"
    with tarfile.open(fileobj=BytesIO(data)) as tar:
        extracted = _extract_tex_files(tar, tex_dir)
    if extracted:
        print(f"Extracted {extracted} .tex file(s) to {tex_dir}")
    else:
        print(f"No .tex files found in the arXiv source tarball for {arxiv_id}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper-url", required=True)
    parser.add_argument("--repro-dir", required=True, type=Path)
    args = parser.parse_args()

    arxiv_id = parse_arxiv_id(args.paper_url)
    if not arxiv_id:
        print("Not an arXiv URL; skipping TeX source fetch.")
        return

    try:
        fetch_tex_source(arxiv_id, args.repro_dir)
    except Exception as exc:
        print(f"::warning::Failed to fetch arXiv TeX source for {arxiv_id}: {exc}")


if __name__ == "__main__":
    main()
