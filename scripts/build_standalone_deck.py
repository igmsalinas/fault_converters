#!/usr/bin/env python3
"""Inline every referenced image into a single self-contained HTML deck.

Reads the CMMSE presentation, replaces each local <img src="..."> path with a
base64 data URI, and writes a standalone HTML file that can be shared as one
file (no external image dependencies). Fonts fall back to Helvetica/Arial where
the commercial display face is not installed.
"""

from __future__ import annotations

import base64
import mimetypes
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
DECK = HERE.parent / "docs" / "presentations" / "CMMSE_2026_presentation.html"
OUT = DECK.with_name("CMMSE_2026_presentation_standalone.html")


def data_uri(path: Path) -> str:
    mime, _ = mimetypes.guess_type(path.name)
    if mime == "image/svg+xml" or path.suffix.lower() == ".svg":
        mime = "image/svg+xml"
    elif mime is None:
        mime = "application/octet-stream"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def main() -> None:
    html = DECK.read_text(encoding="utf-8")
    base = DECK.parent

    seen: dict[str, str] = {}
    missing: list[str] = []

    def replace(match: re.Match[str]) -> str:
        prefix, src, suffix = match.group(1), match.group(2), match.group(3)
        if src.startswith(("data:", "http://", "https://")):
            return match.group(0)
        if src not in seen:
            asset = (base / src).resolve()
            if not asset.is_file():
                missing.append(src)
                seen[src] = src  # leave untouched
            else:
                seen[src] = data_uri(asset)
        return f"{prefix}{seen[src]}{suffix}"

    new_html = re.sub(r'(src=")([^"]+)(")', replace, html)
    OUT.write_text(new_html, encoding="utf-8")

    size_mb = OUT.stat().st_size / 1_048_576
    inlined = sum(1 for v in seen.values() if v.startswith("data:"))
    print(f"Wrote {OUT.relative_to(HERE.parent)} ({size_mb:.2f} MB)")
    print(f"Inlined {inlined} unique images.")
    if missing:
        print("WARNING - could not find:")
        for m in sorted(set(missing)):
            print(f"  {m}")


if __name__ == "__main__":
    main()
