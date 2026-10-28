"""Source files must stay valid UTF-8 without double-encoding (UTF-8 bytes re-read as cp1252
and saved again turn 'ü' into 'Ã¼' and '→' into 'â†’' — it silently corrupts Turkish figure
labels and docstrings)."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# built from code points so this file does not contain the patterns itself
MOJIBAKE = ("Ã", "â€", "â†", "Â§")


def _double_encoded(text: str) -> bool:
    """True when some line is UTF-8 that was re-read as cp1252 (round-trips to different text)."""
    for line in text.lstrip("﻿").splitlines():  # a leading BOM would make encode() fail
        if any(ord(c) > 127 for c in line):
            try:
                if line.encode("cp1252").decode("utf-8") != line:
                    return True
            except (UnicodeEncodeError, UnicodeDecodeError):
                continue
    return False


def test_no_double_encoded_text_in_scripts():
    bad = []
    for p in sorted((ROOT / "scripts").rglob("*.py")) + sorted((ROOT / "scripts").rglob("*.sh")):
        text = p.read_text(encoding="utf-8")
        if any(m in text for m in MOJIBAKE) or _double_encoded(text):
            bad.append(str(p.relative_to(ROOT)))
    assert not bad, f"double-encoded (mojibake) text in: {bad}"
