"""Sum logs/gpu_budget.csv; >= 14 GPU-h triggers the §12 cut-list into DECISIONS_NEEDED.md."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.common import io, logging_setup, paths  # noqa: E402

CUT_TRIGGER_GPU_H = 14.0
HARD_CAP_GPU_H = 16.0


def total_gpu_minutes() -> tuple[float, int]:
    p = paths.budget_csv()
    if not p.exists():
        return 0.0, 0
    rows = [r for r in csv_rows(p)]
    return sum(float(r["gpu_min"]) for r in rows), len(rows)


def csv_rows(p: Path) -> list[dict[str, str]]:
    import csv
    with open(p, encoding="utf-8", newline="\n") as f:
        return list(csv.DictReader(f))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    logging_setup.setup("budget_check")
    minutes, n = total_gpu_minutes()
    hours = minutes / 60.0
    msg = f"gpu_budget.csv: {n} jobs, total {hours:.2f} GPU-h (cut trigger {CUT_TRIGGER_GPU_H}, cap {HARD_CAP_GPU_H})"
    print(msg)
    if hours >= CUT_TRIGGER_GPU_H:
        note = paths.ROOT / "DECISIONS_NEEDED.md"
        entry = (f"\n- [budget] {msg} — apply the spec §11/§12 cut-list item by item; "
                 f"ask the user before each cut.\n")
        with open(note, "a", encoding="utf-8", newline="\n") as f:
            f.write(entry)
        if not args.quiet:
            print("Cut trigger reached; entry appended to DECISIONS_NEEDED.md", file=sys.stderr)
        if hours >= HARD_CAP_GPU_H:  # the cap is above the trigger: report the stronger state
            print("HARD CAP EXCEEDED", file=sys.stderr)
            return 3
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())