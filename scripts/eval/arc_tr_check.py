"""ARC-tr schema check (spec §5.3): the arc_tr task YAML maps the gold answer with
`doc_to_target: "{{choices.label.index(answerKey.lstrip())}}"`. A wrong mapping (label format
1-4 vs A-D, stray whitespace, answerKey missing from the labels) would still produce a valid
n=1172 task and silently score garbage, so verify it against the real dataset first.

Checks on malhajar/arc-tr-v0.2 `test`: row count == 1172, columns question/choices/answerKey,
choices has parallel text/label lists, every answerKey.lstrip() is one of its own labels
(i.e. the YAML expression yields a valid index for every row), and the question/choice text
is non-empty. Exit code 1 on any violation. Prints counts and label formats only.
"""
from __future__ import annotations

import argparse
import collections
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.common import logging_setup  # noqa: E402

DATASET = "malhajar/arc-tr-v0.2"
EXPECTED_ROWS = 1172


def check_rows(rows: list[dict]) -> list[str]:
    """Problems found in the rows (empty list = mapping is sound)."""
    problems: list[str] = []
    if len(rows) != EXPECTED_ROWS:
        problems.append(f"row count {len(rows)} != {EXPECTED_ROWS}")
    missing_key = collections.Counter()
    bad_target = 0
    empty_text = 0
    for r in rows:
        for col in ("question", "choices", "answerKey"):
            if col not in r:
                missing_key[col] += 1
        if missing_key:
            continue
        ch = r["choices"]
        labels, texts = list(ch.get("label", [])), list(ch.get("text", []))
        if not labels or len(labels) != len(texts):
            problems.append("choices.label / choices.text lengths differ or are empty")
            break
        if str(r["answerKey"]).lstrip() not in labels:
            bad_target += 1  # the YAML's .index(...) would raise or pick the wrong item
        if not str(r["question"]).strip() or not all(str(t).strip() for t in texts):
            empty_text += 1
    for col, n in missing_key.items():
        problems.append(f"column '{col}' missing in {n} rows")
    if bad_target:
        problems.append(f"answerKey.lstrip() not in choices.label for {bad_target} rows")
    if empty_text:
        problems.append(f"{empty_text} rows with empty question/choice text")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    args = ap.parse_args()
    logging_setup.setup("arc_tr_check")

    from datasets import load_dataset

    ds = load_dataset(DATASET, split=args.split)
    rows = [dict(r) for r in ds]
    problems = check_rows(rows)
    formats = collections.Counter(
        "".join(r["choices"]["label"]) for r in rows if "choices" in r)
    keys = collections.Counter(str(r.get("answerKey")) for r in rows)
    print(f"{DATASET}[{args.split}]: {len(rows)} rows, columns {sorted(ds.column_names)}")
    print(f"label formats: {dict(formats.most_common(5))}; answerKey values: {dict(keys)}")
    if problems:
        for p in problems:
            print(f"FAIL: {p}", file=sys.stderr)
        return 1
    print("OK: every answerKey maps to a valid choice index via the task YAML expression")
    return 0


if __name__ == "__main__":
    sys.exit(main())
