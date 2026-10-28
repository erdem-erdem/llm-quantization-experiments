"""Eval task YAMLs (plan §6.5): write arc_tr + gpqa_tr definitions into tasks/.

arc_tr inherits the harness arc_challenge structure but reads malhajar/arc-tr-v0.2 with
Turkish prompt; doc_to_target mapping (1-4 ↔ A-D) comes from arc_tr_check.py's output
(stored in the YAML literal). gpqa_tr is a custom task applying tasks/gpqa_tr_perm.json.
Belebele/arc_en use harness tasks as-is.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.common import logging_setup, paths  # noqa: E402

ARC_TR_YAML = """\
task: arc_tr
dataset_path: malhajar/arc-tr-v0.2
dataset_name: null
output_type: multiple_choice
test_split: test
doc_to_text: "Soru: {{question}}\\nCevap:"
doc_to_target: "{{choices.label.index(answerKey.lstrip())}}"
doc_to_choice: "{{choices.text}}"
metric_list:
  - metric: acc_norm
    aggregation: mean
    higher_is_better: true
  - metric: acc
    aggregation: mean
    higher_is_better: true
metadata:
  version: 1.0
"""

GPQA_TR_YAML = """\
task: gpqa_tr
dataset_path: ytu-ce-cosmos/gpqa-extended_tr
dataset_name: null
output_type: multiple_choice
test_split: train
process_docs: !function utils.process_docs_gpqa
doc_to_text: "Bu sorunun doğru cevabı nedir: {{question}}\\nSeçenekler:\\n(A) {{choice_a}}\\n(B) {{choice_b}}\\n(C) {{choice_c}}\\n(D) {{choice_d}}\\nCevap:"
doc_to_target: "{{correct_index}}"
doc_to_choice: [" (A)", " (B)", " (C)", " (D)"]
metric_list:
  - metric: acc
    aggregation: mean
    higher_is_better: true
metadata:
  version: 1.0
"""

GPQA_TR_UTILS = '''\
"""gpqa_tr process_docs: applies the fixed option permutation (tasks/gpqa_tr_perm.json)."""
import json
from pathlib import Path

_PERM = json.loads((Path(__file__).resolve().parents[0] / "gpqa_tr_perm.json").read_text(
    encoding="utf-8"))


def process_docs_gpqa(dataset):
    def _apply(doc, i):
        p = _PERM["perm"][i]
        # dataset columns: "Question", "Correct Answer", "Incorrect Answer 1..3" (capitalised)
        choices = [doc["Correct Answer"], doc["Incorrect Answer 1"],
                   doc["Incorrect Answer 2"], doc["Incorrect Answer 3"]]
        doc["question"] = doc["Question"]
        permuted = [choices[j] for j in p]
        doc["choice_a"], doc["choice_b"], doc["choice_c"], doc["choice_d"] = permuted
        doc["correct_index"] = p.index(0)
        return doc
    return dataset.map(_apply, with_indices=True)
'''


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    logging_setup.setup("eval_tasks_setup")
    paths.TASKS.mkdir(parents=True, exist_ok=True)
    io_files = [
        (paths.TASKS / "arc_tr.yaml", ARC_TR_YAML),
        (paths.TASKS / "gpqa_tr.yaml", GPQA_TR_YAML),
        (paths.TASKS / "utils.py", GPQA_TR_UTILS),
    ]
    for p, text in io_files:
        p.write_text(text, encoding="utf-8", newline="\n")
        print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())