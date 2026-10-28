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
