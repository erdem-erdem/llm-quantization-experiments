"""model_cards.csv merge: concurrent per-model dl: jobs must not lose rows."""
from __future__ import annotations

import sys
import threading
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.common import ids  # noqa: E402
from scripts.t0.download_models import merge_cards  # noqa: E402


def test_concurrent_merges_keep_all_rows(tmp_path):
    out = tmp_path / "model_cards.csv"
    errors: list[Exception] = []

    def run(m):
        try:
            merge_cards(out, [{"model": m, "tie": m in ids.TIED}], [m])
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=run, args=(m,)) for m in ids.MODELS]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert not errors
    df = pd.read_csv(out)
    assert list(df["model"]) == list(ids.MODELS)


def test_merge_replaces_own_model_row(tmp_path):
    out = tmp_path / "model_cards.csv"
    merge_cards(out, [{"model": "q06", "tie": False}], ["q06"])
    merge_cards(out, [{"model": "q06", "tie": True}], ["q06"])
    df = pd.read_csv(out)
    assert len(df) == 1 and bool(df.loc[0, "tie"]) is True
