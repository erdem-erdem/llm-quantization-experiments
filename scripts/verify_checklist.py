"""Spec §12 checklist (plan §10): 10 checks, exit != 0 on failure.

Run after the full remote run (`python scripts/verify_checklist.py`). Missing artifacts
fail loudly. Item 10's "budget last row <= 5 min old" freshness sub-check only applies
with --finalize (later re-runs would otherwise always fail it).

GPQA leak scan (item 9) never prints question text: it only searches report/figure/result
markdown for field names that must never leave the dataset ("question_text",
"correct_answer") — the permutation file stores option indices only, never text.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.common import io, logging_setup, paths  # noqa: E402

TIED = {"q06", "q17", "q4"}
# Field names only — the scan never reads, prints or stores question content. Covers the raw
# GPQA columns ("Incorrect Answer 1" …) and the choice_a… fields the task's process_docs builds.
LEAK_PATTERNS = ["question_text", "correct_answer", "incorrect answer",
                 "choice_a", "choice_b", "choice_c", "choice_d"]
BUDGET_GPU_H_MAX = 16.0


def _json(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


def _read_csv(p: Path) -> list[dict]:
    with open(p, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def c1_tied_embeddings() -> tuple[bool, str]:
    p = paths.RESULTS / "model_cards.csv"
    if not p.exists():
        return False, f"{p} yok"
    rows = {r["model"]: r for r in _read_csv(p)}
    # download_models writes the config field as column "tie"
    got_tied = {m for m, r in rows.items()
                if str(r.get("tie", r.get("tie_word_embeddings", ""))).lower() == "true"}
    if got_tied != TIED:
        return False, f"tied set {sorted(got_tied)} != {sorted(TIED)}"
    dumps = {d.name.split("__")[0]: d for d in
             (paths.RESULTS / "tensor_types").glob("*.csv")} if \
        (paths.RESULTS / "tensor_types").exists() else {}
    for m in sorted(TIED):
        if m not in dumps:
            return False, f"{m}: tensor dump yok"
        names = {r["name"] for r in _read_csv(dumps[m])}
        if "output.weight" in names:
            return False, f"{m}: tied ama dump'ta output.weight var"
    untied = [m for m in rows if m not in TIED and m in dumps]
    if not untied or any("output.weight" not in
                         {r["name"] for r in _read_csv(dumps[m])} for m in untied):
        return False, f"untied modellerde output.weight eksik: {untied}"
    return True, f"tied={sorted(TIED)}, dumps ok"


def c2_minhash_in_manifest() -> tuple[bool, str]:
    p = paths.DATA / "manifest_calib.json"
    if not p.exists():
        return False, f"{p} yok"
    tr = [e for e in _json(p)["entries"] if e.get("lang") == "tr"]
    if not tr or any("minhash_dropped" not in e for e in tr):
        return False, "manifest_calib.json tr girdilerinde minhash_dropped yok"
    return True, f"tr minhash_dropped: {[e['minhash_dropped'] for e in tr]}"


def c3_calib_sha() -> tuple[bool, str]:
    p = paths.DATA / "manifest_calib.json"
    if not p.exists():
        return False, f"{p} yok"
    man = _json(p)
    bad = [name for name, sha in man.get("hashes", {}).items()
           if not (paths.CALIB / name).exists()
           or hashlib.sha256((paths.CALIB / name).read_bytes()).hexdigest() != sha]
    if bad:
        return False, f"sha256 uyuşmayan kalibrasyon dosyaları: {bad}"
    metas = list(paths.ROOT.glob("results/*/run_meta/*.json"))
    gptq = [m for m in metas if '"job": "llmcompress"' in m.read_text(encoding="utf-8")]
    imat = [m for m in metas if "imatrix" in m.name or
            "gguf/imatrix" in m.read_text(encoding="utf-8")]
    if not gptq or not imat:
        return False, f"gptq run_meta {len(gptq)} adet, imatrix run_meta {len(imat)} adet"
    return True, f"kalibrasyon sha'ları taze; gptq={len(gptq)} imatrix={len(imat)} run_meta"


def c4_gpqa_perm() -> tuple[bool, str]:
    p = paths.TASKS / "gpqa_tr_perm.json"
    if not p.exists():
        return False, f"{p} yok"
    perm = _json(p)
    # the embedded sha256 covers the body without the sha256 field (gpqa_perm.py, test_gpqa_perm)
    body = json.dumps({"seed_base": perm.get("seed_base"), "n": perm.get("n"),
                       "perm": perm.get("perm")}, ensure_ascii=False) + "\n"
    sha = hashlib.sha256(body.encode("utf-8")).hexdigest()
    if perm.get("sha256") != sha:
        return False, "perm dosyasının kendi sha256'sı uyuşmuyor"
    if perm.get("n") != 546 or perm.get("seed_base") != 1234:
        return False, f"n={perm.get('n')} seed_base={perm.get('seed_base')}"
    tasks = (paths.CONFIGS / "tasks.yaml").read_text(encoding="utf-8")
    if "arc_tr" not in tasks:
        return False, "tasks.yaml'de arc_tr eşlemesi yok"
    return True, "perm sha + n=546 + arc_tr mapping ok"


def c5_eval_and_generation_logging() -> tuple[bool, str]:
    logs = list(paths.JOBS.glob("*.log")) if paths.JOBS.exists() else []
    eval_logs = [l for l in logs if "__eval__" in l.name]
    if not eval_logs:
        return False, "hiç eval iş günlüğü yok"
    bad = [l.name for l in eval_logs
           if "apply_chat_template=False" not in l.read_text(encoding="utf-8",
                                                             errors="replace")]
    if bad:
        return False, f"apply_chat_template=False eksik: {bad[:3]}"
    gen_ok = any("enable_thinking" in l.read_text(encoding="utf-8", errors="replace")
                 for l in logs)
    return True, f"{len(eval_logs)} eval logu; generation thinking-kapalı kaydı: {gen_ok}"


def c6_dequant_bit_exact() -> tuple[bool, str]:
    m = paths.done_marker("E1", "dequant", "q8__q4_k_m__tr__s0")
    if not m.exists():
        return False, f"{m} yok (dequant_validate bit-exact validator bu işe bağlı)"
    return True, "q8__q4_k_m__tr__s0 dequant .done (validator bit-exact testi içerir)"


def c7_e4_exclusivity_dmon() -> tuple[bool, str]:
    raw = paths.RESULTS / "E4" / "raw"
    if not raw.exists():
        return False, f"{raw} yok"
    metas = list((paths.RESULTS / "E4" / "run_meta").glob("*.json")) \
        if (paths.RESULTS / "E4" / "run_meta").exists() else []
    speed = [m for m in metas if '"job": "speed"' in m.read_text(encoding="utf-8")]
    if not speed:
        return False, "E4 speed run_meta yok"
    missing = []
    for m in speed:
        meta = _json(m)
        # bench.py logs dmon under raw/<cell run_id>, e.g. <key> or serve__<engine>__c<N>
        cells = [d for d in raw.glob(f"{meta['run_key']}*") if (d / "dmon.log").exists()]
        if not cells:
            missing.append(meta["run_key"])
    if missing:
        return False, f"dmon.log eksik: {missing[:5]}"
    return True, f"{len(speed)} speed hücresinde dmon.log var (exclusivity record_run'da)"


def c8_e6_fa_and_needle() -> tuple[bool, str]:
    src = (paths.ROOT / "scripts" / "e6" / "server.py").read_text(encoding="utf-8")
    if '"-fa"' not in src:
        return False, "scripts/e6/server.py komutunda -fa yok"
    kv = paths.RESULTS / "E6" / "kv.json"
    needle = paths.RESULTS / "E6" / "needle.json"
    if not kv.exists() or not needle.exists():
        return False, f"{kv} / {needle} yok"
    n = _json(needle)
    ctxs = {int(k.split(",")[0]) for d in n.values() for k in d}
    if max(ctxs, default=0) > 32768:
        return False, f"needle ctx > 32768: {sorted(ctxs)}"
    return True, f"-fa var; needle ctx'ler {sorted(ctxs)}"


def c9_gpqa_leak_scan() -> tuple[bool, str]:
    targets: list[Path] = []
    # logs/*.log is packed too (logs/jobs is not): lm-eval must not have echoed documents there
    for pat in ("reports/*.md", "reports/**/*.md", "figures/*.md", "results/**/*.md",
                "logs/*.log"):
        targets += list(paths.ROOT.glob(pat))
    hits = []
    for p in targets:
        text = p.read_text(encoding="utf-8", errors="replace").lower()
        for pat in LEAK_PATTERNS:
            if pat in text:
                hits.append(f"{p.name}:{pat}")
    if hits:
        return False, f"olası GPQA sızıntısı (alan adı): {hits}"
    return True, f"{len(targets)} markdown/log dosyası temiz"


def c10_figures_and_budget(finalize: bool) -> tuple[bool, str]:
    problems = []  # figures are not generated by this repo (they ship pre-rendered); budget only
    bud = paths.budget_csv()
    if not bud.exists():
        problems.append("gpu_budget.csv yok")
    else:
        rows = _read_csv(bud)
        total = sum(float(r["gpu_min"]) for r in rows if r.get("gpu_min"))
        if total > 60 * BUDGET_GPU_H_MAX:
            problems.append(f"gpu_budget toplam {total / 60:.1f} GPU-sa > {BUDGET_GPU_H_MAX}")
        if finalize and rows:
            last = datetime.fromisoformat(rows[-1]["ts"]).replace(tzinfo=timezone.utc)
            age = (datetime.now(timezone.utc) - last).total_seconds()
            if age > 300:
                problems.append(f"son budget satırı {age / 60:.0f} dk önce (--finalize)")
    if problems:
        return False, "; ".join(problems)
    return True, "bütçe tamam"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--finalize", action="store_true",
                    help="budget freshness (<=5 min) da denetlensin")
    args = ap.parse_args()
    logging_setup.setup("verify_checklist")

    checks = [
        ("1 tied embeddings", c1_tied_embeddings),
        ("2 minhash drop", c2_minhash_in_manifest),
        ("3 calib sha256", c3_calib_sha),
        ("4 gpqa perm", c4_gpqa_perm),
        ("5 eval/generation logging", c5_eval_and_generation_logging),
        ("6 dequant bit-exact", c6_dequant_bit_exact),
        ("7 E4 dmon", c7_e4_exclusivity_dmon),
        ("8 E6 -fa + needle", c8_e6_fa_and_needle),
        ("9 GPQA leak scan", c9_gpqa_leak_scan),
        ("10 budget", lambda: c10_figures_and_budget(args.finalize)),
    ]
    n_fail = 0
    for name, fn in checks:
        try:
            ok, detail = fn()
        except Exception as exc:  # noqa: BLE001 — a crashing check is a failing check
            ok, detail = False, f"check raised: {exc!r}"
        print(f"[{'PASS' if ok else 'FAIL'}] {name:26s} {detail}")
        n_fail += 0 if ok else 1
    print(f"\nverify_checklist: {10 - n_fail}/10 PASS")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())