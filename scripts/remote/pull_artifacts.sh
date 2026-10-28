#!/usr/bin/env bash
# Pack results for the return trip (RUNBOOK step 7): everything needed for analysis,
# nothing that needs the models. Creates quant_artifacts_<date>.tar.gz.
#
# Included: results/*/samples (per-item correctness vectors: Δ/flips/bootstrap are recomputed
# from them) and results/*/run_meta (verify_checklist c3). Excluded: results/*/raw (lm-eval
# logs, may hold benchmark text — GPQA must never leave the box), models, refs, tmp_deq, .kld.
set -euo pipefail
cd "$(dirname "$0")/../.."

out="quant_artifacts_$(date +%Y%m%d).tar.gz"
want=(results figures reports logs NOTES.md DECISIONS_NEEDED.md
      data/manifest.json data/manifest_calib.json data/LICENSES.md
      env/requirements.lock configs/cli_overrides.json)
have=()
for p in "${want[@]}"; do
  if [[ -e "$p" ]]; then have+=("$p"); else echo "skip (missing): $p"; fi
done

tar -czf "$out" \
  --exclude='models' --exclude='refs' --exclude='tmp_deq' \
  --exclude='*.kld' --exclude='results/*/raw' --exclude='logs/jobs' \
  "${have[@]}"

echo "packed: $out ($(du -h "$out" | cut -f1))"
