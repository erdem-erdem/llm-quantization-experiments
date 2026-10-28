# RUNBOOK — running the experiment queue on a 2× H100 (80 GB) Ubuntu 22.04 machine

This guide takes a fresh Ubuntu 22.04 machine with two H100 80 GB GPUs to a finished experiment
queue. All commands are run from the repository root. Budget: about 6.5–7 GPU-hours of jobs, which is
roughly 3.5–4 hours of wall-clock time on 2 GPUs. The first real measurement lands in
`logs/gpu_budget.csv`; estimates should be updated from it.

**2-GPU plan** (`configs/experiments.yaml` → `gpus`, `gpu_plan`): GPU 0 = the q32 chain + E4-32B + demo +
E6 KV/needle; GPU 1 = q14, q8, q4, q17, q06, E0, E2, E5, E6 lm_head, E4-8B.
Cuts relative to the full plan: no accuracy eval for q32 `q8_0`/`q3_k_m`; Belebele-en only on q4/q8/q32;
E2 = q17 + q4 with the second seed (s1) only on q4; no concurrency 64 in E4; no E5 stage 3 (8B transfer).
The needle trials (5) and the KL text (64K tokens) are unchanged.

## 1. Transfer

```bash
rsync -av --exclude models --exclude refs --exclude data/calib \
    ./ user@host:~/llm-quantization-experiments/
# or a tarball:
# tar --exclude='models' --exclude='refs' --exclude='data/calib' -czf repo.tgz .
```

## 2. Setup

```bash
cd ~/llm-quantization-experiments
bash scripts/remote/setup_remote.sh      # apt + venv + requirements + llama.cpp build
source .venv/bin/activate
export HF_TOKEN=hf_...                   # the token is only ever read from the environment
```

`build_llamacpp.sh` builds the commit in `env/llamacpp_commit.txt` under `llama.cpp/` with CUDA (sm_90)
and installs gguf-py from the same checkout. If access to a gated dataset fails, stop and fix the
token before going on.

## 3. Environment check

```bash
python scripts/check_env.py
python -m pytest -q tests/test_scheduler_gpu.py   # on Linux this also runs the "orphan child process dies" test (expect no skip)
```

It verifies every CLI flag against `--help`; a missing flag is recorded in `configs/cli_overrides.json`.
Fix any FAIL line before continuing.

## 4. Compatibility smoke test

```bash
python scripts/t0/download_models.py --models q06
python scripts/eval/tasks_setup.py
python scripts/eval/arc_tr_check.py    # ARC-tr: 1172 rows + answerKey→index mapping (expect OK)
lm_eval --model vllm --model_args pretrained=models/hf/q06,dtype=bfloat16,gpu_memory_utilization=0.9,max_model_len=4096 \
    --tasks belebele_tur_Latn --num_fewshot 0 --limit 20 --include_path tasks --output_path results/smoke/raw/compat
```

Also check that an FP8 checkpoint loads in vLLM (llm-compressor ↔ vLLM compatibility):

```bash
python scripts/q/llmcompress.py --run-key q06__fp8__none__s0
python scripts/eval/lm_eval_run.py --run-key q06__fp8__none__s0 --exp smoke --engine vllm     --tasks belebele_tr --limit 20
```

A failure means a vLLM / lm-eval version mismatch. If it succeeds, freeze the environment:

```bash
pip freeze > env/requirements.lock
```

The llama.cpp commit is pinned in `env/llamacpp_commit.txt`. Changing versions after this step means
re-running E0.

## 5. Data and model preparation (T0)

```bash
python scripts/t0/download_models.py        # 6 models, writes model_cards.csv
python scripts/t0/havadis_eval.py           # data/havadis_eval.txt (>=100k tokens)
python scripts/t0/calib_sets.py             # data/calib/* (tr,en × s0,s1 + mix) + manifest_calib.json
python scripts/t0/own_texts.py              # demo texts, needles, haystacks
python scripts/t0/gpqa_perm.py              # tasks/gpqa_tr_perm.json (question text is NEVER printed)
python scripts/t0/manifest.py               # data/manifest.json + data/LICENSES.md
python scripts/t0/gguf_convert.py --all     # CPU; in the background: nohup ... &
```

No third-party text is stored in the repository: the calibration sets and the Havadis evaluation text are
rebuilt from Hugging Face with fixed seeds. `data/manifest_calib.json` holds their SHA-256 hashes so you
can check that your regenerated files match.

When the GGUF conversion is done, run the **dequantization identity test**. It is the only proof that the
GGUF→HF name/shape mapping is right: it dequantizes the q06 BF16 GGUF and compares it bit by bit with the
original safetensors (takes minutes):

```bash
python scripts/q/dequant.py --run-key q06__bf16__none__s0 --gguf models/gguf/q06-bf16.gguf
python scripts/q/dequant_validate.py --identity q06
python scripts/q/dequant_validate.py --cleanup q06__bf16__none__s0
```

If it fails, do not start the queue: every GGUF accuracy result depends on this mapping.

Do not start the queue before the GGUF conversion has finished (the DAG depends on the `dl:` and `gguf:`
jobs, but if the BF16 GGUFs are not ready the first quantization jobs fail as soon as they start).

## 6. The queue

Before the first full queue, a cheap check of the GPU path is recommended (about 2 hours). It runs the
produce → dequantize → vLLM eval chain on a single model and also exercises `record_run`, the `.done`
markers and the scheduler:

```bash
python scripts/queue_gen.py --profile smoke --out configs/queue.smoke.json
python scripts/scheduler.py --queue configs/queue.smoke.json
```

```bash
python scripts/queue_gen.py --profile full   # configs/queue.full.json (the DAG is validated)
python scripts/scheduler.py --queue configs/queue.full.json --dry-run   # order + per-GPU timeline
tmux new -s run 'python scripts/scheduler.py --queue configs/queue.full.json'
# monitoring:
tail -f logs/sched.log
python scripts/budget_check.py               # GPU-minute table (upper bound 16 GPU-h)
```

Scheduler: one worker thread per GPU (it pulls the jobs assigned to its GPU in order; two jobs never run on
the same GPU at once), a CPU pool (at most 2 RAM/disk-heavy jobs at a time: GGUF conversion, quantize,
dequant, download), a disk guard (free space ≥ 2× the expected output before dequant/KL/download/GGUF),
resume via `.done` markers, and a graceful drain on SIGINT (running jobs finish, no new ones start).

GPU hygiene: the card is checked before and after every GPU job. The scheduler waits for memory above the
baseline taken at start-up to be released; if memory is still held, it kills **only compute processes that
appeared after that baseline** (vLLM / llama-server leftovers). A card that still will not free up is taken
out of service, its jobs stay `pending`, and the run finishes without hanging (`logs/sched.log` →
"taking it out of service"); free the card and run the same command again. Every job runs in its own
process group; a job over 4 hours is killed and recorded as `failed`. CPU jobs cannot see a GPU
(`CUDA_VISIBLE_DEVICES=""`). On a machine where the GPU is shared, such as a development box, use
`--no-gpu-check`.

If interrupted: run the same command again — completed jobs are skipped.

## 7. Finishing

```bash
python scripts/verify_checklist.py --finalize   # expect 10/10 PASS; E0 gate.json.pass must be true
bash scripts/remote/pull_artifacts.sh           # packs quant_artifacts_<date>.tar.gz
```

The package: `results/ figures/ reports/ logs/ data/manifest*.json data/LICENSES.md env/requirements.lock
configs/cli_overrides.json` (without models/, refs/, tmp_deq/, .kld, and without `results/*/raw`, which
can contain benchmark text — GPQA must never leave the machine).

## Troubleshooting

| Symptom | Fix |
|---|---|
| `check_env.py` FAIL: flag missing | check the llama.cpp commit; look at `cli_overrides.json` |
| vLLM OOM (q32 BF16 evaluation) | expected: the job fails, the scheduler continues with the other branches; the row is reported as `NA` |
| HF gated 401/403 | check the token (`hf auth login`); if access cannot be granted, stop and resolve it before running |
| disk guard waiting | `df -h .` — remove old `tmp_deq/` and `.kld` files |
| a job keeps failing | read `logs/jobs/<exp>__<job>__<key>.log`; restart the scheduler to retry |
