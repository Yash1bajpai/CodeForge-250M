#!/bin/bash
# status_report.sh — one-command training status for humans.
# Pulls metrics.json + kernel state from Kaggle and prints a verdict.
# Usage: ./status_report.sh

CKPT_DS="yashbajpai2027/codeforge-ckpt"
KERNEL="yashbajpai2027/codeforge-250m-train-run-2"
TMP=$(mktemp -d)

# 1) kernel state
ST=$(kaggle kernels status "$KERNEL" 2>/dev/null | sed 's/.*has status "//;s/".*//')

# 2) metrics
kaggle datasets download "$CKPT_DS" -p "$TMP" -f metrics.json --unzip --force >/dev/null 2>&1
python3 "$TMP/../../.$$/report.py" 2>/dev/null || python3 - "$TMP/metrics.json" "$ST" <<'EOF'
import json, sys, math, time, collections

metrics_path, kernel_state = sys.argv[1], sys.argv[2]
try:
    entries = [json.loads(l) for l in open(metrics_path) if l.strip()]
except FileNotFoundError:
    print("NO METRICS YET — training hasn't reached its first checkpoint upload.")
    sys.exit()

trains = [e for e in entries if e.get("event") == "TRAIN"]
vals   = [e for e in entries if e.get("event") == "VAL"]
stops  = [e for e in entries if e.get("event") == "STOP"]
ckpts  = [e for e in entries if e.get("event") == "CKPT"]

print("=" * 62)
print(f"KERNEL STATE : {kernel_state or 'unknown'}")
print(f"STEPS DONE   : {trains[-1]['step'] if trains else 0} / 2898")
if trains:
    t = trains[-1]
    print(f"LATEST TRAIN : loss={t['loss']:.4f}  ppl={t['ppl']:.2f}  lr={t['lr']:.2e}  tokens={t['tokens_seen']:,}")
    # throughput from last two TRAIN entries
    if len(trains) >= 2:
        a, b = trains[-2], trains[-1]
        dt = b["ts"] - a["ts"]
        dsteps = b["step"] - a["step"]
        if dt > 0 and dsteps > 0:
            tok_per_s = (b["tokens_seen"] - a["tokens_seen"]) / dt
            print(f"THROUGHPUT  : {tok_per_s:,.0f} tok/s | {dt/dsteps:.1f} s/step")
if vals:
    v = vals[-1]
    print(f"LATEST VAL   : val_loss={v['val_loss']:.4f} (train {v['train_loss']:.4f}) at step {v['step']}")
    print("VAL HISTORY  : " + " -> ".join(f"{x['val_loss']:.3f}" for x in vals[-8:]))
if ckpts:
    print(f"LAST CKPT    : step {ckpts[-1]['step']}")

# --- verdict ---
print("-" * 62)
verdicts = []
if stops:
    s = stops[-1]
    verdicts.append(f"STOPPED EARLY: {s['reason']} at step {s.get('step','?')} — needs attention")
if vals and len(vals) >= 2:
    if vals[-1]["val_loss"] < vals[-2]["val_loss"]:
        verdicts.append("GOOD: val loss still falling (learning, not memorizing)")
    elif vals[-1]["val_loss"] > vals[-2]["val_loss"] * 1.05:
        verdicts.append("WARNING: val loss rising >5% — watch for overfitting")
    else:
        verdicts.append("OK: val loss flat (plateau or noise)")
if trains:
    ppl = trains[-1]["ppl"]
    if ppl < 1.5:
        verdicts.append("BAD: perplexity below 1.5 mid-run = memorization signature")
    elif ppl < 3.0:
        verdicts.append("NOTE: ppl quite low — check val loss before trusting")
    else:
        verdicts.append(f"HEALTHY: ppl {ppl:.2f} — normal for this stage")
if not verdicts:
    verdicts.append("Too early to judge — need ≥2 val evals")
for v in verdicts:
    print(v)
print("=" * 62)
EOF
