#!/bin/bash
# auto_restart.sh — overnight supervisor for CodeForge training.
# Detached via nohup. Polls every POLL_MIN. Rules:
#   - kernel COMPLETE + stop reason WALL_CLOCK_BUDGET and step < max_steps
#       -> re-push kernel (auto-resumes from checkpoint)
#   - kernel COMPLETE + step >= max_steps -> TRAINING DONE, alert, stop
#   - kernel ERROR, or bad STOP reason (VAL_LOSS_DIVERGENCE,
#     LOSS_FLOOR_MEMORIZATION, TOKEN_COUNT_MISMATCH, STOP_AND_SAVE),
#     or crash-loop (2 fast failures) -> ALERT and stop supervising
# Log: /tmp/opencode/overnight.log   Alerts: /tmp/opencode/ALERT.txt

KERNEL="yashbajpai2027/codeforge-250m-train-run-2"
CKPT_DS="yashbajpai2027/codeforge-ckpt"
PUSH_DIR="/tmp/opencode/train_push"
REPO="/data/data/com.termux/files/home/projects/own_llm/CodeForge-250M_win"
LOG="/tmp/opencode/overnight.log"
ALERT="/tmp/opencode/ALERT.txt"
POLL_MIN=10
MAX_STEPS=2898
MAX_RESTARTS=6

log() { echo "[$(date +%H:%M:%S)] $*" >> "$LOG"; }

alert() {
  echo "[$(date)] $*" > "$ALERT"
  log "ALERT: $*"
  command -v termux-notification >/dev/null && termux-notification \
    --title "CodeForge training" --content "$*" 2>/dev/null
  exit 1
}

restarts=0
last_push_ts=0
log "=== overnight supervisor started (poll ${POLL_MIN}m, max $MAX_RESTARTS restarts) ==="

while true; do
  ST=$(kaggle kernels status "$KERNEL" 2>/dev/null | sed 's/.*has status "//;s/".*//')

  if [ -z "$ST" ]; then
    log "status query failed (network?) — will retry"
    sleep $((POLL_MIN * 60)); continue
  fi

  if [ "$ST" = "KernelWorkerStatus.RUNNING" ] || [ "$ST" = "KernelWorkerStatus.QUEUED" ]; then
    log "ok: $ST"
    sleep $((POLL_MIN * 60)); continue
  fi

  # terminal state — pull metrics to decide
  TMP=$(mktemp -d)
  kaggle datasets download "$CKPT_DS" -p "$TMP" -f metrics.json --unzip --force >/dev/null 2>&1
  python3 - "$TMP/metrics.json" "$ST" "$MAX_STEPS" > "$TMP/decision.txt" <<'EOF'
import json, sys
path, state, max_steps = sys.argv[1], sys.argv[2], int(sys.argv[3])
try:
    entries = [json.loads(l) for l in open(path) if l.strip()]
except FileNotFoundError:
    print("ACTION=ALERT reason=no-metrics-uploaded (crashed before upload?) state=" + state); sys.exit()
trains = [e for e in entries if e.get("event") == "TRAIN"]
stops  = [e for e in entries if e.get("event") == "STOP"]
step = trains[-1]["step"] if trains else 0
if step >= max_steps:
    print(f"ACTION=DONE step={step} — training complete!"); sys.exit()
if state == "KernelWorkerStatus.ERROR":
    print(f"ACTION=ALERT reason=kernel-ERROR at step {step}"); sys.exit()
if stops:
    r = stops[-1].get("reason", "?")
    if r == "WALL_CLOCK_BUDGET":
        print(f"ACTION=RESTART reason=wall-clock step={step}")
    elif r == "EPOCH_COMPLETE_SINGLE_PASS":
        print(f"ACTION=ALERT reason=epoch-ended-early step={step} (data shorter than planned)")
    else:
        print(f"ACTION=ALERT reason=watchdog-stop:{r} step={step}")
else:
    # completed without a STOP event — likely clean wall-clock exit
    print(f"ACTION=RESTART reason=completed-no-stop step={step}")
EOF
  DEC=$(cat "$TMP/decision.txt" 2>/dev/null || echo "ACTION=ALERT reason=decision-parse-failed")
  log "terminal: $ST | $DEC"

  case "$DEC" in
    ACTION=RESTART*)
      # crash-loop guard: ONLY fast failures burn the budget. A session that
      # ran >=30 min (e.g. the 11h wall-clock) is healthy; reset the counter.
      now=$(date +%s)
      if [ "$last_push_ts" -gt 0 ] && [ $((now - last_push_ts)) -lt 1800 ]; then
        restarts=$((restarts + 1))
        log "fast failure detected (up <$((1800/60)) min) — budget now $restarts/$MAX_RESTARTS"
      else
        restarts=0
      fi
      if [ "$restarts" -ge "$MAX_RESTARTS" ]; then
        alert "Crash loop: $MAX_RESTARTS fast failures in a row. Last: $DEC"
      fi
      log "re-pushing kernel (resume) — budget $restarts/$MAX_RESTARTS"
      # sync kernel code from repo so a stale push dir can never resurrect old bugs
      if ! cp -f "$REPO/kaggle/train_kernel.py" "$REPO/kaggle/kernel-metadata.json" "$PUSH_DIR/" 2>> "$LOG"; then
        alert "cannot sync kernel files from $REPO — refusing to push stale code"
      fi
      (cd "$PUSH_DIR" && kaggle kernels push -p . >> "$LOG" 2>&1)
      last_push_ts=$(date +%s)
      ;;
    ACTION=DONE*)
      alert "TRAINING COMPLETE — $DEC"
      ;;
    *)
      alert "$DEC"
      ;;
  esac
  rm -rf "$TMP"
  sleep $((POLL_MIN * 60))
done
