#!/bin/bash
# agy_supervisor.sh — Antigravity/Gemini as the continuous training watchman.
#
# LOOP (every POLL_MIN minutes):
#   1. Poll Kaggle kernel status.
#   2. RUNNING: pull live telemetry (codeforge-telemetry dataset). If new
#      metrics since last poll -> ask agy for a VERDICT (HEALTHY/WARN/CRITICAL)
#      and log it. CRITICAL mid-session -> ALERT (kernel can't be stopped via
#      API; user can kill it from the Kaggle web UI to save quota).
#   3. COMPLETE: pull session-end metrics (codeforge-ckpt). If healthy
#      wall-clock completion and step < MAX_STEPS -> agy sanity-check of the
#      kernel log tail, then re-push the kernel (auto-resume from checkpoint).
#      If step >= MAX_STEPS -> TRAINING COMPLETE alert.
#   4. ERROR / bad STOP reason / agy says CRITICAL at session end -> pull the
#      full kernel log, ask agy for a ROOT-CAUSE DIAGNOSIS, write it to
#      AGY_DIAGNOSIS.md, raise ALERT.txt (+termux notification) and STOP.
#      Fixing is opencode's job; after the fix, restart this script.
#
# Files: /tmp/opencode/AGY_WATCH.log    — heartbeats + verdicts (tail me!)
#        /tmp/opencode/AGY_DIAGNOSIS.md — full diagnosis on failures
#        /tmp/opencode/ALERT.txt        — attention needed
# Env:   ONE_SHOT=1 -> run exactly one cycle then exit (for testing)
#        AGY_BIN   -> agy binary path (default /root/.local/bin/agy)

KERNEL="yashbajpai2027/codeforge-250m-train-run-2"
CKPT_DS="yashbajpai2027/codeforge-ckpt"
TELEM_DS="yashbajpai2027/codeforge-telemetry"
PUSH_DIR="/tmp/opencode/train_push"
REPO="/data/data/com.termux/files/home/projects/own_llm/CodeForge-250M_win"
BASE="/tmp/opencode"
LOG="$BASE/AGY_WATCH.log"
DIAG="$BASE/AGY_DIAGNOSIS.md"
ALERT="$BASE/ALERT.txt"
POLL_MIN=10
MAX_STEPS=2898
AGY="${AGY_BIN:-/root/.local/bin/agy}"
KTOK="$(cat ~/.kaggle/access_token 2>/dev/null)"

log()   { echo "[$(date '+%m-%d %H:%M:%S')] $*" >> "$LOG"; }
trimlog() { tail -n 800 "$LOG" > "$LOG.tmp" 2>/dev/null && mv "$LOG.tmp" "$LOG"; }

alert() {  # terminal: something needs a human/fix — stop supervising
  echo "[$(date)] $*" > "$ALERT"
  log "ALERT: $*"
  command -v termux-notification >/dev/null && termux-notification \
    --title "CodeForge training" --content "$*" 2>/dev/null
  exit 1
}

alert_soft() {  # non-terminal (session still RUNNING): notify once, keep watching
  echo "[$(date)] $*" > "$ALERT"
  log "ALERT(soft): $*"
  command -v termux-notification >/dev/null && termux-notification \
    --title "CodeForge training" --content "$*" 2>/dev/null
}

kstat() { timeout 90 kaggle kernels status "$KERNEL" 2>/dev/null | sed 's/.*has status "//;s/".*//'; }

pull_kernel_log() { # $1 = dest file
  timeout 120 curl -s -H "Authorization: Bearer $KTOK" \
    "https://www.kaggle.com/api/v1/kernels/output?user_name=yashbajpai2027&kernel_slug=codeforge-250m-train-run-2" \
    -o "$BASE/_kout.json" 2>/dev/null
  python3 - "$BASE/_kout.json" "$1" <<'EOF'
import json, sys
try:
    d = json.load(open(sys.argv[1]))
    log = d.get('logNullable') or ''
    lines = [e["data"].rstrip() for e in json.loads(log)]
    open(sys.argv[2], 'w').write('\n'.join(lines))
    print(f"pulled {len(lines)} lines")
except Exception as e:
    print(f"log pull failed: {e}")
EOF
}

agy_verdict() { # $1 = metrics file path -> echoes "VERDICT: X — reason"
  timeout 300 "$AGY" -p "You are the on-call SRE for CodeForge-250M, a 250M-param LLM pretraining run on a Kaggle P100 (single GPU, ~45s/step, 2898 steps total, batch 256 seqs x 2048 tok = 524k tok/step, cosine LR 6e-4 peak, val every 100 steps, TRAIN events logged every 5 steps in JSONL). Read the metrics JSONL at '$1'. Judge: loss trend vs expected (~10.5 start, falling), val_loss trend, throughput, any STOP/CRASH/FINAL events. First line of your reply MUST be exactly 'VERDICT: HEALTHY' or 'VERDICT: WARN' or 'VERDICT: CRITICAL' followed by ' - reason' on the SAME line. Then max 3 short bullets. No preamble." 2>/dev/null
}

agy_diagnose() { # $1 = kernel log, $2 = metrics, $3 = out file -> 0 on success
  local prompt="You are the on-call engineer for CodeForge-250M pretraining (250M LLaMA-style, Kaggle P100). A training session failed. Evidence: (1) full kernel log at '$1'; (2) metrics JSONL at '$2'; (3) training code at $REPO/training/train.py; (4) kernel wrapper at $REPO/kaggle/train_kernel.py. Known past bugs (check if repeat): NCCL DDP desync (dead - single GPU forced), 404 processing race on ckpt dataset pulls (mounts prevent), t+2 double-shift loss bug (fixed in models/architecture.py), per-sample shard loads (fixed via RAM preload), silent from-scratch retrain (blocked by rewind guards), stale-mounted checkpoints (guarded by step comparison). Read the evidence files yourself. Produce a markdown report with sections: ROOT CAUSE (one paragraph), EVIDENCE (exact log lines), FIX (concrete, file:line level), RESTART-SAFE (yes/no + what to verify). Be terse and concrete."
  for attempt in 1 2; do
    timeout 420 "$AGY" -p "$prompt" > "$3" 2>/dev/null
    [ -s "$3" ] && return 0   # non-empty output = usable diagnosis (rc may lie)
    log "agy diagnosis attempt $attempt empty/failed; $( [ $attempt -eq 1 ] && echo retrying || echo giving up )"
    sleep 10
  done
  return 1
}

# --- state across cycles ---
last_telem_md5=""
live_critical_notified=""
restarts=0
last_push_ts=0
last_restart_step=0
stalled_restarts=0
mkdir -p "$BASE"
log "=== agy supervisor started (poll ${POLL_MIN}m) ==="
trimlog

while true; do
  TMP=$(mktemp -d)
  ST=$(kstat)

  if [ -z "$ST" ]; then
    log "status query failed (network?) — retry next cycle"
  elif [ "$ST" = "KernelWorkerStatus.RUNNING" ] || [ "$ST" = "KernelWorkerStatus.QUEUED" ]; then
    # live telemetry check (only when RUNNING — cheap single-file pull)
    if timeout 90 kaggle datasets download "$TELEM_DS" -p "$TMP" -f metrics.json --unzip --force >/dev/null 2>&1 \
       && [ -s "$TMP/metrics.json" ]; then
      md5=$(md5sum "$TMP/metrics.json" | cut -d' ' -f1)
      if [ "$md5" != "$last_telem_md5" ]; then
        last_telem_md5="$md5"
        V=$(agy_verdict "$TMP/metrics.json")
        first=$(echo "$V" | head -1)
        case "$first" in
          VERDICT:\ CRITICAL*)
            log "LIVE CRITICAL: $first"
            if [ -z "$live_critical_notified" ]; then
              live_critical_notified=1
              alert_soft "agy flagged the RUNNING session CRITICAL: $first (kill it on Kaggle web UI if warranted) — supervisor keeps watching"
            fi
            ;;
          VERDICT:*)
            log "live ok: $first"
            ;;
          *)
            log "live (no verdict parsed): $(echo "$V" | head -1 | cut -c1-120)"
            ;;
        esac
      else
        log "ok: $ST (telemetry unchanged)"
      fi
    else
      log "ok: $ST (no telemetry yet — early in session)"
    fi

  else
    # ---- terminal state: COMPLETE or ERROR ----
    timeout 90 kaggle datasets download "$CKPT_DS" -p "$TMP" -f metrics.json --unzip --force >/dev/null 2>&1
    DEC=$(python3 - "$TMP/metrics.json" "$ST" "$MAX_STEPS" <<'EOF'
import json, sys
path, state, max_steps = sys.argv[1], sys.argv[2], int(sys.argv[3])
try:
    entries = [json.loads(l) for l in open(path) if l.strip()]
except FileNotFoundError:
    print("ACTION=ALERT reason=no-metrics-uploaded state=" + state); sys.exit()
step = max((e.get("step", 0) for e in entries if "step" in e), default=0)
crashes = [e for e in entries if e.get("event") == "CRASH"]
exits = [e for e in entries if e.get("event") == "TRAIN_EXIT"]
if step >= max_steps:
    print(f"ACTION=DONE step={step}"); sys.exit()
if state == "KernelWorkerStatus.ERROR":
    print(f"ACTION=DIAGNOSE reason=kernel-ERROR step={step}"); sys.exit()
if crashes:
    print(f"ACTION=DIAGNOSE reason=crash-event step={step} err={crashes[-1].get('error','')[:200]}"); sys.exit()
if exits and exits[-1].get("rc", 0) != 0:
    print(f"ACTION=DIAGNOSE reason=train-exit-rc={exits[-1].get('rc')} step={step}"); sys.exit()
stops = [e for e in entries if e.get("event") == "STOP"]
if stops:
    r = stops[-1].get("reason", "?")
    if r == "WALL_CLOCK_BUDGET":
        print(f"ACTION=RESTART reason=wall-clock step={step}")
    elif r == "EPOCH_COMPLETE_SINGLE_PASS":
        print(f"ACTION=ALERT reason=epoch-ended-early step={step}")
    else:
        print(f"ACTION=DIAGNOSE reason=watchdog-stop:{r} step={step}")
else:
    print(f"ACTION=RESTART reason=completed-no-stop step={step}")
EOF
)
    log "terminal: $ST | $DEC"

    case "$DEC" in
      ACTION=RESTART*)
        # progress-based kill: sessions can outlive the fast-failure window
        # (31+ min) while making near-zero steps — that burns quota forever.
        cur_step=$(grep -o 'step=[0-9]*' <<<"$DEC" | head -1 | cut -d= -f2)
        cur_step=${cur_step:-0}
        prog=$((cur_step - last_restart_step))
        if [ "$last_restart_step" -gt 0 ] && [ "$prog" -lt 10 ]; then
          stalled_restarts=$((stalled_restarts + 1))
          log "stalled restart: +${prog} steps — stall count $stalled_restarts/3"
        else
          stalled_restarts=0
        fi
        last_restart_step=$cur_step
        if [ "$stalled_restarts" -ge 3 ]; then
          alert "3 stalled restarts (each <10 steps of progress). Likely a repeat bug — needs a fix, not another restart."
        fi
        # agy sanity-check of the ended session's log before resuming
        pull_kernel_log "$TMP/kernel.log" >/dev/null 2>&1
        if [ -s "$TMP/kernel.log" ]; then
          V=$(timeout 300 "$AGY" -p "Pre-flight check before auto-resuming a Kaggle training session. This is the FULL log of the just-finished session at '$TMP/kernel.log'. Expected: single-GPU (P100 or T4) training of CodeForge-250M, clean WALL_CLOCK_BUDGET stop or clean completion, checkpoint saved + uploaded. Look for: tracebacks, OOM, NCCL, data loader errors, watchdog stops, suspicious restarts from step 0. First line MUST be 'VERDICT: OK' or 'VERDICT: PROBLEM - reason'. Max 3 bullets after." 2>/dev/null)
          case "$(echo "$V" | head -1)" in
            VERDICT:\ PROBLEM*)
              log "pre-resume check FAILED: $(echo "$V" | head -1)"
              echo "$V" > "$DIAG"
              alert "agy blocked auto-resume: $(echo "$V" | head -1)"
              ;;
          esac
          log "pre-resume check: $(echo "$V" | head -1 | cut -c1-160)"
        fi
        # fast-failure budget (only sessions that died <30 min after push burn it)
        now=$(date +%s)
        if [ "$last_push_ts" -gt 0 ] && [ $((now - last_push_ts)) -lt 1800 ]; then
          restarts=$((restarts + 1))
          log "fast failure — budget $restarts/6"
        else
          restarts=0
        fi
        if [ "$restarts" -ge 6 ]; then
          alert "Crash loop: 6 fast failures. Last: $DEC"
        fi
        mkdir -p "$PUSH_DIR"
        cp -f "$REPO/kaggle/train_kernel.py" "$REPO/kaggle/kernel-metadata.json" "$PUSH_DIR/" 2>> "$LOG"
        (cd "$PUSH_DIR" && timeout 180 kaggle kernels push -p . >> "$LOG" 2>&1)
        last_push_ts=$(date +%s)
        log "kernel re-pushed (resume) — budget $restarts/6"
        ;;
      ACTION=DONE*)
        alert "TRAINING COMPLETE — $DEC (run evals next)"
        ;;
      ACTION=DIAGNOSE*)
        pull_kernel_log "$TMP/kernel.log" >/dev/null 2>&1
        # keep evidence in a stable place that survives this cycle (rm -rf $TMP)
        EV="$BASE/diag_evidence"; mkdir -p "$EV"
        cp -f "$TMP/kernel.log" "$TMP/metrics.json" "$EV/" 2>/dev/null
        if agy_diagnose "$EV/kernel.log" "$EV/metrics.json" "$DIAG"; then
          log "diagnosis written to $DIAG"
          alert "$DEC — agy diagnosis ready in $DIAG"
        else
          { echo "agy diagnosis call failed twice; raw evidence:"; \
            echo "kernel log: $EV/kernel.log"; echo "metrics: $EV/metrics.json"; } > "$DIAG"
          alert "$DEC — agy diagnosis FAILED, raw evidence at $EV"
        fi
        ;;
      *)
        alert "$DEC"
        ;;
    esac
  fi

  rm -rf "$TMP"
  [ -n "$ONE_SHOT" ] && { log "one-shot cycle complete"; break; }
  sleep $((POLL_MIN * 60))
done
