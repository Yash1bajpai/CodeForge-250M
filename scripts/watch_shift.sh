#!/bin/bash
# watch_shift.sh — runs INSIDE opencode's bash tool (blocking).
# Polls the Kaggle training kernel every POLL_SEC (default 600 = 10 min).
# Exits EARLY (alert) on: kernel error, session death, STOP event in metrics,
# or val-loss divergence. Otherwise exits 0 with a one-page report at shift end.
#
# Usage: ./watch_shift.sh [minutes] [kernel-slug]
#   minutes    shift length       (default 60)
#   kernel-slug  default yashbajpai2027/codeforge-250m-train

POLL_SEC="${POLL_SEC:-600}"
SHIFT_MIN="${1:-60}"
KERNEL="${2:-yashbajpai2027/codeforge-250m-train}"
CKPT_DS="yashbajpai2027/codeforge-ckpt"
END=$(( $(date +%s) + SHIFT_MIN * 60 ))
TMP=$(mktemp -d)

log() { echo "[$(date +%H:%M:%S)] $*"; }

get_metrics() {
  # pull fresh metrics.json from the checkpoint dataset (cheap, tiny file)
  kaggle datasets download -d "$CKPT_DS" -p "$TMP/m" --unzip -o -f metrics.json >/dev/null 2>&1
  cat "$TMP/m/metrics.json" 2>/dev/null
}

get_status() {
  kaggle kernels status "$KERNEL" 2>/dev/null
}

alert() {
  echo "ALERT: $1"
  exit 42   # 42 = watchdog alert, opencode acts immediately
}

while true; do
  ST=$(get_status)
  NOW=$(date +%s)

  # 1) kernel health
  case "$ST" in
    *Error*) alert "KERNEL ERROR: $ST" ;;
    *Cancel*) alert "KERNEL CANCELLED: $ST" ;;
  esac

  # 2) metrics: STOP events (watchdog fired on Kaggle side)
  M=$(get_metrics)
  if [ -n "$M" ]; then
    LAST_STOP=$(echo "$M" | grep '"event": "STOP"' | tail -1)
    [ -n "$LAST_STOP" ] && alert "WATCHDOG FIRED REMOTELY: $LAST_STOP"
    LAST_VAL=$(echo "$M" | grep '"event": "VAL"' | tail -4)
    # 4 consecutive rising val losses = divergence (redundant with remote watchdog)
    if [ "$(echo "$M" | grep -c '"event": "VAL"')" -ge 4 ]; then
      V=$(echo "$LAST_VAL" | grep -o '"val_loss": [0-9.]*' | grep -o '[0-9.]*$')
      A=$(echo "$V" | sed -n 1p); B=$(echo "$V" | sed -n 2p)
      C=$(echo "$V" | sed -n 3p); D=$(echo "$V" | sed -n 4p)
      if awk "BEGIN{exit !($A < $B && $B < $C && $C < $D)}"; then
        alert "VAL_LOSS_DIVERGENCE (local check): $A -> $B -> $C -> $D"
      fi
    fi
  fi

  # 3) shift over?
  if [ "$NOW" -ge "$END" ]; then
    echo "SHIFT_END: no incidents in ${SHIFT_MIN}m."
    [ -n "$M" ] && echo "--- last metrics ---" && echo "$M" | tail -5
    echo "--- kernel status: $ST"
    exit 0
  fi

  # sleep until next poll or shift end, whichever first
  SLEEP=$(( NOW + POLL_SEC < END ? POLL_SEC : END - NOW ))
  [ "$SLEEP" -le 0 ] && continue
  log "ok — status: ${ST:-unknown}; sleeping ${SLEEP}s"
  sleep "$SLEEP"
done
