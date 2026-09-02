#!/bin/bash
# kaggle_watchdog.sh — persistent watchdog for this opencode session.
# Polls kernel status every POLL_SEC. Any state change (RUNNING->COMPLETE/ERROR/
# CANCEL, or new version detected) makes it exit IMMEDIATELY with a report,
# which surfaces in the opencode session as an alert.
#
# Usage: ./kaggle_watchdog.sh <kernel-slug> [max_minutes]
#   kernel-slug   e.g. yashbajpai2027/codeforge-data-prep-cpu
#   max_minutes   hard stop (default 720 = 12h)

K="${1:?usage: kaggle_watchdog.sh <kernel-slug> [max_minutes]}"
MAX_MIN="${2:-720}"
POLL_SEC="${POLL_SEC:-120}"
END=$(( $(date +%s) + MAX_MIN * 60 ))
TOK="$(cat ~/.kaggle/access_token)"
TMP=$(mktemp -d)

prev=""

while true; do
  ST=$(kaggle kernels status "$K" 2>/dev/null | sed 's/.*has status "//;s/".*//')
  NOW=$(date +%s)

  if [ -z "$ST" ]; then
    echo "WATCHDOG: status query failed for $K (network or auth) — retrying"
  elif [ "$ST" != "$prev" ]; then
    if [ -z "$prev" ]; then
      echo "WATCHDOG: watching $K — initial state: $ST"
    else
      # state CHANGED mid-watch: this is the alert moment
      echo "WATCHDOG ALERT: $K changed state: $prev -> $ST at $(date +%H:%M:%S)"
      exit 42
    fi
  fi
  prev="$ST"

  case "$ST" in
    *COMPLETE*|*ERROR*|*CANCEL*) echo "WATCHDOG: $K reached terminal state: $ST"; exit 42 ;;
  esac

  if [ "$NOW" -ge "$END" ]; then
    echo "WATCHDOG: max watch time reached ($MAX_MIN min). Last state: $ST"
    exit 0
  fi
  sleep "$POLL_SEC"
done
