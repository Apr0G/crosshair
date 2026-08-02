#!/bin/bash
# Watchdog for an unattended scrape.
#
#   ./scripts/scrape_watchdog.sh [swap_limit_mb] [restart_workers]
#
#   swap_limit_mb    stop the scrape once swap used exceeds this   (default 1024)
#   restart_workers  relaunch with this many workers after stopping;
#                    0 = just stop and stay stopped                (default 0)
#
# Why swap and not free memory: macOS reports "free" in a way that looks alarming
# long before anything is wrong, because inactive and speculative pages are
# reclaimable. Swap actually being written is the point where paging starts costing
# more than the extra workers gain — at that point six thrashing workers finish
# slower than four healthy ones.
#
# Stopping is safe at any moment. Each match is written in ONE transaction, so a
# killed worker leaves no partial rows and the match simply isn't marked processed;
# a later run redoes it. Nothing is corrupted and nothing is lost but that match's
# CPU time.

set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

LIMIT_MB=${1:-1024}
RESTART_WORKERS=${2:-0}
POLL_SECONDS=30
PATTERN="main.py scrape"
LOG="watchdog.log"

log() { printf '%s  %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" | tee -a "$LOG"; }

swap_used_mb() {
  # "total = 1024.00M  used = 210.06M  free = 813.94M  (encrypted)"
  sysctl -n vm.swapusage 2>/dev/null | awk '
    { for (i = 1; i <= NF; i++) if ($i == "used") {
        v = $(i+2); u = substr(v, length(v))
        gsub(/[A-Za-z]/, "", v)
        if (u == "G") v *= 1024
        else if (u == "K") v /= 1024
        printf "%.0f", v; exit
      } }'
}

workers_rss_gb() {
  ps -eo rss,command | grep "[m]ain.py scrape" | grep -v -- "--workers" \
    | awk '{s += $1} END {printf "%.1f", s / 1048576}'
}

worker_count() { pgrep -f "$PATTERN" 2>/dev/null | wc -l | tr -d ' '; }

matches() {
  .venv/bin/python src/main.py status 2>/dev/null \
    | awk -F: '/^Matches/ {gsub(/ /, "", $2); print $2; exit}'
}

stop_scrape() {
  log "stopping scrape (SIGINT, graceful — in-flight match rolls back cleanly)"
  pkill -INT -f "$PATTERN" 2>/dev/null
  for _ in $(seq 1 20); do
    [ "$(worker_count)" -eq 0 ] && break
    sleep 1
  done
  if [ "$(worker_count)" -gt 0 ]; then
    log "still alive after 20s, escalating to SIGTERM"
    pkill -TERM -f "$PATTERN" 2>/dev/null
    sleep 5
  fi
  log "scrape stopped. matches in db: $(matches)"
}

log "=========================================================="
log "watchdog started  |  swap limit ${LIMIT_MB} MB  |  poll ${POLL_SECONDS}s"
if [ "$RESTART_WORKERS" -gt 0 ]; then
  log "on trip: restart with ${RESTART_WORKERS} workers"
else
  log "on trip: stop and exit (pass a second argument to auto-restart)"
fi

if [ "$(worker_count)" -eq 0 ]; then
  log "no scrape running — nothing to watch. exiting."
  exit 0
fi

while true; do
  n=$(worker_count)
  if [ "$n" -eq 0 ]; then
    log "scrape finished on its own. matches in db: $(matches)"
    exit 0
  fi

  swap=$(swap_used_mb); swap=${swap:-0}
  rss=$(workers_rss_gb)
  log "procs=${n}  workers_rss=${rss}GB  swap=${swap}MB  matches=$(matches)"

  if [ "$swap" -gt "$LIMIT_MB" ]; then
    log "SWAP ${swap}MB EXCEEDS ${LIMIT_MB}MB — machine is paging"
    stop_scrape
    if [ "$RESTART_WORKERS" -gt 0 ]; then
      sleep 10
      log "restarting with ${RESTART_WORKERS} workers"
      nohup .venv/bin/python -u src/main.py scrape --source api \
            --workers "$RESTART_WORKERS" \
            --max-players $((RESTART_WORKERS * 8)) \
            --matches-per-player 40 > scrape.log 2>&1 &
      sleep 20
      log "restarted, $(worker_count) processes up"
      RESTART_WORKERS=0   # only rescue once; a second trip means stop for good
    else
      log "watchdog exiting."
      exit 1
    fi
  fi

  sleep "$POLL_SECONDS"
done
