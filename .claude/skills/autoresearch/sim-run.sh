#!/usr/bin/env bash
# sim-run.sh — run a sim-cli command, gating ONLY the heavy world-advancing
# commands (run-to / step / watch / bench) through a global counting semaphore.
# Many independent research agents drive their OWN worlds iteratively (fork,
# fast-forward, inspect, compare); this wrapper bounds how many heavy sims run at
# once so their natural staggering can't accidentally align into an OOM. Light
# reads (pillars/state/eco/find/inspect/brain/decide/...) pass through UNGATED,
# so an agent can route every sim-cli call through sim-run safely.
#
# Semaphore: mkdir-based (atomic; no flock / GNU parallel — macOS-native). Slots
# live in a shared absolute SEMDIR so separate agent worktrees coordinate. A slot
# whose holder PID is dead is reclaimed (an agent killed mid-run can't deadlock).
#
# Env:
#   AUTORESEARCH_SEM        semaphore dir   (default /tmp/autoresearch-sem)
#   AUTORESEARCH_SEM_SLOTS  max concurrent heavy sims (default 8)
#   SIMCLI                  sim-cli binary  (default ./target/release/sim-cli)
#
# Usage:
#   sim-run.sh run-to 500000 --in artifacts/w.bin --threads 1   # gated
#   sim-run.sh pillars --in artifacts/w.bin --text              # ungated passthrough
set -uo pipefail
SEMDIR="${AUTORESEARCH_SEM:-/tmp/autoresearch-sem}"
SLOTS="${AUTORESEARCH_SEM_SLOTS:-8}"
SIMCLI="${SIMCLI:-./target/release/sim-cli}"
[ -x "$SIMCLI" ] || SIMCLI="$(command -v sim-cli 2>/dev/null || echo "$SIMCLI")"

case "${1:-}" in
  run-to|step|watch|bench) : ;;                 # heavy → gate
  *) exec "$SIMCLI" "$@" ;;                      # light → passthrough
esac

mkdir -p "$SEMDIR" 2>/dev/null
HELD=""
release() { [ -n "$HELD" ] && rm -rf "$HELD" 2>/dev/null; HELD=""; }
trap release EXIT INT TERM

acquire() {
  local i pf p
  while :; do
    for i in $(seq 1 "$SLOTS"); do
      if mkdir "$SEMDIR/slot.$i" 2>/dev/null; then
        echo $$ > "$SEMDIR/slot.$i/pid"; HELD="$SEMDIR/slot.$i"; return 0
      fi
    done
    # No free slot: reclaim any whose holder is dead, then back off (jittered).
    for i in $(seq 1 "$SLOTS"); do
      pf="$SEMDIR/slot.$i/pid"
      if [ -f "$pf" ]; then
        p="$(cat "$pf" 2>/dev/null)"
        if [ -n "$p" ] && ! kill -0 "$p" 2>/dev/null; then rm -rf "$SEMDIR/slot.$i" 2>/dev/null; fi
      fi
    done
    sleep $(( (RANDOM % 3) + 1 ))
  done
}

acquire
"$SIMCLI" "$@"; RC=$?
release
exit $RC
