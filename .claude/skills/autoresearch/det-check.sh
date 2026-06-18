#!/usr/bin/env bash
# det-check.sh — authoritative determinism check. Run from INSIDE a built
# worktree (sim-cli at ./target/release/sim-cli, or set $SIMCLI). Prints exactly
# one of: ok | broken-p1 | broken-p2 | error.
#
# TWO parts, because `run-to` IGNORES --threads (the thread count is fixed at
# `new` and serialized into the world, so a cross-thread BYTE compare is invalid
# — two worlds built with different --threads differ in bytes even when the sim
# is identical). So:
#   P1  save/load + process reproducibility: two threads=1 processes advance the
#       SAME world → byte-identical (catches HashMap iteration / RNG / uninit).
#   P2  cross-thread independence: a threads=1 vs a threads=4 world advanced the
#       same number of ticks → identical SEMANTIC fingerprint (state+lineage+top;
#       verified stable across thread counts, no thread-count string in it).
set -uo pipefail
CLI="${SIMCLI:-./target/release/sim-cli}"
[ -x "$CLI" ] || { echo error; exit 0; }
A=artifacts/_det; mkdir -p "$A" 2>/dev/null
fp() {
  $CLI state   --in "$1" --text 2>/dev/null
  $CLI lineage --in "$1" --text 2>/dev/null
  $CLI top energy 40 --in "$1" --text 2>/dev/null
  $CLI top age 40    --in "$1" --text 2>/dev/null
}
# P1
$CLI new --seed 7 --scale 70,400 --threads 1 --out "$A/d1.bin" --no-metrics >/dev/null 2>&1 || { echo error; exit 0; }
cp "$A/d1.bin" "$A/d2.bin"
$CLI run-to 4000 --in "$A/d1.bin" --no-metrics >/dev/null 2>&1
$CLI run-to 4000 --in "$A/d2.bin" --no-metrics >/dev/null 2>&1
cmp -s "$A/d1.bin" "$A/d2.bin" || { echo broken-p1; exit 0; }
# P2
$CLI new --seed 7 --scale 70,400 --threads 4 --out "$A/d4.bin" --no-metrics >/dev/null 2>&1 || { echo error; exit 0; }
$CLI run-to 4000 --in "$A/d4.bin" --no-metrics >/dev/null 2>&1
fp "$A/d1.bin" > "$A/fp1.txt"; fp "$A/d4.bin" > "$A/fp4.txt"
cmp -s "$A/fp1.txt" "$A/fp4.txt" || { echo broken-p2; exit 0; }
echo ok
