#!/usr/bin/env bash
# eval-queue.sh — planner-owned evaluator for the autoresearch loop.
#
# ONE scheduler for all sim compute, with a HARD global concurrency cap, so
# independent experiments never oversubscribe the host (the iteration-1
# bottleneck: ~48 concurrent 500k sims + OOM). Implements the screen→confirm
# verification ladder centrally and trusts determinism (each branch is run
# exactly once; no re-runs).
#
# Pipeline:
#   1. build   — each branch (base first, to warm the sccache dep cache) in its
#                own scratch worktree; a TWO-PART determinism check (see below).
#   2. screen  — one cheap run per branch (screen seed, short horizon); a COARSE
#                DISASTER FILTER only: drop collapse (pop 0) / explosion (pop near
#                cap). NO action_effectiveness filter — short-horizon aeff is
#                noisy and a seed-7 aeff drop can be a seed-2026 rescue; the 500k
#                confirm is the real HOLD gate.
#   3. confirm — survivors (+ base) × all seeds at the full horizon.
#   4. report  — summary.json: per-seed pillars, per-seed deltas, n-survivors,
#                rescued/lost seeds, and the honest seed-for-seed
#                clean_delta_vs_base_common_survivors (with a min-n guard), PLUS a
#                `dropped` array (build-failed / determinism-broken / screened)
#                so the planner can always tell "loser" from "never ran".
#
# DETERMINISM CHECK (two parts — `run-to` ignores --threads; the thread count is
# baked in at `new` and serialized, so byte-cmp across thread counts is invalid):
#   P1 save/load+process reproducibility: two threads=1 processes advance the
#      same world → byte-identical (catches HashMap-iteration / RNG / uninit).
#   P2 thread-count independence: a threads=1 vs threads=4 world advanced equally
#      → identical SEMANTIC fingerprint (state+lineage+top); the project invariant.
#
# All phases pool sims through `xargs -P <cap>` (BSD-xargs, macOS-native; no GNU
# parallel / flock needed). A per-sim watchdog kills runaway (explosion) sims.
# Floats + JSON are handled in python3.
#
# Usage:
#   eval-queue.sh --base-ref <sha> --branches "br1 br2 ..." \
#       [--cap 10] [--build-cap 2] \
#       [--seeds 7,42,123,2026] [--screen-seed 7] \
#       [--screen-to 200000] [--confirm-to 500000] \
#       [--explode-pop 40000] [--min-common 2] [--scratch <dir>] [--keep] [--no-screen]
#
# Output: writes <scratch>/summary.json and prints it; also prints a table.
set -uo pipefail

SELF="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"

sanitize() { printf '%s' "$1" | tr '/' '_' | tr -cd '[:alnum:]_.-'; }

# Semantic fingerprint of a world: state + lineage + top organisms. Stable across
# thread counts (verified); contains no thread-count string. Used for the P2 check.
world_fp() {
  ./target/release/sim-cli state   --in "$1" --text 2>/dev/null
  ./target/release/sim-cli lineage --in "$1" --text 2>/dev/null
  ./target/release/sim-cli top energy 40 --in "$1" --text 2>/dev/null
  ./target/release/sim-cli top age 40    --in "$1" --text 2>/dev/null
}

# ----- internal job modes (invoked by xargs / directly, one per line) --------
# --build-one <branch> <wtdir> <label> <buildout>
if [ "${1:-}" = "--build-one" ]; then
  shift; BRANCH="$1"; WT="$2"; LABEL="$3"; OUT="$4"
  rm -rf "$WT"
  if ! git worktree add --detach "$WT" "$BRANCH" >/dev/null 2>&1; then
    echo "label=$LABEL build=worktree-fail det=na" > "$OUT"; exit 0
  fi
  ( cd "$WT" && cargo build -p sim-cli --release >/dev/null 2>&1 ) \
    || { echo "label=$LABEL build=fail det=na" > "$OUT"; exit 0; }
  cd "$WT" || { echo "label=$LABEL build=fail det=na" > "$OUT"; exit 0; }
  CLI=./target/release/sim-cli
  # P1: save/load + process reproducibility (two threads=1 processes).
  $CLI new --seed 7 --scale 70,400 --threads 1 --out artifacts/d1.bin --no-metrics >/dev/null 2>&1
  cp artifacts/d1.bin artifacts/d2.bin
  $CLI run-to 4000 --in artifacts/d1.bin --no-metrics >/dev/null 2>&1
  $CLI run-to 4000 --in artifacts/d2.bin --no-metrics >/dev/null 2>&1
  P1=broken; cmp -s artifacts/d1.bin artifacts/d2.bin && P1=ok
  # P2: thread-count independence (semantic fingerprint, threads=1 vs threads=4).
  $CLI new --seed 7 --scale 70,400 --threads 4 --out artifacts/d4.bin --no-metrics >/dev/null 2>&1
  $CLI run-to 4000 --in artifacts/d4.bin --no-metrics >/dev/null 2>&1
  world_fp artifacts/d1.bin > artifacts/fp1.txt 2>/dev/null
  world_fp artifacts/d4.bin > artifacts/fp4.txt 2>/dev/null
  P2=broken; cmp -s artifacts/fp1.txt artifacts/fp4.txt && P2=ok
  if [ "$P1" = "ok" ] && [ "$P2" = "ok" ]; then DET=ok
  elif [ "$P1" != "ok" ]; then DET=broken-p1
  else DET=broken-p2; fi
  echo "label=$LABEL build=ok det=$DET" > "$OUT"
  exit 0
fi

# --run-one <wtdir> <seed> <horizon> <outfile> <label> <timeout_secs>
# A runaway population explosion (the eval-time trap) makes run-to crawl; a
# portable watchdog kills it. We classify by the run-to exit code: SIGKILL(137)
# = timeout/explosion; other non-zero = infra ERR; 0 = trust + parse (do NOT
# re-derive completion from a possibly-mid-write world file).
if [ "${1:-}" = "--run-one" ]; then
  shift; WT="$1"; SEED="$2"; HZN="$3"; OUT="$4"; LABEL="$5"; TO="${6:-1800}"
  W="$WT/artifacts/c-$SEED.bin"
  cd "$WT" || { echo "label=$LABEL seed=$SEED pop=ERR plant=NA prey=NA aeff=NA misa=NA slope=NA" > "$OUT"; exit 0; }
  if ! ./target/release/sim-cli new --seed "$SEED" --threads 1 --out "$W" >/dev/null 2>&1; then
    echo "label=$LABEL seed=$SEED pop=ERR plant=NA prey=NA aeff=NA misa=NA slope=NA" > "$OUT"; exit 0
  fi
  ./target/release/sim-cli run-to "$HZN" --in "$W" --threads 1 >/dev/null 2>&1 & RPID=$!
  ( sleep "$TO"; kill -9 "$RPID" 2>/dev/null ) & WD=$!
  wait "$RPID" 2>/dev/null; RC=$?
  kill "$WD" 2>/dev/null; wait "$WD" 2>/dev/null
  if [ "$RC" = "137" ]; then
    echo "label=$LABEL seed=$SEED pop=EXPLODE plant=NA prey=NA aeff=NA misa=NA slope=NA" > "$OUT"; exit 0
  elif [ "$RC" != "0" ]; then
    echo "label=$LABEL seed=$SEED pop=ERR plant=NA prey=NA aeff=NA misa=NA slope=NA" > "$OUT"; exit 0
  fi
  PILL="$(./target/release/sim-cli pillars --in "$W" --text 2>/dev/null)"
  STATE="$(./target/release/sim-cli state --in "$W" --text 2>/dev/null)"
  POP="$(printf '%s\n' "$STATE" | grep -m1 'population' | sed -E 's/.*population *= *([0-9]+).*/\1/')"
  [ -z "$POP" ] && POP=0
  METRICS="$(printf '%s\n' "$PILL" | awk '
    {for(i=1;i<=NF;i++){
      if($i=="plant_consumption_rate")plant=$(i+1);
      if($i=="prey_consumption_rate")prey=$(i+1);
      if($i=="action_effectiveness")aeff=$(i+1);
      if($i=="mi_sa")misa=$(i+1);
      if($i=="learning_slope")slope=$(i+1);
    }}
    END{printf "plant=%s prey=%s aeff=%s misa=%s slope=%s",
      (plant==""?"NA":plant),(prey==""?"NA":prey),(aeff==""?"NA":aeff),
      (misa==""?"NA":misa),(slope==""?"NA":slope)}')"
  echo "label=$LABEL seed=$SEED pop=$POP $METRICS" > "$OUT"
  exit 0
fi

# ----- main ------------------------------------------------------------------
BASE_REF=""; BRANCHES=""; CAP=10; BUILD_CAP=2
SEEDS="7,42,123,2026"; SCREEN_SEED=7; SCREEN_TO=200000; CONFIRM_TO=500000
EXPLODE_POP=40000; MIN_COMMON=2; SCRATCH=""; KEEP=0; DO_SCREEN=1
SCREEN_TO_SECS=900; CONFIRM_TO_SECS=1800
while [ $# -gt 0 ]; do
  case "$1" in
    --base-ref) BASE_REF="$2"; shift 2;;
    --branches) BRANCHES="$2"; shift 2;;
    --cap) CAP="$2"; shift 2;;
    --build-cap) BUILD_CAP="$2"; shift 2;;
    --seeds) SEEDS="$2"; shift 2;;
    --screen-seed) SCREEN_SEED="$2"; shift 2;;
    --screen-to) SCREEN_TO="$2"; shift 2;;
    --confirm-to) CONFIRM_TO="$2"; shift 2;;
    --explode-pop) EXPLODE_POP="$2"; shift 2;;
    --min-common) MIN_COMMON="$2"; shift 2;;
    --aeff-drop) shift 2;;   # deprecated/ignored (no aeff screen filter)
    --scratch) SCRATCH="$2"; shift 2;;
    --keep) KEEP=1; shift;;
    --no-screen) DO_SCREEN=0; shift;;
    *) echo "unknown arg: $1" >&2; exit 2;;
  esac
done
[ -z "$BASE_REF" ] && { echo "ERROR: --base-ref required" >&2; exit 2; }
[ -z "$BRANCHES" ] && { echo "ERROR: --branches required" >&2; exit 2; }

REPO="$(git rev-parse --show-toplevel)"
[ -z "$SCRATCH" ] && SCRATCH="$REPO/artifacts/evalq/run-$$"
case "$SCRATCH$REPO" in *[[:space:]\'\"\\]*)
  echo "ERROR: repo/scratch path contains whitespace or quotes; xargs job lines would break: $SCRATCH" >&2; exit 2;; esac
mkdir -p "$SCRATCH/wt" "$SCRATCH/build" "$SCRATCH/screen" "$SCRATCH/confirm"
SEEDS_SP="$(printf '%s' "$SEEDS" | tr ',' ' ')"

# Shared sccache compile cache across all worktree builds (fix #3).
if command -v sccache >/dev/null 2>&1; then export RUSTC_WRAPPER=sccache; fi

# label list = __base__ + one per branch.
LABELS="__base__"
echo "__base__ $BASE_REF" > "$SCRATCH/labels.txt"
for b in $BRANCHES; do L="$(sanitize "$b")"; LABELS="$LABELS $L"; echo "$L $b" >> "$SCRATCH/labels.txt"; done
label_branch() { awk -v l="$1" '$1==l{print $2; exit}' "$SCRATCH/labels.txt"; }
label_wt() { echo "$SCRATCH/wt/$1"; }

echo ">>> [1/4] build (base first to warm sccache, then ${BUILD_CAP}-way; sccache=$([ -n "${RUSTC_WRAPPER:-}" ] && echo on || echo off))" >&2
# Build base FIRST (serial) so the shared dep cache is warm before fan-out.
"$SELF" --build-one "$BASE_REF" "$(label_wt __base__)" "__base__" "$SCRATCH/build/__base__.txt"
: > "$SCRATCH/buildjobs.txt"
for L in $LABELS; do
  [ "$L" = "__base__" ] && continue
  echo "$(label_branch "$L") $(label_wt "$L") $L $SCRATCH/build/$L.txt" >> "$SCRATCH/buildjobs.txt"
done
[ -s "$SCRATCH/buildjobs.txt" ] && xargs -P "$BUILD_CAP" -L 1 "$SELF" --build-one < "$SCRATCH/buildjobs.txt"
cat "$SCRATCH"/build/*.txt > "$SCRATCH/build.all" 2>/dev/null
echo "    build results:" >&2; sed 's/^/      /' "$SCRATCH/build.all" >&2

# usable = built ok AND determinism ok (det-broken is rejected outright).
USABLE=""
for L in $LABELS; do
  if grep -q "^label=$L build=ok det=ok" "$SCRATCH/build/$L.txt" 2>/dev/null; then USABLE="$USABLE $L"; fi
done
if ! echo "$USABLE" | grep -qw "__base__"; then echo "ERROR: base_ref failed build/determinism; aborting" >&2; exit 1; fi

# ----- screen (coarse disaster filter only: collapse / explosion) ------------
SURV="__base__"
if [ "$DO_SCREEN" = "1" ]; then
  echo ">>> [2/4] screen (seed $SCREEN_SEED → $SCREEN_TO, cap $CAP; disaster filter only)" >&2
  : > "$SCRATCH/screenjobs.txt"
  for L in $USABLE; do
    echo "$(label_wt "$L") $SCREEN_SEED $SCREEN_TO $SCRATCH/screen/$L.txt $L $SCREEN_TO_SECS" >> "$SCRATCH/screenjobs.txt"
  done
  [ -s "$SCRATCH/screenjobs.txt" ] && xargs -P "$CAP" -L 1 "$SELF" --run-one < "$SCRATCH/screenjobs.txt"
  cat "$SCRATCH"/screen/*.txt > "$SCRATCH/screen.all" 2>/dev/null
  echo "    screen results:" >&2; sed 's/^/      /' "$SCRATCH/screen.all" >&2
  SURV="$(python3 - "$SCRATCH/screen.all" "$EXPLODE_POP" <<'PY'
import sys,re
path,explode=sys.argv[1],float(sys.argv[2])
surv=['__base__']
for ln in open(path):
    d=dict(re.findall(r'(\w+)=(\S+)',ln))
    L=d.get('label')
    if not L or L=='__base__': continue
    pop=d.get('pop')
    if pop in ('ERR','EXPLODE'): continue          # infra error or runaway
    try: p=float(pop)
    except: continue
    if p<=0: continue                              # collapse
    if p>explode: continue                         # explosion (cap-bound trap)
    surv.append(L)
print(' '.join(surv))
PY
)"
  echo "    survivors → $SURV" >&2
else
  SURV="$USABLE"
fi

# ----- confirm ---------------------------------------------------------------
echo ">>> [3/4] confirm (seeds $SEEDS → $CONFIRM_TO, cap $CAP)" >&2
: > "$SCRATCH/confirmjobs.txt"
for L in $SURV; do
  for S in $SEEDS_SP; do
    echo "$(label_wt "$L") $S $CONFIRM_TO $SCRATCH/confirm/${L}__${S}.txt $L $CONFIRM_TO_SECS" >> "$SCRATCH/confirmjobs.txt"
  done
done
[ -s "$SCRATCH/confirmjobs.txt" ] && xargs -P "$CAP" -L 1 "$SELF" --run-one < "$SCRATCH/confirmjobs.txt"
cat "$SCRATCH"/confirm/*.txt > "$SCRATCH/confirm.all" 2>/dev/null

# ----- report ----------------------------------------------------------------
echo ">>> [4/4] report" >&2
touch "$SCRATCH/screen.all"
python3 - "$SCRATCH/confirm.all" "$SEEDS" "$SCRATCH/labels.txt" "$SCRATCH/build.all" "$SCRATCH/screen.all" "$EXPLODE_POP" "$MIN_COMMON" "$SCRATCH/summary.json" <<'PY'
import sys,re,json
confirm,seeds_s,labels_path,build_path,screen_path,explode,min_common,outp=sys.argv[1:9]
explode=float(explode); min_common=int(min_common)
seeds=[s for s in seeds_s.split(',') if s]
labelbranch={}
for ln in open(labels_path):
    p=ln.split()
    if len(p)>=2: labelbranch[p[0]]=p[1]
def f(x):
    try: return float(x)
    except: return None
def parse(path):
    out={}
    try:
        for ln in open(path):
            d=dict(re.findall(r'(\w+)=(\S+)',ln)); out.setdefault(d.get('label'),[]).append(d)
    except FileNotFoundError: pass
    return out
build={k:v[0] for k,v in parse(build_path).items() if k}
screen={k:v[0] for k,v in parse(screen_path).items() if k}
# confirm rows: label -> seed -> dict
rows={}
for ln in open(confirm):
    d=dict(re.findall(r'(\w+)=(\S+)',ln))
    if 'label' not in d or 'seed' not in d: continue
    rows.setdefault(d['label'],{})[d['seed']]=d
PILLARS=['plant','prey','aeff','misa','slope']
NAMES={'plant':'plant_consumption_rate','prey':'prey_consumption_rate','aeff':'action_effectiveness','misa':'mi_sa','slope':'learning_slope'}
def survived(d):
    p=f(d.get('pop')); return p is not None and p>0 and f(d.get('aeff')) is not None
base=rows.get('__base__',{})
base_surv=set(s for s in seeds if s in base and survived(base[s]))
def mean(vals):
    vals=[v for v in vals if v is not None]
    return round(sum(vals)/len(vals),6) if vals else None
out={'base_ref':labelbranch.get('__base__'),'seeds':seeds,'base_surviving':sorted(base_surv),
     'branches':[], 'dropped':[]}
# dropped = build-failed / determinism-broken / screened-out (so nothing vanishes)
for L,br in labelbranch.items():
    if L=='__base__': continue
    bd=build.get(L,{})
    if bd.get('build') not in ('ok',None) or not bd:
        out['dropped'].append({'label':L,'branch':br,'stage':'build','reason':bd.get('build','missing')}); continue
    det=bd.get('det')
    if det and det!='ok':
        out['dropped'].append({'label':L,'branch':br,'stage':'determinism','reason':det,
                               'note':'determinism broken — rejected outright'}); continue
    if L in screen and L not in rows:  # screened out (built+det ok but not confirmed)
        sd=screen[L]; pop=sd.get('pop'); p=f(pop)
        reason=('infra-error' if pop=='ERR' else 'explosion' if (pop=='EXPLODE' or (p and p>explode))
                else 'collapse' if (p is not None and p<=0) else 'screened')
        out['dropped'].append({'label':L,'branch':br,'stage':'screen','reason':reason,
                               'screen_pop':pop,'screen_aeff':sd.get('aeff')})
for L,seedmap in rows.items():
    if L=='__base__': continue
    surv=set(s for s in seeds if s in seedmap and survived(seedmap[s]))
    per_seed={}
    for s in seeds:
        if s in seedmap:
            dd=seedmap[s]
            per_seed[s]=('extinct' if not survived(dd) else
                         {k:f(dd.get(k)) for k in PILLARS}|{'pop':f(dd.get('pop'))})
        else: per_seed[s]='missing'
    means={NAMES[k]:mean([f(seedmap[s].get(k)) for s in surv if s in seedmap]) for k in PILLARS}
    common=sorted(base_surv & surv)
    clean={}; per_seed_delta={s:{} for s in common}
    for k in PILLARS:
        ds=[]
        for s in common:
            bv=f(base[s].get(k)); xv=f(seedmap[s].get(k))
            if bv is not None and xv is not None:
                ds.append(xv-bv); per_seed_delta[s][NAMES[k]]=round(xv-bv,6)
        clean[NAMES[k]]=round(sum(ds)/len(ds),6) if ds else None
    out['branches'].append({
        'label':L,'branch':labelbranch.get(L,L),
        'determinism':build.get(L,{}).get('det','?'),
        'n_surviving':len(surv),'surviving':sorted(surv),
        'rescued_seeds':sorted(surv-base_surv),'lost_seeds':sorted(base_surv-surv),
        'common_survivors':common,
        'sufficient_n': len(common)>=min_common,
        'clean_delta_vs_base_common_survivors':clean,   # GATE ON THIS (only if sufficient_n)
        'per_seed_delta_vs_base':per_seed_delta,         # eyeball consistency, not just the mean
        'metrics_mean_n4_DO_NOT_GATE':means,
        'per_seed':per_seed,
    })
json.dump(out,open(outp,'w'),indent=2)
print(json.dumps(out,indent=2))
import sys as _s
print('\n%-28s det  n  resc  nOK  Δslope     Δaeff     Δmisa     Δplant    Δprey'%'branch',file=_s.stderr)
for b in out['branches']:
    c=b['clean_delta_vs_base_common_survivors']
    g=lambda k:('   NA   ' if c.get(k) is None else '%+.5f'%c[k])
    print('%-28s %-4s %d  %-4s %-4s %s %s %s %s %s'%(
        b['label'][:28], b['determinism'], b['n_surviving'],
        (','.join(b['rescued_seeds']) or '-'), ('y' if b['sufficient_n'] else 'LOW'),
        g('learning_slope'),g('action_effectiveness'),g('mi_sa'),
        g('plant_consumption_rate'),g('prey_consumption_rate')),file=_s.stderr)
if out['dropped']:
    print('\n  dropped: '+', '.join('%s(%s:%s)'%(d['label'][:24],d['stage'],d['reason']) for d in out['dropped']),file=_s.stderr)
PY

# ----- cleanup ---------------------------------------------------------------
if [ "$KEEP" = "0" ]; then
  for L in $LABELS; do git worktree remove --force "$(label_wt "$L")" >/dev/null 2>&1; done
  git worktree prune >/dev/null 2>&1
fi
echo ">>> done. summary: $SCRATCH/summary.json" >&2
