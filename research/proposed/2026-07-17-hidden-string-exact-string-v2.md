# 2026-07-17-hidden-string-exact-string-v2: calibrated efficient replication

Status: superseded before execution by the smooth exact-string v3 selection
contract. The hard exact endpoint and causal gates remain valid, but hard exact
rate alone was too sparse to serve as the evolutionary search signal.

## Question

Can NEAT robustly evolve a recurrent brain whose runtime reward updates learn
an unseen four-symbol string under the strict exact-string objective?

## Contract

- Task: `hidden_string_adaptation_v2`.
- Alphabet: `a` through `h`; target length four; no sensory input.
- Deterministic founder: one recurrent hidden unit and eight hidden-to-action
  edges. `end` is excluded from structural mutation.
- Attempts: 32.
- Training: 1,024 fixed targets x two rollout seeds, calibrated as the smallest
  contract preserving full-evaluator population rankings.
- Training measurement: one frozen probe after attempt 32. Selection fitness is
  final exact-string rate only.
- Development: 256 disjoint targets x one rollout, primary treatment only,
  after every 25th generation and on the final generation. Report probes are
  `[0,8,16,32]`.
- Sealed: 1,024 disjoint targets x two rollouts, once after final-winner
  selection. Primary probes are `[0,8,16,32]`; shuffled-reward and
  reset-weights controls receive only the final probe.
- Panels are disjoint and hash-shuffled with exact per-position symbol balance.
  Training and sealed distinct-symbol counts are `96/512/416` for two/three/four
  distinct symbols; development counts are `24/128/104`.
- Evolutionary seeds: 211, 307, 401.
- Population: 64.
- Generations: 1,000.
- Evaluation workers: four per process; three concurrent processes consume 12
  of the machine's 14 cores.
- Artifact directory:
  `artifacts/research/runs/active/2026-07-17-hidden-string-exact-string-v2-1000g/`

The calibration and determinism evidence is recorded in
[`2026-07-17-hidden-string-evaluator-optimization.md`](../archive/experiments/2026-07-17-hidden-string-evaluator-optimization.md).

## Decision rule

The experiment passes only if at least two of three seeds satisfy all of:

1. sealed final exact-string rate at least 0.20;
2. sealed exact-string gain at least 0.15 from the zero-attempt probe;
3. shuffled-reward final exact-string rate at most 0.02;
4. reset-weights final exact-string rate at most 0.02.

## Commands

```bash
./target/release/cli hidden-string plan \
  --seed SEED --population 64 --generations 1000 --workers 4

./target/release/cli hidden-string \
  --seed SEED --population 64 --generations 1000 --workers 4 \
  --out-dir artifacts/research/runs/active/2026-07-17-hidden-string-exact-string-v2-1000g
```

## Result

Not run.
