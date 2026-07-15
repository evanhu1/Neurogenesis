# 2026-07-14-unified-consume-action: split versus unified energy acquisition

Status: proposed

## Question

Does replacing separate `Eat` and `Attack` outputs with one target-dispatched
`Consume` output make the evolutionary learning problem more efficient without
removing meaningful ecological strategy?

## Hypothesis

`Eat` and `Attack` currently address the same adjacent cell and are applicable
to disjoint target classes. Their separation therefore adds a neural
verb-to-target coordination problem without adding a choice between different
consequences in the same state. A unified action should improve early learning
speed and reduce failed contingent actions. It should not by itself eliminate
the distinction between foraging and predation, which still depends on sensing,
navigation, pursuit, and the source of acquired energy.

The hypothesis is falsified if unified control merely increases indiscriminate
attacks, extinction, or seed variance; reduces held-out competence or trophic
diversity; or provides no measurable efficiency benefit.

## Treatment semantics

Control retains five explicit outputs: turn left, turn right, forward, eat, and
attack. Treatment has four: turn left, turn right, forward, and consume. Idle
remains implicit in both arms.

In the treatment, `Consume` resolves against the adjacent forward cell:

- plant: transfer the plant's energy;
- organism: pay the configured attack-attempt cost, then transfer the configured
  amount from victim to attacker when predation is enabled;
- wall or empty cell: fail.

No energy values, sensor inputs, target ranges, tick ordering, or evaluation
rules change. The treatment's fully connected seed topology has 14 x 4 = 56
connections rather than the control's 14 x 5 = 70.

## Contract

- Canonical config: `config/world.toml`
- Control: current split `Eat` and `Attack`
- Treatment: unified target-dispatched `Consume`
- Evolutionary seeds: `7,17,27`
- Training world seeds: `11,29,47,61`
- Held-out world seeds: `101,131,151,181,211,241,271,311`
- Population: 24 genomes
- Generations: 20 (`0..19`)
- Episode horizon: 500 ticks, extended for both arms only if end survival shows
  material censoring
- Evaluation: contemporary-only pairwise, eight opponent exposures and 32
  scored cases per genome
- World: canonical 50x50, 100 founders, 20% plant tiles, predation enabled
- Selection, mutation, crossover, speciation, and worker count: identical
- Artifact roots:
  - `artifacts/research/runs/active/2026-07-14-unified-consume-action/control/`
  - `artifacts/research/runs/active/2026-07-14-unified-consume-action/treatment/`

This is a code-level paired experiment. Build each arm from an explicitly
recorded revision/worktree and save the resolved action contract in its result.

## Measurements

Primary:

- Held-out common-panel checkpoint strength at generations 0, 5, 10, 15, and
  19.
- Chronological later-versus-earlier win fraction and strength slope.
- Generation at which each run first crosses fixed absolute-survival thresholds.

Secondary:

- Population mean, median, and best absolute survival.
- Per-opponent score dispersion and final-checkpoint retention.
- Action effectiveness and contingent-action failure fraction.
- Plant energy, stolen prey energy, kills, and continuous plant/prey intake
  fractions.
- Extinction/end-survivor fraction, gross energy acquired, and spatial coverage.
- Species count, structural innovations, enabled connections, and evaluation
  wall time.

Integrity:

- Identical seed/case schedules and simulator-world counts across arms.
- Deterministic replay per arm.
- Closed energy ledger.
- Direct behavioral inspection of representative forager, predator, omnivore,
  and degenerate policies.

## Decision rule

Prefer unified `Consume` for the default action space only if:

1. it improves aggregate held-out chronological strength, time-to-threshold, or
   final competence without relying on one evolutionary seed;
2. it reduces contingent-action waste as predicted;
3. it does not produce worse extinction, materially greater seed variance, or
   a narrower trophic repertoire; and
4. inspected organisms show coherent target acquisition rather than automatic
   attack spam or another observer-metric exploit.

If competence is statistically/behaviorally indistinguishable but the unified
arm is simpler and no adverse gate fails, prefer it on representation-minimality
grounds and explicitly classify the result as simplification, not improved
evolution.

## Commands and provenance

To be written after the treatment exists. Each arm must persist its exact Git
revision, resolved canonical config, action list, result schema, release command,
stdout/stderr, and checksums under its artifact root.

## Result

Not run.

## Interpretation and next decision

Not run.
