# 2026-07-14-compositional-motor-commands: categorical versus factorized action control

Status: completed; predation hypothesis rejected, composition retained as an
experimental representation

## Question

Does allowing an organism to turn, move, and interact in the same tick remove
the current one-command pursuit bottleneck and make adaptive predation easier to
discover without degrading general survival competence?

## Hypothesis

The categorical controller forces an organism to spend a whole tick on exactly
one of turn, forward, eat, or attack. Because movement resolves before attack,
a prey organism that moves out of the target cell cannot be followed and struck
on that tick. The treatment reinterprets the same five neural outputs as three
independently sampled command groups: orientation (none/left/right), locomotion
(none/forward), and interaction (none/eat/attack). A predator may therefore
emit forward plus attack, enter a cell vacated simultaneously by moving prey,
and attack from its new position.

The mechanism is falsified if the treatment does not increase selective attack
hits, distinct victims, or profitable predation; if it only increases attack
spam or mutual-combat extinction; or if any predation gain comes with worse
held-out survival and chronological progress.

## Treatment semantics

Control retains one categorical draw over implicit idle plus the five explicit
actions. Treatment uses the same five logits and the same temperature, but
draws independently from:

- orientation: implicit none, turn left, turn right;
- locomotion: implicit none, forward;
- interaction: implicit none, eat, attack.

Every tick has one fixed global order in both arms: choose commands from the
same snapshot, apply orientation, resolve movement simultaneously, then resolve
interaction. In the treatment only, a move may enter an organism-occupied cell
when that occupant also wins a simultaneous move out of it; dependency cycles
and swaps are legal. Interactions target the organism's forward cell after
movement. Energy, sensors, network topology, logits, mutation, selection, and
the ecology are unchanged.

## Contract

- Code base revision: `9c604c581509f5e7ebc2c06dd95035b752b3d266`
- Batch source patch SHA-256:
  `ce753b08b53b1bb617be7135af1fb026e680a8404f5365053b8b076cd8594280`
- Canonical config: `config/world.toml`
- Control: `compositional_actions_enabled=false`
- Treatment: `compositional_actions_enabled=true`
- Shared ecology override: `attack_attempt_cost=5`; transfer remains 40
- Evolutionary seeds: `7,17,27`
- Training world seeds: `11,29,47`
- Held-out world seeds: `101,131,151,181`
- Population: 48 genomes
- Generations: 20 (`0..19`)
- Episode horizon: 1,000 ticks
- Evaluation: 12 three-lineage memberships per genome, 24 opponent exposures,
  36 scored cases per genome
- Founders/world: 102 (34 per lineage), 50x50
- Objective: survival times relative advantage, mean aggregation (`cvar=1`)
- Evaluation workers: 14 machine-wide, allocated `5,5,4` across evolutionary
  seeds by `batch`
- Crossplay checkpoints: `0,5,10,15,19`, held-out pairwise assay at 1,000 ticks;
  extended to 2,000 and 4,000 when treatment remained censored
- Estimated simulator work per evolutionary seed: 11,520 worlds and 11,520,000
  world ticks; per arm: 34,560 worlds and 34,560,000 world ticks
- Artifact directory:
  `artifacts/research/runs/completed/2026-07-14-compositional-motor-commands/`

Everything else inherits the canonical configuration and default NEAT
parameters.

## Measurements

Primary endpoint: treatment-minus-control difference in successful attack hits
and privately profitable net attack energy among generation champions and the
population tail (`15..19`), aggregated across all three evolutionary seeds.

Competence endpoints are absolute survival, candidate alive ticks, final
survivor fraction, fitness, held-out chronological later-versus-earlier win
fraction, and final checkpoint crossplay strength. Behavioral endpoints are
attack attempts, misses, hits, kills, distinct victims, repeat-hit fraction,
plant/prey energy intake, action fractions, mean emitted commands per organism
tick, multi-command tick fraction, spatial coverage, and trophic roles.

Integrity checks are matched plans and case schedules, closed energy ledgers,
deterministic smoke replay, population mean/median alongside champions, seed
agreement, end-survival censoring, and inspection of representative policies.

## Decision rule

Prefer compositional control only if at least two of three evolutionary seeds
show more attack hits or stronger net attack balance in the preregistered tail,
the aggregate treatment produces at least one coherent profitable predator,
and held-out survival/crossplay competence does not materially regress. Reject
the mechanism if increased attacking is dominated by misses, reciprocal combat,
extinction, one-seed luck, or lower competence. If commands compose but
predation remains absent, conclude that categorical motor exclusivity was not
the active bottleneck.

## Commands and provenance

The two batch commands differed only in the final flag:

```text
cli batch --out-dir artifacts/research/runs/active/2026-07-14-compositional-motor-commands \
  --experiment <control|treatment> --seeds 7,17,27 --total-workers 14 -- \
  --population 48 --generations 20 --horizon 1000 \
  --lineages-per-world 3 --memberships-per-genome 12 \
  --world-seeds 11,29,47 --founders 102 --world-width 50 \
  --objective survival_times_relative_advantage --cvar 1 \
  --set attack_attempt_cost=5 \
  --set compositional_actions_enabled=<false|true>
```

Both manifests validated schema 23 and the exact within-arm contract. The
batch executable SHA-256 was
`cbcd70d1dcdd2e1cfde7ab704c7413b7f01513db13cac23653ac3e317cf76531`.
The release treatment smoke was run twice and produced byte-identical result
SHA-256
`ae2ad490b43452e0cfb03e506c3e0d5ad4d90d8358f21f3afff359f34463d235`.
Categorical generation-zero competence and energy fields exactly reproduced
the prior cost-5 experiment on all three seeds.

Training summaries used inclusive tail `15..19`. Crossplay used held-out world
seeds `101,131,151,181`, checkpoints `0,5,10,15,19`, and horizons 1,000, 2,000,
and 4,000. Full commands, manifests, logs, results, champion worlds, and
crossplay cells are in the artifact directory.

## Result

Composition caused a very large, seed-robust survival gain but did not make
predation privately profitable. Tail champion means were:

| Arm | Seed | hits | attack precision | net attack energy | best survival | population survival | end survivors | plant energy | commands/tick |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| categorical | 7 | 0.07 | 0.007 | -74.7 | 0.324 | 0.281 | 0.000 | 2,706 | 0.979 |
| compositional | 7 | 0.43 | 0.036 | -330.8 | 0.735 | 0.580 | 0.491 | 23,241 | 2.607 |
| categorical | 17 | 0.05 | 0.001 | -285.9 | 0.345 | 0.283 | 0.005 | 3,423 | 1.000 |
| compositional | 17 | 0.02 | 0.004 | -211.9 | 0.768 | 0.591 | 0.556 | 24,741 | 2.755 |
| categorical | 27 | 26.45 | 0.325 | -255.1 | 0.310 | 0.267 | 0.002 | 2,356 | 0.991 |
| compositional | 27 | 69.82 | 0.195 | -1,413.9 | 0.687 | 0.517 | 0.486 | 23,442 | 2.701 |

Seed 27 emitted 4.4 times as many tail attacks under composition and therefore
landed more hits, but precision fell from 0.325 to 0.195 and private attack
balance became much worse. Seeds 7 and 17 remained effectively nonpredatory.
No treatment seed evolved profitable tail predation, so the primary decision
gate failed.

The competence effect was nevertheless real. At generation 19 the treatment
champions emitted 2.61, 2.79, and 2.77 commands per organism tick; 95–98% of
their ticks contained multiple commands. They covered 96–100% of habitable
space and captured 48–51% of actionable plant supply. Their command allocation
was dominated by turning plus Forward plus Eat. A direct decision inspection,
for example, showed `Forward + Eat` emitted with no food visible. The controller
had learned that a free Eat attempt has no downside once movement no longer has
to be sacrificed for it.

Held-out chronological results were unusually consistent:

| Arm | Seed | later wins at 1,000 | later wins at 4,000 | final beats earlier at 4,000 | gen-19 survival area at 4,000 | gen-19 end survivors at 4,000 |
|---|---:|---:|---:|---:|---:|---:|
| categorical | 7 | 9/10 | 9/10 | 4/4 | 0.079 | 0.000 |
| categorical | 17 | 9/10 | 9/10 | 3/4 | 0.083 | 0.000 |
| categorical | 27 | 9/10 | 9/10 | 3/4 | 0.078 | 0.000 |
| compositional | 7 | 10/10 | 10/10 | 4/4 | 0.477 | 0.392 |
| compositional | 17 | 10/10 | 10/10 | 4/4 | 0.544 | 0.467 |
| compositional | 27 | 10/10 | 10/10 | 4/4 | 0.487 | 0.422 |

Treatment population mean and median competence continued improving through
the `15..19` tail in all seeds. Champion slopes were negative in seeds 7 and 17
because individual champions fluctuated near a newly high regime; held-out
crossplay still ordered every later checkpoint above every earlier checkpoint.
Even 4,000 ticks remained censored for treatment, indicating durable renewable
foraging rather than merely delayed extinction.

## Interpretation and next decision

The categorical action bottleneck was real, but it was not specifically the
active bottleneck for predation. Removing it benefited plant acquisition far
more than pursuit. The structural reason is now precise: under categorical
control, Eat has an opportunity cost because it replaces movement; under
composition, a missed Eat is free, so `move + Eat` every tick is rational. The
treatment therefore converts target-conditioned foraging into high-throughput
blanket harvesting and further strengthens the already dominant plant niche.

Do not enable this treatment as the canonical default yet. Retain the feature
flag and the simultaneous vacated-cell movement implementation, because they
make pursuit representable and produced strong cumulative competence. The next
causal experiment should preserve composition while restoring an interaction
tradeoff—most cleanly, charge a small cost for every interaction attempt, with
the existing attack cost decomposed into the same base interaction cost plus an
attack-specific increment. The gate should require lower Eat spam, retained
held-out survival, and improved attack precision/net balance. Only after that
should the treatment be extended beyond 30 generations to test whether its
perfect 20-generation chronological ordering persists rather than merely
creating a larger early foraging jump.

Verification: `cargo check --workspace`, `make fmt`, and `make lint` passed.
The release deterministic smoke passed. `cargo test --workspace` ran 15 of 16
world-sim tests successfully but exposed one pre-existing stale assertion:
`lethal_attack_conserves_energy_without_spawning_food` expects final predator
energy 59 even though its fixture now charges the canonical 10-energy attack
attempt cost; the engine correctly returns 49. The test was not changed because
the repository instructions reserve test maintenance for the human author.
