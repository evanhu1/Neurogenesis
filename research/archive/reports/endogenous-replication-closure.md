# Endogenous ecological replication: reproduction-only closure

Status: **blocked as a standalone open-endedness driver**. This is a source- and
artifact-grounded closure, not a new positive result. No simulation behavior is
changed by this document.

## Verdict

Restoring energy-conserving in-world reproduction is necessary for any natural
late population renewal, but it is not a materially new open-endedness
mechanism in this repository. The archived engine already had all of the
following at once:

- a brain-selected `Reproduce` action;
- maturity and gestation;
- a live-parent energy debit at conception;
- the identical energy amount as the child's starting energy;
- deterministic placement and canonical tick ordering;
- genome mutation at birth and multi-generation lineages.

That substrate was present throughout the archived 21-experiment campaign. It
did not prevent stable equilibria, bounded competence transitions, complexity
bloat, or construction spam. Reintroducing it alone would therefore replay a
falsified mechanism family. The smallest scientifically distinct reopen must
couple conserved reproduction to a new, payoff-bearing information interaction
whose realized problem structure can grow. Public proof-carrying energy caches
or recursively composable public artifacts are examples; automatic or clonal
births by themselves are not.

This conclusion is deliberately narrower than “reproduction can never help.”
The historical campaign used periodic founder injection, and its old engine did
not emit the current fail-closed per-tick energy ledger. It therefore cannot
serve as the requested clean treatment-versus-frozen causal trial. It is enough
to reject the claim that restoring reproduction alone is a new candidate
algorithm.

## Current engine boundary

Current `Simulation::tick` explicitly creates no births. After intents and
commit it runs only post-commit plasticity in the former Spawn phase and returns
an empty `spawned` vector (`sim-core/src/turn/mod.rs`). `ActionType` has six
values including Idle and no `Reproduce` action (`sim-types/src/lib.rs`).
Lifecycle is energy-only; the serialized maturity, gestation, and maximum-age
genes no longer control birth or death (`sim-core/src/turn/lifecycle.rs`).

The current fail-closed ledger captures organism and food energy before the
tick, internal consumption and predation transfers, explicit losses, and plant
spawn energy. It has no typed birth debit/credit because no birth path exists.
An in-tick parent-to-child transfer would leave total organism energy unchanged,
but an auditable restoration would still need paired birth-debit and
birth-credit facts plus the reproduction event; total closure alone would not
prove provenance.

The only current renewal helper is evaluator-owned opponent respawn. It builds a
fresh organism with `starting_energy` and reports that quantity as
`energy_injected`; no live organism or endogenous escrow is debited. Round 10
correctly treated that path as an external source, not reproduction.

## Historical mechanics prove this is not a new lever

Commit `b28c5df` is the end of the archived 21-experiment campaign. Its
reproduction path is mechanically explicit:

1. `ActionType::Reproduce` is a contingent brain action
   (`sim-types/src/lib.rs` at that commit).
2. The reproduction phase requires maturity and sufficient parent energy, then
   executes `organism.energy -= transfer_energy` at conception
   (`sim-core/src/turn/reproduction.rs`).
3. With the baseline `gestation_ticks = 2`,
   `offspring_transfer_energy = 100 + 100 * gestation_ticks = 300`.
4. On successful completion the spawn request carries that exact stored amount
   as `offspring_starting_energy`.
5. The spawn phase mutates the inherited genome and constructs the child with
   `starting_energy_override: Some(offspring_starting_energy)`
   (`sim-core/src/spawn/organisms.rs`).

Thus a successful birth was an internal 300-unit parent-to-child transfer, not a
300-unit mint. A blocked birth or parent death after conception destroyed the
already-debited investment, so those failure paths were dissipative rather than
generative. Birth mutation and the organism's `Reproduce` decision made this a
strictly stronger evolutionary mechanism than evaluator-triggered clonal
renewal.

The old canonical order was Lifecycle -> Intents -> Reproduction trigger ->
Move Resolution -> Commit/gestation completion -> Age -> Spawn/mutation ->
Plasticity -> consistency/metrics. The current order intentionally removed the
reproduction trigger and birth portions in `a5d3c81`; that cut touched 41 files
across engine, metrics, CLI/server, and web wire surfaces. A faithful restoration
is not a small evaluator switch.

### Important archival confound

The archived `sim-evaluation/config.toml` also requested 200 fresh seed-genome
injections every 100 ticks. Those founders were evaluator/world sources with
full starting energy. Later organism inspection found them to be inert gen-0
re-seeds (median age about 100, zero consumptions, empty brains) rather than the
breeding lineages, but they still invalidate a claim of fully endogenous energy
closure for the old campaign. A clean rerun could disable them with:

```bash
./target/release/sim-cli new --seed 7 \
  --set periodic_injection_interval_turns=0 \
  --set periodic_injection_count=0 \
  --out artifacts/research/runs/completed/open-ended/endogenous-reproduction/seed7.bin
```

That rerun would test demographic renewal, but without a materially new
information interaction it would not reopen the open-endedness route.

## Tail evidence with reproduction active

The strongest archived cases are useful because reproduction and birth mutation
were active while substantially different ecological drivers were tested:

| Archived experiment | Seeds/horizon | Engagement | Tail result |
|---|---:|---|---|
| 0017 zero-sum social color transfer | 4 seeds at 500k; seed 7 to 1M | Dense endogenous energy transfer; healthy populations of 1,079-1,557 | Color spread approached a stable uniform distribution (`R -> 0.11`) and action effectiveness settled. Diversity without continuing novelty. |
| 0018 committed-attack pursuit/evasion | 4 seeds at 500k; seed 7 to 1M | Attacks landed; prey evaded; neurons grew 12 -> 30 and synapses 13 -> 30 through 1M | Mean action effectiveness regressed `0.5613 -> 0.5157`. Continuing complexity was competence-reducing bloat. |
| 0019 brain-controlled display contest | 4 seeds at 500k; seed 7 to 1M | Dense, zero-sum, intransitive, brain-controlled interaction was active and wired with large evolved weights | Seed-7 action effectiveness peaked near `0.46` at 750k then fell to `0.4417` at 1M; complexity also turned over. Bounded novelty. |
| 0020 two-step compositional foraging | seed 7 to 1.5M | Chain success rose to roughly `0.55`; brains grew to about 12 neurons | Learned behavior converged to a lower plateau (`~0.388` versus `~0.458` baseline). Single-seed negative, not robustness evidence. |
| 0021 persistent Build affordance | seed 7 to 1M | Build consumed about 10.5% of actions and maintained about 1,500 walls | Degenerate wall spam; action effectiveness collapsed to `0.013`. Single-seed negative, not robustness evidence. |

Experiments 0017-0019 are the relevant multi-seed adversarial evidence. They
separate three failure modes despite ongoing birth and mutation: fixed-point
diversity, novelty-as-bloat, and an active information game that plateaus. The
single-seed 0020-0021 results only localize failure mechanisms and must not be
presented as seed-robust estimates.

Round 10 adds a complementary causal result on the current pure-clonal engine:
renewal with zero/small kill reward kills every focal, physical reward becomes
an evaluator-energy farm, and intermediate caps sit on a scenario-dependent
viability cliff. It proves that the energy source matters. The archived
reproduction campaign proves that repairing the source alone is not sufficient.

## Why the obvious “small experiment” would not answer the task

An evaluator hook that automatically clones any well-fed parent can close a
parent/child ledger and sustain births. It cannot establish a new sensory-motor
strategy: the organism neither chooses reproduction nor solves a new problem.
Population persistence would be the programmed hook's behavior, and an
evolution-versus-frozen difference would reduce to the existing foraging
objective. That is exactly the standing prompt's metric-gaming/bounded-capability
failure.

A faithful brain-controlled restoration would recover the archived action,
gestation, mutation, event, metric, and wire surfaces. Without a new public
interaction it still tests the old finite game. Adding the smallest credible
information interaction is no longer “reproduction-only”; it is the PPEC or
persistent-public-construction route and should be tested under that route's
controls rather than duplicated here.

## Exact reopen and control contract

Reopen endogenous replication only when birth is coupled to a materially new
problem source. The minimum experiment must satisfy all of the following:

- Every successful birth has a same-tick live-parent or explicit endogenous
  escrow debit equal to the child credit. Plant spawn energy remains the sole
  external source; every tick closes fatally, not by debug assertion.
- The interaction publishes information that another organism must use to earn
  energy; IDs, absolute coordinates, and evaluator secrets are unavailable.
- Treatment is compared with the identical energy-clean reproduction path under
  frozen variation, shuffled/neutral selection, no payoff, and no reproduction.
- Early and late organisms are inspected at the causal interaction, with
  semantic permutation and post-early structure knockouts.
- Qualified discoveries, births, interaction energy, ordinary foraging, and
  competence remain nonzero in geometrically extended tail windows across at
  least the evaluation seed suite and modest food/regrowth/world perturbations.

Until such an interaction exists, the registry entry should read **blocked as
reproduction-only**. The active work belongs to the public-information and
recursive-construction lanes.

## Reproducible source and archived-evidence commands

These commands do not depend on ignored world artifacts:

```bash
# Exact parent debit, stored transfer, child credit, and birth mutation.
git show b28c5df:sim-core/src/turn/reproduction.rs | sed -n '70,190p'
git show b28c5df:sim-core/src/spawn/organisms.rs | sed -n '1,72p'
git show b28c5df:sim-types/src/lib.rs | sed -n '20,75p;360,380p'

# Archived multi-seed/tail records.
git show b28c5df:research/experiments/0017-ecology-social-transfer.md
git show b28c5df:research/experiments/0018-ecology-pursuit-evasion.md
git show b28c5df:research/experiments/0019-ecology-display-contest.md
git show b28c5df:research/experiments/0020-ecology-compositional-foraging.md
git show b28c5df:research/experiments/0021-affordance-construction.md

# Removal boundary and current no-birth contract.
git show --stat a5d3c81
rg -n 'No in-world reproduction|produces no births' sim-core/src/turn/mod.rs
```

Each archived experiment document includes its historical checkout/build/run
command. Their numerical summaries are durable campaign records, not live
reruns performed for this audit.
