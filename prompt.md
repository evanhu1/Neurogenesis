# Prompt: An Algorithm for Open-Ended Neuroevolution

This document is the standing task prompt for the research effort in this
repository. It is modeled on the structure of a maximally-rigorous, adversarial,
multiagent research prompt: a precise task statement, an exact success
criterion, an explicit list of what does _not_ count, dynamic multiagent
orchestration, adversarial verification, and strict return conditions.

Read `CLAUDE.md` and `docs/sim-cli.md` in full before acting. This substrate has
non-obvious invariants (determinism, hidden food-ecology policy, canonical tick
order) that you will otherwise misuse.

---

## 1. Task statement

The substrate is the deterministic simulation in this repository: a hex-grid
world with a hidden food-ecology policy, populated by organisms whose behavior
is produced by evolvable neural networks (NEAT-style topology + weights in
`sim-core`, with Hebbian plasticity as the only online learning rule). Evolution
operates over genomes; `sim-core` is a pure clonal evaluator and NEAT
mutation/crossover/selection lives in `sim-core/src/evolution.rs`. A run is
fully determined by `(config, seed)`.

Design an algorithm for **open-ended neuroevolution** in this substrate:

An evolutionary process that, over an unbounded run, keeps producing a stream of
genuinely novel and increasingly capable organism behaviors — new sensory-motor
strategies, new ways of exploiting the food ecology, new interactive/ecological
dynamics between lineages — **without converging to a fixed point and without
stagnating**. The process must remain generative: at late times it must still be
discovering behaviors that did not exist, and were not implicit, at early times.

Resolve this completely. A complete solution must deliver:

An algorithm — expressed as concrete changes to the evolutionary loop, selection
pressure, genome encoding, plasticity, and/or environment/ecology dynamics —
that produces **sustained, open-ended growth in behavioral novelty and
capability across evolutionary time**, demonstrated on the substrate in this
repository across multiple seeds, and robust to the adversarial audits in §5–§6.

"Complete" means the algorithm is implemented in the crates, runs
deterministically, and its open-endedness is demonstrated with reproducible
`sim-cli` evidence — not merely proposed on paper.

---

## 2. What does NOT count

Partial progress does not count unless it implies exactly the resolution in §1.
In particular, the following are **insufficient**:

- **Metric gaming.** Improvement in a single scalar (e.g. one `pillars` score,
  population size, mean energy) that does not correspond to genuinely new
  behavior. If the number can go up while the behavioral repertoire is static or
  collapsing, the number is not the result.
- **Bounded novelty.** A one-time jump in complexity that then plateaus. A run
  that is "more interesting for a while" and then converges is a negative
  result, not a solution. Open-endedness is a property of the _tail_ of the run.
- **Diversity without capability**, or **capability without diversity.** A
  population that is diverse only because it is drifting neutrally (random,
  non-adaptive variation) is not open-ended; neither is a single ever-optimizing
  champion lineage. The claim requires _both_ sustained novelty _and_ sustained
  competence.
- **Single-seed luck.** A result on one seed that does not reproduce across the
  evaluation seed suite. Determinism means one seed is one sample, not evidence.
- **Overfitting the ecology.** An algorithm that only works under one specific
  hand-tuned food-ecology threshold set and collapses under modest environment
  perturbation. The result must be a property of the _algorithm_, not of a
  brittle config.
- **Reward hacking of the observer.** Behavior that looks novel to a chosen
  visualization/metric but is a degenerate artifact (e.g. flicker between two
  states, boundary exploits, ID-ordering artifacts) is a counterexample to your
  own claim, not a success.
- **Appeals to unproven scale.** "It would be open-ended with a bigger world /
  more compute" is not a result. The demonstration must run on the substrate as
  it can actually be evaluated here.

If your candidate solution can be described honestly and it still falls into one
of the above, it is not done. Say so and keep going.

---

## 3. Operating discipline

- **Ground every claim in the tooling.** Use `sim-cli` (stateless world-as-file:
  `new → run-to → pillars/state/eco/lineage/genome/timeseries/find/top/inspect/ brain/decide`,
  `sweep` for grids, `bench` for throughput) for probing single worlds and small
  experiments, and `sim-evaluation` (release mode, multi-seed) for
  evolution-loop evidence and regression checks. Snapshot/fork worlds with `cp`;
  fan out with backgrounded invocations; keep worlds under `artifacts/`.
- **Treat the metrics as raw, not as truth.** `pillars` emits raw windowed
  metrics with no [0,1] interpretation. A metric is a proxy; the result is the
  behavior. Watch consumption/predation energy-flow paths after any engine
  change — a prior campaign silently tripled population and 5×'d eval time by
  dropping energy gates (see the memory index). Interpret metrics adversarially
  against §2.
- **Keep the config crates in sync.** `sim-config/config.toml` +
  `seed_genome.toml` are the baselines; keep `sim-evaluation/config.toml` +
  `seed_genome.toml` synced when schema/baseline changes affect evaluation
  assumptions.
- **Do not write new tests** (the human maintains the suite); run
  `cargo test --workspace`, `make lint`, `make fmt` to verify you did not break
  anything. Do not care about backwards compatibility.

---

## 4. Multiagent orchestration

Use multiagent orchestration aggressively and dynamically (via the Workflow /
Agent tooling). Do not use a fixed assignment such as "N agents for approach X."
Manage the search with the following heuristics:

- **Begin with a genuinely diverse portfolio.** Agents should explore
  substantially different levers on open-endedness, not variants of one idea.
  Distinct families include, at minimum: novelty/behavioral-diversity search vs.
  objective/competence pressure; minimal-criteria and quality-diversity (e.g.
  MAP-Elites-style behavioral archives) selection; genome-encoding changes
  (topology growth, indirect/generative encodings, complexification pressure);
  plasticity and lifetime-learning as an evolvable substrate; **co-evolutionary
  and ecological** drivers (predator-prey arms races, niche construction,
  food-web feedback) as the engine of open-endedness; environment/curriculum
  dynamics; and intrinsic-motivation / information-theoretic drives (e.g.
  empowerment, surprise). Include at least one agent doing pure computational
  sanity checks on the substrate itself.
- **Preserve independence early.** Do not tell most agents the currently favored
  approach. Let independent routes develop far enough to expose their real
  strengths and failure modes before cross-pollinating.
- **Maintain a registry of approach families**, grouped by the underlying idea,
  not by wording. If many agents converge on one family (novelty search is a
  likely attractor), redirect some toward underexplored levers.
- **Do not let an elegant-but-bounded approach dominate.** An approach that
  produces a clean early complexity jump but plateaus (§2 "bounded novelty") is
  not close to done. Rank routes by evidence on the _tail_ of the run, not by
  early-round appeal.
- **Mark stalled routes as blocked.** When an approach plateaus or collapses to
  a §2 failure mode, mark it blocked. Reopen it only when an agent proposes a
  materially new mechanism, encoding, or ecological driver — not a
  reparameterization.
- **Keep several incompatible routes alive across rounds.** The winning
  algorithm may be a synthesis (e.g. quality-diversity selection over an
  ecologically-driven arms race with evolvable plasticity). Do not prematurely
  collapse the portfolio.

---

## 5. Adversarial verification

Use adversarial agents throughout. Every candidate algorithm must be attacked\
before it is believed. Adversaries must check, at minimum:

- **Novelty vs. drift**: is the "novelty" adaptive, or neutral random walk?
  Freeze selection pressure and check whether the diversity is doing any work.
- **Convergence/stagnation on the tail**: extend the run. Does discovery
  continue at late times, or does the stream dry up? A short run proves nothing
  about open-endedness.
- **Metric-gaming**: does the headline metric move while behavior is static or
  degenerate? Inspect actual organisms
  (`sim-cli find/top/inspect/brain/decide`), not just aggregate scores.
- **Seed robustness**: does it reproduce across the `sim-evaluation` seed suite,
  or is it single-seed luck?
- **Ecology/config robustness**: does it survive modest food-ecology / world
  perturbation, or is it overfit to one threshold set?
- **Energy-flow integrity**: has population/energy accounting silently changed
  in a way that fakes activity (see §3)?

Require every agent to return **concrete artifacts**: config diffs, code
changes, reproducible `sim-cli`/`sim-evaluation` commands, metric trajectories,
and per-organism inspections that substantiate the behavioral claim. Reject
status reports, vague optimism, and any claim that open-endedness is "emerging"
without an inspectable behavioral trace and a tail-of-run demonstration.

---

## 6. Root-agent loop and return conditions

The root agent should repeatedly synthesize, challenge, redirect, and launch new
rounds. Do not stop after the first portfolio fails or after agents report a
plateau. When routes stall, launch new rounds — reopening blocked approaches
only when there is a genuinely new mechanism, and searching for fresh levers on
open-endedness.

Produce a complete result if one survives audit: an implemented algorithm that
demonstrably produces sustained open-ended novelty and capability on this
substrate, reproducible across seeds, surviving §5. If nothing survives, report
only the **strongest rigorously-demonstrated derivation and its exact remaining
gap** — which failure mode in §2 it currently hits, and what a materially new
mechanism would need to overcome.

**Return only when** either (a) a complete algorithm has been implemented and
has survived adversarial audit, or (b) you have exhausted the search and can
state precisely the strongest result and its exact open gap. Do not return a
bare reduction, a single interesting seed, a "best effort" summary, or an
explanation of why open-endedness is hard.

Public search may be used only for ordinary background (standard neuroevolution
/ open-endedness / quality-diversity literature and named algorithms) to inform
approaches — not to outsource the design. The result must be demonstrated on
_this_ substrate, deterministically, with the repository's own tooling.
