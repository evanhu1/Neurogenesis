---
type: Experiment
title: Brain-controlled intransitive display contest (dense cognitive arms race) — the precisely-diagnosed mechanism ALSO plateaus below champion; rejected
description: The diagnosis (iter18) said open-ended intelligence needs a DENSE + BRAIN-CONTROLLED + non-saturating cognitive contest so complexity PAYS. Built exactly that: a brain "display" output, perception of neighbors' displays, dense zero-sum adjacency transfer keyed on sin(Δdisplay). At 500k seed-7 aeff appeared to rise (0.32→0.42, accelerating) — but the 1M test shows it was slow recovery from the perception-dilution hit: aeff peaks ~0.46 @750k then turns over to 0.44, plateauing WELL BELOW champion (0.56). Complexity also turns over. The precisely-diagnosed mechanism fails on both counts (converges AND dilutes). Rejected.
iteration: 19
coordinator: ecology-niche
agent: display-contest
surface_area: ecology-niche
base_ref: ae8903a
git_ref: autoresearch/exp-0019-ecology-display-contest
status: rejected
determinism: ok
seeds: [7, 42, 123, 2026]
metrics: { action_effectiveness: 0.500, seed7_aeff_1M: 0.4417, mean_neurons: 38, mean_synapses: 35 }
baseline_metrics: { action_effectiveness: 0.5613 }
tags: [ecology-niche, cognitive-arms-race, brain-controlled, intransitive, open-endedness, perception-dilution, dead-end]
timestamp: 2026-06-18T00:00:00Z
---

# Hypothesis

iter18's diagnosis: open-ended INTELLIGENCE (not bloat) needs a DENSE +
BRAIN-CONTROLLED + non-saturating cognitive contest so growing brain complexity
PAYS. Build exactly that: (1) a brain "display" output neuron (continuous,
brain-computed each tick, conditioned on perception — not argmax-selected, no
action-slot cost); (2) a sensory channel for neighbors' mean display (sin,cos);
(3) a dense zero-sum adjacency energy transfer keyed on `sin(θ_display_self −
θ_display_neighbor)` (intransitive, brain-controlled). Winning requires
out-computing neighbors → complexity should translate into competence
(intelligence), not inflation.

# Change

`sim-types` + `sim-core/src/brain/{topology,expression,evaluation,sensing,plasticity}.rs`
+ `genome/*` + `turn/{mod,intents,commit}.rs`: display output neuron (ID 2006,
reuses action-edge accumulation), `NeighborDisplaySin/Cos` receptors, and
`apply_display_contest_transfer` (modeled on iter17's social transfer but keyed on
the brain-controlled display). `DISPLAY_CONTEST_STRENGTH=0` ⇒ baseline
byte-identical (verified via cmp). det-check ok (P1+P2). Determinism: neighbor
perception reads a persisted prior-tick display snapshot; transfer is
snapshot-then-apply in index order; no new RNG.

# Result

**REJECTED — converges AND dilutes (the precisely-diagnosed mechanism fails).**
The contest is genuinely active/load-bearing (champion brains wire both display
channels with large evolved weights; energy shows the wide zero-sum dominance
spread). At 500k seed-7 aeff *appeared* to rise with an accelerating slope
(0.318→0.421) — encouraging. **But the 1M test shows that was slow recovery from
the perception-dilution hit:**

| seed 7 | aeff | neurons | synapses |
|---|---|---|---|
| 250k | 0.337 | 11.4 | 20.2 |
| 500k | 0.421 | 17.0 | 24.6 |
| 750k | 0.457 | 17.0 | 35.4 |
| **1M** | **0.442** | 14.0 | 24.4 |

aeff peaks ~0.46 @750k then **turns over to 0.44, plateauing far below champion
0.5613**; complexity peaks then turns over too. Cross-seed @500k: aeff
{7:0.42, 42:0.55, 123:0.54, 2026:0.48} mean **0.50 < champion 0.5613**, with
complexity grown 2–2.5× (neurons 35–46) — i.e. still complexity-up / competence-
below-champion. The two added perception channels impose a dilution cost the
contest never repays.

# Learnings (definitive)

Even the **exact** mechanism the iter18 diagnosis specified — a dense,
brain-controlled, intransitive cognitive contest — **plateaus** (by 1M) and stays
BELOW champion (perception dilution). The 500k "accelerating rise" was slow-
saturation, the same burned-twice pattern (roamer, genome) — the ≥2× horizon
check caught it again. So the conclusion is now airtight from EVERY angle: in this
engine, every mechanism's apparent open-ended rise is slow-saturation that
plateaus, and any mechanism rich enough to host a cognitive contest pays a
perception-dilution cost. Open-ended evolution of INTELLIGENT brains is not
reachable in-loop; it requires a fundamentally richer cognitive substrate (deeper
perception/action that does NOT dilute + an arms race that does NOT saturate) —
a research-scope engine redesign, not an in-loop mechanism.
([[findings/open-endedness-needs-a-dense-substrate-the-binding-constraint-is-the-food-ecology-lock]])

# Reproduce

`git checkout autoresearch/exp-0019-ecology-display-contest; cargo build -p sim-cli --release`;
per-seed `new`+`run-to 1000000`+`pillars`+`find "generation>50" --fields neurons,synapses`.
The seed-7 aeff peaks ~0.46 @750k then plateaus ~0.44 @1M, below champion.

# Citations

[1] diff: `git show autoresearch/exp-0019-ecology-display-contest` (commit c4f4825).
[2] Seed-7 1M trajectory + cross-seed 500k, planner-authoritative, 2026-06-18.
