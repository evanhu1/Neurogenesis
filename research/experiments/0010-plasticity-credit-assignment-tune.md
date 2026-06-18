---
type: Experiment
title: Credit-assignment tune (eligibility_retention 0.90→0.94) — cannibalizes the seed-7 win
description: Lengthening the eligibility window looked good at 250k but regresses at 500k and collapses the seed-7 mi_sa niche; the credit-window knobs trade aeff↔mi_sa per-seed and can't broaden both.
iteration: 10
coordinator: plasticity
agent: credit-assignment-tune
surface_area: plasticity-genome
base_ref: 1dab610
git_ref: autoresearch/exp-0010-plasticity-credit-assignment-tune
status: rejected
determinism: ok
seeds: [7, 42, 123, 2026]
metrics: { plant_consumption_rate: 0.0729, prey_consumption_rate: 0.00236, action_effectiveness: 0.5026, mi_sa: 0.1372, learning_slope: -0.000375 }
baseline_metrics: { plant_consumption_rate: 0.0786, prey_consumption_rate: 0.00235, action_effectiveness: 0.5435, mi_sa: 0.1952, learning_slope: -0.000487 }
delta: { plant_consumption_rate: -0.0057, prey_consumption_rate: 0.00001, action_effectiveness: -0.0409, mi_sa: -0.0580, learning_slope: 0.000112 }
tags: [plasticity, eligibility, dead-end]
timestamp: 2026-06-17T00:00:00Z
---

# Hypothesis
A longer eligibility window (higher eligibility_retention) / weaker decay lets the
gentle three-factor signal consolidate useful structure on MORE seeds (broadening
the seed-7-heavy mi_sa gain).

# Change
Swept `eligibility_retention` ∈ {0.90,0.94,0.97} × `PLASTIC_WEIGHT_DECAY` ∈
{0.001,0.0005,0.002}; chose elig 0.94 (decay held). seed_genome (both copies).

# Result
Dead-end — REGRESSES the champion. eligibility 0.90→0.94 lengthens the credit
window (~10→~17 ticks), which at 250k read as a mild broadener but at the full 500k
horizon **destabilizes the precise sensory→action structure** behind the seed-7
high-mi_sa niche (mi_sa 0.4420→0.2628, seed-7 aeff −0.129). Cross-seed aeff −0.041,
mi_sa −0.058. No grid cell broadened both pillars — every mi_sa lift cost aeff.

# Learnings
The credit-window genes trade aeff↔mi_sa per-seed and **cannot broaden both**; the
champion's GAIN-0.04 band ([[experiments/0009-plasticity-three-factor-tune]]) is the
plasticity sweet spot. The seed-7 mi_sa win is FRAGILE (a longer eligibility window
cannibalizes it). Process note: 250k screens MISLEAD for slow eligibility dynamics —
confirm at 500k. (Also harmonized a pre-existing seed_genome eligibility desync,
sim-config 0.95 vs eval 0.90 — but on this dead-end branch; the champion still
carries the harmless desync, eval uses 0.90.)

# Reproduce
`git checkout 6ccadb1; cargo build -p sim-cli --release`; per-seed `new`+`sim-run run-to 500000`+`pillars`.

# Citations
[1] diff: `git show 6ccadb1`
