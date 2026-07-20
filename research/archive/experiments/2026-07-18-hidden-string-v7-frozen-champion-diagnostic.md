# Hidden-string v7 frozen-champion diagnostic

Status: completed positive mechanistic diagnostic; evolutionary discovery and
fresh confirmation remain unrun

## Question

Does the implemented boundary-contrastive signed-evidence fast readout turn the
failed v6 seed-509 controller into reliable reward-only string adaptation on a
fresh custom panel, with the preregistered causal controls and anti-watermark
lesions behaving correctly?

## Method

The diagnostic reused the persisted v6 seed-509 terminal genome as a slow
recurrent context generator. It did not evolve or select a new genome. The v7
evaluator used:

- 128 fresh balanced targets from custom panel seed
  `6215359696954807601`;
- two fresh generated rollout streams, for 256 cases;
- 32 reward-only attempts per case;
- a fast-weight bound of 1.5;
- boundary keys `d / max(norm(d), 0.05)`;
- the inherited learning rate `0.73519224` as the signed-evidence update scale;
- treatment, all six mandatory causal controls, a raw-live-state key lesion,
  and half/quarter boundary-amplitude lesions.

Command:

```bash
./target/release/cli hidden-string reevaluate \
  artifacts/research/runs/active/hidden-string-boundary-fast-weights-v6/seed-509/neat-hidden-string-run-1784368844161-170/champions/terminal.frozen.json.zst \
  --allow-v6-source --attempts 32 --panel custom \
  --panel-seed 6215359696954807601 --targets 128 --rollouts 2 \
  --condition primary,plasticity-off,symbol-permuted-reward,position-permuted-reward,reset-weights,boundary-pulse-off,dynamics-reset-each-symbol,raw-live-state-key,half-boundary-pulse,quarter-boundary-pulse \
  --out artifacts/research/runs/diagnostics/2026-07-18-hidden-string-v7-frozen-509/reevaluation.json
```

Artifact hashes:

- v7 reevaluation:
  `dc80cfb08b7407fe5524b20fcffd09a55ebc9c8c6602a2ea77f927608af1660d`;
- source v6 result:
  `213dc19074ab6af78468eed2b63f018ad03face5f40d00207a1b84af8736ebf9`;
- source terminal genome:
  `1071535d2a2dc5fad6e8a9c613eff8478236aaac668a1913497497d389c932a1`.

## Result

| Condition | Hard exact | Character accuracy | Sequence probability |
|---|---:|---:|---:|
| Primary contrastive fast memory | 100.00% | 100.00% | 0.2395 |
| Plasticity off | 0.00% | 12.50% | 0.000264 |
| Symbol-permuted reward | 0.00% | 0.00% | 0.000004 |
| Position-permuted reward | 0.00% | 6.25% | 0.000059 |
| Reset fast weights each attempt | 0.391% | 25.00% | 0.000553 |
| Boundary pulse off | 0.00% | 12.50% | 0.000256 |
| Dynamics reset each symbol | 0.00% | 34.38% | 0.001206 |
| Raw live-state key | 52.73% | 87.60% | 0.3498 |
| Half boundary amplitude | 56.25% | 89.06% | 0.0772 |
| Quarter boundary amplitude | 12.50% | 78.13% | 0.0488 |

The primary passed the 90% mechanistic gate and every mandatory causal control
stayed at or below the 0.5% exact ceiling. Raw state retained v6-like
interference and only 52.73% exact, so the matched null contrast contributed
47.27 points rather than merely re-expressing the same readout.

The smallest raw nonzero contrast norm was 0.0485518; after the fixed 0.05
floor, the smallest effective key norm was 0.971036. Reducing the boundary to
half lowered that raw margin to 0.0147209 and exact performance to 56.25%; a
quarter pulse lowered it to 0.0035733 and 12.50% exact. The implementation does
not amplify an arbitrarily small watermark into a unit key.

Direct normalized contrast keys were pairwise orthogonal for this controller.
Every measured off-diagonal actual fast-logit effect after componentwise
clipping was exactly zero; diagonal effects averaged 0.299--0.342 logits per
write. The primary nevertheless clipped 52.28% of component updates, so this
is not yet an efficient or broadly conditioned plasticity result.

## Interpretation and decision

Advance v7 to implementation-machinery determinism and a small evolutionary
discovery screen. This result establishes only that the new generic update can
read a pre-existing v6 context generator and learn fresh strings. It is not an
independent evolutionary replication: the genome and mechanism were selected
after inspecting v6, and the fixed four-step event contract remains favorable.

Before fresh confirmation, require all of the following:

1. worker-count semantic identity under an actual NEAT run;
2. variable pre-boundary lead-ins and variable target lengths under the same
   event-driven update;
3. fresh evolutionary seeds and panels;
4. serialized runtime ownership and evolvable bounded plasticity parameters;
5. a delayed-association transfer task with no task-specific learning change.

Extra uncommitted recurrent ticks between emitted symbols are outside the
current event-driven contract. A later world interface with such ticks needs a
symbolic commit/write-enable event; this diagnostic does not establish
arbitrary wall-clock timing invariance.
# Architecture-audit note

This diagnostic concerns a controller later invalidated by the architecture
bar. It is retained only as historical evidence about the removed scaffold.
