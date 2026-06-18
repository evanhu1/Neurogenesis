---
type: Finding
title: A moving hazard delays but does NOT escape convergence — and a 500k positive slope can be slow convergence, not open-endedness
description: A non-stationary moving lethal hazard (roamer) looked open-ended at 500k (2nd-half aeff slope 4x the champion, all seeds still climbing) but the 1M test shows it converges too — slower and at a LOWER competence level (0.528 vs champion 0.549). The permanent mortality tax caps competence without buying unbounded improvement. Two lessons: a mid-climb 500k slope misleads (confirm at 2x horizon), and a fixed-policy moving threat is not enough for open-endedness — co-evolution is likely needed.
confidence: high
status: active
supported_by: [experiments/0013-ecology-roamer]
seeds: [7, 42, 123, 2026]
tags: [open-endedness, convergence, non-stationary, eval-horizon, process, dead-end]
timestamp: 2026-06-18T00:00:00Z
---

# Question

The system converges under a static niche
([[findings/the-system-converges-it-is-not-open-ended-under-action-effectiveness]]).
Does a **non-stationary** pressure — a moving lethal hazard organisms must flee
using vision ([[experiments/0013-ecology-roamer]]) — produce genuine open-ended
(unbounded, sustained) improvement?

# Answer

**No.** It delays convergence but does not escape it, and converges at a *lower*
competence level than the static champion.

# Evidence (measured)

| horizon | spike champion aeff | roamer(14) aeff | roamer late-slope |
|---|---|---|---|
| 500k | 0.5522 | 0.5229 | **+0.0183/100k** (still climbing — looked open-ended) |
| 1M | 0.5492 (flat) | **0.5282** | **−0.0012/100k (CONVERGED)** |

At 500k the roamer's 2nd-half slope was 4× the champion's and every seed was still
climbing (ending at its max) — strongly suggestive of open-endedness. **The 1M
extension corrected this:** the late-slope (750k–1M) is ~0 — the roamer converged,
just later than the champion, at a lower level (0.528 < 0.549; seed 7 even declined
0.582→0.548). The strong 500k slope was **slow convergence on a harder task**, not
unbounded improvement. roamer_count tuning {6,10,14} never reached the champion's
0.5522 at 500k either.

# Two durable lessons

1. **A positive 2nd-half slope at the canonical horizon is NOT proof of
   open-endedness.** It can be slow convergence on a harder task that simply hasn't
   plateaued yet. *Process rule:* any sustained-improvement / open-endedness claim
   must be confirmed at ≥2× the horizon (1M here flattened a slope that looked
   decisive at 500k). This nearly caused a wrong promotion.
2. **A permanent mortality tax lowers the competence ceiling.** A deterministic-
   but-unpredictable moving threat is largely *unavoidable noise* with no learnable
   counter-skill whose mastery exceeds the champion's competence — so it taxes
   without raising the ceiling. Consistent with
   [[mechanisms/selection-pressure-is-the-bottleneck-for-intelligence]]: pressure
   must reward SKILL, not just kill more.

# Implication for the goal

Both a static skill challenge (spikes) AND a moving hazard (roamer) **converge** —
the program has **not** achieved unbounded open-endedness. Candidate routes that
remain genuinely untried:
- **True co-evolution / Red Queen:** the threat (or prey) itself *evolves*, so the
  target never stops moving — a moving *fixed-policy* hazard is not enough.
- **Growing opportunity, not just mortality:** an environment that keeps opening
  *new* niches/resources to exploit (open-ended *opportunity*), rather than adding
  a fixed tax.
- **Open-endedness as the explicit objective:** select FOR sustained novelty
  directly ([[directions/measure-open-endedness-not-just-static-competence]]),
  rather than hoping a niche produces it as a side effect.

The binding question is no longer "which niche" but "what makes improvement
*unbounded*" — likely co-evolution. The spike champion (`47a6111`) remains the
best program; the roamer is a rejected but instructive dead-end.

# Reproduce

`git checkout autoresearch/exp-0013-ecology-roamer; cargo build -p sim-cli --release`;
per-seed `new --seed S` + `run-to 500000` then `run-to 1000000`; read
`pillars.granular` action_effectiveness and regress the late-window slope at BOTH
horizons — the 500k slope is positive, the 1M late slope is ~0.

# Citations

[1] [[experiments/0013-ecology-roamer]] (branch `autoresearch/exp-0013-ecology-roamer`,
commit 6a2775b), cross-seed 500k + 1M, planner-authoritative, 2026-06-18.
