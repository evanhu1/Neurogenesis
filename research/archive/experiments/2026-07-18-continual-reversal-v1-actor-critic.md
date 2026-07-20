# Continual reversal v1: actor-critic and recurrent feedback channels

Status: implemented, one-seed 500-generation diagnostic
Slug: 2026-07-18-continual-reversal-v1-actor-critic
Date: 2026-07-18

## Question

Does the continuous reversal benchmark expose a limitation of the immediate
one-step REINFORCE rule, and does giving the canonical brain access to its
previous selected action and signed reward-prediction error improve adaptation?

## Generic substrate changes

- A generic value output predicts immediate reward from the current recurrent
  state.
- Actor and critic synapses carry a decaying eligibility trace. The actor trace
  receives the sampled-policy score term; the critic trace receives the value
  derivative. Both update from `reward - value_prediction`.
- Previous-tick action-to-hidden genome edges provide evolvable efference copy.
- Each hidden neuron has an evolvable receptor for the previous signed
  reward-prediction error.

The last two channels are representational affordances, not task-installed
state. The genome must evolve their connections/receptors and decide how to use
them. They are retained because action efference copy and neuromodulatory
prediction-error access are generic agent capabilities. A null ablation in one
run is evidence about discovery/use, not evidence that the channels are
task-specific scaffolding.

## Matched run

Seed 101, population 64, 100 generations, four workers, and the canonical
64/64/256 training/development/sealed lifetime panels:

```bash
./target/release/cli continual-reversal --seed 101 --population 64 \
  --generations 100 --workers 4 \
  --out-dir artifacts/research/runs/active/2026-07-18-continual-reversal-learning-v1-critic-schema11
```

The final winner reached 31.569% sealed accuracy. Matched controls on the same
sealed targets and action draws were:

| Condition | Accuracy | Mean recovery ticks |
|---|---:|---:|
| Primary actor-critic | 31.569% | 42.29 |
| One-step REINFORCE | 22.833% | 47.96 |
| Plasticity off | 12.492% | 56.50 |
| Efference copy off | 31.460% | 42.35 |
| Prediction-error hidden feedback off | 31.338% | 42.50 |

The winner evolved learning rate 0.835, four hidden nodes, four action-feedback
edges, one value edge, and nonzero signed-error receptors. Its eligibility
retention was nevertheless exactly zero. The 8.736-point advantage over the
old rule therefore establishes the learned prediction-error/value route in
this winner, not multi-step synaptic credit assignment. Persistent recurrent
dynamics were strongly causal. Efference copy and hidden prediction-error
feedback had only 0.109- and 0.231-point matched effects, respectively; this run
does not show that evolution materially exploited either channel.

The run also calculated a broad dynamics-reset-each-tick lesion. That control
was subsequently removed from the evaluator because it simultaneously erased
recurrent activity, previous-action state, and prediction-error state and
therefore did not isolate one interpretable mechanism.

The one-step-REINFORCE comparison in the table is also historical. It was
removed from the live evaluator after this initial comparison; new sealed
evaluations no longer calculate it.

## 500-generation extension

The same seed and population were continued as a fresh canonical run for 500
generations after retiring the conflated dynamics-reset control and the
historical one-step comparison:

```bash
./target/release/cli continual-reversal --seed 101 --population 64 \
  --generations 500 --workers 4 \
  --out-dir artifacts/research/runs/active/2026-07-18-continual-reversal-v1-500g
```

The final population winner reached 37.607% training, 37.140% development, and
37.083% sealed accuracy. The best checkpointed development accuracy was 37.512%
at generation 374. Relative to the 100-generation sealed result, another 400
generations gained 5.514 percentage points.

| Condition | Accuracy | Mean recovery ticks | Difference from primary |
|---|---:|---:|---:|
| Primary actor-critic | 37.083% | 39.30 | - |
| Plasticity off | 12.486% | 53.90 | -24.597 pp |
| Efference copy off | 36.973% | 39.46 | -0.110 pp |
| Prediction-error hidden feedback off | 37.029% | 39.34 | -0.053 pp |

The winner evolved the maximum learning rate of 1.0, five hidden nodes, 64
enabled edges, five action-feedback edges, two value edges, and nonzero
signed-error receptors. Eligibility retention again evolved to exactly zero.
The longer run therefore improved the immediate actor-critic policy but did not
discover temporal synaptic credit assignment or material dependence on either
recurrent feedback channel. Their tiny matched lesion effects are below the bar
for a causal claim from one seed.

## Regression checks

- Basic reaction was restored to its original `a`-through-`d` contract after
  an audit found that the rename had accidentally expanded it to eight body
  symbols. A fresh seed-17, population-64 run reached `544/544` training and
  `544/544` unseen holdout accuracy at generation 29 and remained exact through
  generation 99.
- The saturated population-256/species-8 basic-memory champion was reevaluated
  under the current evaluator. It reached 94.873% exact strings on the sealed
  panel; plasticity-off and dynamics-reset-each-symbol both produced effectively
  zero exact strings. The basic-memory learning path remains the historical
  immediate one-step REINFORCE rule.

## Conclusion

Keep the value head and signed prediction-error learning rule. The 500-generation
extension confirms that plasticity is causal and that search can continue to
improve the immediate learner. Eligibility state, action efference copy, and
hidden neuromodulatory receptors remain generic substrate capabilities, but
this one-seed run supplies no material evidence that evolution used them. The
next bottleneck is discovering a representation and update strategy that
improves reversal recovery rather than merely refining the immediate policy.

Artifacts:

- `artifacts/research/runs/active/2026-07-18-continual-reversal-learning-v1-critic-schema11/neat-continual-reversal-1784441839507-38184.json.zst`
- `artifacts/research/runs/active/2026-07-18-continual-reversal-v1-500g/neat-continual-reversal-1784442976256-65430.json.zst`
- `artifacts/research/runs/diagnostics/basic-reaction-original-contract-regression/`
- `artifacts/research/runs/diagnostics/hidden-string-search-pop256-species8-seed101-500/`
