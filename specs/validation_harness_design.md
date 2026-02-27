# Validation Harness Spec — `sim-validation`

Headless harness. Runs sim-core for N ticks, prints diagnostic table at fixed
intervals, writes timeseries CSV. 12 metrics. Goal: tell you whether the
evolutionary loop is producing adaptive behavior.

## Sim-Core Changes

Two additions to sim-core, both behind `cfg(feature = "instrumentation")`.

### 1. ActionRecord

Define in `sim-types`:

```rust
#[cfg(feature = "instrumentation")]
#[derive(Clone)]
pub struct ActionRecord {
    pub organism_id: OrganismId,
    pub selected_action: ActionType,
    pub food_ahead: bool,
    pub food_left: bool,
    pub food_right: bool,
    pub food_behind: bool,
    pub inter_activations: Vec<f32>,   // copy of inter neuron activations post-eval
    pub consumptions_count: u32,       // organism's lifetime total at this tick
}
```

Populated in `build_intent_for_organism` after `evaluate_brain` returns. Sensory
bools derived from the ray scan results already computed. Inter activations
copied from `organism.brain.inter[i].neuron.activation`. `consumptions_count`
copied from `organism.consumptions_count`.

Collection: use `map_init` to build per-thread vecs in `build_intents`, then
flatten into `Simulation::action_records: Vec<ActionRecord>`. Expose
`pub fn drain_action_records(&mut self) -> Vec<ActionRecord>` via
`std::mem::take`.

### 2. Organism accessor

Not feature-gated (useful for debugging/viz too):

```rust
pub fn organisms(&self) -> &[OrganismState] {
    &self.organisms
}
```

Used at reporting time for brain size of living population.

## Crate Layout

```
sim-validation/
  src/
    main.rs           CLI + headless loop
    ledger.rs         Per-organism sidecar, recently-deceased buffer
    metrics.rs        All 12 metric computations
    report.rs         Table printer + CSV writer
```

## CLI

```
sim-validation \
  --config <path>          # default: config/default.toml
  --seed <u64>
  --ticks <u64>            # default: 10000
  --report-every <u64>     # default: 1000
  --min-lifetime <u64>     # default: 30
  --out <path>             # default: artifacts/validation/<ISO8601>_seed_<seed>/
  --baseline               # uniform random actions, bypasses brain eval
```

`--baseline` sets `force_random_actions: true` on `WorldConfig` before running.
`build_intent_for_organism` checks this flag and returns a uniformly sampled
action without calling `evaluate_brain`. Add this bool field to `WorldConfig`,
defaulting to `false`.

## Tick Loop

```
initialize sim
register all initial organisms in ledger

for tick in 1..=total_ticks:
    delta = sim.tick()
    records = sim.drain_action_records()

    for record in records:
        ledger.update(record)

    for spawned in delta.spawned:
        ledger.birth(spawned.id, tick)

    for removed in delta.removed_positions:
        if organism:
            ledger.death(id, tick)

    if tick % report_every == 0:
        compute and print table row
        write row to timeseries CSV
        ledger.clear_interval()
```

## Ledger

### Per-organism sidecar

`HashMap<OrganismId, OrganismEntry>`:

```rust
struct OrganismEntry {
    birth_tick: u64,
    last_consumptions: u32,       // updated each tick from ActionRecord

    // Tier 3 accumulators
    action_counts: [u32; N_ACTIONS],
    joint: [[u32; N_ACTIONS]; 5], // sensory_bin × action
    food_ahead_ticks: u32,
    fwd_when_food_ahead: u32,

    // Inter utilization EMA
    inter_ema: Vec<f32>,
    ema_initialized: bool,
}
```

### Update per tick

```rust
fn update(&mut self, record: ActionRecord) {
    let entry = self.sidecar.get_mut(&record.organism_id);
    let a = action_index(record.selected_action);

    entry.last_consumptions = record.consumptions_count;
    entry.action_counts[a] += 1;

    // Sensory bin: priority ahead > left > right > behind > none
    let bin = if record.food_ahead       { 1 }
              else if record.food_left    { 2 }
              else if record.food_right   { 3 }
              else if record.food_behind  { 4 }
              else                        { 0 };
    entry.joint[bin][a] += 1;

    if record.food_ahead {
        entry.food_ahead_ticks += 1;
        if record.selected_action == Forward {
            entry.fwd_when_food_ahead += 1;
        }
    }

    // Inter utilization EMA (alpha = 0.05)
    if !entry.ema_initialized {
        entry.inter_ema = record.inter_activations.iter().map(|a| a.abs()).collect();
        entry.ema_initialized = true;
    } else {
        for (ema, act) in entry.inter_ema.iter_mut().zip(&record.inter_activations) {
            *ema = 0.95 * *ema + 0.05 * act.abs();
        }
    }
}
```

### Death finalization

```rust
fn death(&mut self, id: OrganismId, tick: u64) {
    let entry = self.sidecar.remove(&id);
    let lifetime = tick - entry.birth_tick;

    if lifetime < self.min_lifetime {
        self.neonatal_deaths += 1;
        return;
    }

    let utilization = if entry.inter_ema.is_empty() { 0.0 }
        else { entry.inter_ema.iter().filter(|v| **v > 0.03).count() as f32
               / entry.inter_ema.len() as f32 };

    self.recently_deceased.push(CompletedLifetime {
        lifetime, consumptions: entry.last_consumptions,
        action_counts: entry.action_counts, joint: entry.joint,
        food_ahead_ticks: entry.food_ahead_ticks,
        fwd_when_food_ahead: entry.fwd_when_food_ahead,
        utilization,
    });
}
```

## Metrics

All Tier 2 and Tier 3 metrics are computed over the `recently_deceased` buffer
for the current interval (organisms with lifetime >= `min_lifetime`). If the
buffer is empty, print `NA` for those columns.

### Tier 1 — Simulation Health

| Metric   | Source                                                       |
| -------- | ------------------------------------------------------------ |
| `pop`    | `delta.metrics.organisms` at interval end                    |
| `births` | sum of `delta.spawned.len()` over interval                   |
| `deaths` | organism removals in `delta.removed_positions` over interval |
| `food`   | count of food entities at interval end                       |

### Tier 2 — Evolutionary Progress

All computed on recently-deceased buffer except brain_size.

| Metric      | Definition                                                                            |
| ----------- | ------------------------------------------------------------------------------------- |
| `life_mean` | mean lifetime ticks                                                                   |
| `life_max`  | max lifetime ticks                                                                    |
| `ate%`      | % with consumptions > 0                                                               |
| `cons_mean` | mean lifetime consumptions                                                            |
| `brain`     | mean `(num_neurons + synapse_count)` over **living** population via `sim.organisms()` |

### Tier 3 — Cognitive/Behavioral

Computed on recently-deceased buffer. Pooled across all organisms in interval.

**P(Fwd\|food):** `Σ fwd_when_food_ahead / Σ food_ahead_ticks`. Random baseline:
`1/N_ACTIONS`.

**H(action):** Pool all `action_counts` across deceased. Compute:
`H = -Σ p(a) log₂ p(a)`. Uniform baseline: `log₂(N_ACTIONS)`.

**MI(S;A):** Pool all `joint` tables across deceased. Compute:
`I = Σ p(s,a) log₂(p(s,a) / (p(s)p(a)))`, skip zero cells. Apply Miller-Madow
correction: `I_corr = I - (K-1)/(2N·ln2)` where K = nonzero cells, N = total
observations. Clamp ≥ 0. Baseline: 0.

**util:** Mean `utilization` across deceased.

## Output

### Table to stdout

```
# _d = deceased (lifetime >= 30), _l = living | baselines: P 0.25, H 2.00, MI 0.00

tick  | pop  |  +  |  -  | food | life_μ| life_max| ate% | cons_μ|brain_l|P(F|fd)|MI(S;A)|H(act)|util
 1000 | 4872 | 312 | 340 | 8210 |  48.2 |     312 | 12.0 |   0.3 |  51.2 |  0.19 |  0.02 | 1.31 | 0.34
 2000 | 4901 | 298 | 285 | 7840 |  52.1 |     408 | 18.0 |   0.5 |  50.8 |  0.24 |  0.05 | 1.25 | 0.37
```

### timeseries.csv

Same columns, one row per interval. Written to `<out>/timeseries.csv`.

### summary.json

```json
{
  "seed": 42,
  "ticks": 10000,
  "baseline": false,
  "state_hash": "...",
  "timeseries": [...]
}
```

`state_hash`: hash of `(population_count, sum_of_organism_ids, total_energy)` at
final tick. For determinism verification: same seed → same hash.

## Tests

**Integration:** fixed seed, 100 ticks, assert specific metric values. Two runs
same seed → identical `summary.json`.
