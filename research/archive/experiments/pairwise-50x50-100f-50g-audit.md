# 50x50 / 100-founder competitive NEAT audit

Aggregate classification: **majority_directional_progress**

This bounded checkpoint audit can support or reject directional coevolution over 50 generations. It cannot by itself demonstrate an unbounded open-ended tail.

## Cross-seed summary

| Seed | Run outcome | Last-vs-first | W-T-L | Alive-tick delta | Relative delta | Later-pair wins | Strength slope | Late health |
|---:|---|---|---:|---:|---:|---:|---:|---|
| 7 | mixed_or_cycling | strong_later_loss | 0-0-8 | -47587.6 | -0.8166 | 60.0% | 0.004455 | active |
| 17 | directional_progress | strong_later_win | 8-0-0 | 46239.6 | 0.9519 | 73.3% | 0.018594 | active |
| 27 | directional_progress | strong_later_win | 8-0-0 | 228967.4 | 1.7676 | 73.3% | 0.040164 | active |

## Seed 7

Crossplay contract: horizons [5000]; seeds [101, 131, 151, 181, 211, 241, 271, 311]; two founder pools; balanced slots; objective `survival_times_relative_advantage`.

### Training checkpoint trajectory

| Gen | Best fit | Mean fit | Best survival | Mean survival | Relative | Species | Crossovers | Expr nodes/conns |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.5441 | 0.0835 | 0.3139 | 0.0696 | 1.6925 | 1 | 21 | 0/45 |
| 10 | 1.5158 | 0.7199 | 0.8961 | 0.5037 | 1.6466 | 1 | 17 | 0/45 |
| 20 | 1.4259 | 0.7145 | 0.8566 | 0.4788 | 1.6135 | 5 | 3 | 1/46 |
| 30 | 1.8111 | 0.6930 | 0.9846 | 0.4714 | 1.8391 | 5 | 7 | 1/46 |
| 40 | 1.7462 | 0.7044 | 0.9598 | 0.4604 | 1.8124 | 5 | 10 | 1/47 |
| 49 | 1.7563 | 0.6761 | 0.9667 | 0.4220 | 1.8130 | 4 | 0 | 1/48 |

### Cross-generation strength at longest horizon

| Gen | Mean relative edge vs checkpoints | Pair wins |
|---:|---:|---:|
| 0 | 0.5495 | 4.0/5 |
| 10 | -0.7343 | 1.0/5 |
| 20 | -0.3826 | 1.0/5 |
| 30 | 0.1777 | 3.0/5 |
| 40 | 0.2059 | 4.0/5 |
| 49 | 0.1838 | 2.0/5 |

### Behavior: first vs last checkpoint in direct crossplay

| Checkpoint | Role | Action eff. | Plant rate | Prey rate | Plant share | Prey share | Kills | Total energy | MI(S,A) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| first | omnivore | 0.113311 | 0.051389 | 0.000686 | 0.9871 | 0.0129 | 54.25 | 102665.5 | 0.295997 |
| last | forager | 0.246747 | 0.084469 | 0.000000 | 1.0000 | 0.0000 | 0.00 | 64645.0 | 0.484593 |

### Late breeding and structural health

Generations 33 onward: mean species 4.47 (range 4-5), mean singleton fraction 0.4265; 212 crossovers across 94.1% of generations; 23/29 structural mutations succeeded, with 10 registry-new mutations. Health classification: **active**.

## Seed 17

Crossplay contract: horizons [5000]; seeds [101, 131, 151, 181, 211, 241, 271, 311]; two founder pools; balanced slots; objective `survival_times_relative_advantage`.

### Training checkpoint trajectory

| Gen | Best fit | Mean fit | Best survival | Mean survival | Relative | Species | Crossovers | Expr nodes/conns |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.0526 | 0.0431 | 0.0487 | 0.0430 | 1.0783 | 1 | 15 | 0/45 |
| 10 | 0.1682 | 0.1204 | 0.1456 | 0.1133 | 1.1553 | 1 | 18 | 0/45 |
| 20 | 0.2456 | 0.1548 | 0.2083 | 0.1497 | 1.1718 | 3 | 12 | 0/45 |
| 30 | 0.2773 | 0.1660 | 0.2241 | 0.1562 | 1.2329 | 4 | 10 | 0/45 |
| 40 | 0.3140 | 0.1675 | 0.2356 | 0.1531 | 1.3230 | 5 | 10 | 0/45 |
| 49 | 0.4557 | 0.2426 | 0.3506 | 0.2191 | 1.1166 | 4 | 0 | 2/49 |

### Cross-generation strength at longest horizon

| Gen | Mean relative edge vs checkpoints | Pair wins |
|---:|---:|---:|
| 0 | -1.1792 | 0.0/5 |
| 10 | 0.0365 | 2.0/5 |
| 20 | 0.3490 | 3.0/5 |
| 30 | 0.5125 | 4.0/5 |
| 40 | 0.5413 | 5.0/5 |
| 49 | -0.2602 | 1.0/5 |

### Behavior: first vs last checkpoint in direct crossplay

| Checkpoint | Role | Action eff. | Plant rate | Prey rate | Plant share | Prey share | Kills | Total energy | MI(S,A) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| first | predator | 0.018934 | 0.000000 | 0.001630 | 0.0000 | 1.0000 | 30.75 | 11706.9 | 0.480527 |
| last | omnivore | 0.840984 | 0.094792 | 0.000015 | 0.9998 | 0.0002 | 1.00 | 132904.1 | 0.027911 |

### Late breeding and structural health

Generations 33 onward: mean species 4.71 (range 4-5), mean singleton fraction 0.0118; 97 crossovers across 88.2% of generations; 33/39 structural mutations succeeded, with 23 registry-new mutations. Health classification: **active**.

## Seed 27

Crossplay contract: horizons [5000]; seeds [101, 131, 151, 181, 211, 241, 271, 311]; two founder pools; balanced slots; objective `survival_times_relative_advantage`.

### Training checkpoint trajectory

| Gen | Best fit | Mean fit | Best survival | Mean survival | Relative | Species | Crossovers | Expr nodes/conns |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.2713 | 0.0604 | 0.1702 | 0.0545 | 1.5767 | 1 | 19 | 0/45 |
| 10 | 0.9317 | 0.3987 | 0.6100 | 0.3081 | 1.5114 | 1 | 19 | 0/45 |
| 20 | 1.5256 | 0.7071 | 0.8767 | 0.4903 | 1.6495 | 3 | 13 | 0/45 |
| 30 | 1.4183 | 0.6771 | 0.8808 | 0.4610 | 1.5972 | 4 | 8 | 0/45 |
| 40 | 1.5753 | 0.6907 | 0.9258 | 0.5101 | 1.6818 | 3 | 13 | 0/45 |
| 49 | 1.3985 | 0.4537 | 0.9289 | 0.3387 | 1.4930 | 4 | 0 | 1/46 |

### Cross-generation strength at longest horizon

| Gen | Mean relative edge vs checkpoints | Pair wins |
|---:|---:|---:|
| 0 | -1.7700 | 0.0/5 |
| 10 | -0.1467 | 1.0/5 |
| 20 | -0.1190 | 3.0/5 |
| 30 | 0.9666 | 5.0/5 |
| 40 | 1.0249 | 4.0/5 |
| 49 | 0.0442 | 2.0/5 |

### Behavior: first vs last checkpoint in direct crossplay

| Checkpoint | Role | Action eff. | Plant rate | Prey rate | Plant share | Prey share | Kills | Total energy | MI(S,A) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| first | forager | 0.131336 | 0.018887 | 0.000000 | 1.0000 | 0.0000 | 0.00 | 5655.0 | 0.147720 |
| last | forager | 0.092781 | 0.041653 | 0.000000 | 1.0000 | 0.0000 | 0.00 | 205395.0 | 0.318401 |

### Late breeding and structural health

Generations 33 onward: mean species 3.29 (range 2-4), mean singleton fraction 0.0343; 214 crossovers across 94.1% of generations; 25/29 structural mutations succeeded, with 16 registry-new mutations. Health classification: **active**.

## Fixed interpretation rubric

A direct win requires later-over-earlier agreement in paired alive ticks, mean alive-tick delta, and relative advantage. A strong direct win wins at least 75% of non-tied cases. A run is `directional_progress` only when the final checkpoint wins directly, at least 60% of all checkpoint pairs favor the later genome, and crossplay strength has a positive generation slope. Thus a late champion beating generation 0 while intermediate ordering cycles is reported as nonmonotonic, not directional progress.

Training fitness is reported only as the within-generation selection trajectory. Cross-generation claims come exclusively from frozen, exact two-genome, slot-balanced crossplay on the stated seeds.
