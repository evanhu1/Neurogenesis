# Historical-opponent pressure experiment

## Question

Does replacing contemporary opponents with a deterministic archive improve
retention across evolutionary generations without changing the 32 scored cases
per genome?

The three matched arms used run seeds 7, 17, and 27:

| Arm | Contemporary slots | Historical slots |
|---|---:|---:|
| Control | 8 | 0 |
| Minimal archive | 6 | 2 |
| Strong archive | 4 | 4 |

Every run used population 24, four training world seeds (`11,29,47,61`), one
5,000-tick horizon, the baseline scenario, a 50×50 world, 100 founders,
predation enabled, mean aggregation, and the same mutation/speciation settings.
Generation 0 necessarily used 8/0 because no prior champion existed. Treatment
archives then contained generation 0 and exact-genome-deduplicated champions
sampled every five generations. All final genomes had exactly 32 scored cases.

## What “win” means

Checkpoint cross-play used generations `0,5,10,15,20,24`. A later checkpoint
wins when its founder lineage accumulated more total alive-ticks than the
earlier lineage in their shared two-genome worlds. Each comparison aggregates
four 5,000-tick world seeds with symmetric slot rotation. Duplicate checkpoint
genomes are counted once. The held-out panel used unseen world seeds
`101,131,151,181`.

## Results

| Panel and comparison | Control | Minimal archive | Strong archive |
|---|---:|---:|---:|
| Training, all later vs earlier | 41/45 (91.1%) | 33/45 (73.3%) | 29/40 (72.5%) |
| Training, final vs all earlier | 14/15 (93.3%) | 10/15 (66.7%) | 8/14 (57.1%) |
| Training, successive checkpoints | 13/15 (86.7%) | 10/15 (66.7%) | 10/14 (71.4%) |
| Held out, all later vs earlier | 41/45 (91.1%) | 33/45 (73.3%) | 31/40 (77.5%) |
| Held out, final vs all earlier | **15/15 (100%)** | 9/15 (60.0%) | 8/14 (57.1%) |
| Held out, successive checkpoints | **14/15 (93.3%)** | 11/15 (73.3%) | 12/14 (85.7%) |

The one reduced denominator is legitimate deduplication: strong seed 17's
generation-15 champion was exactly its generation-10 champion. Its archive
therefore stored generations `0,5,10,20` rather than storing that genome twice.

Final held-out losses show that the treatments did not even reliably retain
performance against their own archived opponents:

| Run | Earlier generations that beat generation 24 |
|---|---|
| Control 7 / 17 / 27 | none / none / none |
| Minimal 7 / 17 / 27 | 0 / 10,20 / 10,15,20 |
| Strong 7 / 17 / 27 | 0,15 / 10 (also duplicate 15) / 10,15,20 |

The behavior diagnostics point in the same direction. Final control champions
were forager, omnivore, omnivore across the three seeds, with mean attack kills
`0, 7.0625, 2`. Minimal finished at `0, 0.5, 0.0625` kills. Every strong-archive
final champion was a pure forager with zero kills. This was not an inability to
discover predation: for example, minimal seed 27 reached 42.4, 44.9, and 40.9
mean kills at generations 10, 15, and 20, then fell to 0.0625 at generation 24.
Strong seed 17 similarly fell from 40.4 kills at generation 15 to zero by
generation 20. The archive composition did not prevent behavioral loss.

Raw final training absolute survival did not reveal a hidden archive advantage.
Across seeds, mean final best absolute survival was 0.703 control, 0.633 minimal,
and 0.763 strong; population mean was 0.375, 0.327, and 0.368. These values are
not common-opponent comparisons, which is why the cross-play result is primary.

## Cost accounting

Fixed cases per genome did not fix simulator cost. A contemporary world scores
two evolving genomes, while a historical world scores only one. Including the
all-contemporary bootstrap, each run simulated:

| Arm | Pairwise worlds | Relative to control |
|---|---:|---:|
| Control | 9,600 | 1.00× |
| Minimal archive | 11,904 | 1.24× |
| Strong archive | 14,208 | 1.48× |

## Conclusion

Reject historical-opponent composition with ordinary mean aggregation as the
next default. It consumed more simulation work, reduced contemporary opponent
breadth, lowered common-panel historical retention, and suppressed or forgot
predatory behavior.

The mechanism explains the failure. Beating an old opponent is not the same as
retaining the old behavior, and a mean lets a candidate sacrifice some archive
matches if other cases compensate. Early in a run, repeated weak archive entries
also dilute contemporary arms-race pressure.

After review, the mechanism was removed completely rather than extended with a
more elaborate archive criterion. The contemporary control already retained its
history on this experiment, so there is no observed forgetting problem that
justifies additional selection machinery. Checkpoint persistence and cross-play
remain as diagnostics; a new anti-forgetting mechanism should only be considered
if longer runs demonstrate an actual recurrence.

## Reproduction and analysis

Each arm directory contains the complete NEAT result, progress JSONL, champion
world, training cross-play, and held-out cross-play. `analyze.mjs` recomputes all
aggregate values above and supports `SUMMARY_ONLY=1` for compact output.

The following is the archival evolution command used for this rejected
experiment. Its historical parameters have intentionally been removed from the
current CLI:

```sh
./target/release/sim-cli neat --seed 7 --population 24 --generations 25 \
  --episode-horizons 5000 --opponents 8 --world-seeds 11,29,47,61 \
  --scenarios baseline --workers 12 --scale 50,100 \
  --set predation_enabled=true --param historical_eval_opponents=2 \
  --param historical_archive_interval=5 \
  --out-dir artifacts/research/runs/completed/historical-opponent-pressure/minimal-seed-7
```

Representative held-out cross-play command:

```sh
./target/release/sim-cli neat crossplay RESULT.json \
  --checkpoints 0,5,10,15,20,24 --horizons 5000 \
  --world-seeds 101,131,151,181
```
