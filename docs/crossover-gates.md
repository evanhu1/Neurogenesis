# Crossover Gates 1-3

This experiment asks one narrow question: **does recombination add evolutionary
value when within-lifetime learning is frozen?** It does not claim open-ended
evolution. Passing these gates only establishes that crossover is a sound and
useful outer-loop operator worth carrying into the next objective.

## Locked comparison

The canonical inputs live in `sim-evaluation/crossover-gate/`. Run exactly two
matched-seed arms:

- `paired_clone`: select the same two parents and inherit one intact parent
  brain.
- `crossover`: start from that same intact carrier brain and replace one
  informative action pathway with the other parent's linked module.

Both arms keep the initiating parent's non-neural genome, use the same
carrier-selection bit, apply the same post-inheritance mutation, and differ
only in whether the linked donor module is imported. Plasticity, meta-mutation,
random actions, periodic founder injection, and every non-brain mutation rate
are disabled. Shared-ancestor founders create shallow homologous variation.
`crossover_probability` controls recombination load per conception; it must be
in `(0, 1]` for a gate run. On a miss, the child inherits the exact paired-clone
A/B carrier and still uses the same offspring mutation seed.

`sim-cli sweep --gate crossover` validates the generic experimental invariants;
`--gate crossover-v1` additionally locks the compiled preset, 64 held-out
seeds, 3,000-tick horizon, and report interval. Both record the source TOML,
parsed effective configurations, and executable SHA-256 in the result JSON.
`invariants_validated` is not an outcome verdict; Gates 1–3 must still be
evaluated from the reported measurements.

## Pre-registered gates

### Gate 1: recombination integrity

Across the crossover arm:

1. At least 100 crossover conceptions are independently audited.
2. Canonical failures, dangling edges, novel pre-mutation brain alleles, and
   operator/auditor contribution mismatches are all zero.
3. At least 95% of successful audited children contain informative brain loci
   from both parents.
4. Replaying an identical crossover world produces byte-identical world and
   metric-sidecar files.

This proves that stable identities align homologous genes and that the
operator produces real, replayable biparental genomes. It does not establish
fitness value. The audit is deliberately a genotype-provenance check over the
encoded graph; module eligibility separately requires an informative donor
allele on an enabled pathway, but the audit rate is not itself a measure of
expressed contribution from both parents.

### Gate 2: lineage viability

The crossover arm must complete without run errors and emit topology for every
successful child. At least 75% of seeds must contain a successfully born,
independently-audited biparental crossover child whose maturity window is fully
observed and that later initiates a successful reproduction as parent A. The
median per-seed maximum generation must be at least three. We report this exact
cohort, its reproductive-child fraction, and a per-seed binary success value
whose false cases include empty/extinct cohorts. We also report per-seed birth
count, maximum generation, and final descendant population. A result driven
only by conceptions, non-recombined children, or newborns that never reproduce
does not pass.

### Gate 3: evolutionary value

The single primary endpoint is the paired same-seed delta in
`successful_births` (`crossover - paired_clone`). Gate 3 passes only when its
two-sided 95% Student-t confidence interval is entirely above zero. The locked
holdout has 64 seed pairs (`1000..=1063`); no seed may be added, removed, or
replaced after inspecting the result.

Guardrails are the paired deltas for `maturity_observed_lineage_success` (a
binary per-seed outcome that treats an empty or extinct cohort as failure) and
`final_descendant_population`: neither may show statistically significant harm
(an interval entirely below zero). The complete-case reproductive-child
fraction, maximum generation, and behavior pillars are supporting diagnostics,
not substitute endpoints. If the primary endpoint fails, favorable secondary
metrics do not rescue Gate 3.

## Canonical command

```bash
./target/release/sim-cli sweep \
  --gate crossover-v1 \
  --config sim-evaluation/crossover-gate/config.toml \
  --grid reproduction_mode=paired_clone,crossover \
  --baseline reproduction_mode=paired_clone \
  --seeds 1000,1001,1002,1003,1004,1005,1006,1007,1008,1009,1010,1011,1012,1013,1014,1015,1016,1017,1018,1019,1020,1021,1022,1023,1024,1025,1026,1027,1028,1029,1030,1031,1032,1033,1034,1035,1036,1037,1038,1039,1040,1041,1042,1043,1044,1045,1046,1047,1048,1049,1050,1051,1052,1053,1054,1055,1056,1057,1058,1059,1060,1061,1062,1063 \
  --to 3000 \
  --report-every 100 \
  --threads 1 \
  --jobs 16 \
  --out-dir artifacts/crossover-gates/sweeps
```

Use previously observed seeds for operator tuning. Once the operator and
canonical inputs are locked, evaluate Gate 3 once on a new held-out seed suite
and retain the result whether it passes or fails.

The final settings were locked from development cohorts before the holdout:
`founder_diversification_rounds=2`, `crossover_probability=0.5`,
`mate_compatibility_threshold=0.6`, and a 3,000-tick horizon. The longer horizon
was chosen because 1,500-tick development effects were not durable.

## Held-out result: operator version 1

Artifact:
`artifacts/crossover-gates/sweeps/sweep-1783629810916-56775.json`.
All 64 pre-registered seed pairs completed, the gate configuration invariants
validated, and the result contains no run errors. The artifact fingerprints the
release executable as
`49e3a03348c9fdbe63d540dd1cd954f3dfb39b38f1328d35147339da2f20c729`.

- **Gate 1: pass.** There were 23,337 independent crossover audits. Canonical
  failures, dangling edges, novel pre-mutation alleles, and operator/auditor
  mismatches were all zero. Of 10,719 successfully born audited children,
  10,465 (97.63%) had informative brain inheritance from both parents. A
  separate 3,000-tick replay produced byte-identical world and metric files
  (`b37f59d89b07a960658aeccde204a466813d6b7f8751e864742f5e4eb8f65ad4`
  and `e10e6bde8e641e3b1ab6b483dd5b9a1c46d6760bc8120993052d671bc70aead6`,
  respectively).
- **Gate 2: fail under the pre-registered rule.** All successful children had
  topology, but only 32/64 seeds (50%) produced a maturity-observed audited
  biparental child that later reproduced, below the 75% threshold. Median
  maximum generation was two, below three. The paired-clone arm itself reached
  all-mode lineage success in only 35/64 seeds and also had median maximum
  generation two; this shows that the absolute Gate-2 threshold was poorly
  calibrated to environmental viability, but it cannot be revised after the
  holdout.
- **Gate 3: fail.** The mean paired birth delta was -75.91 with 95% Student-t CI
  [-317.57, +165.76] (20 wins, 32 ties, 12 losses). The lineage-success
  guardrail was +0.0156 [-0.0546, +0.0859], and final descendant population was
  +1.05 [-8.06, +10.15], so neither guardrail showed significant harm; the
  primary endpoint nevertheless failed.

The result rejects random action-module replacement at the tested 50% rate as
a proven evolutionary improvement in this ecology. A few large recombination
losses outweighed more frequent small wins. Do not reuse this holdout to certify
a revised operator; any version 2 needs a new mechanism, criteria calibrated on
development data, and a new seed suite.
