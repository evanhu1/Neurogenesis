# TASKS

## Coordination Rules

- Keep frontend data types/models in sync with backend data types/models as
  changes are made.
- Update `specs/spec.md` continuously so it stays fully aligned with the
  implemented system behavior.
- Keep turn-runner behavior updates explicitly documented in `specs/spec.md`
  because `specs/TURN_RUNNER_SPEC.md` is not present in this repo.

# Goals

Mutation changes:

- Replace add neuron mutation with Split edge operator, which uniform samples
  one edge and replaces it with edge to new neuron and edge from new neuron to
  the old post neuron target.
- Get rid of the remove neuron mutation.
- Separate mutation rates per gene instead of one global rate. Each mutation
  rate itself is mutable with a global mutation rate mutation rate (1 /
  sqrt(2*sqrt(n))) where n = number of mutable genes. expose each individual
  mutation rate as a config parameter, and make structural mutations much rarer
  than non structural ones (add neuron vs adjust weight).
- Use Gaussian perturb for all continuous parameters — weights, bias, mutation
  rates — instead of +/-1

Brain model changes:

- Make sensory output synapses all excitatory (positive weight)
- Add a interneuron_type field to the interneuron that specifies enum
  ExcitatoryNeuron or InhibitoryNeuron. Assert somewhere that the weight is
  correspondingly positive or negative
- Inductive bias 80/20 distribution of excitatory to inhibitory neurons in the
  random generation of new neurons
- Implement Dale’s law when creating new synapses or mutating them: Each neuron
  has a type ∈ {E, I}. All outgoing weights must share the same sign.
- Change synapse strength distribution to be log-normal, when creating synapses

## Sequential Implementation Tasks

1. **Define the new genome/brain data model surface (before coding logic).**
   - Add an interneuron polarity enum in `sim-types` (excitatory/inhibitory).
   - Add `interneuron_type` to `InterNeuronState`.
   - Replace single `mutation_rate` with explicit per-gene mutation-rate fields
     in `OrganismGenome` and `SeedGenomeConfig`.
   - Include mutation-rate genes for every mutable operator/parameter we intend
     to keep (weights, biases, update-rates, structural ops, scalar traits).

2. **Propagate schema changes everywhere they are serialized/deserialized.**
   - Update `sim-types/src/lib.rs` structs and serde compatibility.
   - Update `config/default.toml` with all individual mutation-rate config keys.
   - Update `web-client/src/types.ts` and default TOML parser to match new
     config/genome fields.
   - Update any UI selectors that still reference the removed
     `genome.mutation_rate`.

3. **Introduce shared helpers for signed/log-normal synapse generation and
   Gaussian perturbation.**
   - Add helper(s) in `sim-core/src/genome.rs` for log-normal magnitude
     sampling.
   - Add helper(s) to derive required sign from neuron source type (sensory
     always `+`, excitatory `+`, inhibitory `-`).
   - Ensure continuous mutations (weights, biases, mutation-rate genes) use
     Gaussian perturbation, not stepwise deltas.

4. **Implement interneuron-type generation and persistence.**
   - Seed generation assigns interneuron types with 80/20 excitatory/inhibitory
     split.
   - New neurons created by structural mutation also use the same 80/20 prior.
   - Keep per-neuron type vectors aligned when neuron count changes.

5. **Refactor structural mutation operators.**
   - Remove the old add-neuron mutation behavior.
   - Remove neuron-removal mutation behavior.
   - Add split-edge mutation:
     - Uniformly sample an existing edge.
     - Remove it.
     - Insert edge `(old_pre -> new_inter)` and edge `(new_inter -> old_post)`.
     - Respect ID bounds, edge uniqueness, sorted edge invariants, and
       neuron-count limits.

6. **Apply per-gene mutation rates with self-adaptive mutation-rate mutation.**
   - Gate each mutation operator by its own mutation-rate gene.
   - Mutate mutation-rate genes themselves using global rate-rate
     `time_constant = 1 / sqrt(2 * sqrt(n))`, `n = number of mutable genes`.
   - Clamp all mutation rates to valid bounds and keep structural rates much
     smaller than non-structural defaults in config.

7. **Enforce Dale’s law and sensory-excitatory constraints in mutation +
   generation paths.**
   - Every newly created/perturbed synapse must keep the sign implied by its
     pre-neuron type.
   - Sensory outgoing synapses must always remain positive.
   - Edge-add and edge-split flows both enforce this invariant.

8. **Add runtime assertions in brain expression for invariant safety.**
   - In `sim-core/src/brain.rs`, assert that sensory outgoing weights are
     positive.
   - Assert that each inter neuron’s outgoing edges match its declared
     `interneuron_type` sign.
   - Keep assertions deterministic and cheap in release (debug assertions where
     appropriate).

9. **Update genome-distance and validation logic for new genes.**
   - Include new mutation-rate genes and interneuron-type differences in
     `genome_distance`.
   - Replace old single-rate config validation with per-field validation for all
     mutation-rate config parameters.
   - Validate any new structural constraints (e.g., vector lengths, sign
     consistency if validated at genome layer).

10. **Repair and extend Rust tests for behavior + determinism.**

- Update existing tests/fixtures/bench configs that currently construct
  `mutation_rate`.
- Add focused tests for:
  - split-edge operator behavior,
  - no neuron-removal mutation,
  - sensory edges always positive,
  - Dale’s law sign enforcement,
  - mutation-rate self-adaptation bounds/stability.
- Keep fixed RNG seeds for reproducibility.

11. **Run full validation and fix regressions.**

- Run `cargo check --workspace`.
- Run `cargo test --workspace`.
- Run frontend type checks/build if TS types changed
  (`cd web-client && npm run build && npm run typecheck`).
- Resolve all compile/test failures caused by schema and behavior updates.

12. **Update project docs/spec to reflect final behavior.**

- Document new mutation operators and mutation-rate gene system.
- Document interneuron polarity model, sensory excitatory constraint, Dale’s
  law, and log-normal synapse init.
- Ensure docs align with final implemented constants/defaults.
