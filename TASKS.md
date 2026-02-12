# TASKS

## Coordination Rules

- Keep frontend data types/models in sync with backend data types/models as
  changes are made.
- Update `specs/spec.md` continuously so it stays fully aligned with the
  implemented system behavior.
- Keep turn-runner behavior updates explicitly documented in `specs/spec.md`
  because `specs/TURN_RUNNER_SPEC.md` is not present in this repo.

## Sequential Implementation Tasks

### 1. Change Interneurons To Per-Interneuron Leaky Integrators

1. Add state and genome fields.
   - Add `update_rate: f32` to `InterNeuronState`.
   - Add `inter_update_rates: Vec<f32>` to `OrganismGenome`.
   - Keep validation aligned with `inter_biases` (`len == num_neurons`).
2. Implement seed initialization for `inter_update_rates`.
   - In `generate_seed_genome`, sample each update rate from a log-uniform
     distribution on `[0.03, 1.0]`.
   - Use deterministic RNG flow so fixed seed behavior is preserved.
3. Implement mutation for per-neuron update rates.
   - Add Gaussian perturbation with stddev near `0.05`.
   - Clamp to a strict non-zero/stable range (same `[0.03, 1.0]` bounds).
4. Update brain evaluation to leaky-integrator dynamics.
   - Compute `z_i(t) = b_i + sensory_to_inter_sum_i + inter_to_inter_sum_i(h(t-1))`.
   - Compute `h_i(t) = (1 - alpha_i) * h_i(t-1) + alpha_i * tanh(z_i(t))`.
   - Ensure the recurrent term reads only `h(t-1)` to avoid order dependence.
5. Add deterministic tests.
   - Seed genome tests for update-rate count and bounds.
   - Mutation tests for clamping and non-zero behavior.
   - Brain-step tests for expected temporal smoothing with different `alpha_i`.
6. Keep docs and protocol-adjacent types aligned.
   - Update any affected spec/docs and shared type mirrors.

### 2. Combine Left/Right Turn Into A Single Turn Action (tanh + deadzone)

1. Replace separate turn actions in action definitions.
   - Collapse `turn_left` and `turn_right` into one `turn` action signal.
   - Update action IDs/enums/constants consistently across crates and client.
2. Change turn output activation and interpretation.
   - Use `tanh` for the turn action output (signed direction signal).
   - Apply a deadzone around `0` (`|signal| <= epsilon` means no turn).
   - Map sign to direction (`signal < 0` left, `signal > 0` right).
3. Update intent generation and turn application.
   - Ensure turning logic consumes the unified signed value.
   - Preserve deterministic behavior and tie-breaking semantics.
4. Update serialization/client handling.
   - Align server protocol, client types, and UI behavior with one turn action.
5. Add/adjust tests.
   - Verify deadzone no-op behavior.
   - Verify signed turning behavior and deterministic outcomes.

### 3. Allow Self Recurrence In Interneurons

1. Relax genome/edge validation rules.
   - Permit `inter -> same inter` edges (self loops).
2. Ensure evaluation semantics remain deterministic.
   - Self-recurrent contribution must come from `h_i(t-1)` only.
   - No same-tick feedback path should depend on neuron iteration order.
3. Update genome generation and mutation behavior.
   - Allow creation/retention of self-recurrent edges where inter->inter edges
     are handled.
4. Add tests.
   - Validation test that self recurrence is accepted.
   - Brain-step test confirming expected self-memory behavior across ticks.

### 4. Allow Bias Term In Action Neurons

1. Extend genome with action biases.
   - Add `action_biases: Vec<f32>` to `OrganismGenome`.
   - Validate count against the number of action neurons.
2. Initialize action biases in seed genomes.
   - Define initialization policy (e.g., zero or small random values) and keep
     deterministic seed behavior.
3. Add mutation support for action biases.
   - Apply mutation gates consistent with existing genome mutation flow.
   - Use bounded/clamped perturbations compatible with current bias handling.
4. Update brain action evaluation.
   - Add action bias term before action activation for each action neuron.
   - Keep existing firing/threshold logic unless explicitly changed by task 2.
5. Update tests and shared models.
   - Add coverage for action bias effect on action outputs.
   - Keep Rust/shared/client type definitions synchronized.
