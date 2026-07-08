//! The developed phenotype network. Environment-agnostic: `step` takes an
//! observation vector and returns action logits. The neuron math, `fast_tanh`,
//! weight clamp, α-from-time-constant, and softmax sampler are lifted verbatim
//! from the current engine (`sim-core/src/brain/{mod,evaluation}.rs` and
//! `genome/scalar.rs`) so ported determinism/behavior is preserved.

use serde::{Deserialize, Serialize};

// Weight clamp band, identical to `sim-core`'s `SYNAPSE_STRENGTH_{MIN,MAX}`.
pub const SYNAPSE_STRENGTH_MIN: f32 = 0.001;
pub const SYNAPSE_STRENGTH_MAX: f32 = 1.5;

// Leaky-integrator time-constant band and softmax constants, identical to
// `sim-core`.
pub const INTER_TIME_CONSTANT_MIN: f32 = 0.1;
pub const INTER_TIME_CONSTANT_MAX: f32 = 10.0;
pub const INTER_LOG_TIME_CONSTANT_MIN: f32 = -std::f32::consts::LN_10;
pub const INTER_LOG_TIME_CONSTANT_MAX: f32 = std::f32::consts::LN_10;
pub const EXPLICIT_IDLE_LOGIT_BIAS: f32 = -0.01;
pub const MIN_ACTION_TEMPERATURE: f32 = 1.0e-6;

/// Padé rational-polynomial `tanh` approximation — byte-identical to
/// `sim-core::brain::fast_tanh`.
#[inline(always)]
pub fn fast_tanh(x: f32) -> f32 {
    if x >= 4.97 {
        return 1.0;
    }
    if x <= -4.97 {
        return -1.0;
    }
    let x2 = x * x;
    let num = x * (135135.0 + x2 * (17325.0 + x2 * (378.0 + x2)));
    let den = 135135.0 + x2 * (62370.0 + x2 * (3150.0 + x2 * 28.0));
    num / den
}

/// Keep an edge's sign, clamp its magnitude into the strength band, mapping
/// exact zero to the minimum — identical to `sim-core::brain::constrain_weight`.
#[inline]
pub fn constrain_weight(weight: f32) -> f32 {
    if weight == 0.0 {
        return SYNAPSE_STRENGTH_MIN;
    }
    weight.signum() * weight.abs().clamp(SYNAPSE_STRENGTH_MIN, SYNAPSE_STRENGTH_MAX)
}

/// Leaky-integrator α from a log time constant — identical to
/// `sim-core::genome::inter_alpha_from_log_time_constant`.
#[inline]
pub fn inter_alpha_from_log_time_constant(log_time_constant: f32) -> f32 {
    let clamped = log_time_constant.clamp(INTER_LOG_TIME_CONSTANT_MIN, INTER_LOG_TIME_CONSTANT_MAX);
    let tau = clamped.exp().clamp(INTER_TIME_CONSTANT_MIN, INTER_TIME_CONSTANT_MAX);
    1.0 - (-1.0 / tau).exp()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NeuronKind {
    Input,
    Hidden,
    Output,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Neuron {
    pub kind: NeuronKind,
    pub bias: f32,
    /// Leaky-integrator rate (hidden neurons only; inputs/outputs ignore it).
    pub alpha: f32,
    // Runtime state (serde-defaulted so a freshly-developed brain starts clean).
    #[serde(default)]
    pub state: f32,
    #[serde(default)]
    pub activation: f32,
    #[serde(default)]
    pub mean_activation: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Edge {
    pub from: u32,
    pub to: u32,
    pub weight: f32,
    /// Per-edge plasticity-rate scale painted by the CPPN's `PLR` output
    /// (hybrid adaptive-HyperNEAT). `1.0` reproduces the global rule exactly.
    pub plasticity_scale: f32,
    #[serde(default)]
    pub eligibility: f32,
    #[serde(default)]
    pub pending: f32,
}

/// A developed brain. Neurons are ordered `[inputs..][hidden..][outputs..]`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BrainNet {
    pub neurons: Vec<Neuron>,
    pub edges: Vec<Edge>,
    pub input_count: u32,
    pub hidden_count: u32,
    pub output_count: u32,
    #[serde(default)]
    pub means_initialized: bool,
    // Scratch buffers, rebuilt lazily; skipped from serialization.
    #[serde(skip)]
    scratch_prev: Vec<f32>,
    #[serde(skip)]
    scratch_inputs: Vec<f32>,
    #[serde(skip)]
    logits: Vec<f32>,
}

impl BrainNet {
    pub fn new(
        neurons: Vec<Neuron>,
        edges: Vec<Edge>,
        input_count: u32,
        hidden_count: u32,
        output_count: u32,
    ) -> Self {
        let out = output_count as usize;
        BrainNet {
            neurons,
            edges,
            input_count,
            hidden_count,
            output_count,
            means_initialized: false,
            scratch_prev: Vec::new(),
            scratch_inputs: Vec::new(),
            logits: vec![0.0; out],
        }
    }

    #[inline]
    fn output_start(&self) -> usize {
        (self.input_count + self.hidden_count) as usize
    }
    #[inline]
    fn hidden_start(&self) -> usize {
        self.input_count as usize
    }

    /// One forward pass: obs vector → action logits. Recurrent hidden neurons
    /// integrate over their *previous* activation (leaky integrator); outputs
    /// read the freshly-updated hidden + input activations, mirroring the
    /// current engine's sensory→inter→action dataflow.
    pub fn step(&mut self, observation: &[f32]) -> &[f32] {
        let n = self.neurons.len();
        let hidden_start = self.hidden_start();
        let output_start = self.output_start();

        // Load observation into input activations.
        for (i, neuron) in self.neurons[..hidden_start].iter_mut().enumerate() {
            neuron.activation = observation.get(i).copied().unwrap_or(0.0);
        }

        // Snapshot previous activations (hidden recurrence reads pre-update).
        self.scratch_prev.clear();
        self.scratch_prev.extend(self.neurons.iter().map(|nn| nn.activation));

        // Accumulate hidden inputs from prev activations.
        self.scratch_inputs.clear();
        self.scratch_inputs.resize(n, 0.0);
        for h in hidden_start..output_start {
            self.scratch_inputs[h] = self.neurons[h].bias;
        }
        for edge in &self.edges {
            let to = edge.to as usize;
            if to >= hidden_start && to < output_start {
                self.scratch_inputs[to] += self.scratch_prev[edge.from as usize] * edge.weight;
            }
        }
        // Commit hidden state/activation.
        for h in hidden_start..output_start {
            let neuron = &mut self.neurons[h];
            let alpha = neuron.alpha;
            neuron.state = (1.0 - alpha) * neuron.state + alpha * self.scratch_inputs[h];
            neuron.activation = fast_tanh(neuron.state);
        }

        // Accumulate output logits from updated hidden + inputs.
        for o in output_start..n {
            self.scratch_inputs[o] = self.neurons[o].bias;
        }
        for edge in &self.edges {
            let to = edge.to as usize;
            if to >= output_start {
                self.scratch_inputs[to] += self.neurons[edge.from as usize].activation * edge.weight;
            }
        }
        self.logits.clear();
        for o in output_start..n {
            let logit = self.scratch_inputs[o];
            self.neurons[o].activation = logit; // outputs are linear pre-softmax
            self.logits.push(logit);
        }
        &self.logits
    }

    pub fn logits(&self) -> &[f32] {
        &self.logits
    }
}

/// Softmax-with-explicit-idle action sampler — the env-agnostic form of
/// `sim-core`'s `sample_action_from_logits`. Returns `None` for the implicit
/// idle outcome, else the winning action-slot index. `sample` is the
/// deterministic `(seed,turn,id)`-hashed uniform draw supplied by the driver.
pub fn sample_action(logits: &[f32], temperature: f32, sample: f32) -> Option<usize> {
    if logits.is_empty() {
        return None;
    }
    let temperature = temperature.max(MIN_ACTION_TEMPERATURE);
    let idle_bias = EXPLICIT_IDLE_LOGIT_BIAS;
    let max_logit = logits.iter().copied().fold(idle_bias, f32::max);
    let mut weights: Vec<f32> = Vec::with_capacity(logits.len());
    let mut weight_sum = 0.0f32;
    for &logit in logits {
        let w = ((logit - max_logit) / temperature).exp();
        weights.push(w);
        weight_sum += w;
    }
    let idle_weight = ((idle_bias - max_logit) / temperature).exp();
    weight_sum += idle_weight;

    let target = sample.clamp(0.0, 1.0 - f32::EPSILON) * weight_sum;
    let mut cumulative = 0.0f32;
    for (idx, &w) in weights.iter().enumerate() {
        cumulative += w;
        if target < cumulative {
            return Some(idx);
        }
    }
    None
}
