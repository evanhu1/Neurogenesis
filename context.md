# NeuroGenesis Context

NeuroGenesis is a deterministic artificial-life simulation project written
primarily in Rust. The `sim-core` crate implements the simulation engine (turn
pipeline, neural brains, genome logic, spawning, and grid/world mechanics),
while companion crates provide server networking and a web client for
visualization.

The goals of this project:

- Neurogenesis is a simulation of the evolution of intelligent brains in a
  digital environment. It contains a hard-coded environment (hex grid, plant
  growth), organisms with a few hard-coded (lifespan in ticks) and majority
  evolvable phenotypes. Each organism is a thin wrapper around a brain which is
  a DAG with sensory, inter, and action neuron nodes. The architecture, hyper
  parameters, and properties of this brain are heavily inspired by biological
  brains. The overarching goal of this project is to first successfully evolve
  the in-world equivalent of a C Elegans nematode, and then to evolve further
  intelligent complexity from there. In a more fantasy like sense, we are trying
  to evolve general intelligence with emergent perception, memory, planning and
  simulation, symbolic / abstract reasoning, language, theory of mind, via
  evolutionary curriculum learning that scales environment complexity with
  cognitive complexity, avoiding convergence to stable niches along the way,
  simulating the evolutionary historical pathway we took from nucleotides to
  genes to cells in the ocean, to polyps and nematode, to fish and then mammal
  and primate and then human.
- Copy the efficient and general algorithms and design of human brains, while
  maximally leveraging the advantages of computational hardware over biological
  hardware. We have one empirical case of how solving for long-horizon survival
  in unpredictable environments is enough to bootstrap general intelligence over
  hundreds of millions of years of evolution, in humans—we are the proof. I want
  to compress this hundred million years of evolution into a simulation, cutting
  the orders of magnitude by abstracting the physical world and leveraging the
  speed, precision, and power of computers, math, and many clever tricks /
  well-designed approximations.
- Build a scalable, and real-world like environment that supports the
  neuroevolutionary process by providing a natural cognitive curriculum, all the
  way from food gathering and navigation to reasoning and long term planning.
  For example, need environments where predictive modeling gives survival
  advantage, while also supporting simple niches like gathering food.

## sim-core/src/brain.rs

```rust
use crate::genome::{
    inter_alpha_from_log_tau, BRAIN_SPACE_MAX, BRAIN_SPACE_MIN, DEFAULT_INTER_LOG_TAU,
    SYNAPSE_STRENGTH_MAX, SYNAPSE_STRENGTH_MIN,
};
use crate::grid::{hex_neighbor, rotate_left, rotate_right};
#[cfg(feature = "profiling")]
use crate::profiling::{self, BrainStage};
use rand::Rng;
use sim_types::{
    ActionNeuronState, ActionType, BrainLocation, BrainState, EntityType, InterNeuronState,
    InterNeuronType, NeuronId, NeuronState, NeuronType, Occupant, OrganismGenome, OrganismId,
    SensoryNeuronState, SensoryReceptor, SynapseEdge,
};
#[cfg(feature = "profiling")]
use std::time::Instant;

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

const DEFAULT_BIAS: f32 = 0.0;
const HEBB_WEIGHT_CLAMP_ENABLED: bool = true;
const DOPAMINE_ENERGY_DELTA_SCALE: f32 = 10.0;
const PLASTIC_WEIGHT_DECAY: f32 = 0.001;
const SYNAPSE_PRUNE_INTERVAL_TICKS: u64 = 10;
const MIN_ENERGY_SENSOR_SCALE: f32 = 1.0;
const ENERGY_SENSOR_CURVE_EXPONENT: f32 = 2.0;
const LOOK_TARGETS: [EntityType; 3] = [EntityType::Food, EntityType::Organism, EntityType::Wall];
const LOOK_RAY_COUNT: usize = SensoryReceptor::LOOK_RAY_OFFSETS.len();
pub(crate) const SENSORY_COUNT: u32 = SensoryReceptor::LOOK_NEURON_COUNT + 1;
pub(crate) const ENERGY_SENSORY_ID: u32 = SensoryReceptor::LOOK_NEURON_COUNT;
pub(crate) const ACTION_COUNT: usize = ActionType::ALL.len();
pub(crate) const ACTION_COUNT_U32: u32 = ACTION_COUNT as u32;
pub(crate) const INTER_ID_BASE: u32 = 1000;
pub(crate) const ACTION_ID_BASE: u32 = 2000;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ResolvedActions {
    pub(crate) selected_action: ActionType,
}

impl Default for ResolvedActions {
    fn default() -> Self {
        Self {
            selected_action: ActionType::Idle,
        }
    }
}

#[derive(Default)]
pub(crate) struct BrainEvaluation {
    pub(crate) resolved_actions: ResolvedActions,
    pub(crate) action_activations: [f32; ACTION_COUNT],
    pub(crate) synapse_ops: u64,
}

/// Reusable scratch buffers for brain evaluation, avoiding per-tick allocations.
pub(crate) struct BrainScratch {
    inter_inputs: Vec<f32>,
    prev_inter: Vec<f32>,
    inter_activations: Vec<f32>,
    action_activations: [f32; ACTION_COUNT],
}

impl BrainScratch {
    pub(crate) fn new() -> Self {
        Self {
            inter_inputs: Vec::new(),
            prev_inter: Vec::new(),
            inter_activations: Vec::new(),
            action_activations: [0.0; ACTION_COUNT],
        }
    }
}

/// Build a BrainState from inherited neuron genes and stored synapse topology.
pub(crate) fn express_genome<R: Rng + ?Sized>(genome: &OrganismGenome, _rng: &mut R) -> BrainState {
    let mut sensory = Vec::with_capacity(SENSORY_COUNT as usize);
    let mut sensory_id = 0_u32;
    for ray_offset in SensoryReceptor::LOOK_RAY_OFFSETS {
        for look_target in LOOK_TARGETS {
            sensory.push(make_sensory_neuron(
                sensory_id,
                SensoryReceptor::LookRay {
                    ray_offset,
                    look_target,
                },
                location_or_default(&genome.sensory_locations, sensory_id as usize),
            ));
            sensory_id = sensory_id.saturating_add(1);
        }
    }
    debug_assert_eq!(sensory_id, ENERGY_SENSORY_ID);
    sensory.push(make_sensory_neuron(
        ENERGY_SENSORY_ID,
        SensoryReceptor::Energy,
        location_or_default(&genome.sensory_locations, ENERGY_SENSORY_ID as usize),
    ));

    let mut inter = Vec::with_capacity(genome.num_neurons as usize);
    for i in 0..genome.num_neurons {
        let idx = i as usize;
        let bias = genome.inter_biases.get(idx).copied().unwrap_or(0.0);
        let log_tau = genome
            .inter_log_taus
            .get(idx)
            .copied()
            .unwrap_or(DEFAULT_INTER_LOG_TAU);
        let alpha = inter_alpha_from_log_tau(log_tau);
        let interneuron_type = genome
            .interneuron_types
            .get(idx)
            .copied()
            .unwrap_or(InterNeuronType::Excitatory);
        inter.push(InterNeuronState {
            neuron: make_neuron(
                NeuronId(INTER_ID_BASE + i),
                NeuronType::Inter,
                bias,
                location_or_default(&genome.inter_locations, idx),
            ),
            interneuron_type,
            alpha,
            synapses: Vec::new(),
        });
    }

    let mut action = Vec::with_capacity(ACTION_COUNT);
    for (idx, action_type) in ActionType::ALL.into_iter().enumerate() {
        let bias = genome.action_biases.get(idx).copied().unwrap_or(0.0);
        action.push(make_action_neuron(
            ACTION_ID_BASE + idx as u32,
            action_type,
            bias,
            location_or_default(&genome.action_locations, idx),
        ));
    }

    wire_birth_synapses_from_genome(genome, &mut sensory, &mut inter);

    let mut brain = BrainState {
        sensory,
        inter,
        action,
        synapse_count: 0,
    };
    refresh_parent_ids_and_synapse_count(&mut brain);
    brain
}

pub(crate) fn action_index(action: ActionType) -> usize {
    match action {
        ActionType::Idle => 0,
        ActionType::TurnLeft => 1,
        ActionType::TurnRight => 2,
        ActionType::Forward => 3,
        ActionType::TurnLeftForward => 4,
        ActionType::TurnRightForward => 5,
        ActionType::Consume => 6,
        ActionType::Reproduce => 7,
    }
}

fn location_or_default(locations: &[BrainLocation], index: usize) -> BrainLocation {
    locations.get(index).copied().unwrap_or(BrainLocation {
        x: 0.5 * (BRAIN_SPACE_MIN + BRAIN_SPACE_MAX),
        y: 0.5 * (BRAIN_SPACE_MIN + BRAIN_SPACE_MAX),
    })
}

fn wire_birth_synapses_from_genome(
    genome: &OrganismGenome,
    sensory: &mut [SensoryNeuronState],
    inter: &mut [InterNeuronState],
) {
    let max_inter_id = INTER_ID_BASE + inter.len() as u32;
    let max_action_id = ACTION_ID_BASE + ACTION_COUNT_U32;

    for edge in &genome.edges {
        let post_is_inter = (INTER_ID_BASE..max_inter_id).contains(&edge.post_neuron_id.0);
        let post_is_action = (ACTION_ID_BASE..max_action_id).contains(&edge.post_neuron_id.0);
        if !(post_is_inter || post_is_action) {
            continue;
        }

        if edge.pre_neuron_id.0 < SENSORY_COUNT {
            let Some(pre) = sensory.get_mut(edge.pre_neuron_id.0 as usize) else {
                continue;
            };
            pre.synapses.push(SynapseEdge {
                pre_neuron_id: edge.pre_neuron_id,
                post_neuron_id: edge.post_neuron_id,
                weight: constrain_weight(edge.weight, 1.0),
                eligibility: 0.0,
            });
            continue;
        }

        if !(INTER_ID_BASE..max_inter_id).contains(&edge.pre_neuron_id.0) {
            continue;
        }
        if edge.pre_neuron_id == edge.post_neuron_id {
            continue;
        }
        let pre_idx = (edge.pre_neuron_id.0 - INTER_ID_BASE) as usize;
        let Some(pre) = inter.get_mut(pre_idx) else {
            continue;
        };
        let required_sign = match pre.interneuron_type {
            InterNeuronType::Excitatory => 1.0,
            InterNeuronType::Inhibitory => -1.0,
        };
        pre.synapses.push(SynapseEdge {
            pre_neuron_id: edge.pre_neuron_id,
            post_neuron_id: edge.post_neuron_id,
            weight: constrain_weight(edge.weight, required_sign),
            eligibility: 0.0,
        });
    }

    for sensory_neuron in sensory.iter_mut() {
        sensory_neuron.synapses.sort_by(|a, b| {
            a.post_neuron_id
                .cmp(&b.post_neuron_id)
                .then_with(|| a.weight.total_cmp(&b.weight))
        });
    }
    for inter_neuron in inter.iter_mut() {
        inter_neuron.synapses.sort_by(|a, b| {
            a.post_neuron_id
                .cmp(&b.post_neuron_id)
                .then_with(|| a.weight.total_cmp(&b.weight))
        });
    }
}

fn make_neuron(
    id: NeuronId,
    neuron_type: NeuronType,
    bias: f32,
    location: BrainLocation,
) -> NeuronState {
    NeuronState {
        neuron_id: id,
        neuron_type,
        bias,
        x: location.x,
        y: location.y,
        activation: 0.0,
        parent_ids: Vec::new(),
    }
}

pub(crate) fn make_sensory_neuron(
    id: u32,
    receptor: SensoryReceptor,
    location: BrainLocation,
) -> SensoryNeuronState {
    SensoryNeuronState {
        neuron: make_neuron(NeuronId(id), NeuronType::Sensory, DEFAULT_BIAS, location),
        receptor,
        synapses: Vec::new(),
    }
}

pub(crate) fn make_action_neuron(
    id: u32,
    action_type: ActionType,
    bias: f32,
    location: BrainLocation,
) -> ActionNeuronState {
    ActionNeuronState {
        neuron: make_neuron(NeuronId(id), NeuronType::Action, bias, location),
        action_type,
    }
}

pub(crate) fn evaluate_brain(
    organism: &mut sim_types::OrganismState,
    world_width: i32,
    occupancy: &[Option<Occupant>],
    vision_distance: u32,
    scratch: &mut BrainScratch,
) -> BrainEvaluation {
    let mut result = BrainEvaluation::default();

    #[cfg(feature = "profiling")]
    let stage_started = Instant::now();
    let ray_scans = scan_rays(
        (organism.q, organism.r),
        organism.facing,
        organism.id,
        world_width,
        occupancy,
        vision_distance,
    );
    #[cfg(feature = "profiling")]
    profiling::record_brain_stage(BrainStage::ScanAhead, stage_started.elapsed());

    let energy_signal = energy_sensor_value(
        organism.energy,
        organism.genome.starting_energy.max(MIN_ENERGY_SENSOR_SCALE),
    );

    #[cfg(feature = "profiling")]
    let stage_started = Instant::now();
    for sensory in &mut organism.brain.sensory {
        sensory.neuron.activation = match &sensory.receptor {
            SensoryReceptor::LookRay {
                ray_offset,
                look_target,
            } => look_ray_signal(&ray_scans, *ray_offset, *look_target),
            SensoryReceptor::Energy => energy_signal,
        };
    }
    #[cfg(feature = "profiling")]
    profiling::record_brain_stage(BrainStage::SensoryEncoding, stage_started.elapsed());

    let brain = &mut organism.brain;

    // Reuse scratch buffers: clear + fill avoids reallocation after first organism
    #[cfg(feature = "profiling")]
    let stage_started = Instant::now();
    scratch.inter_inputs.clear();
    scratch
        .inter_inputs
        .extend(brain.inter.iter().map(|n| n.neuron.bias));
    scratch.prev_inter.clear();
    scratch
        .prev_inter
        .extend(brain.inter.iter().map(|n| n.neuron.activation));
    #[cfg(feature = "profiling")]
    profiling::record_brain_stage(BrainStage::InterSetup, stage_started.elapsed());

    let mut action_inputs = [0.0f32; ACTION_COUNT];
    for (idx, action) in brain.action.iter().enumerate() {
        action_inputs[idx] = action.neuron.bias;
    }

    // Accumulate sensory → inter.
    #[cfg(feature = "profiling")]
    let stage_started = Instant::now();
    for sensory in &brain.sensory {
        result.synapse_ops += accumulate_weighted_inputs(
            &sensory.synapses,
            sensory.neuron.activation,
            INTER_ID_BASE,
            &mut scratch.inter_inputs,
        );
    }

    // Recurrent inter → inter uses previous tick's inter activations.
    for (i, inter) in brain.inter.iter().enumerate() {
        result.synapse_ops += accumulate_weighted_inputs(
            &inter.synapses,
            scratch.prev_inter[i],
            INTER_ID_BASE,
            &mut scratch.inter_inputs,
        );
    }
    #[cfg(feature = "profiling")]
    profiling::record_brain_stage(BrainStage::InterAccumulation, stage_started.elapsed());

    #[cfg(feature = "profiling")]
    let stage_started = Instant::now();
    for (idx, neuron) in brain.inter.iter_mut().enumerate() {
        let alpha = neuron.alpha;
        let previous = scratch.prev_inter[idx];
        let target = scratch.inter_inputs[idx].tanh();
        neuron.neuron.activation = (1.0 - alpha) * previous + alpha * target;
    }
    #[cfg(feature = "profiling")]
    profiling::record_brain_stage(BrainStage::InterActivation, stage_started.elapsed());

    // Inter → action uses this tick's freshly updated inter activations.
    #[cfg(feature = "profiling")]
    let stage_started = Instant::now();
    for sensory in &brain.sensory {
        result.synapse_ops += accumulate_weighted_inputs(
            &sensory.synapses,
            sensory.neuron.activation,
            ACTION_ID_BASE,
            &mut action_inputs,
        );
    }

    for inter in &brain.inter {
        result.synapse_ops += accumulate_weighted_inputs(
            &inter.synapses,
            inter.neuron.activation,
            ACTION_ID_BASE,
            &mut action_inputs,
        );
    }
    #[cfg(feature = "profiling")]
    profiling::record_brain_stage(BrainStage::ActionAccumulation, stage_started.elapsed());

    #[cfg(feature = "profiling")]
    let stage_started = Instant::now();
    for (idx, action) in brain.action.iter_mut().enumerate() {
        action.neuron.activation = sigmoid(action_inputs[idx]);
        result.action_activations[action_index(action.action_type)] = action.neuron.activation;
    }

    result.resolved_actions = resolve_actions(result.action_activations);
    #[cfg(feature = "profiling")]
    profiling::record_brain_stage(BrainStage::ActionActivationResolve, stage_started.elapsed());

    result
}

pub(crate) fn apply_runtime_plasticity(
    organism: &mut sim_types::OrganismState,
    passive_energy_baseline: f32,
    scratch: &mut BrainScratch,
) {
    #[cfg(feature = "profiling")]
    let stage_started = Instant::now();
    let eligibility_retention = organism.genome.eligibility_retention.clamp(0.0, 1.0);
    let weight_prune_threshold = organism.genome.synapse_prune_threshold.max(0.0);
    let should_prune = should_prune_synapses(organism.age_turns, organism.genome.age_of_maturity);
    let is_mature = organism.age_turns >= u64::from(organism.genome.age_of_maturity);

    let brain = &mut organism.brain;
    scratch.inter_activations.clear();
    scratch
        .inter_activations
        .extend(brain.inter.iter().map(|inter| inter.neuron.activation));
    for (idx, action) in brain.action.iter().enumerate() {
        scratch.action_activations[idx] = action.neuron.activation;
    }
    let energy_delta = organism.energy - organism.energy_prev;
    // Baseline-correct the reward signal so passive metabolism alone is neutral.
    let corrected_energy_delta = energy_delta + passive_energy_baseline.max(0.0);
    let dopamine_signal = (corrected_energy_delta / DOPAMINE_ENERGY_DELTA_SCALE).tanh();
    organism.energy_prev = organism.energy;
    let eta = (organism.genome.hebb_eta_baseline + organism.genome.hebb_eta_gain).max(0.0);
    #[cfg(feature = "profiling")]
    profiling::record_brain_stage(BrainStage::PlasticitySetup, stage_started.elapsed());

    #[cfg(feature = "profiling")]
    let stage_started = Instant::now();
    for sensory in &mut brain.sensory {
        tune_synapses(
            &mut sensory.synapses,
            sensory.neuron.activation,
            1.0,
            eta,
            dopamine_signal,
            eligibility_retention,
            is_mature,
            &scratch.inter_activations,
            &scratch.action_activations,
        );
    }
    #[cfg(feature = "profiling")]
    profiling::record_brain_stage(BrainStage::PlasticitySensoryTuning, stage_started.elapsed());

    #[cfg(feature = "profiling")]
    let stage_started = Instant::now();
    for inter in &mut brain.inter {
        let required_sign = match inter.interneuron_type {
            InterNeuronType::Excitatory => 1.0,
            InterNeuronType::Inhibitory => -1.0,
        };
        tune_synapses(
            &mut inter.synapses,
            inter.neuron.activation,
            required_sign,
            eta,
            dopamine_signal,
            eligibility_retention,
            is_mature,
            &scratch.inter_activations,
            &scratch.action_activations,
        );
    }
    #[cfg(feature = "profiling")]
    profiling::record_brain_stage(BrainStage::PlasticityInterTuning, stage_started.elapsed());

    if should_prune {
        #[cfg(feature = "profiling")]
        let stage_started = Instant::now();
        prune_low_weight_synapses(brain, weight_prune_threshold);
        #[cfg(feature = "profiling")]
        profiling::record_brain_stage(BrainStage::PlasticityPrune, stage_started.elapsed());
    }
}

fn should_prune_synapses(age_turns: u64, age_of_maturity: u32) -> bool {
    let maturity_ticks = u64::from(age_of_maturity);
    age_turns >= maturity_ticks && age_turns % SYNAPSE_PRUNE_INTERVAL_TICKS == 0
}

fn tune_synapses(
    edges: &mut [SynapseEdge],
    pre_activation: f32,
    required_sign: f32,
    eta: f32,
    dopamine_signal: f32,
    eligibility_retention: f32,
    is_mature: bool,
    inter_activations: &[f32],
    action_activations: &[f32; ACTION_COUNT],
) {
    for edge in edges {
        let post_activation =
            match post_activation(edge.post_neuron_id, inter_activations, action_activations) {
                Some(value) => value,
                None => continue,
            };

        edge.eligibility =
            eligibility_retention * edge.eligibility + pre_activation * post_activation;
        if !is_mature {
            continue;
        }

        let updated_weight = edge.weight + eta * dopamine_signal * edge.eligibility
            - PLASTIC_WEIGHT_DECAY * edge.weight;
        edge.weight = constrain_weight(updated_weight, required_sign);
    }
}

fn post_activation(
    neuron_id: NeuronId,
    inter_activations: &[f32],
    action_activations: &[f32; ACTION_COUNT],
) -> Option<f32> {
    if neuron_id.0 >= ACTION_ID_BASE {
        let action_idx = (neuron_id.0 - ACTION_ID_BASE) as usize;
        return action_activations.get(action_idx).copied();
    }
    if neuron_id.0 >= INTER_ID_BASE {
        let inter_idx = (neuron_id.0 - INTER_ID_BASE) as usize;
        return inter_activations.get(inter_idx).copied();
    }
    None
}

fn constrain_weight(weight: f32, required_sign: f32) -> f32 {
    if required_sign.is_sign_negative() {
        if weight >= 0.0 {
            return -SYNAPSE_STRENGTH_MIN;
        }
        let magnitude = if HEBB_WEIGHT_CLAMP_ENABLED {
            (-weight).clamp(SYNAPSE_STRENGTH_MIN, SYNAPSE_STRENGTH_MAX)
        } else {
            (-weight).max(SYNAPSE_STRENGTH_MIN)
        };
        -magnitude
    } else {
        if weight <= 0.0 {
            return SYNAPSE_STRENGTH_MIN;
        }
        let magnitude = if HEBB_WEIGHT_CLAMP_ENABLED {
            weight.clamp(SYNAPSE_STRENGTH_MIN, SYNAPSE_STRENGTH_MAX)
        } else {
            weight.max(SYNAPSE_STRENGTH_MIN)
        };
        magnitude
    }
}

fn prune_low_weight_synapses(brain: &mut BrainState, threshold: f32) {
    let mut pruned_any = false;

    for sensory in &mut brain.sensory {
        let before = sensory.synapses.len();
        sensory.synapses.retain(|synapse| {
            synapse.weight.abs() >= threshold || synapse.eligibility.abs() >= (2.0f32 * threshold)
        });
        pruned_any |= sensory.synapses.len() != before;
    }
    for inter in &mut brain.inter {
        let before = inter.synapses.len();
        inter.synapses.retain(|synapse| {
            synapse.weight.abs() >= threshold || synapse.eligibility.abs() >= (2.0f32 * threshold)
        });
        pruned_any |= inter.synapses.len() != before;
    }

    if pruned_any {
        refresh_parent_ids_and_synapse_count(brain);
    }
}

fn refresh_parent_ids_and_synapse_count(brain: &mut BrainState) {
    let inter_len = brain.inter.len();
    let action_len = brain.action.len();
    let mut inter_parent_ids: Vec<Vec<NeuronId>> = vec![Vec::new(); inter_len];
    let mut action_parent_ids: Vec<Vec<NeuronId>> = vec![Vec::new(); action_len];

    for sensory in &brain.sensory {
        let pre_id = sensory.neuron.neuron_id;
        for synapse in &sensory.synapses {
            if synapse.post_neuron_id.0 >= INTER_ID_BASE {
                let inter_idx = synapse.post_neuron_id.0.wrapping_sub(INTER_ID_BASE) as usize;
                if inter_idx < inter_parent_ids.len() {
                    inter_parent_ids[inter_idx].push(pre_id);
                    continue;
                }
            }
            if synapse.post_neuron_id.0 >= ACTION_ID_BASE {
                let action_idx = synapse.post_neuron_id.0.wrapping_sub(ACTION_ID_BASE) as usize;
                if action_idx < action_parent_ids.len() {
                    action_parent_ids[action_idx].push(pre_id);
                }
            }
        }
    }

    for inter in &brain.inter {
        let pre_id = inter.neuron.neuron_id;
        for synapse in &inter.synapses {
            if synapse.post_neuron_id.0 >= INTER_ID_BASE {
                let inter_idx = synapse.post_neuron_id.0.wrapping_sub(INTER_ID_BASE) as usize;
                if inter_idx < inter_parent_ids.len() {
                    inter_parent_ids[inter_idx].push(pre_id);
                    continue;
                }
            }
            if synapse.post_neuron_id.0 >= ACTION_ID_BASE {
                let action_idx = synapse.post_neuron_id.0.wrapping_sub(ACTION_ID_BASE) as usize;
                if action_idx < action_parent_ids.len() {
                    action_parent_ids[action_idx].push(pre_id);
                }
            }
        }
    }

    for (idx, inter) in brain.inter.iter_mut().enumerate() {
        let mut parents = std::mem::take(&mut inter_parent_ids[idx]);
        parents.sort();
        parents.dedup();
        inter.neuron.parent_ids = parents;
    }
    for (idx, action) in brain.action.iter_mut().enumerate() {
        let mut parents = std::mem::take(&mut action_parent_ids[idx]);
        parents.sort();
        parents.dedup();
        action.neuron.parent_ids = parents;
    }

    let synapse_count = brain
        .sensory
        .iter()
        .map(|n| n.synapses.len())
        .sum::<usize>()
        + brain.inter.iter().map(|n| n.synapses.len()).sum::<usize>();
    brain.synapse_count = synapse_count as u32;
}

/// Derives the set of active neuron IDs from a brain's current activation state.
/// Sensory/Inter neurons are active when activation > 0.0.
/// Action neurons use policy-based categorical argmax resolution.
pub fn derive_active_neuron_ids(brain: &BrainState) -> Vec<NeuronId> {
    let mut active = Vec::new();

    for sensory in &brain.sensory {
        if sensory.neuron.activation > 0.0 {
            active.push(sensory.neuron.neuron_id);
        }
    }

    let action_activations: [f32; ACTION_COUNT] =
        std::array::from_fn(|i| brain.action.get(i).map_or(0.0, |n| n.neuron.activation));

    let resolved = resolve_actions(action_activations);
    active.push(
        brain.action[action_index(resolved.selected_action)]
            .neuron
            .neuron_id,
    );

    active
}

fn resolve_actions(activations: [f32; ACTION_COUNT]) -> ResolvedActions {
    let mut best_idx = 0usize;
    let mut best_activation = activations[0];
    for (idx, activation) in activations.iter().copied().enumerate().skip(1) {
        if activation > best_activation {
            best_idx = idx;
            best_activation = activation;
        }
    }

    ResolvedActions {
        selected_action: ActionType::ALL[best_idx],
    }
}

/// Accumulates weighted inputs using arithmetic index resolution.
/// `id_base` is the neuron ID offset for the target layer (e.g. 1000 for inter, 2000 for action).
/// Edges targeting neurons outside the range are skipped via bounds check.
fn accumulate_weighted_inputs(
    edges: &[SynapseEdge],
    source_activation: f32,
    id_base: u32,
    inputs: &mut [f32],
) -> u64 {
    let num_slots = inputs.len();
    let mut synapse_ops = 0;
    for edge in edges {
        let idx = edge.post_neuron_id.0.wrapping_sub(id_base) as usize;
        if idx < num_slots {
            inputs[idx] += source_activation * edge.weight;
            synapse_ops += 1;
        }
    }
    synapse_ops
}

/// Maps energy to [0, 1) with a midpoint of 0.5 at `scale`.
/// Uses a Hill-style curve: v = (r^n) / (1 + r^n), where r = energy / scale.
fn energy_sensor_value(energy: f32, scale: f32) -> f32 {
    let safe_scale = scale.max(MIN_ENERGY_SENSOR_SCALE);
    let ratio = energy.max(0.0) / safe_scale;
    let curved = ratio.powf(ENERGY_SENSOR_CURVE_EXPONENT);
    curved / (1.0 + curved)
}

pub(crate) struct ScanResult {
    pub(crate) target: EntityType,
    pub(crate) signal: f32,
}

type RayScans = [Option<ScanResult>; LOOK_RAY_COUNT];

fn ray_offset_index(ray_offset: i8) -> Option<usize> {
    SensoryReceptor::LOOK_RAY_OFFSETS
        .iter()
        .position(|offset| *offset == ray_offset)
}

fn look_ray_signal(ray_scans: &RayScans, ray_offset: i8, look_target: EntityType) -> f32 {
    let Some(ray_idx) = ray_offset_index(ray_offset) else {
        return 0.0;
    };
    match ray_scans[ray_idx].as_ref() {
        Some(hit) if hit.target == look_target => hit.signal,
        _ => 0.0,
    }
}

fn rotate_facing_by_offset(
    mut facing: sim_types::FacingDirection,
    ray_offset: i8,
) -> sim_types::FacingDirection {
    if ray_offset >= 0 {
        for _ in 0..u8::try_from(ray_offset).unwrap_or(0) {
            facing = rotate_right(facing);
        }
        return facing;
    }

    for _ in 0..ray_offset.unsigned_abs() {
        facing = rotate_left(facing);
    }
    facing
}

/// Scans all fixed look rays relative to `facing`, using occlusion per-ray.
pub(crate) fn scan_rays(
    position: (i32, i32),
    facing: sim_types::FacingDirection,
    organism_id: OrganismId,
    world_width: i32,
    occupancy: &[Option<Occupant>],
    vision_distance: u32,
) -> RayScans {
    std::array::from_fn(|idx| {
        scan_ray(
            position,
            facing,
            SensoryReceptor::LOOK_RAY_OFFSETS[idx],
            organism_id,
            world_width,
            occupancy,
            vision_distance,
        )
    })
}

/// Scans one ray up to `vision_distance` hexes.
/// Returns the closest entity found (with occlusion) and a distance-encoded signal
/// strength: `(max_dist - dist + 1) / max_dist`. Returns `None` if all cells are empty.
fn scan_ray(
    position: (i32, i32),
    facing: sim_types::FacingDirection,
    ray_offset: i8,
    organism_id: OrganismId,
    world_width: i32,
    occupancy: &[Option<Occupant>],
    vision_distance: u32,
) -> Option<ScanResult> {
    let ray_facing = rotate_facing_by_offset(facing, ray_offset);
    let max_dist = vision_distance.max(1);
    let mut current = position;
    for d in 1..=max_dist {
        current = hex_neighbor(current, ray_facing, world_width);
        let idx = current.1 as usize * world_width as usize + current.0 as usize;
        match occupancy[idx] {
            Some(Occupant::Organism(id)) if id == organism_id => {}
            Some(Occupant::Food(_)) => {
                let signal = (max_dist - d + 1) as f32 / max_dist as f32;
                return Some(ScanResult {
                    target: EntityType::Food,
                    signal,
                });
            }
            Some(Occupant::Organism(_)) => {
                let signal = (max_dist - d + 1) as f32 / max_dist as f32;
                return Some(ScanResult {
                    target: EntityType::Organism,
                    signal,
                });
            }
            Some(Occupant::Wall) => {
                let signal = (max_dist - d + 1) as f32 / max_dist as f32;
                return Some(ScanResult {
                    target: EntityType::Wall,
                    signal,
                });
            }
            None => {}
        }
    }
    None
}
```

## sim-core/src/genome.rs

```rust
use crate::brain::{ACTION_COUNT, ACTION_COUNT_U32, ACTION_ID_BASE, INTER_ID_BASE, SENSORY_COUNT};
use crate::SimError;
use rand::Rng;
use rand_distr::{Distribution, StandardNormal};
use sim_types::{
    ActionType, BrainLocation, InterNeuronType, NeuronId, OrganismGenome, SeedGenomeConfig,
    SynapseEdge,
};
use std::cmp::Ordering;
use std::collections::HashSet;

const MIN_MUTATED_VISION_DISTANCE: u32 = 1;
const MAX_MUTATED_VISION_DISTANCE: u32 = 32;
const MIN_MUTATED_AGE_OF_MATURITY: u32 = 0;
const MAX_MUTATED_AGE_OF_MATURITY: u32 = 10_000;
pub(crate) const SYNAPSE_STRENGTH_MAX: f32 = 1.0;
pub(crate) const SYNAPSE_STRENGTH_MIN: f32 = 0.001;
const BIAS_MAX: f32 = 1.0;
const ETA_BASELINE_MIN: f32 = 0.0;
const ETA_BASELINE_MAX: f32 = 0.2;
const ETA_GAIN_MIN: f32 = -1.0;
const ETA_GAIN_MAX: f32 = 1.0;
const ELIGIBILITY_RETENTION_MIN: f32 = 0.0;
const ELIGIBILITY_RETENTION_MAX: f32 = 1.0;
const SYNAPSE_PRUNE_THRESHOLD_MIN: f32 = 0.0;
const SYNAPSE_PRUNE_THRESHOLD_MAX: f32 = 1.0;

const INTER_TYPE_EXCITATORY_PRIOR: f32 = 0.8;
const MUTATION_RATE_ADAPTATION_TAU: f32 = 0.25;
const MUTATION_RATE_MIN: f32 = 1.0e-4;
const MUTATION_RATE_MAX: f32 = 1.0 - MUTATION_RATE_MIN;

const BIAS_PERTURBATION_STDDEV: f32 = 0.15;
const INTER_LOG_TAU_PERTURBATION_STDDEV: f32 = 0.05;
const ELIGIBILITY_RETENTION_PERTURBATION_STDDEV: f32 = 0.05;
const SYNAPSE_PRUNE_THRESHOLD_PERTURBATION_STDDEV: f32 = 0.02;
const LOCATION_PERTURBATION_STDDEV: f32 = 0.75;
pub(crate) const INTER_TAU_MIN: f32 = 0.1;
pub(crate) const INTER_TAU_MAX: f32 = 15.0;
pub(crate) const INTER_LOG_TAU_MIN: f32 = -2.302_585_1;
pub(crate) const INTER_LOG_TAU_MAX: f32 = 2.995_732_3;
pub(crate) const DEFAULT_INTER_LOG_TAU: f32 = 0.0;
pub(crate) const BRAIN_SPACE_MIN: f32 = 0.0;
pub(crate) const BRAIN_SPACE_MAX: f32 = 10.0;
const SPATIAL_PRIOR_LONG_RANGE_FLOOR: f32 = 0.01;
const SYNAPSE_WEIGHT_LOG_NORMAL_MU: f32 = -0.5;
const SYNAPSE_WEIGHT_LOG_NORMAL_SIGMA: f32 = 0.8;

pub(crate) fn generate_seed_genome<R: Rng + ?Sized>(
    config: &SeedGenomeConfig,
    rng: &mut R,
) -> OrganismGenome {
    let num_neurons = config.num_neurons;
    let max_synapses = max_possible_synapses(num_neurons);
    let inter_biases: Vec<f32> = (0..num_neurons).map(|_| sample_initial_bias(rng)).collect();
    let inter_log_taus: Vec<f32> = (0..num_neurons)
        .map(|_| sample_uniform_log_tau(rng))
        .collect();
    let interneuron_types: Vec<InterNeuronType> = (0..num_neurons)
        .map(|_| sample_interneuron_type(rng))
        .collect();
    let inter_locations: Vec<BrainLocation> = (0..num_neurons)
        .map(|_| sample_uniform_location(rng))
        .collect();
    let action_biases: Vec<f32> = ActionType::ALL
        .into_iter()
        .map(|_| sample_initial_bias(rng))
        .collect();
    let sensory_locations: Vec<BrainLocation> = (0..SENSORY_COUNT)
        .map(|_| sample_uniform_location(rng))
        .collect();
    let action_locations: Vec<BrainLocation> = (0..ACTION_COUNT)
        .map(|_| sample_uniform_location(rng))
        .collect();

    let mut genome = OrganismGenome {
        num_neurons,
        num_synapses: config.num_synapses.min(max_synapses),
        spatial_prior_sigma: config.spatial_prior_sigma.max(0.01),
        vision_distance: config.vision_distance,
        starting_energy: config.starting_energy,
        age_of_maturity: config.age_of_maturity,
        hebb_eta_baseline: config.hebb_eta_baseline,
        hebb_eta_gain: config.hebb_eta_gain,
        eligibility_retention: config.eligibility_retention,
        synapse_prune_threshold: config.synapse_prune_threshold,
        mutation_rate_age_of_maturity: config.mutation_rate_age_of_maturity,
        mutation_rate_vision_distance: config.mutation_rate_vision_distance,
        mutation_rate_num_synapses: config.mutation_rate_num_synapses,
        mutation_rate_inter_bias: config.mutation_rate_inter_bias,
        mutation_rate_inter_update_rate: config.mutation_rate_inter_update_rate,
        mutation_rate_action_bias: config.mutation_rate_action_bias,
        mutation_rate_eligibility_retention: config.mutation_rate_eligibility_retention,
        mutation_rate_synapse_prune_threshold: config.mutation_rate_synapse_prune_threshold,
        mutation_rate_neuron_location: config.mutation_rate_neuron_location,
        inter_biases,
        inter_log_taus,
        interneuron_types,
        action_biases,
        sensory_locations,
        inter_locations,
        action_locations,
        edges: Vec::new(),
    };
    sync_synapse_genes_to_target(&mut genome, rng);
    genome
}

fn sample_interneuron_type<R: Rng + ?Sized>(rng: &mut R) -> InterNeuronType {
    if rng.random::<f32>() < INTER_TYPE_EXCITATORY_PRIOR {
        InterNeuronType::Excitatory
    } else {
        InterNeuronType::Inhibitory
    }
}

fn mutate_mutation_rate_genes<R: Rng + ?Sized>(genome: &mut OrganismGenome, rng: &mut R) {
    let mut rates = [
        genome.mutation_rate_age_of_maturity,
        genome.mutation_rate_vision_distance,
        genome.mutation_rate_num_synapses,
        genome.mutation_rate_inter_bias,
        genome.mutation_rate_inter_update_rate,
        genome.mutation_rate_action_bias,
        genome.mutation_rate_eligibility_retention,
        genome.mutation_rate_synapse_prune_threshold,
        genome.mutation_rate_neuron_location,
    ];
    let shared_normal = standard_normal(rng) * MUTATION_RATE_ADAPTATION_TAU;

    for rate in &mut rates {
        let gene_normal = standard_normal(rng) * MUTATION_RATE_ADAPTATION_TAU;
        let adapted = *rate * (shared_normal + gene_normal).exp();
        *rate = adapted.clamp(MUTATION_RATE_MIN, MUTATION_RATE_MAX);
    }

    genome.mutation_rate_age_of_maturity = rates[0];
    genome.mutation_rate_vision_distance = rates[1];
    genome.mutation_rate_num_synapses = rates[2];
    genome.mutation_rate_inter_bias = rates[3];
    genome.mutation_rate_inter_update_rate = rates[4];
    genome.mutation_rate_action_bias = rates[5];
    genome.mutation_rate_eligibility_retention = rates[6];
    genome.mutation_rate_synapse_prune_threshold = rates[7];
    genome.mutation_rate_neuron_location = rates[8];
}

fn align_genome_vectors<R: Rng + ?Sized>(genome: &mut OrganismGenome, rng: &mut R) {
    genome.num_synapses = genome
        .num_synapses
        .min(max_possible_synapses(genome.num_neurons));
    genome.spatial_prior_sigma = genome.spatial_prior_sigma.max(0.01);

    let target_inter_len = genome.num_neurons as usize;

    while genome.inter_biases.len() < target_inter_len {
        genome.inter_biases.push(sample_initial_bias(rng));
    }
    genome.inter_biases.truncate(target_inter_len);

    while genome.inter_log_taus.len() < target_inter_len {
        genome.inter_log_taus.push(sample_uniform_log_tau(rng));
    }
    genome.inter_log_taus.truncate(target_inter_len);

    while genome.interneuron_types.len() < target_inter_len {
        genome.interneuron_types.push(sample_interneuron_type(rng));
    }
    genome.interneuron_types.truncate(target_inter_len);

    while genome.inter_locations.len() < target_inter_len {
        genome.inter_locations.push(sample_uniform_location(rng));
    }
    genome.inter_locations.truncate(target_inter_len);

    if genome.action_biases.len() < ACTION_COUNT {
        genome.action_biases.resize(ACTION_COUNT, 0.0);
    } else if genome.action_biases.len() > ACTION_COUNT {
        genome.action_biases.truncate(ACTION_COUNT);
    }

    while genome.sensory_locations.len() < SENSORY_COUNT as usize {
        genome.sensory_locations.push(sample_uniform_location(rng));
    }
    genome.sensory_locations.truncate(SENSORY_COUNT as usize);

    while genome.action_locations.len() < ACTION_COUNT {
        genome.action_locations.push(sample_uniform_location(rng));
    }
    genome.action_locations.truncate(ACTION_COUNT);

    sanitize_synapse_genes(genome);
}

pub(crate) fn mutate_genome<R: Rng + ?Sized>(genome: &mut OrganismGenome, rng: &mut R) {
    align_genome_vectors(genome, rng);
    mutate_mutation_rate_genes(genome, rng);

    if rng.random::<f32>() < genome.mutation_rate_age_of_maturity {
        genome.age_of_maturity = step_u32(
            genome.age_of_maturity,
            MIN_MUTATED_AGE_OF_MATURITY,
            MAX_MUTATED_AGE_OF_MATURITY,
            rng,
        );
    }

    if rng.random::<f32>() < genome.mutation_rate_vision_distance {
        genome.vision_distance = step_u32(
            genome.vision_distance,
            MIN_MUTATED_VISION_DISTANCE,
            MAX_MUTATED_VISION_DISTANCE,
            rng,
        );
    }

    if rng.random::<f32>() < genome.mutation_rate_num_synapses {
        let max_synapses = max_possible_synapses(genome.num_neurons);
        genome.num_synapses = step_u32(genome.num_synapses, 0, max_synapses, rng);
    }

    if rng.random::<f32>() < genome.mutation_rate_inter_bias && genome.num_neurons > 0 {
        let idx = rng.random_range(0..genome.num_neurons as usize);
        genome.inter_biases[idx] = perturb_clamped(
            genome.inter_biases[idx],
            BIAS_PERTURBATION_STDDEV,
            -BIAS_MAX,
            BIAS_MAX,
            rng,
        );
    }

    if rng.random::<f32>() < genome.mutation_rate_inter_update_rate && genome.num_neurons > 0 {
        let idx = rng.random_range(0..genome.num_neurons as usize);
        genome.inter_log_taus[idx] = perturb_clamped(
            genome.inter_log_taus[idx],
            INTER_LOG_TAU_PERTURBATION_STDDEV,
            INTER_LOG_TAU_MIN,
            INTER_LOG_TAU_MAX,
            rng,
        );
    }

    if rng.random::<f32>() < genome.mutation_rate_action_bias && !genome.action_biases.is_empty() {
        let idx = rng.random_range(0..genome.action_biases.len());
        genome.action_biases[idx] = perturb_clamped(
            genome.action_biases[idx],
            BIAS_PERTURBATION_STDDEV,
            -BIAS_MAX,
            BIAS_MAX,
            rng,
        );
    }

    if rng.random::<f32>() < genome.mutation_rate_eligibility_retention {
        genome.eligibility_retention = perturb_clamped(
            genome.eligibility_retention,
            ELIGIBILITY_RETENTION_PERTURBATION_STDDEV,
            ELIGIBILITY_RETENTION_MIN,
            ELIGIBILITY_RETENTION_MAX,
            rng,
        );
    }

    if rng.random::<f32>() < genome.mutation_rate_synapse_prune_threshold {
        genome.synapse_prune_threshold = perturb_clamped(
            genome.synapse_prune_threshold,
            SYNAPSE_PRUNE_THRESHOLD_PERTURBATION_STDDEV,
            SYNAPSE_PRUNE_THRESHOLD_MIN,
            SYNAPSE_PRUNE_THRESHOLD_MAX,
            rng,
        );
    }

    if rng.random::<f32>() < genome.mutation_rate_neuron_location {
        mutate_random_neuron_location(genome, rng);
    }

    sync_synapse_genes_to_target(genome, rng);
}

fn mutate_random_neuron_location<R: Rng + ?Sized>(genome: &mut OrganismGenome, rng: &mut R) {
    let enabled_inter = genome.num_neurons as usize;
    let total = SENSORY_COUNT as usize + enabled_inter + ACTION_COUNT;
    if total == 0 {
        return;
    }

    let idx = rng.random_range(0..total);
    let location = if idx < SENSORY_COUNT as usize {
        genome.sensory_locations.get_mut(idx)
    } else if idx < SENSORY_COUNT as usize + enabled_inter {
        genome
            .inter_locations
            .get_mut(idx.saturating_sub(SENSORY_COUNT as usize))
    } else {
        genome
            .action_locations
            .get_mut(idx.saturating_sub(SENSORY_COUNT as usize + enabled_inter))
    };

    if let Some(location) = location {
        location.x = perturb_clamped(
            location.x,
            LOCATION_PERTURBATION_STDDEV,
            BRAIN_SPACE_MIN,
            BRAIN_SPACE_MAX,
            rng,
        );
        location.y = perturb_clamped(
            location.y,
            LOCATION_PERTURBATION_STDDEV,
            BRAIN_SPACE_MIN,
            BRAIN_SPACE_MAX,
            rng,
        );
    }
}

fn sync_synapse_genes_to_target<R: Rng + ?Sized>(genome: &mut OrganismGenome, rng: &mut R) {
    sanitize_synapse_genes(genome);

    let target = genome.num_synapses as usize;
    if genome.edges.len() > target {
        remove_random_synapse_genes(genome, genome.edges.len() - target, rng);
    }
    if genome.edges.len() < target {
        add_synapse_genes_with_spatial_prior(genome, target - genome.edges.len(), rng);
    }

    sort_synapse_genes(&mut genome.edges);
    genome.num_synapses = genome.edges.len() as u32;
}

fn sanitize_synapse_genes(genome: &mut OrganismGenome) {
    let num_neurons = genome.num_neurons;
    genome
        .edges
        .retain(|edge| is_valid_synapse_pair(edge.pre_neuron_id, edge.post_neuron_id, num_neurons));

    for edge in &mut genome.edges {
        let required_sign =
            required_pre_sign(edge.pre_neuron_id, num_neurons, &genome.interneuron_types)
                .unwrap_or(1.0);
        edge.weight = constrain_weight_to_sign(edge.weight, required_sign);
        edge.eligibility = 0.0;
    }

    sort_synapse_genes(&mut genome.edges);
    genome.edges.dedup_by(|a, b| {
        a.pre_neuron_id == b.pre_neuron_id && a.post_neuron_id == b.post_neuron_id
    });
}

fn remove_random_synapse_genes<R: Rng + ?Sized>(
    genome: &mut OrganismGenome,
    remove_count: usize,
    rng: &mut R,
) {
    let to_remove = remove_count.min(genome.edges.len());
    for _ in 0..to_remove {
        let idx = rng.random_range(0..genome.edges.len());
        genome.edges.swap_remove(idx);
    }
}

fn add_synapse_genes_with_spatial_prior<R: Rng + ?Sized>(
    genome: &mut OrganismGenome,
    add_count: usize,
    rng: &mut R,
) {
    if add_count == 0 {
        return;
    }

    let mut existing_pairs: HashSet<(u32, u32)> = HashSet::with_capacity(genome.edges.len());
    for edge in &genome.edges {
        existing_pairs.insert((edge.pre_neuron_id.0, edge.post_neuron_id.0));
    }

    let mut weighted_candidates: Vec<(f32, NeuronId, NeuronId)> = Vec::new();

    let num_neurons = genome.num_neurons;
    for sensory_idx in 0..SENSORY_COUNT {
        let pre_id = NeuronId(sensory_idx);
        for post_id in post_ids(num_neurons) {
            if existing_pairs.contains(&(pre_id.0, post_id.0)) {
                continue;
            }
            let probability = connection_probability(genome, pre_id, post_id);
            let priority = weighted_without_replacement_priority(probability, rng);
            weighted_candidates.push((priority, pre_id, post_id));
        }
    }

    for inter_idx in 0..num_neurons {
        let pre_id = NeuronId(INTER_ID_BASE + inter_idx);
        for post_id in post_ids(num_neurons) {
            if !is_valid_synapse_pair(pre_id, post_id, num_neurons) {
                continue;
            }
            if existing_pairs.contains(&(pre_id.0, post_id.0)) {
                continue;
            }
            let probability = connection_probability(genome, pre_id, post_id);
            let priority = weighted_without_replacement_priority(probability, rng);
            weighted_candidates.push((priority, pre_id, post_id));
        }
    }

    weighted_candidates.sort_unstable_by(|a, b| {
        a.0.total_cmp(&b.0)
            .then_with(|| a.1.cmp(&b.1))
            .then_with(|| a.2.cmp(&b.2))
    });

    for &(_, pre_id, post_id) in weighted_candidates.iter().take(add_count) {
        let required_sign =
            required_pre_sign(pre_id, num_neurons, &genome.interneuron_types).unwrap_or(1.0);
        genome.edges.push(SynapseEdge {
            pre_neuron_id: pre_id,
            post_neuron_id: post_id,
            weight: sample_signed_lognormal_weight(required_sign, rng),
            eligibility: 0.0,
        });
    }
}

fn post_ids(num_neurons: u32) -> impl Iterator<Item = NeuronId> {
    let inter = (0..num_neurons).map(|idx| NeuronId(INTER_ID_BASE + idx));
    let actions = (0..ACTION_COUNT_U32).map(|idx| NeuronId(ACTION_ID_BASE + idx));
    inter.chain(actions)
}

fn is_valid_synapse_pair(pre: NeuronId, post: NeuronId, num_neurons: u32) -> bool {
    if !is_valid_pre_id(pre, num_neurons) || !is_valid_post_id(post, num_neurons) {
        return false;
    }

    if is_inter_id(pre, num_neurons) && is_inter_id(post, num_neurons) && pre == post {
        return false;
    }

    true
}

fn is_valid_pre_id(id: NeuronId, num_neurons: u32) -> bool {
    id.0 < SENSORY_COUNT || is_inter_id(id, num_neurons)
}

fn is_valid_post_id(id: NeuronId, num_neurons: u32) -> bool {
    is_inter_id(id, num_neurons)
        || (ACTION_ID_BASE..ACTION_ID_BASE + ACTION_COUNT_U32).contains(&id.0)
}

fn is_inter_id(id: NeuronId, num_neurons: u32) -> bool {
    (INTER_ID_BASE..INTER_ID_BASE + num_neurons).contains(&id.0)
}

fn required_pre_sign(
    pre: NeuronId,
    num_neurons: u32,
    interneuron_types: &[InterNeuronType],
) -> Option<f32> {
    if pre.0 < SENSORY_COUNT {
        return Some(1.0);
    }
    if is_inter_id(pre, num_neurons) {
        let idx = (pre.0 - INTER_ID_BASE) as usize;
        let inter_type = interneuron_types
            .get(idx)
            .copied()
            .unwrap_or(InterNeuronType::Excitatory);
        return Some(match inter_type {
            InterNeuronType::Excitatory => 1.0,
            InterNeuronType::Inhibitory => -1.0,
        });
    }
    None
}

fn sort_synapse_genes(edges: &mut [SynapseEdge]) {
    edges.sort_unstable_by(|a, b| {
        synapse_key_cmp(a, b)
            .then_with(|| a.weight.total_cmp(&b.weight))
            .then_with(|| a.eligibility.total_cmp(&b.eligibility))
    });
}

fn synapse_key_cmp(a: &SynapseEdge, b: &SynapseEdge) -> Ordering {
    a.pre_neuron_id
        .cmp(&b.pre_neuron_id)
        .then_with(|| a.post_neuron_id.cmp(&b.post_neuron_id))
}

fn connection_probability(genome: &OrganismGenome, pre: NeuronId, post: NeuronId) -> f32 {
    let pre_location = neuron_location_or_default(genome, pre);
    let post_location = neuron_location_or_default(genome, post);
    let dx = pre_location.x - post_location.x;
    let dy = pre_location.y - post_location.y;
    let distance_sq = dx * dx + dy * dy;
    let sigma = genome.spatial_prior_sigma.max(0.01);
    let sigma_sq = sigma * sigma;
    let local_bias = (-0.5 * distance_sq / sigma_sq).exp();
    (SPATIAL_PRIOR_LONG_RANGE_FLOOR + (1.0 - SPATIAL_PRIOR_LONG_RANGE_FLOOR) * local_bias)
        .clamp(0.0, 1.0)
}

fn neuron_location_or_default(genome: &OrganismGenome, id: NeuronId) -> BrainLocation {
    if id.0 < SENSORY_COUNT {
        return genome
            .sensory_locations
            .get(id.0 as usize)
            .copied()
            .unwrap_or(default_brain_location());
    }

    if is_inter_id(id, genome.num_neurons) {
        let idx = (id.0 - INTER_ID_BASE) as usize;
        return genome
            .inter_locations
            .get(idx)
            .copied()
            .unwrap_or(default_brain_location());
    }

    if (ACTION_ID_BASE..ACTION_ID_BASE + ACTION_COUNT_U32).contains(&id.0) {
        let idx = (id.0 - ACTION_ID_BASE) as usize;
        return genome
            .action_locations
            .get(idx)
            .copied()
            .unwrap_or(default_brain_location());
    }

    default_brain_location()
}

fn default_brain_location() -> BrainLocation {
    BrainLocation {
        x: 0.5 * (BRAIN_SPACE_MIN + BRAIN_SPACE_MAX),
        y: 0.5 * (BRAIN_SPACE_MIN + BRAIN_SPACE_MAX),
    }
}

fn weighted_without_replacement_priority<R: Rng + ?Sized>(weight: f32, rng: &mut R) -> f32 {
    let clamped_weight = weight.max(f32::MIN_POSITIVE);
    let u = rng.random::<f32>().max(f32::MIN_POSITIVE);
    -u.ln() / clamped_weight
}

fn sample_signed_lognormal_weight<R: Rng + ?Sized>(required_sign: f32, rng: &mut R) -> f32 {
    let z = standard_normal(rng);
    let magnitude = (SYNAPSE_WEIGHT_LOG_NORMAL_MU + SYNAPSE_WEIGHT_LOG_NORMAL_SIGMA * z)
        .exp()
        .clamp(SYNAPSE_STRENGTH_MIN, SYNAPSE_STRENGTH_MAX);
    if required_sign.is_sign_negative() {
        -magnitude
    } else {
        magnitude
    }
}

fn constrain_weight_to_sign(weight: f32, required_sign: f32) -> f32 {
    if required_sign.is_sign_negative() {
        if weight >= 0.0 {
            return -SYNAPSE_STRENGTH_MIN;
        }
        -(-weight).clamp(SYNAPSE_STRENGTH_MIN, SYNAPSE_STRENGTH_MAX)
    } else {
        if weight <= 0.0 {
            return SYNAPSE_STRENGTH_MIN;
        }
        weight.clamp(SYNAPSE_STRENGTH_MIN, SYNAPSE_STRENGTH_MAX)
    }
}

fn step_u32<R: Rng + ?Sized>(value: u32, min: u32, max: u32, rng: &mut R) -> u32 {
    if min >= max {
        return min;
    }
    if value <= min {
        return min.saturating_add(1).min(max);
    }
    if value >= max {
        return max.saturating_sub(1).max(min);
    }
    if rng.random::<bool>() {
        value.saturating_add(1).min(max)
    } else {
        value.saturating_sub(1).max(min)
    }
}

fn sample_uniform_log_tau<R: Rng + ?Sized>(rng: &mut R) -> f32 {
    rng.random_range(INTER_LOG_TAU_MIN..=INTER_LOG_TAU_MAX)
}

fn sample_uniform_location<R: Rng + ?Sized>(rng: &mut R) -> BrainLocation {
    BrainLocation {
        x: rng.random_range(BRAIN_SPACE_MIN..=BRAIN_SPACE_MAX),
        y: rng.random_range(BRAIN_SPACE_MIN..=BRAIN_SPACE_MAX),
    }
}

pub(crate) fn inter_alpha_from_log_tau(log_tau: f32) -> f32 {
    let clamped_log_tau = log_tau.clamp(INTER_LOG_TAU_MIN, INTER_LOG_TAU_MAX);
    let tau = clamped_log_tau.exp().clamp(INTER_TAU_MIN, INTER_TAU_MAX);
    1.0 - (-1.0 / tau).exp()
}

fn sample_initial_bias<R: Rng + ?Sized>(rng: &mut R) -> f32 {
    perturb_clamped(
        0.0,
        BIAS_PERTURBATION_STDDEV * 2.0,
        -BIAS_MAX,
        BIAS_MAX,
        rng,
    )
}

pub(crate) fn prune_disconnected_inter_neurons(_genome: &mut OrganismGenome) {}

fn perturb_clamped<R: Rng + ?Sized>(
    value: f32,
    stddev: f32,
    min: f32,
    max: f32,
    rng: &mut R,
) -> f32 {
    let normal = standard_normal(rng);
    (value + normal * stddev).clamp(min, max)
}

fn standard_normal<R: Rng + ?Sized>(rng: &mut R) -> f32 {
    StandardNormal.sample(rng)
}

fn average_location_distance(a: &[BrainLocation], b: &[BrainLocation], len: usize) -> f32 {
    if len == 0 {
        return 0.0;
    }

    let mut total = 0.0;
    for i in 0..len {
        let la = a
            .get(i)
            .copied()
            .unwrap_or(BrainLocation { x: 5.0, y: 5.0 });
        let lb = b
            .get(i)
            .copied()
            .unwrap_or(BrainLocation { x: 5.0, y: 5.0 });
        total += (la.x - lb.x).abs() + (la.y - lb.y).abs();
    }

    total / len as f32
}

fn centroid(locations: &[BrainLocation], len: usize) -> BrainLocation {
    if len == 0 {
        return BrainLocation { x: 5.0, y: 5.0 };
    }

    let mut sum_x = 0.0;
    let mut sum_y = 0.0;
    for i in 0..len {
        let location = locations
            .get(i)
            .copied()
            .unwrap_or(BrainLocation { x: 5.0, y: 5.0 });
        sum_x += location.x;
        sum_y += location.y;
    }

    BrainLocation {
        x: sum_x / len as f32,
        y: sum_y / len as f32,
    }
}

fn synapse_gene_distance(a: &[SynapseEdge], b: &[SynapseEdge]) -> f32 {
    let mut distance = 0.0;

    let mut a_idx = 0usize;
    let mut b_idx = 0usize;

    while a_idx < a.len() && b_idx < b.len() {
        match synapse_key_cmp(&a[a_idx], &b[b_idx]) {
            Ordering::Equal => {
                distance += (a[a_idx].weight - b[b_idx].weight).abs();
                a_idx += 1;
                b_idx += 1;
            }
            Ordering::Less => {
                distance += 1.0 + a[a_idx].weight.abs();
                a_idx += 1;
            }
            Ordering::Greater => {
                distance += 1.0 + b[b_idx].weight.abs();
                b_idx += 1;
            }
        }
    }

    while a_idx < a.len() {
        distance += 1.0 + a[a_idx].weight.abs();
        a_idx += 1;
    }
    while b_idx < b.len() {
        distance += 1.0 + b[b_idx].weight.abs();
        b_idx += 1;
    }

    distance
}

/// L1 genome distance: scalar traits + mutation-rate genes + topology + brain geometry.
pub(crate) fn genome_distance(a: &OrganismGenome, b: &OrganismGenome) -> f32 {
    let mut dist = (a.num_neurons as f32 - b.num_neurons as f32).abs()
        + (a.num_synapses as f32 - b.num_synapses as f32).abs()
        + (a.spatial_prior_sigma - b.spatial_prior_sigma).abs()
        + (a.vision_distance as f32 - b.vision_distance as f32).abs()
        + (a.starting_energy - b.starting_energy).abs()
        + (a.age_of_maturity as f32 - b.age_of_maturity as f32).abs()
        + (a.hebb_eta_baseline - b.hebb_eta_baseline).abs()
        + (a.hebb_eta_gain - b.hebb_eta_gain).abs()
        + (a.eligibility_retention - b.eligibility_retention).abs()
        + (a.synapse_prune_threshold - b.synapse_prune_threshold).abs();

    let a_rates = [
        a.mutation_rate_age_of_maturity,
        a.mutation_rate_vision_distance,
        a.mutation_rate_num_synapses,
        a.mutation_rate_inter_bias,
        a.mutation_rate_inter_update_rate,
        a.mutation_rate_action_bias,
        a.mutation_rate_eligibility_retention,
        a.mutation_rate_synapse_prune_threshold,
        a.mutation_rate_neuron_location,
    ];
    let b_rates = [
        b.mutation_rate_age_of_maturity,
        b.mutation_rate_vision_distance,
        b.mutation_rate_num_synapses,
        b.mutation_rate_inter_bias,
        b.mutation_rate_inter_update_rate,
        b.mutation_rate_action_bias,
        b.mutation_rate_eligibility_retention,
        b.mutation_rate_synapse_prune_threshold,
        b.mutation_rate_neuron_location,
    ];
    for i in 0..a_rates.len() {
        dist += (a_rates[i] - b_rates[i]).abs();
    }

    let max_enabled = a.num_neurons.max(b.num_neurons) as usize;
    for i in 0..max_enabled {
        let ta = a
            .interneuron_types
            .get(i)
            .copied()
            .unwrap_or(InterNeuronType::Excitatory);
        let tb = b
            .interneuron_types
            .get(i)
            .copied()
            .unwrap_or(InterNeuronType::Excitatory);
        if ta != tb {
            dist += 1.0;
        }
    }

    dist += synapse_gene_distance(&a.edges, &b.edges);

    dist += average_location_distance(
        &a.sensory_locations,
        &b.sensory_locations,
        SENSORY_COUNT as usize,
    );
    dist += average_location_distance(&a.action_locations, &b.action_locations, ACTION_COUNT);
    dist += average_location_distance(&a.inter_locations, &b.inter_locations, max_enabled.max(1));

    let a_inter_centroid = centroid(&a.inter_locations, a.num_neurons as usize);
    let b_inter_centroid = centroid(&b.inter_locations, b.num_neurons as usize);
    dist += (a_inter_centroid.x - b_inter_centroid.x).abs()
        + (a_inter_centroid.y - b_inter_centroid.y).abs();

    dist
}

fn validate_rate(name: &str, rate: f32) -> Result<(), SimError> {
    if (0.0..=1.0).contains(&rate) {
        Ok(())
    } else {
        Err(SimError::InvalidConfig(format!(
            "{name} must be within [0, 1]"
        )))
    }
}

fn max_possible_synapses(num_neurons: u32) -> u32 {
    let pre_count = u64::from(SENSORY_COUNT + num_neurons);
    let post_count = u64::from(num_neurons + ACTION_COUNT_U32);
    let all_pairs = pre_count.saturating_mul(post_count);
    let max = all_pairs.saturating_sub(u64::from(num_neurons));
    max.min(u64::from(u32::MAX)) as u32
}

pub(crate) fn validate_seed_genome_config(config: &SeedGenomeConfig) -> Result<(), SimError> {
    if !(ETA_BASELINE_MIN..=ETA_BASELINE_MAX).contains(&config.hebb_eta_baseline) {
        return Err(SimError::InvalidConfig(format!(
            "hebb_eta_baseline must be within [{ETA_BASELINE_MIN}, {ETA_BASELINE_MAX}]"
        )));
    }
    if !(ETA_GAIN_MIN..=ETA_GAIN_MAX).contains(&config.hebb_eta_gain) {
        return Err(SimError::InvalidConfig(format!(
            "hebb_eta_gain must be within [{ETA_GAIN_MIN}, {ETA_GAIN_MAX}]"
        )));
    }
    if !(ELIGIBILITY_RETENTION_MIN..=ELIGIBILITY_RETENTION_MAX)
        .contains(&config.eligibility_retention)
    {
        return Err(SimError::InvalidConfig(format!(
            "eligibility_retention must be within [{ELIGIBILITY_RETENTION_MIN}, {ELIGIBILITY_RETENTION_MAX}]"
        )));
    }
    if !(SYNAPSE_PRUNE_THRESHOLD_MIN..=SYNAPSE_PRUNE_THRESHOLD_MAX)
        .contains(&config.synapse_prune_threshold)
    {
        return Err(SimError::InvalidConfig(format!(
            "synapse_prune_threshold must be within [{SYNAPSE_PRUNE_THRESHOLD_MIN}, {SYNAPSE_PRUNE_THRESHOLD_MAX}]"
        )));
    }

    if config.age_of_maturity > MAX_MUTATED_AGE_OF_MATURITY {
        return Err(SimError::InvalidConfig(format!(
            "age_of_maturity must be <= {MAX_MUTATED_AGE_OF_MATURITY}"
        )));
    }
    if !config.starting_energy.is_finite() || config.starting_energy <= 0.0 {
        return Err(SimError::InvalidConfig(
            "starting_energy must be greater than zero".to_owned(),
        ));
    }

    validate_rate(
        "mutation_rate_age_of_maturity",
        config.mutation_rate_age_of_maturity,
    )?;
    validate_rate(
        "mutation_rate_vision_distance",
        config.mutation_rate_vision_distance,
    )?;
    validate_rate(
        "mutation_rate_num_synapses",
        config.mutation_rate_num_synapses,
    )?;
    validate_rate("mutation_rate_inter_bias", config.mutation_rate_inter_bias)?;
    validate_rate(
        "mutation_rate_inter_update_rate",
        config.mutation_rate_inter_update_rate,
    )?;
    validate_rate(
        "mutation_rate_action_bias",
        config.mutation_rate_action_bias,
    )?;
    validate_rate(
        "mutation_rate_eligibility_retention",
        config.mutation_rate_eligibility_retention,
    )?;
    validate_rate(
        "mutation_rate_synapse_prune_threshold",
        config.mutation_rate_synapse_prune_threshold,
    )?;
    validate_rate(
        "mutation_rate_neuron_location",
        config.mutation_rate_neuron_location,
    )?;

    if config.vision_distance < MIN_MUTATED_VISION_DISTANCE {
        return Err(SimError::InvalidConfig(
            "vision_distance must be >= 1".to_owned(),
        ));
    }
    if config.vision_distance > MAX_MUTATED_VISION_DISTANCE {
        return Err(SimError::InvalidConfig(format!(
            "vision_distance must be <= {}",
            MAX_MUTATED_VISION_DISTANCE
        )));
    }

    if !config.spatial_prior_sigma.is_finite() || config.spatial_prior_sigma <= 0.0 {
        return Err(SimError::InvalidConfig(
            "spatial_prior_sigma must be > 0".to_owned(),
        ));
    }

    let max_synapses = max_possible_synapses(config.num_neurons);
    if config.num_synapses > max_synapses {
        return Err(SimError::InvalidConfig(format!(
            "num_synapses must be <= {max_synapses} for num_neurons={}",
            config.num_neurons
        )));
    }

    Ok(())
}
```

## sim-core/src/spawn.rs

```rust
use crate::brain::express_genome;
use crate::genome::{
    generate_seed_genome, genome_distance, mutate_genome, prune_disconnected_inter_neurons,
};
use crate::grid::{opposite_direction, world_capacity};
use crate::Simulation;
use rand::seq::SliceRandom;
use rand::Rng;
use serde::{Deserialize, Serialize};
use sim_types::{
    FacingDirection, FoodId, FoodState, Occupant, OrganismGenome, OrganismId, OrganismState,
    SpeciesId,
};

const DEFAULT_TERRAIN_THRESHOLD: f64 = 0.86;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct FoodRegrowthEvent {
    pub(crate) due_turn: u64,
    #[serde(default)]
    pub(crate) tie_break: u64,
    pub(crate) tile_idx: usize,
    pub(crate) generation: u32,
}

#[derive(Clone)]
pub(crate) struct ReproductionSpawn {
    pub(crate) parent_genome: OrganismGenome,
    pub(crate) parent_species_id: SpeciesId,
    pub(crate) parent_facing: FacingDirection,
    pub(crate) q: i32,
    pub(crate) r: i32,
}

#[derive(Clone)]
pub(crate) enum SpawnRequestKind {
    Reproduction(ReproductionSpawn),
}

#[derive(Clone)]
pub(crate) struct SpawnRequest {
    pub(crate) kind: SpawnRequestKind,
}

impl Simulation {
    pub(crate) fn resolve_spawn_requests(&mut self, queue: &[SpawnRequest]) -> Vec<OrganismState> {
        let mut spawned = Vec::new();
        for request in queue {
            let organism = match &request.kind {
                SpawnRequestKind::Reproduction(reproduction) => {
                    let mut child_genome = reproduction.parent_genome.clone();
                    mutate_genome(&mut child_genome, &mut self.rng);
                    prune_disconnected_inter_neurons(&mut child_genome);

                    let threshold = self.config.speciation_threshold;
                    let child_species_id = {
                        let within_lineage = self
                            .species_registry
                            .get(&reproduction.parent_species_id)
                            .map(|root| genome_distance(&child_genome, root) <= threshold)
                            .unwrap_or(false);
                        if within_lineage {
                            reproduction.parent_species_id
                        } else {
                            let id = self.alloc_species_id();
                            self.species_registry.insert(id, child_genome.clone());
                            id
                        }
                    };

                    let brain = express_genome(&child_genome, &mut self.rng);
                    OrganismState {
                        id: self.alloc_organism_id(),
                        species_id: child_species_id,
                        q: reproduction.q,
                        r: reproduction.r,
                        age_turns: 0,
                        facing: opposite_direction(reproduction.parent_facing),
                        energy: child_genome.starting_energy,
                        energy_prev: child_genome.starting_energy,
                        consumptions_count: 0,
                        reproductions_count: 0,
                        brain,
                        genome: child_genome,
                    }
                }
            };

            if self.add_organism(organism.clone()) {
                spawned.push(organism);
            }
        }

        spawned
    }

    pub(crate) fn spawn_initial_population(&mut self) {
        let seed_config = self.config.seed_genome_config.clone();

        let mut open_positions = self.empty_positions();
        open_positions.shuffle(&mut self.rng);
        let initial_population = self.target_population().min(open_positions.len());

        for _ in 0..initial_population {
            let (q, r) = open_positions
                .pop()
                .expect("initial population requires at least one unique cell per organism");
            let id = self.alloc_organism_id();
            let genome = generate_seed_genome(&seed_config, &mut self.rng);
            let brain = express_genome(&genome, &mut self.rng);

            // Seed genomes are independently random — each gets its own species.
            // Species assignment via genome distance only matters for mutation-derived offspring.
            let species_id = self.alloc_species_id();
            self.species_registry.insert(species_id, genome.clone());

            let facing = self.random_facing();
            let organism = OrganismState {
                id,
                species_id,
                q,
                r,
                age_turns: 0,
                facing,
                energy: genome.starting_energy,
                energy_prev: genome.starting_energy,
                consumptions_count: 0,
                reproductions_count: 0,
                brain,
                genome,
            };
            let added = self.add_organism(organism);
            debug_assert!(added);
        }
    }

    fn random_facing(&mut self) -> FacingDirection {
        FacingDirection::ALL[self.rng.random_range(0..FacingDirection::ALL.len())]
    }

    fn alloc_organism_id(&mut self) -> OrganismId {
        let id = OrganismId(self.next_organism_id);
        self.next_organism_id += 1;
        id
    }

    fn alloc_food_id(&mut self) -> FoodId {
        let id = FoodId(self.next_food_id);
        self.next_food_id += 1;
        id
    }

    fn target_population(&self) -> usize {
        let max_population = self.config.num_organisms as usize;
        let available_cells = if self.terrain_map.is_empty() {
            world_capacity(self.config.world_width)
        } else {
            self.terrain_map.iter().filter(|blocked| !**blocked).count()
        };
        max_population.min(available_cells)
    }

    pub(crate) fn initialize_terrain(&mut self) {
        let width = self.config.world_width;
        let terrain_seed = self.seed ^ 0xA5A5_A5A5_u64;
        self.terrain_map = if (self.config.terrain_threshold as f64 - DEFAULT_TERRAIN_THRESHOLD)
            .abs()
            > f64::EPSILON
        {
            build_terrain_map_with_threshold(
                width,
                width,
                self.config.terrain_noise_scale as f64,
                terrain_seed,
                self.config.terrain_threshold as f64,
            )
        } else {
            build_terrain_map(
                width,
                width,
                self.config.terrain_noise_scale as f64,
                terrain_seed,
            )
        };
        for (idx, blocked) in self.terrain_map.iter().copied().enumerate() {
            if blocked {
                self.occupancy[idx] = Some(Occupant::Wall);
            }
        }
    }

    pub(crate) fn initialize_food_ecology(&mut self) {
        let capacity = world_capacity(self.config.world_width);
        self.food_fertility = build_fertility_map(self.config.world_width, self.seed, &self.config);
        debug_assert_eq!(self.food_fertility.len(), capacity);
        self.food_regrowth_generation = vec![0; capacity];
        self.food_regrowth_queue.clear();
    }

    pub(crate) fn seed_initial_food_supply(&mut self) {
        self.ensure_food_ecology_state();
        for cell_idx in 0..self.food_fertility.len() {
            if self.occupancy[cell_idx].is_some() {
                continue;
            }
            if self.accept_fertility_sample(cell_idx) {
                let _ = self.spawn_food_at_cell(cell_idx);
            }
        }
    }

    pub(crate) fn bootstrap_food_regrowth_queue(&mut self) {
        self.ensure_food_ecology_state();
        for idx in 0..self.food_fertility.len() {
            let delay = self.regrowth_delay_for_tile(idx);
            self.schedule_food_regrowth_with_delay(idx, delay);
        }
    }

    pub(crate) fn replenish_food_supply(&mut self) -> Vec<FoodState> {
        self.ensure_food_ecology_state();
        let mut spawned = Vec::new();

        loop {
            let Some(next_event) = self.food_regrowth_queue.first().copied() else {
                break;
            };
            if next_event.due_turn > self.turn {
                break;
            }
            let event = self
                .food_regrowth_queue
                .pop_first()
                .expect("first event must still be present");
            if self.food_regrowth_generation[event.tile_idx] != event.generation {
                continue;
            }
            if self.occupancy[event.tile_idx].is_none()
                && self.accept_fertility_sample(event.tile_idx)
            {
                if let Some(food) = self.spawn_food_at_cell(event.tile_idx) {
                    spawned.push(food);
                }
            }
            let delay = self.regrowth_delay_for_tile(event.tile_idx);
            self.schedule_food_regrowth_with_delay(event.tile_idx, delay);
        }

        spawned
    }

    fn ensure_food_ecology_state(&mut self) {
        let capacity = world_capacity(self.config.world_width);
        let valid_fertility = self.food_fertility.len() == capacity;
        let valid_generations = self.food_regrowth_generation.len() == capacity;
        if valid_fertility && valid_generations {
            return;
        }

        self.initialize_food_ecology();
        for idx in 0..capacity {
            let delay = self.regrowth_delay_for_tile(idx);
            self.schedule_food_regrowth_with_delay(idx, delay);
        }
    }

    fn spawn_food_at_cell(&mut self, cell_idx: usize) -> Option<FoodState> {
        if self.occupancy[cell_idx].is_some() {
            return None;
        }
        let width = self.config.world_width as usize;
        let q = (cell_idx % width) as i32;
        let r = (cell_idx / width) as i32;
        let food = FoodState {
            id: self.alloc_food_id(),
            q,
            r,
            energy: self.config.food_energy,
        };
        self.occupancy[cell_idx] = Some(Occupant::Food(food.id));
        self.foods.push(food.clone());
        Some(food)
    }

    fn schedule_food_regrowth_with_delay(&mut self, tile_idx: usize, delay: u64) {
        let generation = self.food_regrowth_generation[tile_idx].saturating_add(1);
        self.food_regrowth_generation[tile_idx] = generation;
        self.food_regrowth_queue.insert(FoodRegrowthEvent {
            due_turn: self.turn.saturating_add(delay),
            tie_break: hash_2d(tile_idx as i64, generation as i64, self.seed),
            tile_idx,
            generation,
        });
    }

    fn regrowth_delay_for_tile(&mut self, _tile_idx: usize) -> u64 {
        let interval = self.config.food_regrowth_interval.max(1);
        let base_delay = self.rng.random_range(1..=interval);
        (f64::from(base_delay) / f64::from(self.config.plant_growth_speed)).ceil() as u64
    }

    fn accept_fertility_sample(&mut self, tile_idx: usize) -> bool {
        let chance = self.fertility_value(tile_idx);
        self.rng.random::<f32>() <= chance
    }

    fn fertility_value(&self, tile_idx: usize) -> f32 {
        self.food_fertility[tile_idx] as f32 / u16::MAX as f32
    }

    fn empty_positions(&self) -> Vec<(i32, i32)> {
        let width = self.config.world_width as i32;
        let mut positions = Vec::new();
        for r in 0..width {
            for q in 0..width {
                if self.occupant_at(q, r).is_none() {
                    positions.push((q, r));
                }
            }
        }
        positions
    }
}

fn build_fertility_map(world_width: u32, seed: u64, config: &sim_types::WorldConfig) -> Vec<u16> {
    let width = world_width as usize;
    let mut fertility = Vec::with_capacity(width * width);
    for r in 0..width {
        for q in 0..width {
            let x = q as f64 * config.food_fertility_noise_scale as f64;
            let y = r as f64 * config.food_fertility_noise_scale as f64;
            let value = fractal_perlin_2d(x, y, seed);
            let normalized = ((value + 1.0) * 0.5).clamp(0.0, 1.0) as f32;
            let shifted = config.food_fertility_floor
                + (1.0 - config.food_fertility_floor)
                    * normalized.powf(config.food_fertility_exponent);
            let encoded = (shifted.clamp(0.0, 1.0) * u16::MAX as f32).round() as u16;
            fertility.push(encoded);
        }
    }
    fertility
}

pub(crate) fn build_terrain_map(width: u32, height: u32, scale: f64, seed: u64) -> Vec<bool> {
    build_terrain_map_with_threshold(width, height, scale, seed, DEFAULT_TERRAIN_THRESHOLD)
}

fn build_terrain_map_with_threshold(
    width: u32,
    height: u32,
    scale: f64,
    seed: u64,
    terrain_threshold: f64,
) -> Vec<bool> {
    let width = width as usize;
    let height = height as usize;
    let mut blocked = Vec::with_capacity(width * height);
    for r in 0..height {
        for q in 0..width {
            let x = q as f64 * scale;
            let y = r as f64 * scale;
            let value = fractal_perlin_2d(x, y, seed);
            let normalized = ((value + 1.0) * 0.5).clamp(0.0, 1.0);
            blocked.push(normalized > terrain_threshold);
        }
    }
    blocked
}

fn fractal_perlin_2d(x: f64, y: f64, seed: u64) -> f64 {
    const OCTAVES: usize = 1;
    let mut amplitude = 1.0_f64;
    let mut frequency = 1.0_f64;
    let mut total = 0.0_f64;
    let mut weight = 0.0_f64;

    for octave in 0..OCTAVES {
        let octave_seed =
            seed.wrapping_add(0x9E37_79B9_7F4A_7C15_u64.wrapping_mul(octave as u64 + 1));
        total += amplitude * perlin_2d(x * frequency, y * frequency, octave_seed);
        weight += amplitude;
        amplitude *= 0.5;
        frequency *= 2.0;
    }

    if weight == 0.0 {
        0.0
    } else {
        total / weight
    }
}

fn perlin_2d(x: f64, y: f64, seed: u64) -> f64 {
    let x0 = x.floor() as i64;
    let y0 = y.floor() as i64;
    let x1 = x0 + 1;
    let y1 = y0 + 1;

    let dx = x - x0 as f64;
    let dy = y - y0 as f64;

    let n00 = grad(hash_2d(x0, y0, seed), dx, dy);
    let n10 = grad(hash_2d(x1, y0, seed), dx - 1.0, dy);
    let n01 = grad(hash_2d(x0, y1, seed), dx, dy - 1.0);
    let n11 = grad(hash_2d(x1, y1, seed), dx - 1.0, dy - 1.0);

    let u = fade(dx);
    let v = fade(dy);
    let nx0 = lerp(n00, n10, u);
    let nx1 = lerp(n01, n11, u);
    lerp(nx0, nx1, v)
}

fn hash_2d(x: i64, y: i64, seed: u64) -> u64 {
    let mut z = seed
        ^ (x as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
        ^ (y as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z ^= z >> 30;
    z = z.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z ^= z >> 27;
    z = z.wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    z
}

fn grad(hash: u64, x: f64, y: f64) -> f64 {
    match (hash & 7) as u8 {
        0 => x + y,
        1 => x - y,
        2 => -x + y,
        3 => -x - y,
        4 => x,
        5 => -x,
        6 => y,
        _ => -y,
    }
}

fn fade(t: f64) -> f64 {
    t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
}

fn lerp(a: f64, b: f64, t: f64) -> f64 {
    a + t * (b - a)
}
```

## sim-core/src/turn.rs

```rust
use crate::brain::{action_index, apply_runtime_plasticity, evaluate_brain, BrainScratch};
use crate::grid::{hex_neighbor, opposite_direction, rotate_left, rotate_right, wrap_position};
use crate::spawn::{ReproductionSpawn, SpawnRequest, SpawnRequestKind};
use crate::Simulation;
#[cfg(feature = "profiling")]
use crate::{profiling, profiling::TurnPhase};
use rayon::prelude::*;
use sim_types::{
    ActionType, EntityId, FacingDirection, FoodState, Occupant, OrganismFacing, OrganismId,
    OrganismMove, OrganismState, RemovedEntityPosition, SpeciesId, TickDelta, WorldConfig,
};
use std::cmp::Ordering;
use std::collections::{BTreeMap, HashSet};
#[cfg(feature = "profiling")]
use std::time::Instant;

const FOOD_ENERGY_METABOLISM_DIVISOR: f32 = 1000.0;

#[derive(Clone, Copy)]
struct SnapshotOrganismState {
    q: i32,
    r: i32,
    facing: FacingDirection,
}

#[derive(Clone)]
struct TurnSnapshot {
    world_width: i32,
    organism_count: usize,
    organism_ids: Vec<OrganismId>,
    organism_states: Vec<SnapshotOrganismState>,
}

#[derive(Clone, Copy)]
struct OrganismIntent {
    idx: usize,
    id: OrganismId,
    from: (i32, i32),
    facing_after_actions: FacingDirection,
    wants_move: bool,
    wants_reproduce: bool,
    move_target: Option<(i32, i32)>,
    move_confidence: f32,
    action_cost_count: u8,
    synapse_ops: u64,
}

#[derive(Clone, Copy)]
struct MoveCandidate {
    actor_idx: usize,
    actor_id: OrganismId,
    from: (i32, i32),
    target: (i32, i32),
    confidence: f32,
}

#[derive(Clone, Copy)]
struct MoveResolution {
    actor_idx: usize,
    actor_id: OrganismId,
    from: (i32, i32),
    to: (i32, i32),
}

#[derive(Default)]
struct CommitResult {
    moves: Vec<OrganismMove>,
    facing_updates: Vec<OrganismFacing>,
    removed_positions: Vec<RemovedEntityPosition>,
    food_spawned: Vec<FoodState>,
    consumptions: u64,
    predations: u64,
    actions_applied: u64,
}

impl Simulation {
    pub(crate) fn tick(&mut self) -> TickDelta {
        #[cfg(feature = "profiling")]
        let tick_started = Instant::now();

        #[cfg(feature = "profiling")]
        let phase_started = Instant::now();
        let (starvations, starved_removed_positions) = self.lifecycle_phase();
        #[cfg(feature = "profiling")]
        profiling::record_turn_phase(TurnPhase::Lifecycle, phase_started.elapsed());

        #[cfg(feature = "profiling")]
        let phase_started = Instant::now();
        let snapshot = self.build_turn_snapshot();
        #[cfg(feature = "profiling")]
        profiling::record_turn_phase(TurnPhase::Snapshot, phase_started.elapsed());

        #[cfg(feature = "profiling")]
        let phase_started = Instant::now();
        let intents = self.build_intents(&snapshot);
        #[cfg(feature = "profiling")]
        profiling::record_turn_phase(TurnPhase::Intents, phase_started.elapsed());

        let synapse_ops = intents.iter().map(|intent| intent.synapse_ops).sum::<u64>();

        let mut spawn_requests = Vec::new();
        #[cfg(feature = "profiling")]
        let phase_started = Instant::now();
        let successful_reproduction = Self::reproduction_phase(
            &mut self.organisms,
            &intents,
            &self.occupancy,
            snapshot.world_width,
            self.config.reproduction_energy_cost,
            &mut spawn_requests,
        );
        #[cfg(feature = "profiling")]
        profiling::record_turn_phase(TurnPhase::Reproduction, phase_started.elapsed());

        let reproductions = successful_reproduction
            .iter()
            .filter(|reproduced| **reproduced)
            .count() as u64;
        #[cfg(feature = "profiling")]
        let phase_started = Instant::now();
        let resolutions = self.resolve_moves(
            &snapshot,
            &self.occupancy,
            &intents,
            &successful_reproduction,
        );
        #[cfg(feature = "profiling")]
        profiling::record_turn_phase(TurnPhase::MoveResolution, phase_started.elapsed());

        #[cfg(feature = "profiling")]
        let phase_started = Instant::now();
        let commit = self.commit_phase(&snapshot, &intents, &resolutions, &successful_reproduction);
        #[cfg(feature = "profiling")]
        profiling::record_turn_phase(TurnPhase::Commit, phase_started.elapsed());

        #[cfg(feature = "profiling")]
        let phase_started = Instant::now();
        self.increment_age_for_survivors();
        #[cfg(feature = "profiling")]
        profiling::record_turn_phase(TurnPhase::Age, phase_started.elapsed());

        #[cfg(feature = "profiling")]
        let phase_started = Instant::now();
        let spawned = self.resolve_spawn_requests(&spawn_requests);
        #[cfg(feature = "profiling")]
        profiling::record_turn_phase(TurnPhase::Spawn, phase_started.elapsed());

        #[cfg(feature = "profiling")]
        let phase_started = Instant::now();
        self.prune_extinct_species();
        #[cfg(feature = "profiling")]
        profiling::record_turn_phase(TurnPhase::PruneSpecies, phase_started.elapsed());

        #[cfg(feature = "profiling")]
        let phase_started = Instant::now();
        self.debug_assert_consistent_state();
        #[cfg(feature = "profiling")]
        profiling::record_turn_phase(TurnPhase::ConsistencyCheck, phase_started.elapsed());

        #[cfg(feature = "profiling")]
        let phase_started = Instant::now();
        self.turn = self.turn.saturating_add(1);
        self.metrics.turns = self.turn;
        self.metrics.synapse_ops_last_turn = synapse_ops;
        self.metrics.actions_applied_last_turn = commit.actions_applied + reproductions;
        self.metrics.consumptions_last_turn = commit.consumptions;
        self.metrics.predations_last_turn = commit.predations;
        self.metrics.total_consumptions += commit.consumptions;
        self.metrics.reproductions_last_turn = reproductions;
        self.metrics.starvations_last_turn = starvations;
        self.refresh_population_metrics();

        let mut removed_positions = commit.removed_positions;
        removed_positions.extend(starved_removed_positions);
        #[cfg(feature = "profiling")]
        profiling::record_turn_phase(TurnPhase::MetricsAndDelta, phase_started.elapsed());

        #[cfg(feature = "profiling")]
        profiling::record_tick_total(tick_started.elapsed());

        TickDelta {
            turn: self.turn,
            moves: commit.moves,
            facing_updates: commit.facing_updates,
            removed_positions,
            spawned,
            food_spawned: commit.food_spawned,
            metrics: self.metrics.clone(),
        }
    }

    fn build_turn_snapshot(&self) -> TurnSnapshot {
        let len = self.organisms.len();
        let mut organism_ids = Vec::with_capacity(len);
        let mut organism_states = Vec::with_capacity(len);

        for organism in &self.organisms {
            organism_ids.push(organism.id);
            organism_states.push(SnapshotOrganismState {
                q: organism.q,
                r: organism.r,
                facing: organism.facing,
            });
        }

        TurnSnapshot {
            world_width: self.config.world_width as i32,
            organism_count: len,
            organism_ids,
            organism_states,
        }
    }

    fn build_intents(&mut self, snapshot: &TurnSnapshot) -> Vec<OrganismIntent> {
        let occupancy = &self.occupancy;
        let food_energy = self.config.food_energy;

        if self.should_parallelize_intents(snapshot.organism_count) {
            let intent_threads = self.intent_parallelism();
            let world_width = snapshot.world_width;
            let organism_ids = &snapshot.organism_ids;
            let organism_states = &snapshot.organism_states;
            return crate::install_with_intent_pool(intent_threads, || {
                self.organisms
                    .par_iter_mut()
                    .enumerate()
                    .map_init(BrainScratch::new, |scratch, (idx, organism)| {
                        build_intent_for_organism(
                            idx,
                            organism,
                            food_energy,
                            world_width,
                            occupancy,
                            organism_states[idx],
                            organism_ids[idx],
                            scratch,
                        )
                    })
                    .collect()
            });
        }

        let mut intents = Vec::with_capacity(snapshot.organism_count);
        let mut scratch = BrainScratch::new();
        for idx in 0..snapshot.organism_count {
            intents.push(build_intent_for_organism(
                idx,
                &mut self.organisms[idx],
                food_energy,
                snapshot.world_width,
                occupancy,
                snapshot.organism_states[idx],
                snapshot.organism_ids[idx],
                &mut scratch,
            ));
        }
        intents
    }

    fn resolve_moves(
        &self,
        snapshot: &TurnSnapshot,
        occupancy: &[Option<Occupant>],
        intents: &[OrganismIntent],
        successful_reproduction: &[bool],
    ) -> Vec<MoveResolution> {
        let w = snapshot.world_width as usize;
        let world_cells = occupancy.len();
        let mut best_by_cell: Vec<Option<MoveCandidate>> = vec![None; world_cells];

        for intent in intents {
            if !intent.wants_move {
                continue;
            }
            if successful_reproduction[intent.idx] {
                continue;
            }
            let Some(target) = intent.move_target else {
                continue;
            };
            let cell_idx = target.1 as usize * w + target.0 as usize;
            if matches!(
                occupancy[cell_idx],
                Some(Occupant::Organism(_)) | Some(Occupant::Wall)
            ) {
                continue;
            }
            let candidate = MoveCandidate {
                actor_idx: intent.idx,
                actor_id: intent.id,
                from: intent.from,
                target,
                confidence: intent.move_confidence,
            };
            match &best_by_cell[cell_idx] {
                Some(current)
                    if compare_move_candidates(&candidate, current) != Ordering::Greater => {}
                _ => best_by_cell[cell_idx] = Some(candidate),
            }
        }

        let mut winners: Vec<MoveCandidate> = best_by_cell.into_iter().flatten().collect();
        winners.sort_by_key(|w| w.actor_idx);

        winners
            .into_iter()
            .map(|winner| MoveResolution {
                actor_idx: winner.actor_idx,
                actor_id: winner.actor_id,
                from: winner.from,
                to: winner.target,
            })
            .collect()
    }

    fn commit_phase(
        &mut self,
        snapshot: &TurnSnapshot,
        intents: &[OrganismIntent],
        resolutions: &[MoveResolution],
        successful_reproduction: &[bool],
    ) -> CommitResult {
        let world_width = snapshot.world_width;
        let world_width_usize = world_width as usize;

        let mut facing_updates = Vec::new();
        let mut actions_applied = 0_u64;
        let action_energy_cost = self.config.move_action_energy_cost;
        for (idx, intent) in intents.iter().enumerate() {
            debug_assert_eq!(intent.idx, idx);
            let organism = &mut self.organisms[idx];
            if organism.facing != intent.facing_after_actions {
                facing_updates.push(OrganismFacing {
                    id: organism.id,
                    facing: intent.facing_after_actions,
                });
            }
            organism.facing = intent.facing_after_actions;
            let action_count = u64::from(intent.action_cost_count);
            actions_applied += action_count;
            if action_count > 0 {
                organism.energy -= action_energy_cost * intent.action_cost_count as f32;
            }
        }

        let org_count = self.organisms.len();
        let food_count = self.foods.len();
        let mut bite_targets = vec![None; org_count];
        for (idx, intent) in intents.iter().enumerate() {
            if !intent.wants_move || successful_reproduction[idx] {
                continue;
            }
            let Some((target_q, target_r)) = intent.move_target else {
                continue;
            };
            let target_idx = target_r as usize * world_width_usize + target_q as usize;
            if let Some(Occupant::Organism(prey_id)) = self.occupancy[target_idx] {
                bite_targets[idx] = Some(prey_id);
            }
        }

        let mut move_to: Vec<Option<(i32, i32)>> = vec![None; org_count];
        let mut consumed_food = vec![false; food_count];
        let mut removed_positions = Vec::new();
        let mut consumptions = 0_u64;
        let mut predations = 0_u64;
        let mut dead_organisms = vec![false; org_count];
        for resolution in resolutions {
            move_to[resolution.actor_idx] = Some(resolution.to);
            let from_idx =
                resolution.from.1 as usize * world_width_usize + resolution.from.0 as usize;
            let to_idx = resolution.to.1 as usize * world_width_usize + resolution.to.0 as usize;
            debug_assert_eq!(
                self.occupancy[from_idx],
                Some(Occupant::Organism(resolution.actor_id))
            );
            debug_assert!(!matches!(
                self.occupancy[to_idx],
                Some(Occupant::Organism(_)) | Some(Occupant::Wall)
            ));

            if let Some(Occupant::Food(food_id)) = self.occupancy[to_idx] {
                if let Some(food_idx) = food_index_by_id(&self.foods, food_id) {
                    consumed_food[food_idx] = true;
                    let food = &self.foods[food_idx];
                    removed_positions.push(RemovedEntityPosition {
                        entity_id: EntityId::Food(food_id),
                        q: food.q,
                        r: food.r,
                    });
                    self.organisms[resolution.actor_idx].energy += food.energy;
                    self.organisms[resolution.actor_idx].consumptions_count = self.organisms
                        [resolution.actor_idx]
                        .consumptions_count
                        .saturating_add(1);
                    consumptions += 1;
                }
            }

            self.occupancy[from_idx] = None;
            self.occupancy[to_idx] = Some(Occupant::Organism(resolution.actor_id));
            let organism = &mut self.organisms[resolution.actor_idx];
            organism.q = resolution.to.0;
            organism.r = resolution.to.1;
        }

        for (idx, intent) in intents.iter().enumerate() {
            if !intent.wants_move || dead_organisms[idx] {
                continue;
            }
            if successful_reproduction[idx] {
                continue;
            }
            if move_to[idx].is_some() {
                continue;
            }
            let Some((target_q, target_r)) = intent.move_target else {
                continue;
            };
            let Some(prey_id) = bite_targets[idx] else {
                continue;
            };
            let target_idx = target_r as usize * world_width_usize + target_q as usize;

            match self.occupancy[target_idx] {
                Some(Occupant::Organism(current_prey_id)) if current_prey_id == prey_id => {
                    let Some(prey_idx) = organism_index_by_id(&self.organisms, prey_id) else {
                        continue;
                    };
                    if idx == prey_idx || dead_organisms[prey_idx] {
                        continue;
                    }

                    let drain = self.organisms[prey_idx]
                        .energy
                        .min(self.config.food_energy * 2.0)
                        .max(0.0);
                    if drain <= 0.0 {
                        continue;
                    }

                    if idx < prey_idx {
                        let (left, right) = self.organisms.split_at_mut(prey_idx);
                        let predator = &mut left[idx];
                        let prey = &mut right[0];
                        prey.energy -= drain;
                        predator.energy += drain;
                        predator.consumptions_count = predator.consumptions_count.saturating_add(1);
                    } else if idx > prey_idx {
                        let (left, right) = self.organisms.split_at_mut(idx);
                        let prey = &mut left[prey_idx];
                        let predator = &mut right[0];
                        prey.energy -= drain;
                        predator.energy += drain;
                        predator.consumptions_count = predator.consumptions_count.saturating_add(1);
                    } else {
                        continue;
                    }

                    consumptions += 1;
                    predations += 1;

                    if self.organisms[prey_idx].energy <= 0.0 && !dead_organisms[prey_idx] {
                        dead_organisms[prey_idx] = true;
                        removed_positions.push(RemovedEntityPosition {
                            entity_id: EntityId::Organism(prey_id),
                            q: self.organisms[prey_idx].q,
                            r: self.organisms[prey_idx].r,
                        });
                        self.occupancy[target_idx] = None;
                    }
                }
                None => {}
                Some(Occupant::Food(_)) => {}
                Some(Occupant::Organism(_)) => {}
                Some(Occupant::Wall) => {}
            }
        }

        if dead_organisms.iter().any(|dead| *dead) {
            let mut survivors = Vec::with_capacity(self.organisms.len());
            for (idx, organism) in self.organisms.drain(..).enumerate() {
                if !dead_organisms[idx] {
                    survivors.push(organism);
                }
            }
            self.organisms = survivors;
        }

        let mut new_foods = Vec::with_capacity(food_count);
        for (idx, food) in self.foods.drain(..).enumerate() {
            if !consumed_food[idx] {
                new_foods.push(food);
            }
        }
        self.foods = new_foods;

        let food_spawned = self.replenish_food_supply();

        let moves = resolutions
            .iter()
            .map(|resolution| OrganismMove {
                id: resolution.actor_id,
                from: resolution.from,
                to: resolution.to,
            })
            .collect();

        CommitResult {
            moves,
            facing_updates,
            removed_positions,
            food_spawned,
            consumptions,
            predations,
            actions_applied,
        }
    }

    fn reproduction_phase(
        organisms: &mut [OrganismState],
        intents: &[OrganismIntent],
        occupancy: &[Option<Occupant>],
        world_width: i32,
        reproduction_energy_cost: f32,
        spawn_requests: &mut Vec<SpawnRequest>,
    ) -> Vec<bool> {
        let mut reserved_spawn_cells = HashSet::new();
        let mut successful_reproduction = vec![false; organisms.len()];

        for intent in intents {
            let org_idx = intent.idx;
            let organism = &mut organisms[org_idx];
            if !intent.wants_reproduce {
                continue;
            }

            let parent_energy = organism.energy;
            if parent_energy < reproduction_energy_cost {
                continue;
            }
            let maturity_age = u64::from(organism.genome.age_of_maturity);
            if organism.age_turns < maturity_age {
                continue;
            }
            let parent_q = organism.q;
            let parent_r = organism.r;
            let parent_facing = organism.facing;
            let parent_species_id = organism.species_id;
            let parent_genome = organism.genome.clone();

            let (q, r) = reproduction_target(world_width, parent_q, parent_r, parent_facing);
            if occupancy_snapshot_cell(occupancy, world_width, q, r).is_some()
                || reserved_spawn_cells.contains(&(q, r))
            {
                continue;
            }

            spawn_requests.push(SpawnRequest {
                kind: SpawnRequestKind::Reproduction(ReproductionSpawn {
                    parent_genome,
                    parent_species_id,
                    parent_facing,
                    q,
                    r,
                }),
            });
            reserved_spawn_cells.insert((q, r));
            organism.energy -= reproduction_energy_cost;
            organism.reproductions_count = organism.reproductions_count.saturating_add(1);
            successful_reproduction[org_idx] = true;
        }

        successful_reproduction
    }

    fn lifecycle_phase(&mut self) -> (u64, Vec<RemovedEntityPosition>) {
        let max_age = self.config.max_organism_age as u64;
        let world_width = self.config.world_width as usize;
        let mut dead = vec![false; self.organisms.len()];
        let mut starved_positions = Vec::new();

        for (idx, organism) in self.organisms.iter_mut().enumerate() {
            let metabolism_energy_cost = organism_metabolism_energy_cost(&self.config, organism);
            organism.energy -= metabolism_energy_cost;
            if organism.energy <= 0.0 || organism.age_turns >= max_age {
                dead[idx] = true;
                let cell_idx = organism.r as usize * world_width + organism.q as usize;
                self.occupancy[cell_idx] = None;
                starved_positions.push(RemovedEntityPosition {
                    entity_id: EntityId::Organism(organism.id),
                    q: organism.q,
                    r: organism.r,
                });
            }
        }

        let starvation_count = starved_positions.len() as u64;
        if starvation_count == 0 {
            return (0, starved_positions);
        }

        let mut new_organisms = Vec::with_capacity(self.organisms.len());
        for (idx, organism) in self.organisms.drain(..).enumerate() {
            if !dead[idx] {
                new_organisms.push(organism);
            }
        }
        self.organisms = new_organisms;

        (starvation_count, starved_positions)
    }
}

fn organism_metabolism_energy_cost(config: &WorldConfig, organism: &OrganismState) -> f32 {
    organism_metabolism_energy_cost_from_food_energy(config.food_energy, organism)
}

fn organism_metabolism_energy_cost_from_food_energy(
    food_energy: f32,
    organism: &OrganismState,
) -> f32 {
    // `num_neurons` tracks enabled interneurons. Sensory neurons are concrete runtime nodes.
    let neuron_count = organism.genome.num_neurons as f32 + organism.brain.sensory.len() as f32;
    let vision_distance_cost = organism.genome.vision_distance as f32 / 3.0;
    let neuron_energy_cost = food_energy / FOOD_ENERGY_METABOLISM_DIVISOR;
    neuron_energy_cost * (neuron_count + vision_distance_cost)
}

impl Simulation {
    fn increment_age_for_survivors(&mut self) {
        for organism in &mut self.organisms {
            organism.age_turns = organism.age_turns.saturating_add(1);
        }
    }

    pub(crate) fn refresh_population_metrics(&mut self) {
        self.metrics.organisms = self.organisms.len() as u32;
        self.metrics.total_species_created = self.next_species_id;
        self.metrics.species_counts = self.compute_species_counts();
    }

    fn compute_species_counts(&self) -> BTreeMap<SpeciesId, u32> {
        let mut species_counts = BTreeMap::new();
        for organism in &self.organisms {
            let count = species_counts.entry(organism.species_id).or_insert(0_u32);
            *count = count.saturating_add(1);
        }
        species_counts
    }
}

fn compare_move_candidates(a: &MoveCandidate, b: &MoveCandidate) -> Ordering {
    a.confidence
        .total_cmp(&b.confidence)
        .then_with(|| b.actor_id.cmp(&a.actor_id))
}

fn build_intent_for_organism(
    idx: usize,
    organism: &mut OrganismState,
    food_energy: f32,
    world_width: i32,
    occupancy: &[Option<Occupant>],
    snapshot_state: SnapshotOrganismState,
    organism_id: OrganismId,
    scratch: &mut BrainScratch,
) -> OrganismIntent {
    let vision_distance = organism.genome.vision_distance;
    #[cfg(feature = "profiling")]
    let brain_eval_started = Instant::now();
    let evaluation = evaluate_brain(organism, world_width, occupancy, vision_distance, scratch);
    #[cfg(feature = "profiling")]
    profiling::record_brain_eval_total(brain_eval_started.elapsed());

    let passive_energy_baseline =
        organism_metabolism_energy_cost_from_food_energy(food_energy, organism);
    #[cfg(feature = "profiling")]
    let plasticity_started = Instant::now();
    apply_runtime_plasticity(organism, passive_energy_baseline, scratch);
    #[cfg(feature = "profiling")]
    profiling::record_brain_plasticity_total(plasticity_started.elapsed());

    let selected_action = evaluation.resolved_actions.selected_action;
    let selected_action_activation = evaluation.action_activations[action_index(selected_action)];
    let (facing_after_actions, wants_move, wants_reproduce, move_target) =
        intent_from_selected_action(selected_action, snapshot_state, world_width);
    let move_confidence = if wants_move {
        selected_action_activation
    } else {
        0.0
    };

    OrganismIntent {
        idx,
        id: organism_id,
        from: (snapshot_state.q, snapshot_state.r),
        facing_after_actions,
        wants_move,
        wants_reproduce,
        move_target,
        move_confidence,
        action_cost_count: u8::from(selected_action != ActionType::Idle),
        synapse_ops: evaluation.synapse_ops,
    }
}

fn intent_from_selected_action(
    selected_action: ActionType,
    snapshot_state: SnapshotOrganismState,
    world_width: i32,
) -> (FacingDirection, bool, bool, Option<(i32, i32)>) {
    let from = (snapshot_state.q, snapshot_state.r);
    let current_facing = snapshot_state.facing;

    match selected_action {
        ActionType::Idle => (current_facing, false, false, None),
        ActionType::TurnLeft => (rotate_left(current_facing), false, false, None),
        ActionType::TurnRight => (rotate_right(current_facing), false, false, None),
        ActionType::Forward => (
            current_facing,
            true,
            false,
            Some(hex_neighbor(from, current_facing, world_width)),
        ),
        ActionType::TurnLeftForward => {
            let facing = rotate_left(current_facing);
            (
                facing,
                true,
                false,
                Some(hex_neighbor(from, facing, world_width)),
            )
        }
        ActionType::TurnRightForward => {
            let facing = rotate_right(current_facing);
            (
                facing,
                true,
                false,
                Some(hex_neighbor(from, facing, world_width)),
            )
        }
        ActionType::Consume => (current_facing, false, false, None),
        ActionType::Reproduce => (current_facing, false, true, None),
    }
}

fn occupancy_snapshot_cell(
    occupancy: &[Option<Occupant>],
    world_width: i32,
    q: i32,
    r: i32,
) -> Option<Occupant> {
    let (q, r) = wrap_position((q, r), world_width);
    let idx = r as usize * world_width as usize + q as usize;
    occupancy[idx]
}

fn organism_index_by_id(organisms: &[OrganismState], id: OrganismId) -> Option<usize> {
    organisms.binary_search_by_key(&id, |o| o.id).ok()
}

fn food_index_by_id(foods: &[FoodState], id: sim_types::FoodId) -> Option<usize> {
    foods.binary_search_by_key(&id, |food| food.id).ok()
}

fn reproduction_target(
    world_width: i32,
    parent_q: i32,
    parent_r: i32,
    parent_facing: FacingDirection,
) -> (i32, i32) {
    let opposite_facing = opposite_direction(parent_facing);
    hex_neighbor((parent_q, parent_r), opposite_facing, world_width)
}
```

## config/default.toml

```toml
world_width = 500
steps_per_second = 5
num_organisms = 5000
food_energy = 20
move_action_energy_cost = 1.0
plant_growth_speed = 0.05
food_regrowth_interval = 20
food_fertility_noise_scale = 0.012
food_fertility_exponent = 5.0
food_fertility_floor = 0.001
terrain_noise_scale = 0.01
terrain_threshold = 0.70
speciation_threshold = 20.0

[seed_genome_config]
num_neurons = 20
num_synapses = 50
spatial_prior_sigma = 3.0
vision_distance = 10
starting_energy = 250.0
age_of_maturity = 50
hebb_eta_baseline = 0.001
hebb_eta_gain = 0.01
eligibility_retention = 0.95
synapse_prune_threshold = 0.05
mutation_rate_age_of_maturity = 0.05
mutation_rate_vision_distance = 0.05
mutation_rate_num_synapses = 0.05
mutation_rate_inter_bias = 0.1
mutation_rate_inter_update_rate = 0.1
mutation_rate_action_bias = 0.1
mutation_rate_eligibility_retention = 0.05
mutation_rate_synapse_prune_threshold = 0.05
mutation_rate_neuron_location = 0.1
```
