use super::*;
use types::TickDelta;

pub(super) trait IntoEnergy {
    fn into_energy(self) -> u32;
}

impl IntoEnergy for i32 {
    fn into_energy(self) -> u32 {
        self.max(0) as u32
    }
}

impl IntoEnergy for u32 {
    fn into_energy(self) -> u32 {
        self
    }
}

impl IntoEnergy for f32 {
    fn into_energy(self) -> u32 {
        self.max(0.0).round() as u32
    }
}

impl IntoEnergy for f64 {
    fn into_energy(self) -> u32 {
        self.max(0.0).round() as u32
    }
}

fn finalize_test_organism(organism: OrganismState) -> OrganismState {
    organism
}

pub(super) fn test_genome() -> OrganismGenome {
    OrganismGenome::test_fixture()
}

pub(super) fn stable_test_config() -> WorldConfig {
    WorldConfig::test_fixture()
}

pub(super) fn test_config(world_width: u32, num_organisms: u32) -> WorldConfig {
    let mut config = stable_test_config();
    config.world_width = world_width;
    config.num_organisms = num_organisms;
    config
}

fn forced_brain(
    wants_move: bool,
    turn_left: bool,
    turn_right: bool,
    confidence: f32,
) -> BrainState {
    let preferred_action = if wants_move {
        ActionType::Forward
    } else if turn_left && !turn_right {
        ActionType::TurnLeft
    } else if turn_right && !turn_left {
        ActionType::TurnRight
    } else {
        ActionType::Idle
    };
    forced_brain_with_action(preferred_action, confidence)
}

fn forced_brain_with_action(preferred_action: ActionType, confidence: f32) -> BrainState {
    let sensory = vec![make_sensory_neuron(
        0,
        SensoryReceptor::Symbol { symbol: Symbol::A },
    )];
    let inter_id = NeuronId(1000);
    let inter_bias = confidence;
    let preferred_symbol = match preferred_action {
        ActionType::Idle => Symbol::A,
        ActionType::TurnLeft => Symbol::B,
        ActionType::TurnRight => Symbol::C,
        ActionType::Forward => Symbol::D,
        ActionType::Attack => Symbol::End,
    };
    let inter_synapses: Vec<SynapseEdge> = Symbol::ALL
        .into_iter()
        .enumerate()
        .map(|(idx, symbol)| SynapseEdge {
            pre_neuron_id: inter_id,
            post_neuron_id: NeuronId(2000 + idx as u32),
            timing: types::SynapseTiming::CurrentTick,
            pre_inter_index: None,
            post_inter_index: None,
            weight: if symbol == preferred_symbol {
                8.0
            } else {
                -8.0
            },
            eligibility: 0.0,
            pending_coactivation: 0.0,
        })
        .collect();
    let synapse_count = inter_synapses.len() as u32;
    let inter_state = inter_bias.clamp(-0.999_999, 0.999_999).atanh();
    let inter = vec![InterNeuronState {
        neuron: NeuronState {
            neuron_id: inter_id,
            neuron_type: NeuronType::Inter,
            bias: inter_bias,
            activation: confidence,
        },
        state: inter_state,
        alpha: 1.0,
        synapses: inter_synapses,
        output_synapse_start: 0,
    }];
    let action: Vec<_> = Symbol::ALL
        .into_iter()
        .enumerate()
        .map(|(idx, symbol)| make_action_neuron(2000 + idx as u32, symbol))
        .collect();
    BrainState {
        sensory,
        inter,
        action,
        recurrent_synapses: Vec::new(),
        previous_inter_activations: vec![0.0],
        synapse_count,
        sensory_mean_activation: vec![0.0],
        inter_mean_activation: vec![0.0],
        action_mean_activation: vec![0.0; Symbol::COUNT],
        means_initialized: false,
    }
}

pub(super) fn make_single_action_organism(
    id: u64,
    q: i32,
    r: i32,
    facing: FacingDirection,
    preferred_action: ActionType,
    confidence: f32,
    energy: impl IntoEnergy,
) -> OrganismState {
    let energy = energy.into_energy();
    let initial_energy = if energy == 0 { 10 } else { energy };
    finalize_test_organism(OrganismState {
        id: OrganismId(id),
        species_id: types::SpeciesId(id),
        q,
        r,
        generation: 0,
        age_turns: 0,
        facing,
        energy: initial_energy,
        energy_at_last_sensing: initial_energy,
        energy_flow_last_tick: 0,
        successful_attacks_count: 0,
        last_action_taken: ActionType::Idle,
        last_action_symbol: Symbol::A,
        last_action_mask: 0,
        #[cfg(feature = "instrumentation")]
        instrumentation: Default::default(),
        brain: forced_brain_with_action(preferred_action, confidence),
        genome: test_genome(),
    })
}

/// Build an organism whose action biases deterministically emit exactly the
/// requested categorical action. Extreme finite biases keep these tests
/// independent of the deterministic sampling draw while still exercising the
/// real brain-evaluation and intent-building pipeline.
pub(super) fn make_categorical_organism(
    id: u64,
    q: i32,
    r: i32,
    facing: FacingDirection,
    commands: &[ActionType],
    energy: impl IntoEnergy,
) -> OrganismState {
    make_categorical_organism_with_bias(id, q, r, facing, commands, 100.0, energy)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn make_categorical_organism_with_bias(
    id: u64,
    q: i32,
    r: i32,
    facing: FacingDirection,
    commands: &[ActionType],
    command_bias: f32,
    energy: impl IntoEnergy,
) -> OrganismState {
    let mut organism = make_single_action_organism(id, q, r, facing, ActionType::Idle, 0.0, energy);
    organism.genome.brain.action_biases.fill(-100.0);
    if let Some(command) = commands.first() {
        let symbol = match command {
            ActionType::Idle => Symbol::A,
            ActionType::TurnLeft => Symbol::B,
            ActionType::TurnRight => Symbol::C,
            ActionType::Forward => Symbol::D,
            ActionType::Attack => Symbol::End,
        };
        organism.genome.brain.action_biases[brain::action_index(symbol)] = command_bias;
    } else {
        organism.genome.brain.action_biases[brain::action_index(Symbol::A)] = command_bias;
    }
    organism
}

#[allow(clippy::too_many_arguments)]
pub(super) fn make_organism(
    id: u64,
    q: i32,
    r: i32,
    facing: FacingDirection,
    wants_move: bool,
    turn_left: bool,
    turn_right: bool,
    confidence: f32,
    energy: impl IntoEnergy,
) -> OrganismState {
    let energy = energy.into_energy();
    let initial_energy = if energy == 0 { 10 } else { energy };
    finalize_test_organism(OrganismState {
        id: OrganismId(id),
        species_id: types::SpeciesId(id),
        q,
        r,
        generation: 0,
        age_turns: 0,
        facing,
        energy: initial_energy,
        energy_at_last_sensing: initial_energy,
        energy_flow_last_tick: 0,
        successful_attacks_count: 0,
        last_action_taken: ActionType::Idle,
        last_action_symbol: Symbol::A,
        last_action_mask: 0,
        #[cfg(feature = "instrumentation")]
        instrumentation: Default::default(),
        brain: forced_brain(wants_move, turn_left, turn_right, confidence),
        genome: test_genome(),
    })
}

pub(super) fn configure_sim(sim: &mut Simulation, mut organisms: Vec<OrganismState>) {
    organisms.sort_by_key(|organism| organism.id);
    sim.organisms = organisms;
    sim.next_organism_id = sim
        .organisms
        .iter()
        .map(|organism| organism.id.0)
        .max()
        .map_or(0, |max_id| max_id + 1);
    sim.occupancy = vec![None; crate::grid::world_capacity(sim.config.world_width)];
    for (idx, blocked) in sim.terrain_map.iter().copied().enumerate() {
        if blocked {
            sim.occupancy[idx] = Some(Occupant::Wall);
        }
    }
    for organism in &sim.organisms {
        let idx = sim.cell_index(organism.q, organism.r);
        assert!(
            sim.occupancy[idx].is_none(),
            "test setup should not overlap"
        );
        sim.occupancy[idx] = Some(Occupant::Organism(organism.id));
    }
    sim.turn = 0;
    sim.metrics = MetricsSnapshot::default();
    sim.refresh_population_metrics();
}

pub(super) fn tick_once(sim: &mut Simulation) -> TickDelta {
    sim.tick()
}

#[allow(clippy::type_complexity)]
pub(super) fn move_map(delta: &TickDelta) -> HashMap<OrganismId, ((i32, i32), (i32, i32))> {
    delta
        .moves
        .iter()
        .map(|movement| (movement.id, (movement.from, movement.to)))
        .collect()
}

pub(super) fn assert_no_overlap(sim: &Simulation) {
    let mut seen = HashSet::new();
    for organism in &sim.organisms {
        assert!(
            seen.insert((organism.q, organism.r)),
            "organisms should not overlap",
        );
        let idx = sim.cell_index(organism.q, organism.r);
        assert_eq!(sim.occupancy[idx], Some(Occupant::Organism(organism.id)));
    }
    assert_eq!(
        sim.organisms.len() + sim.terrain_map.iter().filter(|blocked| **blocked).count(),
        sim.occupancy.iter().flatten().count()
    );
}
