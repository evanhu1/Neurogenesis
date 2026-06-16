use super::*;
use crate::spawn::{ReproductionSpawn, SpawnRequest};
use sim_types::TickDelta;

pub(super) trait IntoEnergy {
    fn into_energy(self) -> f32;
}

impl IntoEnergy for i32 {
    fn into_energy(self) -> f32 {
        self as f32
    }
}

impl IntoEnergy for u32 {
    fn into_energy(self) -> f32 {
        self as f32
    }
}

impl IntoEnergy for f32 {
    fn into_energy(self) -> f32 {
        self
    }
}

impl IntoEnergy for f64 {
    fn into_energy(self) -> f32 {
        self as f32
    }
}

fn finalize_test_organism(mut organism: OrganismState) -> OrganismState {
    let coeff = stable_test_config().body_mass_metabolic_cost_coeff;
    crate::metabolism::refresh_organism_base_metabolic_cost(&mut organism, coeff);
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
        SensoryReceptor::VisionRay {
            ray_offset: 0,
            channel: sim_types::VisionChannel::Green,
        },
        BrainLocation { x: 0.0, y: 0.0 },
    )];
    let inter_id = NeuronId(1000);
    let inter_bias = confidence;
    let inter_synapses: Vec<SynapseEdge> = ActionType::ALL
        .iter()
        .copied()
        .enumerate()
        .map(|(idx, action_type)| SynapseEdge {
            pre_neuron_id: inter_id,
            post_neuron_id: NeuronId(2000 + idx as u32),
            weight: if action_type == preferred_action {
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
            x: 1.0,
            y: 1.0,
            activation: confidence,
        },
        state: inter_state,
        alpha: 1.0,
        synapses: inter_synapses,
        action_synapse_start: 0,
    }];
    let action: Vec<_> = ActionType::ALL
        .iter()
        .copied()
        .enumerate()
        .map(|(idx, action_type)| {
            make_action_neuron(
                2000 + idx as u32,
                action_type,
                BrainLocation {
                    x: 2.0,
                    y: idx as f32,
                },
            )
        })
        .collect();

    BrainState {
        sensory,
        inter,
        action,
        synapse_count,
        sensory_mean_activation: vec![0.0],
        inter_mean_activation: vec![0.0],
        action_mean_activation: vec![0.0; ActionType::ALL.len()],
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
    let initial_energy = if energy <= 0.0 { 10.0 } else { energy };
    finalize_test_organism(OrganismState {
        id: OrganismId(id),
        species_id: sim_types::SpeciesId(id),
        q,
        r,
        generation: 0,
        age_turns: 0,
        facing,
        energy: initial_energy,
        health: initial_energy,
        max_health: initial_energy,
        energy_at_last_sensing: initial_energy,
        damage_taken_last_turn: 0.0,
        is_gestating: false,
        consumptions_count: 0,
        plant_consumptions_count: 0,
        prey_consumptions_count: 0,
        reproductions_count: 0,
        last_action_taken: ActionType::Idle,
        base_metabolic_cost: 0.0,
        #[cfg(feature = "instrumentation")]
        instrumentation: Default::default(),
        brain: forced_brain_with_action(preferred_action, confidence),
        genome: test_genome(),
    })
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
    let initial_energy = if energy <= 0.0 { 10.0 } else { energy };
    finalize_test_organism(OrganismState {
        id: OrganismId(id),
        species_id: sim_types::SpeciesId(id),
        q,
        r,
        generation: 0,
        age_turns: 0,
        facing,
        energy: initial_energy,
        health: initial_energy,
        max_health: initial_energy,
        energy_at_last_sensing: initial_energy,
        damage_taken_last_turn: 0.0,
        is_gestating: false,
        consumptions_count: 0,
        plant_consumptions_count: 0,
        prey_consumptions_count: 0,
        reproductions_count: 0,
        last_action_taken: ActionType::Idle,
        base_metabolic_cost: 0.0,
        #[cfg(feature = "instrumentation")]
        instrumentation: Default::default(),
        brain: forced_brain(wants_move, turn_left, turn_right, confidence),
        genome: test_genome(),
    })
}

pub(super) fn reproduction_request_from_parent(
    sim: &Simulation,
    parent_id: OrganismId,
) -> SpawnRequest {
    let parent = sim
        .organisms
        .iter()
        .find(|organism| organism.id == parent_id)
        .expect("parent should exist for reproduction request");
    let (q, r) = crate::grid::hex_neighbor(
        (parent.q, parent.r),
        crate::grid::opposite_direction(parent.facing),
        sim.config.world_width as i32,
    );
    SpawnRequest::Reproduction(Box::new(ReproductionSpawn {
        parent_genome: parent.genome.clone(),
        parent_generation: parent.generation,
        parent_species_id: parent.species_id,
        parent_facing: parent.facing,
        offspring_starting_energy: sim_types::offspring_transfer_energy(
            parent.genome.lifecycle.gestation_ticks,
        ),
        q,
        r,
    }))
}

pub(super) fn reproduction_request_at(
    sim: &Simulation,
    parent_id: OrganismId,
    q: i32,
    r: i32,
) -> SpawnRequest {
    let parent = sim
        .organisms
        .iter()
        .find(|organism| organism.id == parent_id)
        .expect("parent should exist for reproduction request");
    SpawnRequest::Reproduction(Box::new(ReproductionSpawn {
        parent_genome: parent.genome.clone(),
        parent_generation: parent.generation,
        parent_species_id: parent.species_id,
        parent_facing: parent.facing,
        offspring_starting_energy: sim_types::offspring_transfer_energy(
            parent.genome.lifecycle.gestation_ticks,
        ),
        q,
        r,
    }))
}

pub(super) fn configure_sim(sim: &mut Simulation, mut organisms: Vec<OrganismState>) {
    let coeff = sim.config.body_mass_metabolic_cost_coeff;
    for organism in &mut organisms {
        crate::metabolism::refresh_organism_base_metabolic_cost(organism, coeff);
    }
    organisms.sort_by_key(|organism| organism.id);
    sim.organisms = organisms;
    sim.pending_actions = vec![PendingActionState::default(); sim.organisms.len()];
    sim.foods.clear();
    sim.next_food_id = 0;
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
    sim.initialize_food_ecology();
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
    for food in &sim.foods {
        assert!(seen.insert((food.q, food.r)), "entities should not overlap",);
        let idx = sim.cell_index(food.q, food.r);
        assert_eq!(sim.occupancy[idx], Some(Occupant::Food(food.id)));
    }
    assert_eq!(
        sim.organisms.len()
            + sim.foods.len()
            + sim.terrain_map.iter().filter(|blocked| **blocked).count(),
        sim.occupancy.iter().flatten().count()
    );
}
