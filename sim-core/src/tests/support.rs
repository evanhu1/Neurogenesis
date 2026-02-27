use super::*;
use crate::spawn::{ReproductionSpawn, SpawnRequest, SpawnRequestKind};

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

pub(super) fn test_genome() -> OrganismGenome {
    let default_loc = BrainLocation { x: 5.0, y: 5.0 };
    OrganismGenome {
        num_neurons: 1,
        num_synapses: 0,
        spatial_prior_sigma: 3.5,
        vision_distance: 2,
        starting_energy: 100.0,
        age_of_maturity: 0,
        hebb_eta_gain: 0.0,
        eligibility_retention: 0.9,
        synapse_prune_threshold: 0.01,
        mutation_rate_age_of_maturity: 0.0,
        mutation_rate_vision_distance: 0.0,
        mutation_rate_inter_bias: 0.0,
        mutation_rate_inter_update_rate: 0.0,
        mutation_rate_action_bias: 0.0,
        mutation_rate_eligibility_retention: 0.0,
        mutation_rate_synapse_prune_threshold: 0.0,
        mutation_rate_neuron_location: 0.0,
        mutation_rate_synapse_weight_perturbation: 0.0,
        mutation_rate_add_synapse: 0.0,
        mutation_rate_remove_synapse: 0.0,
        mutation_rate_add_neuron_split_edge: 0.0,
        inter_biases: vec![0.0],
        inter_log_time_constants: vec![0.0],
        action_biases: vec![0.0; ActionType::ALL.len()],
        sensory_locations: vec![default_loc; crate::brain::SENSORY_COUNT as usize],
        inter_locations: vec![default_loc],
        action_locations: vec![default_loc; ActionType::ALL.len()],
        edges: Vec::new(),
    }
}

pub(super) fn stable_test_config() -> WorldConfig {
    WorldConfig {
        world_width: 10,
        steps_per_second: 5,
        num_organisms: 10,
        periodic_injection_interval_turns: 0,
        periodic_injection_count: 0,
        food_energy: 50.0,
        move_action_energy_cost: 1.0,
        action_temperature: 0.5,
        food_regrowth_interval: 10,
        food_regrowth_jitter: 2,
        terrain_noise_scale: 0.02,
        terrain_threshold: 1.0,
        max_organism_age: 500,
        global_mutation_rate_modifier: 1.0,
        meta_mutation_enabled: true,
        runtime_plasticity_enabled: true,
        seed_genome_config: SeedGenomeConfig {
            num_neurons: 1,
            num_synapses: 0,
            spatial_prior_sigma: 3.5,
            vision_distance: 2,
            starting_energy: 100.0,
            age_of_maturity: 0,
            hebb_eta_gain: 0.0,
            eligibility_retention: 0.9,
            synapse_prune_threshold: 0.01,
            mutation_rate_age_of_maturity: 0.0,
            mutation_rate_vision_distance: 0.0,
            mutation_rate_inter_bias: 0.0,
            mutation_rate_inter_update_rate: 0.0,
            mutation_rate_action_bias: 0.0,
            mutation_rate_eligibility_retention: 0.0,
            mutation_rate_synapse_prune_threshold: 0.0,
            mutation_rate_neuron_location: 0.0,
            mutation_rate_synapse_weight_perturbation: 0.0,
            mutation_rate_add_synapse: 0.0,
            mutation_rate_remove_synapse: 0.0,
            mutation_rate_add_neuron_split_edge: 0.0,
        },
    }
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
    let sensory = vec![make_sensory_neuron(
        0,
        SensoryReceptor::LookRay {
            ray_offset: 0,
            look_target: EntityType::Food,
        },
        BrainLocation { x: 0.0, y: 0.0 },
    )];
    let inter_id = NeuronId(1000);
    let inter_bias = confidence;
    let preferred_action = if wants_move {
        ActionType::Forward
    } else if turn_left && !turn_right {
        ActionType::TurnLeft
    } else if turn_right && !turn_left {
        ActionType::TurnRight
    } else {
        ActionType::Idle
    };
    let inter_synapses: Vec<SynapseEdge> = ActionType::ALL
        .into_iter()
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
    let inter = vec![InterNeuronState {
        neuron: NeuronState {
            neuron_id: inter_id,
            neuron_type: NeuronType::Inter,
            bias: inter_bias,
            x: 1.0,
            y: 1.0,
            activation: confidence,
            parent_ids: Vec::new(),
        },
        alpha: 1.0,
        synapses: inter_synapses,
    }];
    let mut action: Vec<_> = ActionType::ALL
        .into_iter()
        .enumerate()
        .map(|(idx, action_type)| {
            make_action_neuron(
                2000 + idx as u32,
                action_type,
                0.0,
                BrainLocation {
                    x: 2.0,
                    y: idx as f32,
                },
            )
        })
        .collect();
    for action_neuron in &mut action {
        action_neuron.neuron.parent_ids = vec![inter_id];
    }

    BrainState {
        sensory,
        inter,
        action,
        synapse_count,
    }
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
    OrganismState {
        id: OrganismId(id),
        q,
        r,
        generation: 0,
        age_turns: 0,
        facing,
        energy: initial_energy,
        energy_prev: initial_energy,
        dopamine: 0.0,
        consumptions_count: 0,
        reproductions_count: 0,
        last_action_taken: ActionType::Idle,
        brain: forced_brain(wants_move, turn_left, turn_right, confidence),
        genome: test_genome(),
    }
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
    SpawnRequest {
        kind: SpawnRequestKind::Reproduction(ReproductionSpawn {
            parent_genome: parent.genome.clone(),
            parent_generation: parent.generation,
            parent_facing: parent.facing,
            q,
            r,
        }),
    }
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
    SpawnRequest {
        kind: SpawnRequestKind::Reproduction(ReproductionSpawn {
            parent_genome: parent.genome.clone(),
            parent_generation: parent.generation,
            parent_facing: parent.facing,
            q,
            r,
        }),
    }
}

pub(super) fn configure_sim(sim: &mut Simulation, mut organisms: Vec<OrganismState>) {
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
    sim.food_fertility.clear();
    sim.food_regrowth_due_turn.clear();
    sim.food_regrowth_schedule.clear();
    sim.metrics = MetricsSnapshot::default();
    sim.refresh_population_metrics();
}

pub(super) fn tick_once(sim: &mut Simulation) -> TickDelta {
    sim.step_n(1).into_iter().next().expect("exactly one delta")
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
