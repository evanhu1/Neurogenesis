use super::*;
use crate::spawn::{ReproductionSpawn, SpawnRequest, SpawnRequestKind};
use std::cmp::Ordering;

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
    OrganismGenome {
        num_neurons: 1,
        vision_distance: 2,
        age_of_maturity: 0,
        hebb_eta_baseline: 0.0,
        hebb_eta_gain: 0.0,
        eligibility_decay_lambda: 0.9,
        synapse_prune_threshold: 0.01,
        mutation_rate_age_of_maturity: 0.0,
        mutation_rate_vision_distance: 0.0,
        mutation_rate_add_edge: 0.0,
        mutation_rate_remove_edge: 0.0,
        mutation_rate_split_edge: 0.0,
        mutation_rate_inter_bias: 0.0,
        mutation_rate_inter_update_rate: 0.0,
        mutation_rate_action_bias: 0.0,
        mutation_rate_eligibility_decay_lambda: 0.0,
        mutation_rate_synapse_prune_threshold: 0.0,
        inter_biases: vec![0.0],
        inter_log_taus: vec![0.0],
        interneuron_types: vec![InterNeuronType::Excitatory],
        action_biases: vec![0.0; ActionType::ALL.len()],
        edges: vec![],
    }
}

pub(super) fn stable_test_config() -> WorldConfig {
    WorldConfig {
        world_width: 10,
        steps_per_second: 5,
        num_organisms: 10,
        starting_energy: 100.0,
        food_energy: 50.0,
        reproduction_energy_cost: 100.0,
        move_action_energy_cost: 1.0,
        neuron_metabolism_cost: 0.25,
        plant_growth_speed: 1.0,
        food_regrowth_interval: 10,
        food_fertility_noise_scale: 0.045,
        food_fertility_exponent: 1.8,
        food_fertility_floor: 0.04,
        max_organism_age: 500,
        max_num_neurons: 1,
        speciation_threshold: 50.0,
        seed_genome_config: SeedGenomeConfig {
            num_neurons: 1,
            num_synapses: 0,
            vision_distance: 2,
            age_of_maturity: 0,
            hebb_eta_baseline: 0.0,
            hebb_eta_gain: 0.0,
            eligibility_decay_lambda: 0.9,
            synapse_prune_threshold: 0.01,
            mutation_rate_age_of_maturity: 0.0,
            mutation_rate_vision_distance: 0.0,
            mutation_rate_add_edge: 0.0,
            mutation_rate_remove_edge: 0.0,
            mutation_rate_split_edge: 0.0,
            mutation_rate_inter_bias: 0.0,
            mutation_rate_inter_update_rate: 0.0,
            mutation_rate_action_bias: 0.0,
            mutation_rate_eligibility_decay_lambda: 0.0,
            mutation_rate_synapse_prune_threshold: 0.0,
        },
    }
}

pub(super) fn test_config(world_width: u32, num_organisms: u32) -> WorldConfig {
    let mut config = stable_test_config();
    config.world_width = world_width;
    config.num_organisms = num_organisms;
    config
}

pub(super) fn compare_snapshots(a: &WorldSnapshot, b: &WorldSnapshot) -> Ordering {
    let snapshot_a = serde_json::to_string(a).expect("serialize snapshot A");
    let snapshot_b = serde_json::to_string(b).expect("serialize snapshot B");
    snapshot_a.cmp(&snapshot_b)
}

fn forced_brain(
    wants_move: bool,
    turn_left: bool,
    turn_right: bool,
    confidence: f32,
) -> BrainState {
    let sensory = vec![make_sensory_neuron(
        0,
        SensoryReceptor::Look {
            look_target: EntityType::Food,
        },
    )];
    let inter_id = NeuronId(1000);
    let inter_bias = 1.0;
    let inter_synapses = vec![
        SynapseEdge {
            pre_neuron_id: NeuronId(1000),
            post_neuron_id: NeuronId(2000),
            weight: if wants_move { 8.0 } else { -8.0 },
            eligibility: 0.0,
        },
        SynapseEdge {
            pre_neuron_id: NeuronId(1000),
            post_neuron_id: NeuronId(2001),
            weight: if turn_left && !turn_right {
                -8.0
            } else if turn_right && !turn_left {
                8.0
            } else {
                0.0
            },
            eligibility: 0.0,
        },
        SynapseEdge {
            pre_neuron_id: NeuronId(1000),
            post_neuron_id: NeuronId(2002),
            weight: -8.0,
            eligibility: 0.0,
        },
        SynapseEdge {
            pre_neuron_id: NeuronId(1000),
            post_neuron_id: NeuronId(2003),
            weight: -8.0,
            eligibility: 0.0,
        },
    ];
    let inter = vec![InterNeuronState {
        neuron: NeuronState {
            neuron_id: inter_id,
            neuron_type: NeuronType::Inter,
            bias: inter_bias,
            activation: confidence,
            parent_ids: Vec::new(),
        },
        interneuron_type: InterNeuronType::Excitatory,
        alpha: 1.0,
        synapses: inter_synapses,
    }];
    let mut action = vec![
        make_action_neuron(2000, ActionType::MoveForward, 0.0),
        make_action_neuron(2001, ActionType::Turn, 0.0),
        make_action_neuron(2002, ActionType::Consume, 0.0),
        make_action_neuron(2003, ActionType::Reproduce, 0.0),
        make_action_neuron(2004, ActionType::Dopamine, 0.0),
    ];
    for action_neuron in &mut action {
        if action_neuron.action_type != ActionType::Dopamine {
            action_neuron.neuron.parent_ids = vec![inter_id];
        }
    }

    BrainState {
        sensory,
        inter,
        action,
        synapse_count: 4,
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
    OrganismState {
        id: OrganismId(id),
        species_id: SpeciesId(0),
        q,
        r,
        age_turns: 0,
        facing,
        energy: if energy <= 0.0 { 10.0 } else { energy },
        consumptions_count: 0,
        reproductions_count: 0,
        brain: forced_brain(wants_move, turn_left, turn_right, confidence),
        genome: test_genome(),
    }
}

pub(super) fn make_food(id: u64, q: i32, r: i32, energy: impl IntoEnergy) -> FoodState {
    FoodState {
        id: FoodId(id),
        q,
        r,
        energy: energy.into_energy(),
    }
}

pub(super) fn enable_reproduce_action(organism: &mut OrganismState) {
    let inter_id = organism.brain.inter[0].neuron.neuron_id;
    let mut found = false;
    for synapse in &mut organism.brain.inter[0].synapses {
        if synapse.post_neuron_id == NeuronId(2003) {
            synapse.weight = 8.0;
            found = true;
            break;
        }
    }
    if !found {
        let inter_id = organism.brain.inter[0].neuron.neuron_id;
        organism.brain.inter[0].synapses.push(SynapseEdge {
            pre_neuron_id: inter_id,
            post_neuron_id: NeuronId(2003),
            weight: 8.0,
            eligibility: 0.0,
        });
        organism.brain.inter[0]
            .synapses
            .sort_by(|a, b| a.post_neuron_id.cmp(&b.post_neuron_id));
        organism.brain.synapse_count = organism.brain.synapse_count.saturating_add(1);
    }
    if let Some(action) = organism
        .brain
        .action
        .iter_mut()
        .find(|action| action.action_type == ActionType::Reproduce)
    {
        action.neuron.parent_ids = vec![inter_id];
    }
}

pub(super) fn enable_consume_action(organism: &mut OrganismState) {
    let inter_id = organism.brain.inter[0].neuron.neuron_id;
    let mut found = false;
    for synapse in &mut organism.brain.inter[0].synapses {
        if synapse.post_neuron_id == NeuronId(2002) {
            synapse.weight = 8.0;
            found = true;
            break;
        }
    }
    if !found {
        organism.brain.inter[0].synapses.push(SynapseEdge {
            pre_neuron_id: inter_id,
            post_neuron_id: NeuronId(2002),
            weight: 8.0,
            eligibility: 0.0,
        });
        organism.brain.inter[0]
            .synapses
            .sort_by(|a, b| a.post_neuron_id.cmp(&b.post_neuron_id));
        organism.brain.synapse_count = organism.brain.synapse_count.saturating_add(1);
    }
    if let Some(action) = organism
        .brain
        .action
        .iter_mut()
        .find(|action| action.action_type == ActionType::Consume)
    {
        action.neuron.parent_ids = vec![inter_id];
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
            parent_species_id: parent.species_id,
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
            parent_species_id: parent.species_id,
            parent_facing: parent.facing,
            q,
            r,
        }),
    }
}

pub(super) fn configure_sim(sim: &mut Simulation, mut organisms: Vec<OrganismState>) {
    organisms.sort_by_key(|organism| organism.id);
    sim.organisms = organisms;
    sim.foods.clear();
    sim.next_food_id = 0;
    sim.next_organism_id = sim
        .organisms
        .iter()
        .map(|organism| organism.id.0)
        .max()
        .map_or(0, |max_id| max_id + 1);
    sim.occupancy = vec![None; crate::grid::world_capacity(sim.config.world_width)];
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
    sim.food_regrowth_generation.clear();
    sim.food_regrowth_queue.clear();
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
        sim.organisms.len() + sim.foods.len(),
        sim.occupancy.iter().flatten().count()
    );
}
