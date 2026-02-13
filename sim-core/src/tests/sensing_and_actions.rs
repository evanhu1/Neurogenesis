use super::support::*;
use super::*;
use crate::brain::{
    action_index, apply_runtime_plasticity, evaluate_brain, express_genome, BrainScratch,
};

#[test]
fn scan_ahead_returns_organism_with_distance_signal() {
    let cfg = test_config(5, 2);
    let mut sim = Simulation::new(cfg, 7).expect("simulation should initialize");

    sim.organisms[0].q = 2;
    sim.organisms[0].r = 2;
    sim.organisms[0].facing = FacingDirection::East;
    sim.organisms[1].q = 4;
    sim.organisms[1].r = 2;

    sim.occupancy.fill(None);
    for org in &sim.organisms {
        let idx = sim.cell_index(org.q, org.r).expect("in-bounds test setup");
        sim.occupancy[idx] = Some(Occupant::Organism(org.id));
    }

    // vision_distance=2, organism at distance 2 → signal = (2 - 2 + 1)/2 = 0.5
    let result = scan_ahead(
        (2, 2),
        FacingDirection::East,
        sim.organisms[0].id,
        sim.config.world_width as i32,
        &sim.occupancy,
        2,
    );
    let result = result.expect("should detect organism");
    assert_eq!(result.target, EntityType::Organism);
    assert!((result.signal - 0.5).abs() < f32::EPSILON);
}

#[test]
fn scan_ahead_returns_oob_at_boundary() {
    let cfg = test_config(5, 2);
    let mut sim = Simulation::new(cfg, 7).expect("simulation should initialize");

    sim.organisms[0].q = 2;
    sim.organisms[0].r = 2;
    sim.organisms[0].facing = FacingDirection::East;
    sim.organisms[1].q = 0;
    sim.organisms[1].r = 0;

    sim.occupancy.fill(None);
    for org in &sim.organisms {
        let idx = sim.cell_index(org.q, org.r).expect("in-bounds test setup");
        sim.occupancy[idx] = Some(Occupant::Organism(org.id));
    }

    // vision_distance=3, OOB at distance 3 → signal = (3 - 3 + 1)/3 = 1/3
    let result = scan_ahead(
        (2, 2),
        FacingDirection::East,
        sim.organisms[0].id,
        sim.config.world_width as i32,
        &sim.occupancy,
        3,
    );
    let result = result.expect("should detect out of bounds");
    assert_eq!(result.target, EntityType::OutOfBounds);
    assert!((result.signal - 1.0 / 3.0).abs() < f32::EPSILON);
}

#[test]
fn scan_ahead_returns_none_for_empty_path() {
    let cfg = test_config(5, 2);
    let mut sim = Simulation::new(cfg, 7).expect("simulation should initialize");

    sim.organisms[0].q = 2;
    sim.organisms[0].r = 2;
    sim.organisms[0].facing = FacingDirection::NorthWest;
    sim.organisms[1].q = 4;
    sim.organisms[1].r = 4;

    sim.occupancy.fill(None);
    for org in &sim.organisms {
        let idx = sim.cell_index(org.q, org.r).expect("in-bounds test setup");
        sim.occupancy[idx] = Some(Occupant::Organism(org.id));
    }

    // Looking NorthWest from (2,2): distance 1 → (2,1), distance 2 → (2,0) — both empty
    let result = scan_ahead(
        (2, 2),
        FacingDirection::NorthWest,
        sim.organisms[0].id,
        sim.config.world_width as i32,
        &sim.occupancy,
        2,
    );
    assert!(result.is_none());
}

#[test]
fn scan_ahead_detects_food_with_distance_signal() {
    let cfg = test_config(5, 1);
    let mut sim = Simulation::new(cfg, 76).expect("simulation should initialize");
    configure_sim(
        &mut sim,
        vec![make_organism(
            0,
            2,
            2,
            FacingDirection::East,
            false,
            false,
            false,
            0.1,
            10.0,
        )],
    );
    let added = sim.add_food(make_food(0, 4, 2, sim.config.food_energy));
    assert!(added);

    // vision_distance=2, food at distance 2 → signal = (2 - 2 + 1)/2 = 0.5
    let result = scan_ahead(
        (2, 2),
        FacingDirection::East,
        OrganismId(0),
        sim.config.world_width as i32,
        &sim.occupancy,
        2,
    );
    let result = result.expect("should detect food");
    assert_eq!(result.target, EntityType::Food);
    assert!((result.signal - 0.5).abs() < f32::EPSILON);
}

#[test]
fn scan_ahead_adjacent_entity_returns_max_signal() {
    let cfg = test_config(5, 1);
    let mut sim = Simulation::new(cfg, 76).expect("simulation should initialize");
    configure_sim(
        &mut sim,
        vec![make_organism(
            0,
            2,
            2,
            FacingDirection::East,
            false,
            false,
            false,
            0.1,
            10.0,
        )],
    );
    let added = sim.add_food(make_food(0, 3, 2, sim.config.food_energy));
    assert!(added);

    // vision_distance=3, food at distance 1 → signal = (3 - 1 + 1)/3 = 1.0
    let result = scan_ahead(
        (2, 2),
        FacingDirection::East,
        OrganismId(0),
        sim.config.world_width as i32,
        &sim.occupancy,
        3,
    );
    let result = result.expect("should detect food");
    assert_eq!(result.target, EntityType::Food);
    assert!((result.signal - 1.0).abs() < f32::EPSILON);
}

#[test]
fn scan_ahead_occlusion_closer_entity_blocks_farther() {
    let cfg = test_config(7, 1);
    let mut sim = Simulation::new(cfg, 76).expect("simulation should initialize");
    configure_sim(
        &mut sim,
        vec![make_organism(
            0,
            1,
            3,
            FacingDirection::East,
            false,
            false,
            false,
            0.1,
            10.0,
        )],
    );
    // Food at distance 2, organism at distance 4 — food should occlude organism
    sim.add_food(make_food(0, 3, 3, sim.config.food_energy));
    sim.add_organism(make_organism(
        1,
        5,
        3,
        FacingDirection::East,
        false,
        false,
        false,
        0.1,
        10.0,
    ));

    let result = scan_ahead(
        (1, 3),
        FacingDirection::East,
        OrganismId(0),
        sim.config.world_width as i32,
        &sim.occupancy,
        5,
    );
    let result = result.expect("should detect food (closer)");
    assert_eq!(result.target, EntityType::Food);
    // distance 2, max 5 → signal = (5 - 2 + 1)/5 = 0.8
    assert!((result.signal - 0.8).abs() < f32::EPSILON);
}

#[test]
fn turn_actions_rotate_facing() {
    assert_eq!(
        facing_after_turn(FacingDirection::East, TurnChoice::Left),
        FacingDirection::NorthEast
    );
    assert_eq!(
        facing_after_turn(FacingDirection::East, TurnChoice::Right),
        FacingDirection::SouthEast
    );
    assert_eq!(
        facing_after_turn(FacingDirection::East, TurnChoice::None),
        FacingDirection::East
    );
}

#[test]
fn move_and_turn_can_be_composed_in_same_turn() {
    let cfg = test_config(6, 1);
    let mut sim = Simulation::new(cfg, 75).expect("simulation should initialize");
    configure_sim(
        &mut sim,
        vec![make_organism(
            0,
            2,
            2,
            FacingDirection::East,
            true,
            true,
            false,
            0.8,
            10.0,
        )],
    );

    let delta = tick_once(&mut sim);
    assert_eq!(delta.moves.len(), 1);
    assert_eq!(delta.moves[0].id, OrganismId(0));
    assert_eq!(delta.moves[0].to, (3, 1));
    assert_eq!(delta.metrics.reproductions_last_turn, 0);

    let organism = sim
        .organisms
        .iter()
        .find(|organism| organism.id == OrganismId(0))
        .expect("organism should remain alive");
    assert_eq!(organism.facing, FacingDirection::NorthEast);
}

#[test]
fn reproduce_blocks_move_and_turn() {
    let mut cfg = test_config(7, 1);
    cfg.reproduction_energy_cost = 5.0;
    let mut sim = Simulation::new(cfg, 76).expect("simulation should initialize");
    let mut organism = make_organism(0, 3, 3, FacingDirection::East, true, true, false, 0.8, 10.0);
    enable_reproduce_action(&mut organism);
    configure_sim(&mut sim, vec![organism]);

    let delta = tick_once(&mut sim);
    assert!(delta.moves.is_empty());
    assert_eq!(delta.metrics.reproductions_last_turn, 1);
    assert_eq!(delta.spawned.len(), 1);
    assert_eq!((delta.spawned[0].q, delta.spawned[0].r), (2, 3));

    let organism = sim
        .organisms
        .iter()
        .find(|organism| organism.id == OrganismId(0))
        .expect("organism should remain alive");
    assert_eq!((organism.q, organism.r), (3, 3));
    assert_eq!(organism.facing, FacingDirection::East);
}

#[test]
fn turn_action_has_deadzone_around_zero() {
    let mut organism = OrganismState {
        id: OrganismId(0),
        species_id: SpeciesId(0),
        q: 1,
        r: 1,
        age_turns: 0,
        facing: FacingDirection::East,
        energy: 10.0,
        consumptions_count: 0,
        reproductions_count: 0,
        genome: OrganismGenome {
            num_neurons: 0,
            vision_distance: 1,
            hebb_eta_baseline: 0.0,
            hebb_eta_gain: 0.0,
            eligibility_decay_lambda: 0.9,
            synapse_prune_threshold: 0.01,
            mutation_rate_vision_distance: 0.0,
            mutation_rate_weight: 0.0,
            mutation_rate_add_edge: 0.0,
            mutation_rate_remove_edge: 0.0,
            mutation_rate_split_edge: 0.0,
            mutation_rate_inter_bias: 0.0,
            mutation_rate_inter_update_rate: 0.0,
            mutation_rate_action_bias: 0.0,
            mutation_rate_eligibility_decay_lambda: 0.0,
            mutation_rate_synapse_prune_threshold: 0.0,
            inter_biases: vec![],
            inter_log_taus: vec![],
            interneuron_types: vec![],
            action_biases: vec![0.0, 0.05, 0.0, 0.0],
            edges: vec![],
        },
        brain: BrainState {
            sensory: vec![],
            inter: vec![],
            action: vec![],
            synapse_count: 0,
        },
    };
    organism.brain = express_genome(&organism.genome);
    let occupancy = vec![None; 9];
    let mut scratch = BrainScratch::new();
    let vision_distance = organism.genome.vision_distance;
    let evaluation = evaluate_brain(&mut organism, 3, &occupancy, vision_distance, &mut scratch);
    assert_eq!(evaluation.resolved_actions.turn, TurnChoice::None);

    organism.genome.action_biases[1] = 0.4;
    organism.brain = express_genome(&organism.genome);
    let vision_distance = organism.genome.vision_distance;
    let evaluation = evaluate_brain(&mut organism, 3, &occupancy, vision_distance, &mut scratch);
    assert_eq!(evaluation.resolved_actions.turn, TurnChoice::Right);

    organism.genome.action_biases[1] = -0.4;
    organism.brain = express_genome(&organism.genome);
    let vision_distance = organism.genome.vision_distance;
    let evaluation = evaluate_brain(&mut organism, 3, &occupancy, vision_distance, &mut scratch);
    assert_eq!(evaluation.resolved_actions.turn, TurnChoice::Left);
}

#[test]
fn self_recurrent_interneuron_uses_previous_activation_with_leaky_update() {
    let genome = OrganismGenome {
        num_neurons: 1,
        vision_distance: 1,
        hebb_eta_baseline: 0.0,
        hebb_eta_gain: 0.0,
        eligibility_decay_lambda: 0.9,
        synapse_prune_threshold: 0.01,
        mutation_rate_vision_distance: 0.0,
        mutation_rate_weight: 0.0,
        mutation_rate_add_edge: 0.0,
        mutation_rate_remove_edge: 0.0,
        mutation_rate_split_edge: 0.0,
        mutation_rate_inter_bias: 0.0,
        mutation_rate_inter_update_rate: 0.0,
        mutation_rate_action_bias: 0.0,
        mutation_rate_eligibility_decay_lambda: 0.0,
        mutation_rate_synapse_prune_threshold: 0.0,
        inter_biases: vec![0.0],
        inter_log_taus: vec![(1.0 / std::f32::consts::LN_2).ln()],
        interneuron_types: vec![InterNeuronType::Excitatory],
        action_biases: vec![0.0; ActionType::ALL.len()],
        edges: vec![SynapseEdge {
            pre_neuron_id: NeuronId(1000),
            post_neuron_id: NeuronId(1000),
            weight: 2.0,
            eligibility: 0.0,
        }],
    };
    let mut organism = OrganismState {
        id: OrganismId(0),
        species_id: SpeciesId(0),
        q: 1,
        r: 1,
        age_turns: 0,
        facing: FacingDirection::East,
        energy: 10.0,
        consumptions_count: 0,
        reproductions_count: 0,
        brain: express_genome(&genome),
        genome,
    };
    organism.brain.inter[0].neuron.activation = 1.0;

    let mut occupancy = vec![None; 9];
    occupancy[4] = Some(Occupant::Organism(organism.id));
    let mut scratch = BrainScratch::new();
    let vision_distance = organism.genome.vision_distance;
    let _ = evaluate_brain(&mut organism, 3, &occupancy, vision_distance, &mut scratch);

    let expected = 0.5 * 1.0 + 0.5 * 2.0_f32.tanh();
    assert!((organism.brain.inter[0].neuron.activation - expected).abs() < 1e-5);
}

#[test]
fn action_biases_drive_actions_without_incoming_edges() {
    let genome = OrganismGenome {
        num_neurons: 0,
        vision_distance: 1,
        hebb_eta_baseline: 0.0,
        hebb_eta_gain: 0.0,
        eligibility_decay_lambda: 0.9,
        synapse_prune_threshold: 0.01,
        mutation_rate_vision_distance: 0.0,
        mutation_rate_weight: 0.0,
        mutation_rate_add_edge: 0.0,
        mutation_rate_remove_edge: 0.0,
        mutation_rate_split_edge: 0.0,
        mutation_rate_inter_bias: 0.0,
        mutation_rate_inter_update_rate: 0.0,
        mutation_rate_action_bias: 0.0,
        mutation_rate_eligibility_decay_lambda: 0.0,
        mutation_rate_synapse_prune_threshold: 0.0,
        inter_biases: vec![],
        inter_log_taus: vec![],
        interneuron_types: vec![],
        action_biases: vec![5.0, 0.0, 0.0, 6.0],
        edges: vec![],
    };
    let mut organism = OrganismState {
        id: OrganismId(0),
        species_id: SpeciesId(0),
        q: 1,
        r: 1,
        age_turns: 0,
        facing: FacingDirection::East,
        energy: 10.0,
        consumptions_count: 0,
        reproductions_count: 0,
        brain: express_genome(&genome),
        genome,
    };
    let mut occupancy = vec![None; 9];
    occupancy[4] = Some(Occupant::Organism(organism.id));
    let mut scratch = BrainScratch::new();
    let vision_distance = organism.genome.vision_distance;
    let evaluation = evaluate_brain(&mut organism, 3, &occupancy, vision_distance, &mut scratch);

    assert!(evaluation.resolved_actions.wants_reproduce);
    assert!(!evaluation.resolved_actions.wants_move);
}

#[test]
fn oja_update_adjusts_weight_and_eligibility_for_active_synapse() {
    let genome = OrganismGenome {
        num_neurons: 1,
        vision_distance: 1,
        hebb_eta_baseline: 0.1,
        hebb_eta_gain: 0.0,
        eligibility_decay_lambda: 0.9,
        synapse_prune_threshold: 0.0,
        mutation_rate_vision_distance: 0.0,
        mutation_rate_weight: 0.0,
        mutation_rate_add_edge: 0.0,
        mutation_rate_remove_edge: 0.0,
        mutation_rate_split_edge: 0.0,
        mutation_rate_inter_bias: 0.0,
        mutation_rate_inter_update_rate: 0.0,
        mutation_rate_action_bias: 0.0,
        mutation_rate_eligibility_decay_lambda: 0.0,
        mutation_rate_synapse_prune_threshold: 0.0,
        inter_biases: vec![1.0],
        inter_log_taus: vec![crate::genome::INTER_LOG_TAU_MIN],
        interneuron_types: vec![InterNeuronType::Excitatory],
        action_biases: vec![0.0; ActionType::ALL.len()],
        edges: vec![SynapseEdge {
            pre_neuron_id: NeuronId(1000),
            post_neuron_id: NeuronId(2000),
            weight: 1.0,
            eligibility: 0.0,
        }],
    };
    let mut organism = OrganismState {
        id: OrganismId(0),
        species_id: SpeciesId(0),
        q: 1,
        r: 1,
        age_turns: 0,
        facing: FacingDirection::East,
        energy: 10.0,
        consumptions_count: 0,
        reproductions_count: 0,
        brain: express_genome(&genome),
        genome,
    };

    let mut occupancy = vec![None; 9];
    occupancy[4] = Some(Occupant::Organism(organism.id));
    let mut scratch = BrainScratch::new();
    let vision_distance = organism.genome.vision_distance;
    let _ = evaluate_brain(
        &mut organism,
        3,
        &occupancy,
        vision_distance,
        &mut scratch,
    );

    let pre = organism.brain.inter[0].neuron.activation;
    let post = organism.brain.action[action_index(ActionType::MoveForward)]
        .neuron
        .activation;
    apply_runtime_plasticity(&mut organism, 500);

    let edge = &organism.brain.inter[0].synapses[0];
    let expected_eligibility = pre * post;
    let expected_weight = 1.0 + 0.1 * post * (pre - post * 1.0);
    assert!((edge.eligibility - expected_eligibility).abs() < 1e-5);
    assert!((edge.weight - expected_weight).abs() < 1e-5);
    assert!(edge.weight > 0.0);
}

#[test]
fn synapse_pruning_runs_on_schedule_and_removes_low_eligibility_edges() {
    let genome = OrganismGenome {
        num_neurons: 0,
        vision_distance: 1,
        hebb_eta_baseline: 0.0,
        hebb_eta_gain: 0.0,
        eligibility_decay_lambda: 0.9,
        synapse_prune_threshold: 0.1,
        mutation_rate_vision_distance: 0.0,
        mutation_rate_weight: 0.0,
        mutation_rate_add_edge: 0.0,
        mutation_rate_remove_edge: 0.0,
        mutation_rate_split_edge: 0.0,
        mutation_rate_inter_bias: 0.0,
        mutation_rate_inter_update_rate: 0.0,
        mutation_rate_action_bias: 0.0,
        mutation_rate_eligibility_decay_lambda: 0.0,
        mutation_rate_synapse_prune_threshold: 0.0,
        inter_biases: vec![],
        inter_log_taus: vec![],
        interneuron_types: vec![],
        action_biases: vec![0.0; ActionType::ALL.len()],
        edges: vec![SynapseEdge {
            pre_neuron_id: NeuronId(0),
            post_neuron_id: NeuronId(2000),
            weight: 1.0,
            eligibility: 0.0,
        }],
    };
    let mut organism = OrganismState {
        id: OrganismId(0),
        species_id: SpeciesId(0),
        q: 1,
        r: 1,
        age_turns: 100,
        facing: FacingDirection::East,
        energy: 10.0,
        consumptions_count: 0,
        reproductions_count: 0,
        brain: express_genome(&genome),
        genome,
    };
    let mut occupancy = vec![None; 9];
    occupancy[4] = Some(Occupant::Organism(organism.id));
    let mut scratch = BrainScratch::new();
    let vision_distance = organism.genome.vision_distance;
    let _ = evaluate_brain(
        &mut organism,
        3,
        &occupancy,
        vision_distance,
        &mut scratch,
    );

    apply_runtime_plasticity(&mut organism, 500);
    assert!(organism.brain.sensory[0].synapses.is_empty());
    assert_eq!(organism.brain.synapse_count, 0);
}
