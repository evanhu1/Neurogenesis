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
        let idx = sim.cell_index(org.q, org.r);
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
fn scan_ahead_wraps_across_boundary() {
    let cfg = test_config(5, 2);
    let mut sim = Simulation::new(cfg, 7).expect("simulation should initialize");

    sim.organisms[0].q = 2;
    sim.organisms[0].r = 2;
    sim.organisms[0].facing = FacingDirection::East;
    // Wrap sequence from (2,2) East in width 5:
    // d1 -> (3,2), d2 -> (4,2), d3 -> (0,2)
    sim.organisms[1].q = 0;
    sim.organisms[1].r = 2;

    sim.occupancy.fill(None);
    for org in &sim.organisms {
        let idx = sim.cell_index(org.q, org.r);
        sim.occupancy[idx] = Some(Occupant::Organism(org.id));
    }

    // vision_distance=3, wrapped organism at distance 3 -> signal = (3 - 3 + 1)/3 = 1/3
    let result = scan_ahead(
        (2, 2),
        FacingDirection::East,
        sim.organisms[0].id,
        sim.config.world_width as i32,
        &sim.occupancy,
        3,
    );
    let result = result.expect("should detect wrapped organism");
    assert_eq!(result.target, EntityType::Organism);
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
        let idx = sim.cell_index(org.q, org.r);
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
fn reproduce_blocks_move_but_allows_turn() {
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
    // Turn still applies even when reproduction succeeds
    assert_eq!(organism.facing, FacingDirection::NorthEast);
}

#[test]
fn reproduce_failure_allows_move_and_turn() {
    let mut cfg = test_config(7, 1);
    cfg.reproduction_energy_cost = 20.0;
    let mut sim = Simulation::new(cfg, 1076).expect("simulation should initialize");
    let mut organism = make_organism(0, 3, 3, FacingDirection::East, true, true, false, 0.8, 10.0);
    enable_reproduce_action(&mut organism);
    configure_sim(&mut sim, vec![organism]);

    let delta = tick_once(&mut sim);
    assert_eq!(delta.metrics.reproductions_last_turn, 0);
    assert_eq!(delta.moves.len(), 1);
    assert_eq!(delta.moves[0].id, OrganismId(0));
    assert_eq!(delta.moves[0].to, (4, 2));

    let organism = sim
        .organisms
        .iter()
        .find(|organism| organism.id == OrganismId(0))
        .expect("organism should remain alive");
    assert_eq!(organism.facing, FacingDirection::NorthEast);
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
            inter_biases: vec![],
            inter_log_taus: vec![],
            interneuron_types: vec![],
            action_biases: vec![0.0, 0.05, 0.0, 0.0, 0.0],
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
        age_of_maturity: 100,
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
        inter_biases: vec![],
        inter_log_taus: vec![],
        interneuron_types: vec![],
        action_biases: vec![5.0, 0.0, 0.0, 6.0, 0.0],
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
    assert!(evaluation.resolved_actions.wants_move);
}

#[test]
fn oja_update_adjusts_weight_and_eligibility_for_active_synapse() {
    let genome = OrganismGenome {
        num_neurons: 1,
        vision_distance: 1,
        age_of_maturity: 0,
        hebb_eta_baseline: 0.0,
        hebb_eta_gain: 0.2,
        eligibility_decay_lambda: 0.9,
        synapse_prune_threshold: 0.0,
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
    let _ = evaluate_brain(&mut organism, 3, &occupancy, vision_distance, &mut scratch);

    let pre = organism.brain.inter[0].neuron.activation;
    let post = organism.brain.action[action_index(ActionType::MoveForward)]
        .neuron
        .activation;
    let dopamine = organism.brain.action[action_index(ActionType::Dopamine)]
        .neuron
        .activation;
    apply_runtime_plasticity(&mut organism, &mut scratch);

    let eta = 0.2 * dopamine;
    let edge = &organism.brain.inter[0].synapses[0];
    let oja_gradient = post * (pre - post * 1.0);
    let expected_eligibility = oja_gradient;
    let expected_weight = 1.0 + eta * expected_eligibility;
    assert!((edge.eligibility - expected_eligibility).abs() < 1e-5);
    assert!((edge.weight - expected_weight).abs() < 1e-5);
    assert!(edge.weight > 0.0);
}

#[test]
fn synapse_pruning_requires_weight_and_eligibility_below_threshold() {
    let genome = OrganismGenome {
        num_neurons: 0,
        vision_distance: 1,
        age_of_maturity: 100,
        hebb_eta_baseline: 0.0,
        hebb_eta_gain: 0.0,
        eligibility_decay_lambda: 0.9,
        synapse_prune_threshold: 0.1,
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
        inter_biases: vec![],
        inter_log_taus: vec![],
        interneuron_types: vec![],
        action_biases: vec![0.0; ActionType::ALL.len()],
        edges: vec![SynapseEdge {
            pre_neuron_id: NeuronId(0),
            post_neuron_id: NeuronId(2000),
            weight: 0.05,
            eligibility: 2.0,
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

    let mut scratch = BrainScratch::new();
    apply_runtime_plasticity(&mut organism, &mut scratch);
    assert_eq!(organism.brain.sensory[0].synapses.len(), 1);
    assert_eq!(organism.brain.synapse_count, 1);
}

#[test]
fn synapse_pruning_runs_on_schedule_and_removes_edges_when_both_are_low() {
    let genome = OrganismGenome {
        num_neurons: 0,
        vision_distance: 1,
        age_of_maturity: 100,
        hebb_eta_baseline: 0.0,
        hebb_eta_gain: 0.0,
        eligibility_decay_lambda: 0.9,
        synapse_prune_threshold: 0.1,
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
        inter_biases: vec![],
        inter_log_taus: vec![],
        interneuron_types: vec![],
        action_biases: vec![0.0; ActionType::ALL.len()],
        edges: vec![SynapseEdge {
            pre_neuron_id: NeuronId(0),
            post_neuron_id: NeuronId(2000),
            weight: 0.05,
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

    let mut scratch = BrainScratch::new();
    apply_runtime_plasticity(&mut organism, &mut scratch);
    assert!(organism.brain.sensory[0].synapses.is_empty());
    assert_eq!(organism.brain.synapse_count, 0);
}

#[test]
fn synapse_pruning_waits_until_age_of_maturity() {
    let genome = OrganismGenome {
        num_neurons: 0,
        vision_distance: 1,
        age_of_maturity: 150,
        hebb_eta_baseline: 0.0,
        hebb_eta_gain: 0.0,
        eligibility_decay_lambda: 0.9,
        synapse_prune_threshold: 0.1,
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
        inter_biases: vec![],
        inter_log_taus: vec![],
        interneuron_types: vec![],
        action_biases: vec![0.0; ActionType::ALL.len()],
        edges: vec![SynapseEdge {
            pre_neuron_id: NeuronId(0),
            post_neuron_id: NeuronId(2000),
            weight: 0.05,
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

    let mut scratch = BrainScratch::new();
    apply_runtime_plasticity(&mut organism, &mut scratch);
    assert_eq!(organism.brain.sensory[0].synapses.len(), 1);
    assert_eq!(organism.brain.synapse_count, 1);
}

fn run_single_oja_step_with_dopamine_bias(dopamine_bias: f32) -> (f32, f32, f32, f32) {
    let mut action_biases = vec![0.0; ActionType::ALL.len()];
    action_biases[action_index(ActionType::Dopamine)] = dopamine_bias;

    let genome = OrganismGenome {
        num_neurons: 1,
        vision_distance: 1,
        age_of_maturity: 0,
        hebb_eta_baseline: 0.0,
        hebb_eta_gain: 0.2,
        eligibility_decay_lambda: 0.9,
        synapse_prune_threshold: 0.0,
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
        inter_biases: vec![1.0],
        inter_log_taus: vec![crate::genome::INTER_LOG_TAU_MIN],
        interneuron_types: vec![InterNeuronType::Excitatory],
        action_biases,
        edges: vec![SynapseEdge {
            pre_neuron_id: NeuronId(1000),
            post_neuron_id: NeuronId(2000),
            weight: 0.5,
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
    let _ = evaluate_brain(&mut organism, 3, &occupancy, vision_distance, &mut scratch);

    let dopamine_signal = organism.brain.action[action_index(ActionType::Dopamine)]
        .neuron
        .activation;
    let weight_before = organism.brain.inter[0].synapses[0].weight;
    apply_runtime_plasticity(&mut organism, &mut scratch);

    let edge = &organism.brain.inter[0].synapses[0];
    (
        weight_before,
        edge.weight,
        edge.eligibility,
        dopamine_signal,
    )
}

#[test]
fn dopamine_neuron_negative_signal_de_reinforces_oja_weight_update() {
    let (weight_before, positive_weight_after, positive_eligibility, positive_dopamine_signal) =
        run_single_oja_step_with_dopamine_bias(2.0);
    let (_, negative_weight_after, negative_eligibility, negative_dopamine_signal) =
        run_single_oja_step_with_dopamine_bias(-2.0);

    assert!(positive_dopamine_signal > 0.9);
    assert!(negative_dopamine_signal < -0.9);

    assert!(positive_weight_after > weight_before);
    assert!(negative_weight_after < weight_before);

    assert!((positive_eligibility - negative_eligibility).abs() < 1e-6);
}

fn make_runtime_learning_organism(
    id: u64,
    hebb_eta_gain: f32,
    initial_weight: f32,
) -> OrganismState {
    let mut action_biases = vec![0.0; ActionType::ALL.len()];
    action_biases[action_index(ActionType::MoveForward)] = -0.7;
    action_biases[action_index(ActionType::Dopamine)] = 2.0;

    let genome = OrganismGenome {
        num_neurons: 1,
        vision_distance: 1,
        age_of_maturity: 1_000,
        hebb_eta_baseline: 0.0,
        hebb_eta_gain,
        eligibility_decay_lambda: 0.9,
        synapse_prune_threshold: 0.0,
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
        inter_biases: vec![1.0],
        inter_log_taus: vec![crate::genome::INTER_LOG_TAU_MIN],
        interneuron_types: vec![InterNeuronType::Excitatory],
        action_biases,
        edges: vec![SynapseEdge {
            pre_neuron_id: NeuronId(1000),
            post_neuron_id: NeuronId(2000),
            weight: initial_weight,
            eligibility: 0.0,
        }],
    };

    OrganismState {
        id: OrganismId(id),
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
    }
}

#[test]
fn hebbian_runtime_plasticity_changes_behavior_within_lifetime() {
    const WORLD_WIDTH: i32 = 3;
    const INITIAL_WEIGHT: f32 = 0.2;
    const TRAINING_STEPS: usize = 8;

    let mut learner = make_runtime_learning_organism(0, 0.8, INITIAL_WEIGHT);
    let mut control = make_runtime_learning_organism(1, 0.0, INITIAL_WEIGHT);

    let occupancy = vec![None; (WORLD_WIDTH * WORLD_WIDTH) as usize];
    let mut learner_scratch = BrainScratch::new();
    let mut control_scratch = BrainScratch::new();

    let learner_vision = learner.genome.vision_distance;
    let learner_baseline = evaluate_brain(
        &mut learner,
        WORLD_WIDTH,
        &occupancy,
        learner_vision,
        &mut learner_scratch,
    );
    let control_vision = control.genome.vision_distance;
    let control_baseline = evaluate_brain(
        &mut control,
        WORLD_WIDTH,
        &occupancy,
        control_vision,
        &mut control_scratch,
    );
    let baseline_learner_move =
        learner_baseline.action_activations[action_index(ActionType::MoveForward)];
    let baseline_control_move =
        control_baseline.action_activations[action_index(ActionType::MoveForward)];
    assert!(baseline_learner_move < 0.5);
    assert!((baseline_learner_move - baseline_control_move).abs() < 1e-6);

    for _ in 0..TRAINING_STEPS {
        let learner_vision = learner.genome.vision_distance;
        let _ = evaluate_brain(
            &mut learner,
            WORLD_WIDTH,
            &occupancy,
            learner_vision,
            &mut learner_scratch,
        );
        apply_runtime_plasticity(&mut learner, &mut learner_scratch);

        let control_vision = control.genome.vision_distance;
        let _ = evaluate_brain(
            &mut control,
            WORLD_WIDTH,
            &occupancy,
            control_vision,
            &mut control_scratch,
        );
        apply_runtime_plasticity(&mut control, &mut control_scratch);
    }

    let learner_vision = learner.genome.vision_distance;
    let learner_probe = evaluate_brain(
        &mut learner,
        WORLD_WIDTH,
        &occupancy,
        learner_vision,
        &mut learner_scratch,
    );
    let control_vision = control.genome.vision_distance;
    let control_probe = evaluate_brain(
        &mut control,
        WORLD_WIDTH,
        &occupancy,
        control_vision,
        &mut control_scratch,
    );
    let learner_move = learner_probe.action_activations[action_index(ActionType::MoveForward)];
    let control_move = control_probe.action_activations[action_index(ActionType::MoveForward)];

    assert!(learner_probe.resolved_actions.wants_move);
    assert!(!control_probe.resolved_actions.wants_move);
    assert!(learner_move > control_move + 0.1);

    let learner_weight = learner.brain.inter[0].synapses[0].weight;
    let control_weight = control.brain.inter[0].synapses[0].weight;
    assert!(learner_weight > INITIAL_WEIGHT);
    assert!((control_weight - INITIAL_WEIGHT).abs() < 1e-6);
    assert!((learner.genome.edges[0].weight - INITIAL_WEIGHT).abs() < 1e-6);
}

#[test]
fn oja_update_reduces_uncorrelated_synapse_toward_minimum_strength() {
    const WORLD_WIDTH: i32 = 3;
    const INITIAL_WEIGHT: f32 = 1.5;
    const TRAINING_STEPS: usize = 16;

    let mut action_biases = vec![0.0; ActionType::ALL.len()];
    action_biases[action_index(ActionType::MoveForward)] = 1.0;
    action_biases[action_index(ActionType::Dopamine)] = 2.0;

    let genome = OrganismGenome {
        num_neurons: 0,
        vision_distance: 1,
        age_of_maturity: 1_000,
        hebb_eta_baseline: 0.0,
        hebb_eta_gain: 1.0,
        eligibility_decay_lambda: 0.9,
        synapse_prune_threshold: 0.0,
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
        inter_biases: vec![],
        inter_log_taus: vec![],
        interneuron_types: vec![],
        action_biases,
        edges: vec![SynapseEdge {
            pre_neuron_id: NeuronId(0),
            post_neuron_id: NeuronId(2000),
            weight: INITIAL_WEIGHT,
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

    let occupancy = vec![None; (WORLD_WIDTH * WORLD_WIDTH) as usize];
    let mut scratch = BrainScratch::new();

    // With empty occupancy, LookFood stays ~0 while MoveForward remains active via bias.
    let vision_distance = organism.genome.vision_distance;
    let baseline = evaluate_brain(
        &mut organism,
        WORLD_WIDTH,
        &occupancy,
        vision_distance,
        &mut scratch,
    );
    assert!(organism.brain.sensory[0].neuron.activation.abs() < 1e-6);
    assert!(baseline.action_activations[action_index(ActionType::MoveForward)] > 0.5);

    let mut previous = organism.brain.sensory[0].synapses[0].weight;
    for _ in 0..TRAINING_STEPS {
        let vision_distance = organism.genome.vision_distance;
        let _ = evaluate_brain(
            &mut organism,
            WORLD_WIDTH,
            &occupancy,
            vision_distance,
            &mut scratch,
        );
        apply_runtime_plasticity(&mut organism, &mut scratch);

        let current = organism.brain.sensory[0].synapses[0].weight;
        assert!(current <= previous + 1e-6);
        previous = current;
    }

    let final_weight = organism.brain.sensory[0].synapses[0].weight;
    // The implementation clamps magnitude to a positive floor, so "zero" means near this minimum.
    assert!(final_weight <= 0.002);
    assert!(final_weight < INITIAL_WEIGHT * 0.01);
}
