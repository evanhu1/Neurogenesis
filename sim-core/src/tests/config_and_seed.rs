use super::support::{stable_test_config, test_genome};
use super::*;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

#[test]
fn config_validation_rejects_negative_global_mutation_rate_modifier() {
    let mut cfg = stable_test_config();
    cfg.global_mutation_rate_modifier = -0.1;
    let err = Simulation::new(cfg, 1).expect_err("config should be rejected");
    assert!(err.to_string().contains("global_mutation_rate_modifier"));
}

#[test]
fn config_validation_rejects_non_positive_action_temperature() {
    let mut cfg = stable_test_config();
    cfg.action_temperature = 0.0;
    let err = Simulation::new(cfg, 1).expect_err("config should be rejected");
    assert!(err.to_string().contains("action_temperature"));
}

#[test]
fn seed_genome_initializes_spatial_vectors_within_brain_space() {
    let mut cfg = stable_test_config();
    cfg.seed_genome_config.num_neurons = 4;
    cfg.seed_genome_config.num_synapses = 12;

    let mut rng = ChaCha8Rng::seed_from_u64(99);
    let genome = crate::genome::generate_seed_genome(&cfg.seed_genome_config, &mut rng);

    assert_eq!(genome.num_neurons, 4);
    assert_eq!(genome.inter_biases.len(), 4);
    assert_eq!(genome.inter_log_time_constants.len(), 4);
    assert_eq!(genome.inter_locations.len(), 4);
    assert_eq!(
        genome.sensory_locations.len(),
        crate::brain::SENSORY_COUNT as usize
    );
    assert_eq!(genome.action_locations.len(), ActionType::ALL.len());
    assert_eq!(genome.edges.len(), genome.num_synapses as usize);

    for location in genome
        .sensory_locations
        .iter()
        .chain(genome.inter_locations.iter())
        .chain(genome.action_locations.iter())
    {
        assert!(
            (crate::genome::BRAIN_SPACE_MIN..=crate::genome::BRAIN_SPACE_MAX).contains(&location.x)
        );
        assert!(
            (crate::genome::BRAIN_SPACE_MIN..=crate::genome::BRAIN_SPACE_MAX).contains(&location.y)
        );
    }
}

#[test]
fn mutate_genome_matches_synapse_target_and_mutates_location_in_bounds() {
    let mut genome = test_genome();
    genome.num_neurons = 2;
    genome.num_synapses = 8;
    genome.mutation_rate_neuron_location = 1.0;

    let original_locations: Vec<BrainLocation> = genome
        .sensory_locations
        .iter()
        .chain(genome.inter_locations.iter())
        .chain(genome.action_locations.iter())
        .copied()
        .collect();
    let mut rng = ChaCha8Rng::seed_from_u64(7);

    for _ in 0..32 {
        crate::genome::mutate_genome(&mut genome, 1.0, true, &mut rng);
    }

    assert_eq!(genome.num_synapses, genome.edges.len() as u32);
    assert_eq!(genome.num_synapses, 8);
    assert_eq!(genome.edges.len(), genome.num_synapses as usize);

    let current_locations: Vec<BrainLocation> = genome
        .sensory_locations
        .iter()
        .chain(genome.inter_locations.iter())
        .chain(genome.action_locations.iter())
        .copied()
        .collect();
    assert!(
        original_locations
            .iter()
            .zip(current_locations.iter())
            .any(|(before, after)| before != after),
        "expected at least one neuron location to change",
    );

    for location in genome
        .sensory_locations
        .iter()
        .chain(genome.inter_locations.iter())
        .chain(genome.action_locations.iter())
    {
        assert!(
            (crate::genome::BRAIN_SPACE_MIN..=crate::genome::BRAIN_SPACE_MAX).contains(&location.x)
        );
        assert!(
            (crate::genome::BRAIN_SPACE_MIN..=crate::genome::BRAIN_SPACE_MAX).contains(&location.y)
        );
    }
}

#[test]
fn mutate_genome_sanitizes_synapse_genes_and_does_not_trim_excess() {
    let mut genome = test_genome();
    genome.num_neurons = 2;
    genome.inter_biases = vec![0.0; 2];
    genome.inter_log_time_constants = vec![0.0; 2];
    genome.inter_locations = vec![BrainLocation { x: 5.0, y: 5.0 }; 2];
    genome.num_synapses = 1;
    genome.edges = vec![
        SynapseEdge {
            pre_neuron_id: NeuronId(crate::brain::ACTION_ID_BASE),
            post_neuron_id: NeuronId(crate::brain::INTER_ID_BASE),
            weight: 0.5,
            eligibility: 4.0,
            pending_coactivation: 0.0,
        },
        SynapseEdge {
            pre_neuron_id: NeuronId(0),
            post_neuron_id: NeuronId(crate::brain::INTER_ID_BASE),
            weight: -0.5,
            eligibility: 3.0,
            pending_coactivation: 0.0,
        },
        SynapseEdge {
            pre_neuron_id: NeuronId(0),
            post_neuron_id: NeuronId(crate::brain::INTER_ID_BASE),
            weight: 0.8,
            eligibility: 2.0,
            pending_coactivation: 0.0,
        },
        SynapseEdge {
            pre_neuron_id: NeuronId(crate::brain::INTER_ID_BASE + 1),
            post_neuron_id: NeuronId(crate::brain::ACTION_ID_BASE),
            weight: 0.7,
            eligibility: 1.0,
            pending_coactivation: 0.0,
        },
        SynapseEdge {
            pre_neuron_id: NeuronId(crate::brain::INTER_ID_BASE),
            post_neuron_id: NeuronId(crate::brain::INTER_ID_BASE),
            weight: 0.3,
            eligibility: 5.0,
            pending_coactivation: 0.0,
        },
    ];

    let mut rng = ChaCha8Rng::seed_from_u64(42);
    crate::genome::mutate_genome(&mut genome, 1.0, true, &mut rng);

    assert_eq!(genome.num_synapses, 2);
    assert_eq!(genome.edges.len(), 2);

    let first = &genome.edges[0];
    let second = &genome.edges[1];
    assert_eq!(first.pre_neuron_id, NeuronId(0));
    assert_eq!(first.post_neuron_id, NeuronId(crate::brain::INTER_ID_BASE));
    assert_eq!(first.eligibility, 0.0);
    assert!(
        first.weight.abs() >= crate::genome::SYNAPSE_STRENGTH_MIN
            && first.weight.abs() <= crate::genome::SYNAPSE_STRENGTH_MAX
    );

    assert_eq!(
        second.pre_neuron_id,
        NeuronId(crate::brain::INTER_ID_BASE + 1)
    );
    assert_eq!(
        second.post_neuron_id,
        NeuronId(crate::brain::ACTION_ID_BASE)
    );
    assert_eq!(second.eligibility, 0.0);
    assert!(
        second.weight.abs() >= crate::genome::SYNAPSE_STRENGTH_MIN
            && second.weight.abs() <= crate::genome::SYNAPSE_STRENGTH_MAX
    );
}

#[test]
fn mutate_genome_can_apply_add_neuron_split_edge_operator() {
    let mut genome = test_genome();
    genome.num_neurons = 1;
    genome.inter_biases = vec![0.0];
    genome.inter_log_time_constants = vec![0.0];
    genome.inter_locations = vec![BrainLocation { x: 5.0, y: 5.0 }];
    genome.edges = vec![SynapseEdge {
        pre_neuron_id: NeuronId(0),
        post_neuron_id: NeuronId(crate::brain::ACTION_ID_BASE),
        weight: 0.5,
        eligibility: 0.0,
        pending_coactivation: 0.0,
    }];
    genome.num_synapses = 1;
    genome.mutation_rate_age_of_maturity = 0.0;
    genome.mutation_rate_vision_distance = 0.0;
    genome.mutation_rate_inter_bias = 0.0;
    genome.mutation_rate_inter_update_rate = 0.0;
    genome.mutation_rate_action_bias = 0.0;
    genome.mutation_rate_eligibility_retention = 0.0;
    genome.mutation_rate_synapse_prune_threshold = 0.0;
    genome.mutation_rate_neuron_location = 0.0;
    genome.mutation_rate_synapse_weight_perturbation = 0.0;
    genome.mutation_rate_add_synapse = 0.0;
    genome.mutation_rate_remove_synapse = 0.0;
    genome.mutation_rate_add_neuron_split_edge = 1.0;

    let mut rng = ChaCha8Rng::seed_from_u64(12345);
    let initial_neurons = genome.num_neurons;
    let initial_synapses = genome.num_synapses;
    for _ in 0..64 {
        crate::genome::mutate_genome(&mut genome, 1.0, true, &mut rng);
        if genome.num_neurons > initial_neurons {
            break;
        }
    }

    assert!(
        genome.num_neurons > initial_neurons,
        "expected add-neuron split-edge mutation to be applied",
    );
    assert!(genome.num_synapses > initial_synapses);
}

#[test]
fn mutate_add_neuron_split_edge_replaces_synapse_and_extends_inter_vectors() {
    let mut genome = test_genome();
    genome.num_neurons = 1;
    genome.inter_biases = vec![0.0];
    genome.inter_log_time_constants = vec![0.2];
    genome.inter_locations = vec![BrainLocation { x: 2.0, y: 3.0 }];
    genome.sensory_locations =
        vec![BrainLocation { x: 1.0, y: 1.0 }; crate::brain::SENSORY_COUNT as usize];
    genome.action_locations = vec![BrainLocation { x: 9.0, y: 9.0 }; ActionType::ALL.len()];
    genome.edges = vec![SynapseEdge {
        pre_neuron_id: NeuronId(0),
        post_neuron_id: NeuronId(crate::brain::ACTION_ID_BASE),
        weight: 0.6,
        eligibility: 0.7,
        pending_coactivation: 0.0,
    }];
    genome.num_synapses = 1;

    let mut rng = ChaCha8Rng::seed_from_u64(1234);
    crate::genome::mutate_add_neuron_split_edge(&mut genome, &mut rng);

    let new_inter_id = NeuronId(crate::brain::INTER_ID_BASE + 1);
    assert_eq!(genome.num_neurons, 2);
    assert_eq!(genome.inter_biases.len(), 2);
    assert_eq!(genome.inter_log_time_constants.len(), 2);
    assert_eq!(genome.inter_locations.len(), 2);
    assert_eq!(genome.num_synapses, 2);
    assert_eq!(genome.edges.len(), 2);
    let incoming = genome
        .edges
        .iter()
        .find(|edge| edge.pre_neuron_id == NeuronId(0) && edge.post_neuron_id == new_inter_id)
        .expect("split mutation should create pre->new edge");
    let outgoing = genome
        .edges
        .iter()
        .find(|edge| {
            edge.pre_neuron_id == new_inter_id
                && edge.post_neuron_id == NeuronId(crate::brain::ACTION_ID_BASE)
        })
        .expect("split mutation should create new->post edge");
    assert_eq!(incoming.weight, 1.0);
    assert_eq!(outgoing.weight, 0.6);
    assert!(!genome.edges.iter().any(|edge| {
        edge.pre_neuron_id == NeuronId(0)
            && edge.post_neuron_id == NeuronId(crate::brain::ACTION_ID_BASE)
    }));
}

#[test]
fn mutate_add_synapse_adds_new_edge_and_increments_target() {
    let mut genome = test_genome();
    genome.num_neurons = 1;
    genome.inter_biases = vec![0.0];
    genome.inter_log_time_constants = vec![0.0];
    genome.inter_locations = vec![BrainLocation { x: 5.0, y: 5.0 }];
    genome.edges = vec![SynapseEdge {
        pre_neuron_id: NeuronId(0),
        post_neuron_id: NeuronId(crate::brain::ACTION_ID_BASE),
        weight: 0.4,
        eligibility: 0.0,
        pending_coactivation: 0.0,
    }];
    genome.num_synapses = 1;

    let mut rng = ChaCha8Rng::seed_from_u64(2026);
    crate::genome::mutate_add_synapse(&mut genome, &mut rng);

    assert_eq!(genome.num_synapses, 2);
    assert_eq!(genome.edges.len(), 2);
    let unique_pairs = genome
        .edges
        .iter()
        .map(|edge| (edge.pre_neuron_id, edge.post_neuron_id))
        .collect::<HashSet<_>>();
    assert_eq!(unique_pairs.len(), genome.edges.len());
}

#[test]
fn mutate_remove_synapse_removes_edge_and_decrements_target() {
    let mut genome = test_genome();
    genome.num_neurons = 1;
    genome.inter_biases = vec![0.0];
    genome.inter_log_time_constants = vec![0.0];
    genome.inter_locations = vec![BrainLocation { x: 5.0, y: 5.0 }];
    genome.edges = vec![
        SynapseEdge {
            pre_neuron_id: NeuronId(0),
            post_neuron_id: NeuronId(crate::brain::INTER_ID_BASE),
            weight: 0.4,
            eligibility: 0.0,
            pending_coactivation: 0.0,
        },
        SynapseEdge {
            pre_neuron_id: NeuronId(crate::brain::INTER_ID_BASE),
            post_neuron_id: NeuronId(crate::brain::ACTION_ID_BASE),
            weight: -0.7,
            eligibility: 0.0,
            pending_coactivation: 0.0,
        },
    ];
    genome.num_synapses = 2;

    let mut rng = ChaCha8Rng::seed_from_u64(7);
    crate::genome::mutate_remove_synapse(&mut genome, &mut rng);

    assert_eq!(genome.num_synapses, 1);
    assert_eq!(genome.edges.len(), 1);
}

#[test]
fn seed_num_synapses_is_clamped_to_possible_pairs() {
    let mut cfg = stable_test_config();
    cfg.seed_genome_config.num_neurons = 3;
    cfg.seed_genome_config.num_synapses = 9_999;

    let mut rng = ChaCha8Rng::seed_from_u64(123);
    let genome = crate::genome::generate_seed_genome(&cfg.seed_genome_config, &mut rng);

    let all_pairs = (crate::brain::SENSORY_COUNT + genome.num_neurons)
        * (genome.num_neurons + crate::brain::ACTION_COUNT_U32);
    let max_pairs = all_pairs.saturating_sub(genome.num_neurons);
    assert_eq!(genome.num_synapses, max_pairs);
    assert_eq!(genome.edges.len(), max_pairs as usize);
}

#[test]
fn global_mutation_rate_modifier_does_not_change_inherited_mutation_rate_genes() {
    let mut genome_a = test_genome();

    genome_a.mutation_rate_age_of_maturity = 0.17;
    genome_a.mutation_rate_vision_distance = 0.13;
    genome_a.mutation_rate_inter_bias = 0.09;
    genome_a.mutation_rate_inter_update_rate = 0.11;
    genome_a.mutation_rate_action_bias = 0.07;
    genome_a.mutation_rate_eligibility_retention = 0.05;
    genome_a.mutation_rate_synapse_prune_threshold = 0.03;
    genome_a.mutation_rate_neuron_location = 0.19;
    genome_a.mutation_rate_synapse_weight_perturbation = 0.23;
    genome_a.mutation_rate_add_synapse = 0.31;
    genome_a.mutation_rate_remove_synapse = 0.27;
    genome_a.mutation_rate_add_neuron_split_edge = 0.29;
    let mut genome_b = genome_a.clone();

    let mut rng_a = ChaCha8Rng::seed_from_u64(404);
    let mut rng_b = ChaCha8Rng::seed_from_u64(404);

    crate::genome::mutate_genome(&mut genome_a, 1.0, true, &mut rng_a);
    crate::genome::mutate_genome(&mut genome_b, 10.0, true, &mut rng_b);

    assert_eq!(
        genome_a.mutation_rate_age_of_maturity,
        genome_b.mutation_rate_age_of_maturity
    );
    assert_eq!(
        genome_a.mutation_rate_vision_distance,
        genome_b.mutation_rate_vision_distance
    );
    assert_eq!(
        genome_a.mutation_rate_inter_bias,
        genome_b.mutation_rate_inter_bias
    );
    assert_eq!(
        genome_a.mutation_rate_inter_update_rate,
        genome_b.mutation_rate_inter_update_rate
    );
    assert_eq!(
        genome_a.mutation_rate_action_bias,
        genome_b.mutation_rate_action_bias
    );
    assert_eq!(
        genome_a.mutation_rate_eligibility_retention,
        genome_b.mutation_rate_eligibility_retention
    );
    assert_eq!(
        genome_a.mutation_rate_synapse_prune_threshold,
        genome_b.mutation_rate_synapse_prune_threshold
    );
    assert_eq!(
        genome_a.mutation_rate_neuron_location,
        genome_b.mutation_rate_neuron_location
    );
    assert_eq!(
        genome_a.mutation_rate_synapse_weight_perturbation,
        genome_b.mutation_rate_synapse_weight_perturbation
    );
    assert_eq!(
        genome_a.mutation_rate_add_synapse,
        genome_b.mutation_rate_add_synapse
    );
    assert_eq!(
        genome_a.mutation_rate_remove_synapse,
        genome_b.mutation_rate_remove_synapse
    );
    assert_eq!(
        genome_a.mutation_rate_add_neuron_split_edge,
        genome_b.mutation_rate_add_neuron_split_edge
    );
}

#[test]
fn meta_mutation_disabled_keeps_mutation_rate_genes_unchanged() {
    let mut genome = test_genome();
    genome.mutation_rate_age_of_maturity = 0.17;
    genome.mutation_rate_vision_distance = 0.13;
    genome.mutation_rate_inter_bias = 0.09;
    genome.mutation_rate_inter_update_rate = 0.11;
    genome.mutation_rate_action_bias = 0.07;
    genome.mutation_rate_eligibility_retention = 0.05;
    genome.mutation_rate_synapse_prune_threshold = 0.03;
    genome.mutation_rate_neuron_location = 0.19;
    genome.mutation_rate_synapse_weight_perturbation = 0.23;
    genome.mutation_rate_add_synapse = 0.31;
    genome.mutation_rate_remove_synapse = 0.27;
    genome.mutation_rate_add_neuron_split_edge = 0.29;
    let baseline = genome.clone();

    let mut rng = ChaCha8Rng::seed_from_u64(404);
    crate::genome::mutate_genome(&mut genome, 1.0, false, &mut rng);

    assert_eq!(
        genome.mutation_rate_age_of_maturity,
        baseline.mutation_rate_age_of_maturity
    );
    assert_eq!(
        genome.mutation_rate_vision_distance,
        baseline.mutation_rate_vision_distance
    );
    assert_eq!(
        genome.mutation_rate_inter_bias,
        baseline.mutation_rate_inter_bias
    );
    assert_eq!(
        genome.mutation_rate_inter_update_rate,
        baseline.mutation_rate_inter_update_rate
    );
    assert_eq!(
        genome.mutation_rate_action_bias,
        baseline.mutation_rate_action_bias
    );
    assert_eq!(
        genome.mutation_rate_eligibility_retention,
        baseline.mutation_rate_eligibility_retention
    );
    assert_eq!(
        genome.mutation_rate_synapse_prune_threshold,
        baseline.mutation_rate_synapse_prune_threshold
    );
    assert_eq!(
        genome.mutation_rate_neuron_location,
        baseline.mutation_rate_neuron_location
    );
    assert_eq!(
        genome.mutation_rate_synapse_weight_perturbation,
        baseline.mutation_rate_synapse_weight_perturbation
    );
    assert_eq!(
        genome.mutation_rate_add_synapse,
        baseline.mutation_rate_add_synapse
    );
    assert_eq!(
        genome.mutation_rate_remove_synapse,
        baseline.mutation_rate_remove_synapse
    );
    assert_eq!(
        genome.mutation_rate_add_neuron_split_edge,
        baseline.mutation_rate_add_neuron_split_edge
    );
}

#[test]
fn config_validation_rejects_invalid_terrain_threshold() {
    let mut cfg = stable_test_config();
    cfg.terrain_threshold = 1.5;
    let err = Simulation::new(cfg, 7).expect_err("config should be rejected");
    assert!(err.to_string().contains("terrain_threshold"));
}

#[test]
fn terrain_map_is_seed_deterministic_and_threshold_controlled() {
    let mut low_threshold_cfg = stable_test_config();
    low_threshold_cfg.world_width = 48;
    low_threshold_cfg.num_organisms = 1;
    low_threshold_cfg.terrain_noise_scale = 0.02;
    low_threshold_cfg.terrain_threshold = 0.60;
    let mut high_threshold_cfg = low_threshold_cfg.clone();
    high_threshold_cfg.terrain_threshold = 0.95;

    let sim_a = Simulation::new(low_threshold_cfg.clone(), 2026).expect("simulation should init");
    let sim_b = Simulation::new(low_threshold_cfg, 2026).expect("simulation should init");
    let sim_c = Simulation::new(high_threshold_cfg, 2026).expect("simulation should init");
    let sim_other_seed = Simulation::new(stable_test_config_with_terrain(0.60), 2027)
        .expect("simulation should init");

    assert_eq!(sim_a.terrain_map, sim_b.terrain_map);
    assert_ne!(sim_a.terrain_map, sim_other_seed.terrain_map);

    let low_blocked = sim_a.terrain_map.iter().filter(|blocked| **blocked).count();
    let high_blocked = sim_c.terrain_map.iter().filter(|blocked| **blocked).count();
    assert!(
        low_blocked > high_blocked,
        "lower threshold should produce more blocked terrain: low={low_blocked} high={high_blocked}",
    );
}

#[test]
fn stochastic_action_sampling_is_deterministic_for_repeated_runs() {
    let mut cfg = stable_test_config();
    cfg.world_width = 30;
    cfg.num_organisms = 120;
    cfg.action_temperature = 1.2;

    let mut run_a = Simulation::new(cfg.clone(), 2026).expect("simulation should initialize");
    let mut run_b = Simulation::new(cfg, 2026).expect("simulation should initialize");

    for _ in 0..25 {
        let _ = run_a.tick();
        let _ = run_b.tick();
    }

    assert_eq!(run_a.snapshot(), run_b.snapshot());
}

fn stable_test_config_with_terrain(terrain_threshold: f32) -> WorldConfig {
    let mut cfg = stable_test_config();
    cfg.world_width = 48;
    cfg.num_organisms = 1;
    cfg.terrain_noise_scale = 0.02;
    cfg.terrain_threshold = terrain_threshold;
    cfg
}
