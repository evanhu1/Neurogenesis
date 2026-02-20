use super::support::{stable_test_config, test_genome};
use super::*;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

#[test]
fn config_validation_rejects_out_of_range_vision_distance_mutation_rate() {
    let mut cfg = stable_test_config();
    cfg.seed_genome_config.mutation_rate_vision_distance = 1.5;
    let err = Simulation::new(cfg, 1).expect_err("config should be rejected");
    assert!(err.to_string().contains("mutation_rate_vision_distance"));
}

#[test]
fn config_validation_rejects_out_of_range_split_edge_mutation_rate() {
    let mut cfg = stable_test_config();
    cfg.seed_genome_config.mutation_rate_add_neuron_split_edge = 1.5;
    let err = Simulation::new(cfg, 1).expect_err("config should be rejected");
    assert!(err
        .to_string()
        .contains("mutation_rate_add_neuron_split_edge"));
}

#[test]
fn config_validation_rejects_non_positive_seed_starting_energy() {
    let mut cfg = stable_test_config();
    cfg.seed_genome_config.starting_energy = 0.0;
    let err = Simulation::new(cfg, 1).expect_err("config should be rejected");
    assert!(err.to_string().contains("starting_energy"));
}

#[test]
fn config_validation_rejects_non_positive_action_temperature() {
    let mut cfg = stable_test_config();
    cfg.action_temperature = 0.0;
    let err = Simulation::new(cfg, 1).expect_err("config should be rejected");
    assert!(err.to_string().contains("action_temperature"));
}

#[test]
fn config_validation_rejects_negative_action_selection_margin() {
    let mut cfg = stable_test_config();
    cfg.action_selection_margin = Some(-0.1);
    let err = Simulation::new(cfg, 1).expect_err("config should be rejected");
    assert!(err.to_string().contains("action_selection_margin"));
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
    assert_eq!(genome.interneuron_types.len(), 4);
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
        crate::genome::mutate_genome(&mut genome, &mut rng);
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
    genome.interneuron_types = vec![InterNeuronType::Excitatory, InterNeuronType::Inhibitory];
    genome.inter_locations = vec![BrainLocation { x: 5.0, y: 5.0 }; 2];
    genome.num_synapses = 1;
    genome.edges = vec![
        SynapseEdge {
            pre_neuron_id: NeuronId(crate::brain::ACTION_ID_BASE),
            post_neuron_id: NeuronId(crate::brain::INTER_ID_BASE),
            weight: 0.5,
            eligibility: 4.0,
        },
        SynapseEdge {
            pre_neuron_id: NeuronId(0),
            post_neuron_id: NeuronId(crate::brain::INTER_ID_BASE),
            weight: -0.5,
            eligibility: 3.0,
        },
        SynapseEdge {
            pre_neuron_id: NeuronId(0),
            post_neuron_id: NeuronId(crate::brain::INTER_ID_BASE),
            weight: 0.8,
            eligibility: 2.0,
        },
        SynapseEdge {
            pre_neuron_id: NeuronId(crate::brain::INTER_ID_BASE + 1),
            post_neuron_id: NeuronId(crate::brain::ACTION_ID_BASE),
            weight: 0.7,
            eligibility: 1.0,
        },
        SynapseEdge {
            pre_neuron_id: NeuronId(crate::brain::INTER_ID_BASE),
            post_neuron_id: NeuronId(crate::brain::INTER_ID_BASE),
            weight: 0.3,
            eligibility: 5.0,
        },
    ];

    let mut rng = ChaCha8Rng::seed_from_u64(42);
    crate::genome::mutate_genome(&mut genome, &mut rng);

    assert_eq!(genome.num_synapses, 2);
    assert_eq!(genome.edges.len(), 2);

    let first = &genome.edges[0];
    let second = &genome.edges[1];
    assert_eq!(first.pre_neuron_id, NeuronId(0));
    assert_eq!(first.post_neuron_id, NeuronId(crate::brain::INTER_ID_BASE));
    assert_eq!(first.eligibility, 0.0);
    assert!(first.weight > 0.0);

    assert_eq!(
        second.pre_neuron_id,
        NeuronId(crate::brain::INTER_ID_BASE + 1)
    );
    assert_eq!(
        second.post_neuron_id,
        NeuronId(crate::brain::ACTION_ID_BASE)
    );
    assert_eq!(second.eligibility, 0.0);
    assert!(second.weight < 0.0);
}

#[test]
fn mutate_genome_can_perturb_inherited_synapse_weights_without_flipping_sign() {
    let mut genome = test_genome();
    genome.num_neurons = 2;
    genome.inter_biases = vec![0.0; 2];
    genome.inter_log_time_constants = vec![0.0; 2];
    genome.interneuron_types = vec![InterNeuronType::Excitatory, InterNeuronType::Inhibitory];
    genome.inter_locations = vec![BrainLocation { x: 5.0, y: 5.0 }; 2];
    genome.mutation_rate_synapse_weight_perturbation = 1.0;
    genome.edges = vec![
        SynapseEdge {
            pre_neuron_id: NeuronId(0),
            post_neuron_id: NeuronId(crate::brain::INTER_ID_BASE),
            weight: 0.3,
            eligibility: 0.0,
        },
        SynapseEdge {
            pre_neuron_id: NeuronId(crate::brain::INTER_ID_BASE + 1),
            post_neuron_id: NeuronId(crate::brain::ACTION_ID_BASE),
            weight: -0.45,
            eligibility: 0.0,
        },
    ];
    genome.num_synapses = genome.edges.len() as u32;

    let initial_weights: Vec<(NeuronId, NeuronId, f32)> = genome
        .edges
        .iter()
        .map(|edge| (edge.pre_neuron_id, edge.post_neuron_id, edge.weight))
        .collect();

    let mut rng = ChaCha8Rng::seed_from_u64(17);
    for _ in 0..16 {
        crate::genome::mutate_genome(&mut genome, &mut rng);
    }

    assert_eq!(genome.edges.len(), 2);
    let mut changed = false;
    for (pre, post, initial_weight) in initial_weights {
        let edge = genome
            .edges
            .iter()
            .find(|edge| edge.pre_neuron_id == pre && edge.post_neuron_id == post)
            .expect("edge should remain present");
        if (edge.weight - initial_weight).abs() > 1.0e-6 {
            changed = true;
        }
        assert_eq!(
            edge.weight.is_sign_negative(),
            initial_weight.is_sign_negative()
        );
        assert!(
            edge.weight.abs() >= crate::genome::SYNAPSE_STRENGTH_MIN
                && edge.weight.abs() <= crate::genome::SYNAPSE_STRENGTH_MAX
        );
    }
    assert!(
        changed,
        "at least one inherited edge weight should be perturbed"
    );
}

#[test]
fn mutate_genome_can_apply_add_neuron_split_edge_operator() {
    let mut genome = test_genome();
    genome.num_neurons = 1;
    genome.inter_biases = vec![0.0];
    genome.inter_log_time_constants = vec![0.0];
    genome.interneuron_types = vec![InterNeuronType::Excitatory];
    genome.inter_locations = vec![BrainLocation { x: 5.0, y: 5.0 }];
    genome.edges = vec![SynapseEdge {
        pre_neuron_id: NeuronId(0),
        post_neuron_id: NeuronId(crate::brain::ACTION_ID_BASE),
        weight: 0.5,
        eligibility: 0.0,
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
    genome.mutation_rate_add_neuron_split_edge = 1.0;

    let mut rng = ChaCha8Rng::seed_from_u64(12345);
    let initial_neurons = genome.num_neurons;
    let initial_synapses = genome.num_synapses;
    for _ in 0..64 {
        crate::genome::mutate_genome(&mut genome, &mut rng);
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
    genome.interneuron_types = vec![InterNeuronType::Excitatory];
    genome.inter_locations = vec![BrainLocation { x: 2.0, y: 3.0 }];
    genome.sensory_locations =
        vec![BrainLocation { x: 1.0, y: 1.0 }; crate::brain::SENSORY_COUNT as usize];
    genome.action_locations = vec![BrainLocation { x: 9.0, y: 9.0 }; ActionType::ALL.len()];
    genome.edges = vec![SynapseEdge {
        pre_neuron_id: NeuronId(0),
        post_neuron_id: NeuronId(crate::brain::ACTION_ID_BASE),
        weight: 0.6,
        eligibility: 0.7,
    }];
    genome.num_synapses = 1;

    let mut rng = ChaCha8Rng::seed_from_u64(1234);
    crate::genome::mutate_add_neuron_split_edge(&mut genome, &mut rng);

    let new_inter_id = NeuronId(crate::brain::INTER_ID_BASE + 1);
    assert_eq!(genome.num_neurons, 2);
    assert_eq!(genome.inter_biases.len(), 2);
    assert_eq!(genome.inter_log_time_constants.len(), 2);
    assert_eq!(genome.interneuron_types.len(), 2);
    assert_eq!(genome.inter_locations.len(), 2);
    assert_eq!(genome.num_synapses, 2);
    assert_eq!(genome.edges.len(), 2);
    assert_eq!(genome.interneuron_types[1], InterNeuronType::Excitatory);
    assert!(genome
        .edges
        .iter()
        .any(|edge| edge.pre_neuron_id == NeuronId(0) && edge.post_neuron_id == new_inter_id));
    assert!(genome.edges.iter().any(|edge| {
        edge.pre_neuron_id == new_inter_id
            && edge.post_neuron_id == NeuronId(crate::brain::ACTION_ID_BASE)
    }));
    assert!(!genome.edges.iter().any(|edge| {
        edge.pre_neuron_id == NeuronId(0)
            && edge.post_neuron_id == NeuronId(crate::brain::ACTION_ID_BASE)
    }));
}

#[test]
fn mutate_add_neuron_split_edge_respects_required_dale_signs() {
    let mut genome = test_genome();
    genome.num_neurons = 2;
    genome.inter_biases = vec![0.0; 2];
    genome.inter_log_time_constants = vec![0.0, 0.3];
    genome.interneuron_types = vec![InterNeuronType::Excitatory, InterNeuronType::Inhibitory];
    genome.inter_locations = vec![
        BrainLocation { x: 4.0, y: 4.0 },
        BrainLocation { x: 6.0, y: 6.0 },
    ];
    genome.edges = vec![SynapseEdge {
        pre_neuron_id: NeuronId(crate::brain::INTER_ID_BASE + 1),
        post_neuron_id: NeuronId(crate::brain::ACTION_ID_BASE + 1),
        weight: -0.45,
        eligibility: 0.2,
    }];
    genome.num_synapses = genome.edges.len() as u32;

    let mut rng = ChaCha8Rng::seed_from_u64(55);
    crate::genome::mutate_add_neuron_split_edge(&mut genome, &mut rng);

    let old_pre = NeuronId(crate::brain::INTER_ID_BASE + 1);
    let new_inter_id = NeuronId(crate::brain::INTER_ID_BASE + 2);
    let old_post = NeuronId(crate::brain::ACTION_ID_BASE + 1);

    assert_eq!(genome.num_neurons, 3);
    assert_eq!(genome.interneuron_types[2], InterNeuronType::Inhibitory);

    let pre_to_new = genome
        .edges
        .iter()
        .find(|edge| edge.pre_neuron_id == old_pre && edge.post_neuron_id == new_inter_id)
        .expect("pre->new split edge should exist");
    let new_to_post = genome
        .edges
        .iter()
        .find(|edge| edge.pre_neuron_id == new_inter_id && edge.post_neuron_id == old_post)
        .expect("new->post split edge should exist");
    assert!(pre_to_new.weight < 0.0);
    assert!(new_to_post.weight < 0.0);
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
fn genome_distance_captures_geometry_and_synapse_traits() {
    let mut a = test_genome();
    let mut b = a.clone();

    assert_eq!(crate::genome::genome_distance(&a, &b), 0.0);

    b.num_synapses = b.num_synapses.saturating_add(3);
    b.sensory_locations[0].x += 2.0;
    let distance = crate::genome::genome_distance(&a, &b);
    assert!(distance > 0.0);

    a.inter_locations[0].y += 4.0;
    let second_distance = crate::genome::genome_distance(&a, &b);
    assert!(second_distance > distance);
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
    cfg.action_selection_margin = None;

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
