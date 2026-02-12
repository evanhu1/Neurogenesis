use super::support::*;
use super::*;
use crate::genome::{INTER_LOG_TAU_MAX, INTER_LOG_TAU_MIN};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::cmp::Ordering;

fn mutation_rates(genome: &OrganismGenome) -> [f32; 8] {
    [
        genome.mutation_rate_vision_distance,
        genome.mutation_rate_weight,
        genome.mutation_rate_add_edge,
        genome.mutation_rate_remove_edge,
        genome.mutation_rate_split_edge,
        genome.mutation_rate_inter_bias,
        genome.mutation_rate_inter_update_rate,
        genome.mutation_rate_action_bias,
    ]
}

#[test]
fn deterministic_seed() {
    let cfg = stable_test_config();
    let mut a = Simulation::new(cfg.clone(), 42).expect("simulation A should initialize");
    let mut b = Simulation::new(cfg, 42).expect("simulation B should initialize");
    a.step_n(30);
    b.step_n(30);
    assert_eq!(
        compare_snapshots(&a.snapshot(), &b.snapshot()),
        Ordering::Equal
    );
}

#[test]
fn population_metrics_track_species_counts() {
    let cfg = test_config(4, 4);
    let mut sim = Simulation::new(cfg, 99).expect("simulation should initialize");

    configure_sim(
        &mut sim,
        vec![
            OrganismState {
                age_turns: 2,
                species_id: SpeciesId(0),
                ..make_organism(0, 0, 0, FacingDirection::East, false, false, false, 0.1, 5)
            },
            OrganismState {
                age_turns: 4,
                species_id: SpeciesId(0),
                ..make_organism(1, 1, 0, FacingDirection::East, false, false, false, 0.1, 5)
            },
            OrganismState {
                age_turns: 6,
                species_id: SpeciesId(1),
                ..make_organism(2, 2, 0, FacingDirection::East, false, false, false, 0.1, 5)
            },
            OrganismState {
                age_turns: 10,
                species_id: SpeciesId(1),
                ..make_organism(3, 3, 0, FacingDirection::East, false, false, false, 0.1, 5)
            },
        ],
    );

    let _ = tick_once(&mut sim);
    let species_counts = &sim.metrics().species_counts;
    assert_eq!(species_counts.get(&SpeciesId(0)).copied(), Some(2));
    assert_eq!(species_counts.get(&SpeciesId(1)).copied(), Some(2));
    assert_eq!(species_counts.values().sum::<u32>(), 4);
}

#[test]
fn different_seed_changes_state() {
    let cfg = stable_test_config();
    let mut a = Simulation::new(cfg.clone(), 42).expect("simulation A should initialize");
    let mut b = Simulation::new(cfg, 43).expect("simulation B should initialize");
    a.step_n(10);
    b.step_n(10);
    assert_ne!(
        compare_snapshots(&a.snapshot(), &b.snapshot()),
        Ordering::Equal
    );
}

#[test]
fn config_validation_rejects_zero_world_width() {
    let mut cfg = stable_test_config();
    cfg.world_width = 0;
    let err = Simulation::new(cfg, 1).expect_err("expected invalid config error");
    assert!(err.to_string().contains("world_width"));
}

#[test]
fn config_validation_rejects_mutation_rate_out_of_range() {
    let mut cfg = stable_test_config();
    cfg.seed_genome_config.mutation_rate_weight = 1.5;
    let err = Simulation::new(cfg, 1).expect_err("expected invalid config error");
    assert!(err.to_string().contains("mutation_rate_weight"));
}

#[test]
fn config_validation_rejects_max_num_neurons_out_of_range() {
    let mut cfg = stable_test_config();
    cfg.max_num_neurons = 0;
    let err = Simulation::new(cfg, 1).expect_err("expected invalid config error");
    assert!(err.to_string().contains("max_num_neurons"));
}

#[test]
fn config_validation_rejects_zero_vision_distance() {
    let mut cfg = stable_test_config();
    cfg.seed_genome_config.vision_distance = 0;
    let err = Simulation::new(cfg, 1).expect_err("expected invalid config error");
    assert!(err.to_string().contains("vision_distance"));
}

#[test]
fn population_is_capped_by_world_capacity_without_overlap() {
    let mut cfg = stable_test_config();
    cfg.world_width = 3;
    cfg.num_organisms = 20;
    cfg.seed_genome_config.num_neurons = 0;
    cfg.seed_genome_config.num_synapses = 0;
    let sim = Simulation::new(cfg, 3).expect("simulation should initialize");
    assert_eq!(sim.organisms.len(), 9);
    assert_eq!(
        sim.occupancy.iter().filter(|cell| cell.is_some()).count(),
        9
    );
}

#[test]
fn initial_food_population_matches_coverage_divisor() {
    let cfg = test_config(20, 4);
    let expected = (20_usize * 20) / cfg.food_coverage_divisor as usize;
    let sim = Simulation::new(cfg, 55).expect("simulation should initialize");
    assert_eq!(sim.foods.len(), expected);
}

#[test]
fn validate_state_accepts_fresh_simulation() {
    let sim = Simulation::new(stable_test_config(), 55).expect("simulation should initialize");
    sim.validate_state()
        .expect("freshly initialized simulation state should validate");
}

#[test]
fn validate_state_rejects_invalid_occupancy_mapping() {
    let mut sim = Simulation::new(stable_test_config(), 88).expect("simulation should initialize");
    sim.occupancy.fill(None);
    let err = sim
        .validate_state()
        .expect_err("occupancy mismatch should fail validation");
    assert!(err.to_string().contains("occupancy vector"));
}

#[test]
fn seed_genome_initializes_log_taus_and_action_biases() {
    let cfg = stable_test_config();
    let mut rng = ChaCha8Rng::seed_from_u64(1234);
    let genome =
        crate::genome::generate_seed_genome(&cfg.seed_genome_config, cfg.max_num_neurons, &mut rng);

    assert_eq!(genome.inter_log_taus.len(), cfg.max_num_neurons as usize);
    assert!(genome
        .inter_log_taus
        .iter()
        .all(|log_tau| *log_tau >= INTER_LOG_TAU_MIN && *log_tau <= INTER_LOG_TAU_MAX));
    assert!(genome
        .action_biases
        .iter()
        .all(|bias| *bias >= -1.0 && *bias <= 1.0));
}

#[test]
fn inter_log_tau_mutation_clamps_to_strict_bounds() {
    let mut cfg = stable_test_config();
    cfg.seed_genome_config.num_neurons = 4;
    cfg.max_num_neurons = 6;
    cfg.seed_genome_config.num_synapses = 0;
    cfg.seed_genome_config.mutation_rate_inter_update_rate = 1.0;

    let mut rng = ChaCha8Rng::seed_from_u64(2026);
    let mut genome =
        crate::genome::generate_seed_genome(&cfg.seed_genome_config, cfg.max_num_neurons, &mut rng);
    genome.mutation_rate_inter_update_rate = 1.0;

    for _ in 0..500 {
        crate::genome::mutate_genome(&mut genome, cfg.max_num_neurons, &mut rng);
        assert!(genome
            .inter_log_taus
            .iter()
            .all(|log_tau| *log_tau >= INTER_LOG_TAU_MIN && *log_tau <= INTER_LOG_TAU_MAX));
    }
}

#[test]
fn split_edge_operator_replaces_edge_and_adds_interneuron() {
    let mut genome = test_genome();
    genome.inter_biases = vec![0.0, 0.0];
    genome.inter_log_taus = vec![0.0, 0.0];
    genome.interneuron_types = vec![InterNeuronType::Excitatory, InterNeuronType::Excitatory];
    genome.edges = vec![SynapseEdge {
        pre_neuron_id: NeuronId(0),
        post_neuron_id: NeuronId(2000),
        weight: 0.5,
    }];
    genome.mutation_rate_split_edge = 1.0;

    let mut rng = ChaCha8Rng::seed_from_u64(11);
    for _ in 0..30 {
        crate::genome::mutate_genome(&mut genome, 2, &mut rng);
        if genome.num_neurons == 2 {
            break;
        }
    }

    assert_eq!(genome.num_neurons, 2);
    assert_eq!(genome.edges.len(), 2);
    assert!(!genome
        .edges
        .iter()
        .any(|edge| edge.pre_neuron_id == NeuronId(0) && edge.post_neuron_id == NeuronId(2000)));
    assert!(genome
        .edges
        .iter()
        .any(|edge| edge.pre_neuron_id == NeuronId(0) && edge.post_neuron_id == NeuronId(1001)));
    assert!(genome
        .edges
        .iter()
        .any(|edge| edge.pre_neuron_id == NeuronId(1001) && edge.post_neuron_id == NeuronId(2000)));
    let split_outgoing = genome
        .edges
        .iter()
        .find(|edge| edge.pre_neuron_id == NeuronId(1001) && edge.post_neuron_id == NeuronId(2000))
        .expect("split edge should create outgoing edge from new interneuron");
    match genome.interneuron_types[1] {
        InterNeuronType::Excitatory => assert!(split_outgoing.weight > 0.0),
        InterNeuronType::Inhibitory => assert!(split_outgoing.weight < 0.0),
    }
}

#[test]
fn mutate_genome_repairs_legacy_wrong_sign_edges() {
    let mut genome = test_genome();
    genome.interneuron_types = vec![InterNeuronType::Inhibitory];
    genome.edges = vec![SynapseEdge {
        pre_neuron_id: NeuronId(1000),
        post_neuron_id: NeuronId(2000),
        weight: 0.4,
    }];
    genome.mutation_rate_vision_distance = 0.0;
    genome.mutation_rate_weight = 0.0;
    genome.mutation_rate_add_edge = 0.0;
    genome.mutation_rate_remove_edge = 0.0;
    genome.mutation_rate_split_edge = 0.0;
    genome.mutation_rate_inter_bias = 0.0;
    genome.mutation_rate_inter_update_rate = 0.0;
    genome.mutation_rate_action_bias = 0.0;

    let mut rng = ChaCha8Rng::seed_from_u64(77);
    crate::genome::mutate_genome(&mut genome, 1, &mut rng);
    assert!(
        genome.edges[0].weight < 0.0,
        "inhibitory outgoing edge should be normalized to a negative weight"
    );
}

#[test]
fn no_neuron_removal_mutation_occurs() {
    let mut cfg = stable_test_config();
    cfg.seed_genome_config.num_neurons = 3;
    cfg.max_num_neurons = 3;
    cfg.seed_genome_config.num_synapses = 4;
    cfg.seed_genome_config.mutation_rate_remove_edge = 1.0;
    cfg.seed_genome_config.mutation_rate_split_edge = 1.0;

    let mut rng = ChaCha8Rng::seed_from_u64(99);
    let mut genome =
        crate::genome::generate_seed_genome(&cfg.seed_genome_config, cfg.max_num_neurons, &mut rng);
    let initial = genome.num_neurons;

    for _ in 0..300 {
        crate::genome::mutate_genome(&mut genome, cfg.max_num_neurons, &mut rng);
        assert_eq!(genome.num_neurons, initial);
    }
}

#[test]
fn sensory_edges_remain_positive_through_mutation() {
    let mut cfg = stable_test_config();
    cfg.seed_genome_config.num_neurons = 5;
    cfg.max_num_neurons = 8;
    cfg.seed_genome_config.num_synapses = 30;
    cfg.seed_genome_config.mutation_rate_weight = 1.0;
    cfg.seed_genome_config.mutation_rate_add_edge = 1.0;
    cfg.seed_genome_config.mutation_rate_split_edge = 1.0;

    let mut rng = ChaCha8Rng::seed_from_u64(2024);
    let mut genome =
        crate::genome::generate_seed_genome(&cfg.seed_genome_config, cfg.max_num_neurons, &mut rng);

    for _ in 0..250 {
        crate::genome::mutate_genome(&mut genome, cfg.max_num_neurons, &mut rng);
        assert!(genome
            .edges
            .iter()
            .filter(|edge| edge.pre_neuron_id.0 < 4)
            .all(|edge| edge.weight > 0.0));
    }
}

#[test]
fn dale_law_signs_hold_for_all_interneuron_outgoing_edges() {
    let mut cfg = stable_test_config();
    cfg.seed_genome_config.num_neurons = 6;
    cfg.max_num_neurons = 10;
    cfg.seed_genome_config.num_synapses = 40;
    cfg.seed_genome_config.mutation_rate_weight = 1.0;
    cfg.seed_genome_config.mutation_rate_add_edge = 1.0;
    cfg.seed_genome_config.mutation_rate_split_edge = 1.0;

    let mut rng = ChaCha8Rng::seed_from_u64(777);
    let mut genome =
        crate::genome::generate_seed_genome(&cfg.seed_genome_config, cfg.max_num_neurons, &mut rng);

    for _ in 0..250 {
        crate::genome::mutate_genome(&mut genome, cfg.max_num_neurons, &mut rng);
        for edge in &genome.edges {
            if edge.pre_neuron_id.0 < 1000 || edge.pre_neuron_id.0 >= 1000 + genome.num_neurons {
                continue;
            }
            let idx = (edge.pre_neuron_id.0 - 1000) as usize;
            let neuron_type = genome
                .interneuron_types
                .get(idx)
                .copied()
                .unwrap_or(InterNeuronType::Excitatory);
            match neuron_type {
                InterNeuronType::Excitatory => assert!(edge.weight > 0.0),
                InterNeuronType::Inhibitory => assert!(edge.weight < 0.0),
            }
        }
    }
}

#[test]
fn mutation_rate_self_adaptation_stays_bounded_and_changes_rates() {
    let mut cfg = stable_test_config();
    cfg.seed_genome_config.num_neurons = 4;
    cfg.max_num_neurons = 8;
    cfg.seed_genome_config.num_synapses = 10;
    cfg.seed_genome_config.mutation_rate_vision_distance = 0.2;
    cfg.seed_genome_config.mutation_rate_weight = 0.3;
    cfg.seed_genome_config.mutation_rate_add_edge = 0.2;
    cfg.seed_genome_config.mutation_rate_remove_edge = 0.2;
    cfg.seed_genome_config.mutation_rate_split_edge = 0.1;
    cfg.seed_genome_config.mutation_rate_inter_bias = 0.25;
    cfg.seed_genome_config.mutation_rate_inter_update_rate = 0.25;
    cfg.seed_genome_config.mutation_rate_action_bias = 0.25;

    let mut rng = ChaCha8Rng::seed_from_u64(8080);
    let mut genome =
        crate::genome::generate_seed_genome(&cfg.seed_genome_config, cfg.max_num_neurons, &mut rng);
    let initial_rates = mutation_rates(&genome);

    for _ in 0..200 {
        crate::genome::mutate_genome(&mut genome, cfg.max_num_neurons, &mut rng);
    }

    let final_rates = mutation_rates(&genome);

    assert!(final_rates
        .iter()
        .all(|rate| rate.is_finite() && *rate > 0.0 && *rate < 1.0));
    assert!(final_rates
        .iter()
        .zip(initial_rates.iter())
        .any(|(a, b)| (a - b).abs() > 1e-5));
}

#[test]
fn mutation_rate_self_adaptation_recovers_from_hard_boundaries() {
    let mut genome = test_genome();
    genome.mutation_rate_vision_distance = 0.0;
    genome.mutation_rate_weight = 1.0;
    genome.mutation_rate_add_edge = 0.0;
    genome.mutation_rate_remove_edge = 1.0;
    genome.mutation_rate_split_edge = 0.0;
    genome.mutation_rate_inter_bias = 1.0;
    genome.mutation_rate_inter_update_rate = 0.0;
    genome.mutation_rate_action_bias = 1.0;

    let mut rng = ChaCha8Rng::seed_from_u64(123_456);
    crate::genome::mutate_genome(&mut genome, 1, &mut rng);

    assert!(mutation_rates(&genome)
        .iter()
        .all(|rate| rate.is_finite() && *rate > 0.0 && *rate < 1.0));
}

#[test]
fn mutation_rate_self_adaptation_long_horizon_stays_inside_open_interval() {
    let mut cfg = stable_test_config();
    cfg.seed_genome_config.num_neurons = 4;
    cfg.max_num_neurons = 8;
    cfg.seed_genome_config.num_synapses = 10;
    cfg.seed_genome_config.mutation_rate_vision_distance = 0.25;
    cfg.seed_genome_config.mutation_rate_weight = 0.25;
    cfg.seed_genome_config.mutation_rate_add_edge = 0.25;
    cfg.seed_genome_config.mutation_rate_remove_edge = 0.25;
    cfg.seed_genome_config.mutation_rate_split_edge = 0.25;
    cfg.seed_genome_config.mutation_rate_inter_bias = 0.25;
    cfg.seed_genome_config.mutation_rate_inter_update_rate = 0.25;
    cfg.seed_genome_config.mutation_rate_action_bias = 0.25;

    let mut rng = ChaCha8Rng::seed_from_u64(777_777);
    let mut genome =
        crate::genome::generate_seed_genome(&cfg.seed_genome_config, cfg.max_num_neurons, &mut rng);

    for _ in 0..10_000 {
        crate::genome::mutate_genome(&mut genome, cfg.max_num_neurons, &mut rng);
        assert!(mutation_rates(&genome)
            .iter()
            .all(|rate| rate.is_finite() && *rate > 0.0 && *rate < 1.0));
    }
}

#[test]
fn mutation_rate_self_adaptation_long_horizon_avoids_boundary_pileup() {
    let mut cfg = stable_test_config();
    cfg.seed_genome_config.num_neurons = 4;
    cfg.max_num_neurons = 8;
    cfg.seed_genome_config.num_synapses = 10;
    cfg.seed_genome_config.mutation_rate_vision_distance = 0.25;
    cfg.seed_genome_config.mutation_rate_weight = 0.25;
    cfg.seed_genome_config.mutation_rate_add_edge = 0.25;
    cfg.seed_genome_config.mutation_rate_remove_edge = 0.25;
    cfg.seed_genome_config.mutation_rate_split_edge = 0.25;
    cfg.seed_genome_config.mutation_rate_inter_bias = 0.25;
    cfg.seed_genome_config.mutation_rate_inter_update_rate = 0.25;
    cfg.seed_genome_config.mutation_rate_action_bias = 0.25;

    const LINEAGE_COUNT: u64 = 64;
    const GENERATIONS: usize = 3_000;
    const NEAR_ZERO: f32 = 1.0e-3;
    const NEAR_ONE: f32 = 0.999;

    let mut exact_zero_count = 0usize;
    let mut exact_one_count = 0usize;
    let mut near_zero_count = 0usize;
    let mut near_one_count = 0usize;
    let mut total_rates = 0usize;
    let mut final_rate_sum = 0.0f64;

    for lineage_seed in 0..LINEAGE_COUNT {
        let mut rng = ChaCha8Rng::seed_from_u64(20_000 + lineage_seed);
        let mut genome = crate::genome::generate_seed_genome(
            &cfg.seed_genome_config,
            cfg.max_num_neurons,
            &mut rng,
        );
        for _ in 0..GENERATIONS {
            crate::genome::mutate_genome(&mut genome, cfg.max_num_neurons, &mut rng);
        }

        for rate in mutation_rates(&genome) {
            if rate == 0.0 {
                exact_zero_count += 1;
            }
            if rate == 1.0 {
                exact_one_count += 1;
            }
            if rate <= NEAR_ZERO {
                near_zero_count += 1;
            }
            if rate >= NEAR_ONE {
                near_one_count += 1;
            }
            final_rate_sum += f64::from(rate);
            total_rates += 1;
        }
    }

    let near_zero_fraction = near_zero_count as f64 / total_rates as f64;
    let near_one_fraction = near_one_count as f64 / total_rates as f64;
    let mean_final_rate = final_rate_sum / total_rates as f64;

    assert_eq!(
        exact_zero_count, 0,
        "exact_zero_count={exact_zero_count}, exact_one_count={exact_one_count}, near_zero_fraction={near_zero_fraction}, near_one_fraction={near_one_fraction}, mean_final_rate={mean_final_rate}"
    );
    assert_eq!(
        exact_one_count, 0,
        "exact_zero_count={exact_zero_count}, exact_one_count={exact_one_count}, near_zero_fraction={near_zero_fraction}, near_one_fraction={near_one_fraction}, mean_final_rate={mean_final_rate}"
    );
    assert!(
        near_zero_fraction < 0.35,
        "exact_zero_count={exact_zero_count}, exact_one_count={exact_one_count}, near_zero_fraction={near_zero_fraction}, near_one_fraction={near_one_fraction}, mean_final_rate={mean_final_rate}"
    );
    assert!(
        near_one_fraction < 0.05,
        "exact_zero_count={exact_zero_count}, exact_one_count={exact_one_count}, near_zero_fraction={near_zero_fraction}, near_one_fraction={near_one_fraction}, mean_final_rate={mean_final_rate}"
    );
    assert!(
        mean_final_rate < 0.25,
        "exact_zero_count={exact_zero_count}, exact_one_count={exact_one_count}, near_zero_fraction={near_zero_fraction}, near_one_fraction={near_one_fraction}, mean_final_rate={mean_final_rate}"
    );
}
