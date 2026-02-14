use super::support::{stable_test_config, test_genome};
use super::*;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

#[test]
fn config_validation_rejects_out_of_range_num_synapse_mutation_rate() {
    let mut cfg = stable_test_config();
    cfg.seed_genome_config.mutation_rate_num_synapses = 1.5;
    let err = Simulation::new(cfg, 1).expect_err("config should be rejected");
    assert!(err.to_string().contains("mutation_rate_num_synapses"));
}

#[test]
fn seed_genome_initializes_spatial_vectors_within_brain_space() {
    let mut cfg = stable_test_config();
    cfg.max_num_neurons = 6;
    cfg.seed_genome_config.num_neurons = 4;
    cfg.seed_genome_config.num_synapses = 12;

    let mut rng = ChaCha8Rng::seed_from_u64(99);
    let genome =
        crate::genome::generate_seed_genome(&cfg.seed_genome_config, cfg.max_num_neurons, &mut rng);

    assert_eq!(genome.num_neurons, 4);
    assert_eq!(genome.inter_biases.len(), 6);
    assert_eq!(genome.inter_log_taus.len(), 6);
    assert_eq!(genome.interneuron_types.len(), 6);
    assert_eq!(genome.inter_locations.len(), 6);
    assert_eq!(
        genome.sensory_locations.len(),
        crate::brain::SENSORY_COUNT as usize
    );
    assert_eq!(genome.action_locations.len(), ActionType::ALL.len());

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
fn mutate_genome_can_mutate_num_synapses_and_location_in_bounds() {
    let mut genome = test_genome();
    genome.num_neurons = 2;
    genome.num_synapses = 0;
    genome.mutation_rate_num_synapses = 1.0;
    genome.mutation_rate_neuron_location = 1.0;

    let original_location = genome.sensory_locations[0];
    let mut rng = ChaCha8Rng::seed_from_u64(7);

    for _ in 0..32 {
        crate::genome::mutate_genome(&mut genome, 8, &mut rng);
    }

    assert!(genome.num_synapses > 0);
    assert_ne!(genome.sensory_locations[0], original_location);

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
fn seed_num_synapses_is_clamped_to_possible_pairs() {
    let mut cfg = stable_test_config();
    cfg.max_num_neurons = 3;
    cfg.seed_genome_config.num_neurons = 3;
    cfg.seed_genome_config.num_synapses = 9_999;

    let mut rng = ChaCha8Rng::seed_from_u64(123);
    let genome =
        crate::genome::generate_seed_genome(&cfg.seed_genome_config, cfg.max_num_neurons, &mut rng);

    let all_pairs = (crate::brain::SENSORY_COUNT + genome.num_neurons)
        * (genome.num_neurons + crate::brain::ACTION_COUNT_U32);
    let max_pairs = all_pairs.saturating_sub(genome.num_neurons);
    assert_eq!(genome.num_synapses, max_pairs);
}
