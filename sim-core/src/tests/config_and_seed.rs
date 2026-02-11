use super::support::*;
use super::*;
use std::cmp::Ordering;

#[test]
fn deterministic_seed() {
    let cfg = WorldConfig::default();
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
    let cfg = WorldConfig::default();
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
    let cfg = WorldConfig {
        world_width: 0,
        ..WorldConfig::default()
    };
    let err = Simulation::new(cfg, 1).expect_err("expected invalid config error");
    assert!(err.to_string().contains("world_width"));
}

#[test]
fn config_validation_rejects_mutation_chance_out_of_range() {
    let mut cfg = WorldConfig::default();
    cfg.seed_species_config.mutation_chance = 1.5;
    let err = Simulation::new(cfg, 1).expect_err("expected invalid config error");
    assert!(err.to_string().contains("mutation_chance"));
}

#[test]
fn config_validation_rejects_zero_vision_distance() {
    let mut cfg = WorldConfig::default();
    cfg.seed_species_config.vision_distance = 0;
    let err = Simulation::new(cfg, 1).expect_err("expected invalid config error");
    assert!(err.to_string().contains("vision_distance"));
}

#[test]
fn population_is_capped_by_world_capacity_without_overlap() {
    let mut cfg = WorldConfig {
        world_width: 3,
        num_organisms: 20,
        ..WorldConfig::default()
    };
    cfg.seed_species_config.num_neurons = 0;
    cfg.seed_species_config.num_synapses = 0;
    let sim = Simulation::new(cfg, 3).expect("simulation should initialize");
    assert_eq!(sim.organisms.len(), 9);
    assert_eq!(
        sim.occupancy.iter().filter(|cell| cell.is_some()).count(),
        9
    );
}

#[test]
fn initial_food_population_targets_ten_percent_of_world_tiles() {
    let cfg = test_config(20, 4);
    let sim = Simulation::new(cfg, 55).expect("simulation should initialize");
    assert_eq!(sim.foods.len(), (20_usize * 20) / 10);
}
