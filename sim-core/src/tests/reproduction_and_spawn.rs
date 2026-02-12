use super::support::*;
use super::*;

#[test]
fn spawn_queue_order_is_deterministic_under_limited_space() {
    let cfg = test_config(2, 3);
    let mut sim = Simulation::new(cfg, 19).expect("simulation should initialize");
    configure_sim(
        &mut sim,
        vec![
            make_organism(
                0,
                0,
                0,
                FacingDirection::NorthEast,
                true,
                false,
                false,
                0.9,
                0,
            ),
            make_organism(1, 1, 0, FacingDirection::West, false, false, false, 0.1, 0),
            make_organism(2, 0, 1, FacingDirection::East, false, false, false, 0.1, 0),
        ],
    );

    let spawned = sim.resolve_spawn_requests(&[
        reproduction_request_at(&sim, OrganismId(0), 1, 1),
        reproduction_request_at(&sim, OrganismId(1), 1, 1),
    ]);

    assert_eq!(spawned.len(), 1);
    let child = sim
        .organisms
        .iter()
        .find(|organism| organism.id == OrganismId(3))
        .expect("first spawn request should consume final empty slot");
    assert_eq!((child.q, child.r), (1, 1));
}

#[test]
fn reproduction_offspring_brain_runtime_state_is_reset() {
    let mut cfg = test_config(8, 1);
    cfg.seed_genome_config.mutation_rate_weight = 0.0;
    let mut sim = Simulation::new(cfg, 31).expect("simulation should initialize");
    configure_sim(
        &mut sim,
        vec![make_organism(
            0,
            3,
            3,
            FacingDirection::East,
            true,
            true,
            false,
            0.8,
            0,
        )],
    );

    let spawned =
        sim.resolve_spawn_requests(&[reproduction_request_from_parent(&sim, OrganismId(0))]);

    assert_eq!(spawned.len(), 1);
    let child = &spawned[0];
    assert!(child
        .brain
        .sensory
        .iter()
        .all(|n| n.neuron.activation == 0.0));
    assert!(child.brain.inter.iter().all(|n| n.neuron.activation == 0.0));
    assert!(child
        .brain
        .action
        .iter()
        .all(|n| n.neuron.activation == 0.0));
    assert_eq!(child.facing, FacingDirection::West);
}

#[test]
fn reproduction_spawn_is_opposite_of_parent_facing() {
    let mut cfg = test_config(40, 1);
    cfg.center_spawn_min_fraction = 0.45;
    cfg.center_spawn_max_fraction = 0.55;
    cfg.seed_genome_config.mutation_rate_weight = 0.0;
    let mut sim = Simulation::new(cfg, 30).expect("simulation should initialize");
    configure_sim(
        &mut sim,
        vec![make_organism(
            0,
            1,
            1,
            FacingDirection::East,
            false,
            false,
            false,
            0.2,
            0,
        )],
    );

    let spawned =
        sim.resolve_spawn_requests(&[reproduction_request_from_parent(&sim, OrganismId(0))]);
    assert_eq!(spawned.len(), 1);
    let child = &spawned[0];
    assert_eq!((child.q, child.r), (0, 1));
}

#[test]
fn reproduce_action_requires_enough_energy() {
    let cfg = test_config(7, 1);
    let mut sim = Simulation::new(cfg, 71).expect("simulation should initialize");
    let mut parent = make_organism(
        0,
        3,
        3,
        FacingDirection::East,
        false,
        false,
        false,
        0.7,
        5.0,
    );
    enable_reproduce_action(&mut parent);
    configure_sim(&mut sim, vec![parent]);

    let delta = tick_once(&mut sim);
    assert_eq!(delta.metrics.reproductions_last_turn, 0);
    assert!(delta.spawned.is_empty());
    let parent_after = sim
        .organisms
        .iter()
        .find(|organism| organism.id == OrganismId(0))
        .expect("parent should remain alive");
    assert_eq!(parent_after.reproductions_count, 0);
    assert_eq!(parent_after.energy, 4.0);
}

#[test]
fn reproduce_action_fails_when_spawn_cell_blocked() {
    let cfg = test_config(7, 2);
    let mut sim = Simulation::new(cfg, 72).expect("simulation should initialize");
    let mut parent = make_organism(
        0,
        3,
        3,
        FacingDirection::East,
        false,
        false,
        false,
        0.7,
        30.0,
    );
    enable_reproduce_action(&mut parent);
    configure_sim(
        &mut sim,
        vec![
            parent,
            make_organism(
                1,
                2,
                3,
                FacingDirection::West,
                false,
                false,
                false,
                0.2,
                10.0,
            ),
        ],
    );

    let delta = tick_once(&mut sim);
    assert_eq!(delta.metrics.reproductions_last_turn, 0);
    assert!(delta.spawned.is_empty());
    let parent_after = sim
        .organisms
        .iter()
        .find(|organism| organism.id == OrganismId(0))
        .expect("parent should remain alive");
    assert_eq!(parent_after.reproductions_count, 0);
    assert_eq!(parent_after.energy, 29.0);
}

#[test]
fn reproduce_action_succeeds_when_energy_and_space_allow_it() {
    let mut cfg = test_config(7, 1);
    cfg.reproduction_energy_cost = 20.0;
    let mut sim = Simulation::new(cfg, 73).expect("simulation should initialize");
    let mut parent = make_organism(
        0,
        3,
        3,
        FacingDirection::East,
        false,
        false,
        false,
        0.7,
        30.0,
    );
    enable_reproduce_action(&mut parent);
    configure_sim(&mut sim, vec![parent]);

    let delta = tick_once(&mut sim);
    assert_eq!(delta.metrics.reproductions_last_turn, 1);
    assert_eq!(delta.spawned.len(), 1);
    assert_eq!((delta.spawned[0].q, delta.spawned[0].r), (2, 3));

    let parent_after = sim
        .organisms
        .iter()
        .find(|organism| organism.id == OrganismId(0))
        .expect("parent should remain alive");
    assert_eq!(parent_after.reproductions_count, 1);
    assert_eq!(parent_after.energy, 9.0);
}

#[test]
fn reproduce_action_fails_when_turn_upkeep_leaves_insufficient_energy() {
    let cfg = test_config(7, 1);
    let mut sim = Simulation::new(cfg, 75).expect("simulation should initialize");
    let mut parent = make_organism(
        0,
        3,
        3,
        FacingDirection::East,
        false,
        false,
        false,
        0.7,
        100.0,
    );
    enable_reproduce_action(&mut parent);
    configure_sim(&mut sim, vec![parent]);

    let delta = tick_once(&mut sim);
    assert_eq!(delta.metrics.reproductions_last_turn, 0);
    assert!(delta.spawned.is_empty());
    let parent_after = sim
        .organisms
        .iter()
        .find(|organism| organism.id == OrganismId(0))
        .expect("parent should remain alive");
    assert_eq!(parent_after.reproductions_count, 0);
    assert_eq!(parent_after.energy, 99.0);
}

#[test]
fn reproduction_can_create_new_species_via_genome_distance() {
    let mut cfg = test_config(7, 1);
    cfg.reproduction_energy_cost = 20.0;
    // High mutation pressure ensures child genome diverges from parent
    cfg.seed_genome_config.mutation_rate_weight = 1.0;
    cfg.seed_genome_config.mutation_rate_add_edge = 1.0;
    cfg.seed_genome_config.mutation_rate_split_edge = 1.0;
    // Very low threshold so even a small mutation creates a new species
    cfg.speciation_threshold = 0.001;
    let mut sim = Simulation::new(cfg, 88).expect("simulation should initialize");

    let mut parent = make_organism(
        0,
        3,
        3,
        FacingDirection::East,
        false,
        false,
        false,
        0.7,
        30.0,
    );
    // Give parent a genome with high mutation pressure so child will diverge
    parent.genome.mutation_rate_weight = 1.0;
    parent.genome.mutation_rate_add_edge = 1.0;
    parent.genome.mutation_rate_split_edge = 1.0;
    enable_reproduce_action(&mut parent);
    configure_sim(&mut sim, vec![parent]);

    let delta = tick_once(&mut sim);
    assert_eq!(delta.metrics.reproductions_last_turn, 1);
    assert_eq!(delta.spawned.len(), 1);
    // With threshold=0.001 and high mutation pressure, child should be a new species
    assert_ne!(delta.spawned[0].species_id, SpeciesId(0));
    assert!(sim.species_registry.len() >= 2);
}
