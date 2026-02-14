use super::support::*;
use super::*;

#[test]
fn starvation_and_reproduction_interact_in_same_turn() {
    let mut cfg = test_config(6, 4);
    cfg.reproduction_energy_cost = 5.0;
    let mut sim = Simulation::new(cfg, 18).expect("simulation should initialize");
    let mut reproducer =
        make_organism(0, 1, 1, FacingDirection::East, true, false, false, 0.9, 6.0);
    enable_reproduce_action(&mut reproducer);
    configure_sim(
        &mut sim,
        vec![
            reproducer,
            make_organism(
                1,
                2,
                1,
                FacingDirection::West,
                false,
                false,
                false,
                0.1,
                3.0,
            ),
            make_organism(
                2,
                0,
                0,
                FacingDirection::East,
                false,
                false,
                false,
                0.2,
                1.0,
            ),
            make_organism(
                3,
                4,
                4,
                FacingDirection::West,
                false,
                false,
                false,
                0.2,
                6.0,
            ),
        ],
    );

    let delta = tick_once(&mut sim);
    assert_eq!(delta.metrics.consumptions_last_turn, 0);
    assert_eq!(delta.metrics.reproductions_last_turn, 1);
    assert_eq!(delta.metrics.starvations_last_turn, 1);
    assert_eq!(
        delta
            .removed_positions
            .iter()
            .map(|entry| entry.entity_id)
            .collect::<Vec<_>>(),
        vec![EntityId::Organism(OrganismId(2))]
    );
    assert_eq!(delta.spawned.len(), 1);
    assert_eq!((delta.spawned[0].q, delta.spawned[0].r), (0, 1));
    assert_eq!(sim.organisms.len(), 4);
    let reproducer = sim
        .organisms
        .iter()
        .find(|organism| organism.id == OrganismId(0))
        .expect("reproducer should survive");
    assert_eq!(reproducer.consumptions_count, 0);
    assert_eq!(reproducer.reproductions_count, 1);
    // 6.0 - 1.0 (turn upkeep) - 5.0 (reproduction) - 2.0 (move + reproduce action costs) = -2.0
    assert_eq!(reproducer.energy, -2.0);
}

#[test]
fn starvation_does_not_spawn_replacements() {
    let cfg = test_config(6, 2);
    let mut sim = Simulation::new(cfg, 29).expect("simulation should initialize");
    configure_sim(
        &mut sim,
        vec![
            make_organism(
                0,
                1,
                1,
                FacingDirection::East,
                false,
                false,
                false,
                0.2,
                1.0,
            ),
            make_organism(
                1,
                4,
                4,
                FacingDirection::West,
                false,
                false,
                false,
                0.2,
                5.0,
            ),
        ],
    );

    let delta = tick_once(&mut sim);
    assert_eq!(delta.metrics.starvations_last_turn, 1);
    assert!(delta.spawned.is_empty());
    assert_eq!(
        delta
            .removed_positions
            .iter()
            .map(|entry| entry.entity_id)
            .collect::<Vec<_>>(),
        vec![EntityId::Organism(OrganismId(0))]
    );
    assert_eq!(sim.organisms.len(), 1);
}

#[test]
fn move_energy_cost_applies_to_attempted_moves() {
    let cfg = test_config(5, 2);
    let mut sim = Simulation::new(cfg, 74).expect("simulation should initialize");
    configure_sim(
        &mut sim,
        vec![
            make_organism(
                0,
                0,
                1,
                FacingDirection::East,
                true,
                false,
                false,
                0.6,
                10.0,
            ),
            make_organism(
                1,
                1,
                0,
                FacingDirection::SouthEast,
                true,
                false,
                false,
                0.6,
                10.0,
            ),
        ],
    );

    let _ = tick_once(&mut sim);
    let winner = sim
        .organisms
        .iter()
        .find(|organism| organism.id == OrganismId(0))
        .expect("winner should survive");
    let loser = sim
        .organisms
        .iter()
        .find(|organism| organism.id == OrganismId(1))
        .expect("loser should survive");
    assert_eq!(winner.energy, 8.0);
    assert_eq!(loser.energy, 8.0);
}

#[test]
fn no_overlap_invariant_holds_after_mixed_turn() {
    let cfg = test_config(6, 4);
    let mut sim = Simulation::new(cfg, 20).expect("simulation should initialize");
    configure_sim(
        &mut sim,
        vec![
            make_organism(0, 1, 1, FacingDirection::East, true, false, false, 0.9, 1),
            make_organism(1, 2, 1, FacingDirection::West, false, false, false, 0.1, 0),
            make_organism(2, 0, 0, FacingDirection::East, false, false, false, 0.2, 1),
            make_organism(3, 4, 4, FacingDirection::West, true, false, false, 0.4, 0),
        ],
    );

    let _ = tick_once(&mut sim);
    assert_no_overlap(&sim);
}

#[test]
fn targeted_complex_resolution_snapshot_is_deterministic() {
    let cfg = test_config(6, 4);

    let scenario = vec![
        make_organism(0, 1, 1, FacingDirection::East, true, false, false, 0.9, 1),
        make_organism(
            1,
            2,
            1,
            FacingDirection::SouthEast,
            true,
            false,
            false,
            0.7,
            0,
        ),
        make_organism(2, 2, 2, FacingDirection::West, true, false, false, 0.6, 1),
        make_organism(
            3,
            1,
            2,
            FacingDirection::NorthEast,
            true,
            false,
            false,
            0.8,
            0,
        ),
    ];

    let mut a = Simulation::new(cfg.clone(), 21).expect("simulation should initialize");
    configure_sim(&mut a, scenario.clone());
    a.step_n(3);
    let a_snapshot = serde_json::to_string(&a.snapshot()).expect("serialize snapshot");

    let mut b = Simulation::new(cfg, 21).expect("simulation should initialize");
    configure_sim(&mut b, scenario);
    b.step_n(3);
    let b_snapshot = serde_json::to_string(&b.snapshot()).expect("serialize snapshot");

    assert_eq!(a_snapshot, b_snapshot);
}
