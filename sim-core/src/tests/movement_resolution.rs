use super::support::*;
use super::*;

#[test]
fn move_into_cell_vacated_same_turn_succeeds() {
    let cfg = test_config(5, 2);
    let mut sim = Simulation::new(cfg, 11).expect("simulation should initialize");
    configure_sim(
        &mut sim,
        vec![
            make_organism(0, 1, 1, FacingDirection::East, true, false, false, 0.8, 0),
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
        ],
    );

    let delta = tick_once(&mut sim);
    let moves = move_map(&delta);
    assert_eq!(moves.len(), 2);
    assert_eq!(moves.get(&OrganismId(0)), Some(&((1, 1), (2, 1))));
    assert_eq!(moves.get(&OrganismId(1)), Some(&((2, 1), (2, 2))));
    assert_eq!(delta.metrics.consumptions_last_turn, 0);
}

#[test]
fn two_organism_swap_resolves_deterministically() {
    let cfg = test_config(5, 2);
    let mut sim = Simulation::new(cfg, 12).expect("simulation should initialize");
    configure_sim(
        &mut sim,
        vec![
            make_organism(0, 1, 1, FacingDirection::East, true, false, false, 0.4, 0),
            make_organism(1, 2, 1, FacingDirection::West, true, false, false, 0.3, 0),
        ],
    );

    let delta = tick_once(&mut sim);
    let moves = move_map(&delta);
    assert_eq!(moves.get(&OrganismId(0)), Some(&((1, 1), (2, 1))));
    assert_eq!(moves.get(&OrganismId(1)), Some(&((2, 1), (1, 1))));
    assert_eq!(delta.metrics.consumptions_last_turn, 0);
}

#[test]
fn multi_attacker_single_target_uses_confidence_winner() {
    let cfg = test_config(5, 2);
    let mut sim = Simulation::new(cfg, 13).expect("simulation should initialize");
    configure_sim(
        &mut sim,
        vec![
            make_organism(0, 0, 1, FacingDirection::East, true, false, false, 0.9, 0),
            make_organism(
                1,
                1,
                0,
                FacingDirection::SouthEast,
                true,
                false,
                false,
                0.1,
                0,
            ),
        ],
    );

    let delta = tick_once(&mut sim);
    let moves = move_map(&delta);
    assert_eq!(moves.len(), 1);
    assert_eq!(moves.get(&OrganismId(0)), Some(&((0, 1), (1, 1))));
}

#[test]
fn multi_attacker_single_target_tie_breaks_by_id() {
    let cfg = test_config(5, 2);
    let mut sim = Simulation::new(cfg, 14).expect("simulation should initialize");
    configure_sim(
        &mut sim,
        vec![
            make_organism(0, 0, 1, FacingDirection::East, true, false, false, 0.5, 0),
            make_organism(
                1,
                1,
                0,
                FacingDirection::SouthEast,
                true,
                false,
                false,
                0.5,
                0,
            ),
        ],
    );

    let delta = tick_once(&mut sim);
    let moves = move_map(&delta);
    assert_eq!(moves.len(), 1);
    assert_eq!(moves.get(&OrganismId(0)), Some(&((0, 1), (1, 1))));
}

#[test]
fn attacker_vs_escaping_target_has_no_consumption_when_target_escapes() {
    let cfg = test_config(6, 2);
    let mut sim = Simulation::new(cfg, 15).expect("simulation should initialize");
    configure_sim(
        &mut sim,
        vec![
            make_organism(0, 1, 1, FacingDirection::East, true, false, false, 0.7, 0),
            make_organism(1, 2, 1, FacingDirection::East, true, false, false, 0.6, 0),
        ],
    );

    let delta = tick_once(&mut sim);
    let moves = move_map(&delta);
    assert_eq!(moves.get(&OrganismId(0)), Some(&((1, 1), (2, 1))));
    assert_eq!(moves.get(&OrganismId(1)), Some(&((2, 1), (3, 1))));
    assert_eq!(delta.metrics.consumptions_last_turn, 0);
    assert!(sim
        .organisms
        .iter()
        .any(|organism| organism.id == OrganismId(1)));
}

#[test]
fn multi_node_cycle_resolves_without_conflict() {
    let cfg = test_config(6, 3);
    let mut sim = Simulation::new(cfg, 16).expect("simulation should initialize");
    configure_sim(
        &mut sim,
        vec![
            make_organism(
                0,
                1,
                1,
                FacingDirection::SouthEast,
                true,
                false,
                false,
                0.7,
                0,
            ),
            make_organism(1, 2, 1, FacingDirection::West, true, false, false, 0.6, 0),
            make_organism(
                2,
                1,
                2,
                FacingDirection::NorthEast,
                true,
                false,
                false,
                0.5,
                0,
            ),
        ],
    );

    let delta = tick_once(&mut sim);
    let moves = move_map(&delta);
    assert_eq!(moves.len(), 3);
    assert_eq!(moves.get(&OrganismId(0)), Some(&((1, 1), (1, 2))));
    assert_eq!(moves.get(&OrganismId(2)), Some(&((1, 2), (2, 1))));
    assert_eq!(moves.get(&OrganismId(1)), Some(&((2, 1), (1, 1))));
    assert_eq!(delta.metrics.consumptions_last_turn, 0);
}

#[test]
fn contested_occupied_target_where_occupant_remains_uses_consume_path() {
    let cfg = test_config(5, 3);
    let mut sim = Simulation::new(cfg, 17).expect("simulation should initialize");
    configure_sim(
        &mut sim,
        vec![
            make_organism(0, 1, 1, FacingDirection::East, true, false, false, 0.9, 6.0),
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
                1,
                2,
                FacingDirection::NorthEast,
                true,
                false,
                false,
                0.2,
                0,
            ),
        ],
    );

    let delta = tick_once(&mut sim);
    let moves = move_map(&delta);
    assert_eq!(moves.len(), 1);
    assert_eq!(moves.get(&OrganismId(0)), Some(&((1, 1), (2, 1))));
    assert_eq!(delta.metrics.consumptions_last_turn, 1);
    assert!(sim
        .organisms
        .iter()
        .all(|organism| organism.id != OrganismId(1)));
    let predator = sim
        .organisms
        .iter()
        .find(|organism| organism.id == OrganismId(0))
        .expect("predator should survive");
    assert_eq!(predator.energy, 6.0);
}

#[test]
fn move_into_food_consumes_and_replenishes_food_supply() {
    let mut cfg = test_config(5, 1);
    cfg.food_energy = 7.0;
    let mut sim = Simulation::new(cfg, 101).expect("simulation should initialize");
    configure_sim(
        &mut sim,
        vec![make_organism(
            0,
            1,
            1,
            FacingDirection::East,
            true,
            false,
            false,
            0.9,
            10.0,
        )],
    );
    let added = sim.add_food(make_food(0, 2, 1, 7.0));
    assert!(added, "food setup should succeed");
    sim.next_food_id = 1;

    let delta = tick_once(&mut sim);
    let moves = move_map(&delta);
    assert_eq!(moves.get(&OrganismId(0)), Some(&((1, 1), (2, 1))));
    assert!(delta.removed_positions.is_empty());
    assert_eq!(delta.food_removed_positions.len(), 1);
    assert_eq!(
        (
            delta.food_removed_positions[0].q,
            delta.food_removed_positions[0].r
        ),
        (2, 1)
    );
    let target_food = (5_usize * 5) / sim.config.food_coverage_divisor as usize;
    assert_eq!(delta.food_spawned.len(), target_food);
    assert_eq!(delta.metrics.consumptions_last_turn, 1);
    assert_eq!(sim.foods.len(), target_food);
    assert_eq!(
        sim.occupant_at(2, 1),
        Some(Occupant::Organism(OrganismId(0)))
    );

    let predator = sim
        .organisms
        .iter()
        .find(|organism| organism.id == OrganismId(0))
        .expect("predator should survive");
    assert_eq!(predator.energy, 15.0);
}
