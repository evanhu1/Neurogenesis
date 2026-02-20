use super::support::*;
use super::*;

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
    assert!(moves.is_empty());
    assert_eq!(delta.metrics.consumptions_last_turn, 1);
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
    assert_eq!(moves.get(&OrganismId(0)), None);
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
    assert_eq!(moves.len(), 0);
    assert_eq!(delta.metrics.consumptions_last_turn, 2);
}

#[test]
fn contested_occupied_target_where_occupant_remains_triggers_passive_bite() {
    let cfg = test_config(5, 3);
    let predator = make_organism(0, 1, 1, FacingDirection::East, true, false, false, 0.9, 6.0);
    let prey = make_organism(
        1,
        2,
        1,
        FacingDirection::West,
        false,
        false,
        false,
        0.1,
        3.0,
    );
    let blocker = make_organism(
        2,
        1,
        2,
        FacingDirection::NorthEast,
        false,
        false,
        false,
        0.2,
        0,
    );
    let neuron_energy_cost = cfg.food_energy / 100.0;
    let predator_metabolism = neuron_energy_cost
        * (predator.genome.num_neurons as f32
            + predator.brain.sensory.len() as f32
            + predator.genome.vision_distance as f32);
    let prey_metabolism = neuron_energy_cost
        * (prey.genome.num_neurons as f32
            + prey.brain.sensory.len() as f32
            + prey.genome.vision_distance as f32);
    let prey_energy_after_metabolism = prey.energy - prey_metabolism;
    let expected_energy =
        6.0 - predator_metabolism - cfg.move_action_energy_cost + prey_energy_after_metabolism;
    let mut sim = Simulation::new(cfg, 17).expect("simulation should initialize");
    configure_sim(&mut sim, vec![predator, prey, blocker]);

    let delta = tick_once(&mut sim);
    let moves = move_map(&delta);
    assert_eq!(moves.len(), 0);
    assert_eq!(delta.metrics.consumptions_last_turn, 1);
    assert_eq!(delta.metrics.predations_last_turn, 1);
    assert!(!sim
        .organisms
        .iter()
        .any(|organism| organism.id == OrganismId(1)));
    assert!(delta
        .removed_positions
        .iter()
        .any(|pos| pos.entity_id == EntityId::Organism(OrganismId(1))));
    let predator = sim
        .organisms
        .iter()
        .find(|organism| organism.id == OrganismId(0))
        .expect("predator should survive");
    assert_eq!(predator.energy, expected_energy);
}

#[test]
fn move_into_food_consumes_and_schedules_regrowth() {
    let mut cfg = test_config(5, 1);
    cfg.food_energy = 7.0;
    cfg.food_regrowth_interval = 2;
    cfg.plant_growth_speed = 1.0;
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
    assert_eq!(delta.removed_positions.len(), 1);
    assert_eq!(
        delta.removed_positions[0].entity_id,
        EntityId::Food(FoodId(0))
    );
    assert_eq!(
        (delta.removed_positions[0].q, delta.removed_positions[0].r),
        (2, 1)
    );
    assert!(delta.food_spawned.is_empty());
    assert_eq!(delta.metrics.consumptions_last_turn, 1);
    assert_eq!(delta.metrics.predations_last_turn, 0);
    assert!(sim.foods.is_empty());
    assert_eq!(
        sim.occupant_at(2, 1),
        Some(Occupant::Organism(OrganismId(0)))
    );

    let predator = sim
        .organisms
        .iter()
        .find(|organism| organism.id == OrganismId(0))
        .expect("predator should survive");
    assert_eq!(predator.energy, 15.72);
}

#[test]
fn food_regrows_naturally_based_on_fertility_schedule() {
    let mut cfg = test_config(5, 1);
    cfg.food_regrowth_interval = 5;
    cfg.plant_growth_speed = 1.0;
    cfg.food_fertility_floor = 1.0;

    let mut sim = Simulation::new(cfg, 102).expect("simulation should initialize");
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
            0.9,
            10.0,
        )],
    );

    // No initial food — verify tiles grow food on their own via the
    // perpetual regrowth cycle without any consumption trigger.
    assert!(sim.foods.is_empty());

    let mut total_spawned = 0;
    for _ in 0..10 {
        let delta = tick_once(&mut sim);
        total_spawned += delta.food_spawned.len();
    }
    assert!(
        total_spawned > 0,
        "food should regrow on empty tiles via the fertility schedule",
    );
    assert!(!sim.foods.is_empty());
}

#[test]
fn move_resolution_blocks_wall_cells() {
    let cfg = test_config(5, 1);
    let mut sim = Simulation::new(cfg, 203).expect("simulation should initialize");
    let wall_idx = sim.cell_index(2, 1);
    sim.terrain_map[wall_idx] = true;
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
            0,
        )],
    );

    let delta = tick_once(&mut sim);
    assert!(delta.moves.is_empty(), "wall should block movement");
    assert_eq!(sim.occupancy[wall_idx], Some(Occupant::Wall));
    let organism = sim
        .organisms
        .iter()
        .find(|item| item.id == OrganismId(0))
        .expect("organism should exist");
    assert_eq!((organism.q, organism.r), (1, 1));
}
