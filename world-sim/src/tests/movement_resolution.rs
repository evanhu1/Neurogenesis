use super::support::*;
use super::*;

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
fn food_ecology_cycles_consumption_and_regrowth_on_fertile_tile() {
    let mut cfg = test_config(6, 1);
    cfg.food_regrowth_interval = 2;

    let mut sim = Simulation::new(cfg, 102).expect("simulation should initialize");
    configure_sim(
        &mut sim,
        vec![make_single_action_organism(
            0,
            1,
            1,
            FacingDirection::East,
            ActionType::Eat,
            0.9,
            100.0,
        )],
    );

    let target = (2, 1);
    let target_idx = sim.cell_index(target.0, target.1);
    sim.food_tiles = vec![false; sim.occupancy.len()];
    sim.food_tiles[target_idx] = true;
    sim.food_regrowth_due_turn = vec![u64::MAX; sim.occupancy.len()];
    sim.food_regrowth_schedule.clear();
    sim.foods.push(types::FoodState {
        id: FoodId(0),
        q: target.0,
        r: target.1,
        energy: sim.config.food_energy,
        visual: types::plant_visual(),
    });
    sim.occupancy[target_idx] = Some(Occupant::Food(FoodId(0)));
    sim.next_food_id = 1;

    let mut regrowths = 0;
    for _ in 0..8 {
        let delta = tick_once(&mut sim);
        regrowths += delta
            .food_spawned
            .iter()
            .filter(|food| (food.q, food.r) == target)
            .count();
    }

    let eater = sim
        .organisms
        .iter()
        .find(|organism| organism.id == OrganismId(0))
        .expect("organism should survive ecology cycle");
    assert!(
        eater.consumptions_count >= 2,
        "ecology cycle should let the same fertile tile feed repeated consumptions",
    );
    assert!(
        regrowths >= 2,
        "ecology cycle should regrow food on the fertile tile multiple times",
    );
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

#[test]
fn eat_only_interacts_with_food() {
    let cfg = test_config(5, 2);
    let mut sim = Simulation::new(cfg, 204).expect("simulation should initialize");
    configure_sim(
        &mut sim,
        vec![
            make_single_action_organism(0, 1, 1, FacingDirection::East, ActionType::Eat, 0.9, 50.0),
            make_single_action_organism(
                1,
                2,
                1,
                FacingDirection::East,
                ActionType::Idle,
                0.1,
                50.0,
            ),
        ],
    );

    let delta = tick_once(&mut sim);
    assert_eq!(delta.metrics.consumptions_last_turn, 0);
    let prey = sim
        .organisms
        .iter()
        .find(|organism| organism.id == OrganismId(1))
        .expect("prey should still exist");
    assert_eq!(prey.energy, 49);
}

#[test]
fn attack_only_interacts_with_organisms_and_kills_at_zero_energy() {
    let mut cfg = test_config(5, 2);
    cfg.predation_enabled = true;
    let mut sim = Simulation::new(cfg, 205).expect("simulation should initialize");
    let predator = make_single_action_organism(
        0,
        1,
        1,
        FacingDirection::East,
        ActionType::Attack,
        0.9,
        50.0,
    );
    let mut prey =
        make_single_action_organism(1, 2, 1, FacingDirection::East, ActionType::Idle, 0.1, 50.0);
    prey.energy = 10;
    prey.energy_at_last_sensing = 10;
    configure_sim(&mut sim, vec![predator, prey]);

    let delta = tick_once(&mut sim);
    assert_eq!(delta.metrics.predations_last_turn, 1);
    assert!(sim
        .organisms
        .iter()
        .all(|organism| organism.id != OrganismId(1)));
}

#[test]
fn lethal_attack_conserves_energy_without_spawning_food() {
    let mut cfg = test_config(5, 2);
    cfg.predation_enabled = true;
    let mut sim = Simulation::new(cfg, 206).expect("simulation should initialize");
    let mut prey =
        make_single_action_organism(1, 2, 1, FacingDirection::East, ActionType::Idle, 0.1, 50.0);
    prey.energy = 10;
    prey.energy_at_last_sensing = 10;
    let predator = make_single_action_organism(
        0,
        1,
        1,
        FacingDirection::East,
        ActionType::Attack,
        0.9,
        50.0,
    );
    configure_sim(&mut sim, vec![predator, prey]);

    let delta = tick_once(&mut sim);
    assert_eq!(delta.metrics.predations_last_turn, 1);
    assert_eq!(delta.metrics.consumptions_last_turn, 1);
    assert!(sim
        .organisms
        .iter()
        .all(|organism| organism.id != OrganismId(1)));
    assert!(delta.food_spawned.is_empty());
    assert_eq!(sim.organisms[0].energy, 59);
}
