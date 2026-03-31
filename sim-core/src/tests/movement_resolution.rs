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
    cfg.food_regrowth_jitter = 0;

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
    sim.food_fertility = vec![false; sim.occupancy.len()];
    sim.food_fertility[target_idx] = true;
    sim.food_regrowth_due_turn = vec![u64::MAX; sim.occupancy.len()];
    sim.food_regrowth_schedule.clear();
    sim.foods.push(sim_types::FoodState {
        id: FoodId(0),
        q: target.0,
        r: target.1,
        energy: sim.config.food_energy,
        kind: sim_types::FoodKind::Plant,
    });
    sim.occupancy[target_idx] = Some(Occupant::Food(FoodId(0)));
    sim.next_food_id = 1;

    let mut regrowths = 0;
    for _ in 0..8 {
        let delta = tick_once(&mut sim);
        regrowths += delta
            .food_spawned
            .iter()
            .filter(|food| (food.q, food.r) == target && food.kind == sim_types::FoodKind::Plant)
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
fn moving_onto_spikes_applies_same_tick_health_damage() {
    let cfg = test_config(5, 1);
    let mut sim = Simulation::new(cfg, 2031).expect("simulation should initialize");
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
            50.0,
        )],
    );
    let spike_idx = sim.cell_index(2, 1);
    sim.spike_map[spike_idx] = true;

    let delta = tick_once(&mut sim);
    assert_eq!(
        move_map(&delta).get(&OrganismId(0)),
        Some(&((1, 1), (2, 1)))
    );
    let organism = sim
        .organisms
        .iter()
        .find(|item| item.id == OrganismId(0))
        .expect("organism should survive spike damage");
    assert_eq!((organism.q, organism.r), (2, 1));
    assert_eq!(organism.damage_taken_last_turn, 5.0);
    assert_eq!(organism.health, 45.0);
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
    assert_eq!(prey.damage_taken_last_turn, 0.0);
}

#[test]
fn attack_only_interacts_with_organisms_and_applies_damage() {
    let cfg = test_config(5, 2);
    let mut sim = Simulation::new(cfg, 205).expect("simulation should initialize");
    configure_sim(
        &mut sim,
        vec![
            make_single_action_organism(
                0,
                1,
                1,
                FacingDirection::East,
                ActionType::Attack,
                0.9,
                50.0,
            ),
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

    let _ = tick_once(&mut sim);
    let prey = sim
        .organisms
        .iter()
        .find(|organism| organism.id == OrganismId(1))
        .expect("prey should survive a single attack");
    assert_eq!(prey.damage_taken_last_turn, 25.0);
    assert_eq!(prey.health, 25.0);
    assert!(prey.energy > 0.0);
}

#[test]
fn lethal_attack_spawns_corpse_food_without_feeding_attacker() {
    let cfg = test_config(5, 2);
    let mut sim = Simulation::new(cfg, 206).expect("simulation should initialize");
    let mut prey =
        make_single_action_organism(1, 2, 1, FacingDirection::East, ActionType::Idle, 0.1, 50.0);
    prey.health = 5.0;
    prey.max_health = 50.0;
    configure_sim(
        &mut sim,
        vec![
            make_single_action_organism(
                0,
                1,
                1,
                FacingDirection::East,
                ActionType::Attack,
                0.9,
                50.0,
            ),
            prey,
        ],
    );

    let delta = tick_once(&mut sim);
    assert_eq!(delta.metrics.predations_last_turn, 1);
    assert_eq!(delta.metrics.consumptions_last_turn, 0);
    assert!(sim
        .organisms
        .iter()
        .all(|organism| organism.id != OrganismId(1)));
    assert!(delta
        .food_spawned
        .iter()
        .any(|food| food.kind == sim_types::FoodKind::Corpse && (food.q, food.r) == (2, 1)));
}
