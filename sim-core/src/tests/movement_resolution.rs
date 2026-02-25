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
    assert_eq!(delta.metrics.consumptions_last_turn, 2);
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
    assert_eq!(delta.metrics.consumptions_last_turn, 3);
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

#[test]
fn dopamine_stays_near_zero_when_idle_without_events() {
    let mut cfg = test_config(5, 1);
    cfg.action_selection_margin = Some(0.0);
    let mut sim = Simulation::new(cfg, 301).expect("simulation should initialize");
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
            100.0,
        )],
    );

    let _ = tick_once(&mut sim);
    let dopamine_t1 = sim
        .organisms
        .iter()
        .find(|organism| organism.id == OrganismId(0))
        .expect("organism should exist")
        .dopamine;
    let _ = tick_once(&mut sim);
    let dopamine_t2 = sim
        .organisms
        .iter()
        .find(|organism| organism.id == OrganismId(0))
        .expect("organism should exist")
        .dopamine;

    assert!(
        dopamine_t1.abs() < 1.0e-4,
        "idle baseline dopamine should stay near zero, got {dopamine_t1}"
    );
    assert!(
        dopamine_t2.abs() < 1.0e-4,
        "idle baseline dopamine should stay near zero, got {dopamine_t2}"
    );
}

#[test]
fn dopamine_becomes_positive_after_food_consumption() {
    let cfg = test_config(6, 1);
    let mut sim = Simulation::new(cfg, 302).expect("simulation should initialize");
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
            0.95,
            100.0,
        )],
    );

    let food_id = FoodId(0);
    sim.foods.push(sim_types::FoodState {
        id: food_id,
        q: 2,
        r: 1,
        energy: 50.0,
    });
    let food_idx = sim.cell_index(2, 1);
    sim.occupancy[food_idx] = Some(Occupant::Food(food_id));
    sim.next_food_id = 1;

    let first = tick_once(&mut sim);
    assert_eq!(first.metrics.consumptions_last_turn, 1);

    let dopamine = sim
        .organisms
        .iter()
        .find(|organism| organism.id == OrganismId(0))
        .expect("organism should exist")
        .dopamine;
    assert!(
        dopamine > 0.2,
        "food gain should produce a meaningful positive dopamine signal, got {dopamine}"
    );
}

#[test]
fn plant_consumption_biomass_depletion_has_deterministic_jitter() {
    let cfg = test_config(6, 1);
    let mut sim_a = Simulation::new(cfg.clone(), 701).expect("simulation should initialize");
    let mut sim_b = Simulation::new(cfg.clone(), 702).expect("simulation should initialize");
    let mut sim_c = Simulation::new(cfg, 701).expect("simulation should initialize");

    let setup = |sim: &mut Simulation| {
        configure_sim(
            sim,
            vec![make_organism(
                0,
                1,
                1,
                FacingDirection::East,
                true,
                false,
                false,
                0.95,
                100.0,
            )],
        );

        let food_id = FoodId(0);
        sim.foods.push(sim_types::FoodState {
            id: food_id,
            q: 2,
            r: 1,
            energy: 5.0,
        });
        let food_idx = sim.cell_index(2, 1);
        sim.occupancy[food_idx] = Some(Occupant::Food(food_id));
        sim.next_food_id = 1;

        let capacity = crate::grid::world_capacity(sim.config.world_width);
        sim.food_fertility = vec![0; capacity];
        sim.biomass = vec![0.0; capacity];
        sim.biomass[food_idx] = 1.0;
    };
    setup(&mut sim_a);
    setup(&mut sim_b);
    setup(&mut sim_c);

    let _ = tick_once(&mut sim_a);
    let _ = tick_once(&mut sim_b);
    let _ = tick_once(&mut sim_c);

    let biomass_idx_a = sim_a.cell_index(2, 1);
    let biomass_idx_b = sim_b.cell_index(2, 1);
    let biomass_idx_c = sim_c.cell_index(2, 1);
    let biomass_a = sim_a.biomass[biomass_idx_a];
    let biomass_b = sim_b.biomass[biomass_idx_b];
    let biomass_c = sim_c.biomass[biomass_idx_c];

    assert!(
        (0.0625..=0.4375).contains(&biomass_a),
        "jittered biomass depletion should leave expected residue, got {biomass_a}",
    );
    assert!(
        (0.0625..=0.4375).contains(&biomass_b),
        "jittered biomass depletion should leave expected residue, got {biomass_b}",
    );
    assert!(
        (biomass_a - biomass_c).abs() < 1.0e-6,
        "same seed/setup should produce identical jittered depletion: {biomass_a} vs {biomass_c}",
    );
    assert!(
        (biomass_a - biomass_b).abs() > 1.0e-6,
        "different seeds should produce different jittered depletion: {biomass_a} vs {biomass_b}",
    );
}

#[test]
fn food_consumption_locks_organism_for_one_turn() {
    let mut cfg = test_config(6, 1);
    cfg.plant_growth_speed = 0.01;
    cfg.food_fertility_floor = 0.0;
    cfg.action_selection_margin = Some(0.0);
    let mut sim = Simulation::new(cfg, 902).expect("simulation should initialize");
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
            0.95,
            100.0,
        )],
    );

    let food_id = FoodId(0);
    sim.foods.push(sim_types::FoodState {
        id: food_id,
        q: 2,
        r: 1,
        energy: 5.0,
    });
    let food_idx = sim.cell_index(2, 1);
    sim.occupancy[food_idx] = Some(Occupant::Food(food_id));
    sim.next_food_id = 1;

    let first = tick_once(&mut sim);
    assert_eq!(first.moves.len(), 1);
    assert_eq!(first.moves[0].from, (1, 1));
    assert_eq!(first.moves[0].to, (2, 1));

    let second = tick_once(&mut sim);
    assert!(
        second.moves.is_empty(),
        "consume lock should force one full tick of immobility",
    );
    let organism_after_second = sim
        .organisms
        .iter()
        .find(|organism| organism.id == OrganismId(0))
        .expect("organism should survive");
    assert_eq!((organism_after_second.q, organism_after_second.r), (2, 1));

    let third = tick_once(&mut sim);
    let moves = move_map(&third);
    assert_eq!(moves.get(&OrganismId(0)), Some(&((2, 1), (3, 1))));
}

#[test]
fn dopamine_becomes_negative_when_other_organism_bites_prey() {
    let cfg = test_config(6, 2);
    let mut sim = Simulation::new(cfg, 303).expect("simulation should initialize");
    configure_sim(
        &mut sim,
        vec![
            make_organism(
                0,
                1,
                1,
                FacingDirection::East,
                true,
                false,
                false,
                0.95,
                120.0,
            ),
            make_organism(
                1,
                2,
                1,
                FacingDirection::East,
                false,
                false,
                false,
                0.9,
                120.0,
            ),
        ],
    );

    let first = tick_once(&mut sim);
    assert_eq!(first.metrics.predations_last_turn, 1);

    let prey_dopamine = sim
        .organisms
        .iter()
        .find(|organism| organism.id == OrganismId(1))
        .expect("prey should survive")
        .dopamine;
    assert!(
        prey_dopamine < -0.5,
        "being bitten should produce a meaningful negative dopamine signal, got {prey_dopamine}"
    );
}
