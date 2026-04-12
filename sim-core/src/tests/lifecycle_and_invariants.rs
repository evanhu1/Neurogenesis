use super::support::*;
use super::*;

#[test]
fn state_invariants_hold_across_multi_turn_mixed_ecology_and_spawn_flow() {
    let mut cfg = test_config(8, 5);
    cfg.food_regrowth_interval = 2;
    cfg.food_regrowth_jitter = 0;

    let mut sim = Simulation::new(cfg, 20).expect("simulation should initialize");
    let wall_idx = sim.cell_index(3, 1);
    let spike_idx = sim.cell_index(2, 1);
    sim.terrain_map[wall_idx] = true;
    sim.occupancy[wall_idx] = Some(Occupant::Wall);
    sim.spike_map[spike_idx] = true;

    configure_sim(
        &mut sim,
        vec![
            make_single_action_organism(
                0,
                1,
                1,
                FacingDirection::East,
                ActionType::Forward,
                0.9,
                100.0,
            ),
            make_single_action_organism(
                1,
                5,
                3,
                FacingDirection::East,
                ActionType::Attack,
                0.9,
                100.0,
            ),
            {
                let mut prey = make_single_action_organism(
                    2,
                    6,
                    3,
                    FacingDirection::West,
                    ActionType::Idle,
                    0.1,
                    100.0,
                );
                prey.health = 20.0;
                prey.max_health = 100.0;
                prey
            },
            make_single_action_organism(
                3,
                1,
                4,
                FacingDirection::East,
                ActionType::Eat,
                0.9,
                100.0,
            ),
            make_single_action_organism(
                4,
                4,
                5,
                FacingDirection::East,
                ActionType::Reproduce,
                0.9,
                900.0,
            ),
        ],
    );

    let food_idx = sim.cell_index(2, 4);
    sim.food_fertility = vec![false; sim.occupancy.len()];
    sim.food_fertility[food_idx] = true;
    sim.food_regrowth_due_turn = vec![u64::MAX; sim.occupancy.len()];
    sim.food_regrowth_schedule.clear();
    sim.foods.push(sim_types::FoodState {
        id: FoodId(0),
        q: 2,
        r: 4,
        energy: sim.config().food_energy,
        kind: sim_types::FoodKind::Plant,
        visual: sim_types::food_visual(sim_types::FoodKind::Plant),
    });
    sim.occupancy[food_idx] = Some(Occupant::Food(FoodId(0)));
    sim.next_food_id = 1;

    sim.validate_state()
        .expect("mixed scenario should start from a valid state");

    let mut saw_predation = false;
    let mut saw_consumption = false;
    let mut saw_reproduction = false;
    for _ in 0..10 {
        let delta = tick_once(&mut sim);
        saw_predation |= delta.metrics.predations_last_turn > 0;
        saw_consumption |= delta.metrics.consumptions_last_turn > 0;
        saw_reproduction |= delta.metrics.reproductions_last_turn > 0;
        sim.validate_state()
            .expect("mixed scenario should preserve simulation invariants");
        assert_no_overlap(&sim);
    }

    assert!(saw_predation, "scenario should exercise predation");
    assert!(saw_consumption, "scenario should exercise food consumption");
    assert!(saw_reproduction, "scenario should exercise reproduction");
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
