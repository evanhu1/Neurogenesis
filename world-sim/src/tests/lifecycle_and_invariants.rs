use super::support::*;
use super::*;

#[test]
fn state_invariants_hold_across_multi_turn_movement_and_attack_flow() {
    let mut cfg = test_config(8, 5);
    cfg.predation_enabled = true;
    cfg.attack_attempt_cost = 1;
    cfg.attack_energy_transfer = 20;
    cfg.runtime_plasticity_enabled = false;

    let mut sim = Simulation::new(cfg, 20).expect("simulation should initialize");
    let wall_idx = sim.cell_index(3, 1);
    sim.terrain_map[wall_idx] = true;
    sim.occupancy[wall_idx] = Some(Occupant::Wall);

    configure_sim(
        &mut sim,
        vec![
            make_compositional_organism(
                0,
                1,
                1,
                FacingDirection::East,
                &[ActionType::Forward],
                100.0,
            ),
            make_compositional_organism(
                1,
                5,
                3,
                FacingDirection::East,
                &[ActionType::Attack],
                100.0,
            ),
            {
                let mut prey =
                    make_compositional_organism(2, 6, 3, FacingDirection::West, &[], 100.0);
                prey.energy = 20;
                prey.energy_at_last_sensing = 20;
                prey
            },
            make_compositional_organism(
                3,
                1,
                4,
                FacingDirection::East,
                &[ActionType::Forward],
                100.0,
            ),
            make_compositional_organism(
                4,
                4,
                5,
                FacingDirection::East,
                &[ActionType::TurnLeft, ActionType::Forward],
                900.0,
            ),
        ],
    );

    sim.validate_state()
        .expect("competitive scenario should start from a valid state");

    let mut saw_predation = false;
    for _ in 0..10 {
        let delta = tick_once(&mut sim);
        saw_predation |= delta.metrics.predations_last_turn > 0;
        sim.validate_state()
            .expect("competitive scenario should preserve simulation invariants");
        assert_no_overlap(&sim);
    }

    assert!(saw_predation, "scenario should exercise predation");
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
    a.advance_n(3);
    let a_snapshot = serde_json::to_string(&a.snapshot()).expect("serialize snapshot");

    let mut b = Simulation::new(cfg, 21).expect("simulation should initialize");
    configure_sim(&mut b, scenario);
    b.advance_n(3);
    let b_snapshot = serde_json::to_string(&b.snapshot()).expect("serialize snapshot");

    assert_eq!(a_snapshot, b_snapshot);
}
