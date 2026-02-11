use super::support::*;
use super::*;

#[test]
fn look_sensor_returns_binary_occupancy() {
    let cfg = test_config(5, 2);
    let mut sim = Simulation::new(cfg, 7).expect("simulation should initialize");

    sim.organisms[0].q = 2;
    sim.organisms[0].r = 2;
    sim.organisms[0].facing = FacingDirection::East;
    sim.organisms[1].q = 3;
    sim.organisms[1].r = 2;

    sim.occupancy.fill(None);
    for org in &sim.organisms {
        let idx = sim.cell_index(org.q, org.r).expect("in-bounds test setup");
        sim.occupancy[idx] = Some(CellEntity::Organism(org.id));
    }

    let signal = look_sensor_value(
        (2, 2),
        FacingDirection::East,
        sim.organisms[0].id,
        sim.config.world_width as i32,
        &sim.occupancy,
    );
    assert_eq!(signal, 1.0);

    let empty_signal = look_sensor_value(
        (2, 2),
        FacingDirection::NorthWest,
        sim.organisms[0].id,
        sim.config.world_width as i32,
        &sim.occupancy,
    );
    assert_eq!(empty_signal, 0.0);
}

#[test]
fn look_sensor_detects_food_as_occupied() {
    let cfg = test_config(5, 1);
    let mut sim = Simulation::new(cfg, 76).expect("simulation should initialize");
    configure_sim(
        &mut sim,
        vec![make_organism(
            0,
            2,
            2,
            FacingDirection::East,
            false,
            false,
            false,
            0.1,
            10.0,
        )],
    );
    let added = sim.add_food(make_food(0, 3, 2, sim.config.food_energy));
    assert!(added);

    let signal = look_sensor_value(
        (2, 2),
        FacingDirection::East,
        OrganismId(0),
        sim.config.world_width as i32,
        &sim.occupancy,
    );
    assert_eq!(signal, 1.0);
}

#[test]
fn turn_actions_rotate_facing() {
    assert_eq!(
        facing_after_turn(FacingDirection::East, true, false),
        FacingDirection::NorthEast
    );
    assert_eq!(
        facing_after_turn(FacingDirection::East, false, true),
        FacingDirection::SouthEast
    );
    assert_eq!(
        facing_after_turn(FacingDirection::East, true, true),
        FacingDirection::East
    );
}

#[test]
fn only_highest_activation_action_runs_each_turn() {
    let cfg = test_config(6, 1);
    let mut sim = Simulation::new(cfg, 75).expect("simulation should initialize");
    configure_sim(
        &mut sim,
        vec![make_organism(
            0,
            2,
            2,
            FacingDirection::East,
            true,
            true,
            false,
            0.8,
            10.0,
        )],
    );

    let delta = tick_once(&mut sim);
    assert_eq!(delta.moves.len(), 1);
    assert_eq!(delta.moves[0].id, OrganismId(0));
    assert_eq!(delta.moves[0].to, (3, 2));
    assert_eq!(delta.metrics.reproductions_last_turn, 0);

    let organism = sim
        .organisms
        .iter()
        .find(|organism| organism.id == OrganismId(0))
        .expect("organism should remain alive");
    assert_eq!(organism.facing, FacingDirection::East);
}
