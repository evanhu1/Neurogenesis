use super::support::*;
use super::*;

fn organism(sim: &Simulation, id: u64) -> &OrganismState {
    sim.organisms
        .iter()
        .find(|organism| organism.id == OrganismId(id))
        .expect("organism should still be alive")
}

fn competitive_config(world_width: u32, num_organisms: u32) -> WorldConfig {
    let mut cfg = test_config(world_width, num_organisms);
    cfg.predation_enabled = true;
    cfg.attack_attempt_cost = 1;
    cfg.attack_energy_transfer = 20;
    cfg.runtime_plasticity_enabled = false;
    cfg
}

fn categorical_config(world_width: u32, num_organisms: u32) -> WorldConfig {
    competitive_config(world_width, num_organisms)
}

#[test]
fn overlapping_move_destination_uses_confidence_winner() {
    let cfg = test_config(5, 2);
    let mut sim = Simulation::new(cfg, 13).expect("simulation should initialize");
    configure_sim(
        &mut sim,
        vec![
            make_categorical_organism_with_bias(
                0,
                0,
                1,
                FacingDirection::East,
                &[ActionType::Forward],
                100.0,
                50,
            ),
            make_categorical_organism_with_bias(
                1,
                1,
                0,
                FacingDirection::SouthEast,
                &[ActionType::Forward],
                50.0,
                50,
            ),
        ],
    );

    let moves = move_map(&tick_once(&mut sim));
    assert_eq!(moves.len(), 1);
    assert_eq!(moves.get(&OrganismId(0)), Some(&((0, 1), (1, 1))));
    assert_no_overlap(&sim);
}

#[test]
fn overlapping_move_destination_tie_breaks_by_lowest_id() {
    let cfg = test_config(5, 2);
    let mut sim = Simulation::new(cfg, 14).expect("simulation should initialize");
    configure_sim(
        &mut sim,
        vec![
            make_categorical_organism_with_bias(
                0,
                0,
                1,
                FacingDirection::East,
                &[ActionType::Forward],
                100.0,
                50,
            ),
            make_categorical_organism_with_bias(
                1,
                1,
                0,
                FacingDirection::SouthEast,
                &[ActionType::Forward],
                100.0,
                50,
            ),
        ],
    );

    let moves = move_map(&tick_once(&mut sim));
    assert_eq!(moves.len(), 1);
    assert_eq!(moves.get(&OrganismId(0)), Some(&((0, 1), (1, 1))));
    assert_no_overlap(&sim);
}

#[test]
fn occupied_target_blocks_move_when_occupant_stays() {
    let cfg = categorical_config(6, 2);
    let mut sim = Simulation::new(cfg, 15).expect("simulation should initialize");
    configure_sim(
        &mut sim,
        vec![
            make_categorical_organism(0, 1, 1, FacingDirection::East, &[ActionType::Forward], 50),
            make_categorical_organism(1, 2, 1, FacingDirection::East, &[], 50),
        ],
    );

    let delta = tick_once(&mut sim);
    assert!(delta.moves.is_empty());
    assert_eq!((organism(&sim, 0).q, organism(&sim, 0).r), (1, 1));
    assert_no_overlap(&sim);
}

#[test]
fn categorical_move_cannot_enter_cell_vacated_in_same_tick() {
    let cfg = categorical_config(6, 2);
    let mut sim = Simulation::new(cfg, 16).expect("simulation should initialize");
    configure_sim(
        &mut sim,
        vec![
            make_categorical_organism(0, 1, 1, FacingDirection::East, &[ActionType::Forward], 50),
            make_categorical_organism(1, 2, 1, FacingDirection::East, &[ActionType::Forward], 50),
        ],
    );

    let moves = move_map(&tick_once(&mut sim));
    assert_eq!(moves.get(&OrganismId(0)), None);
    assert_eq!(moves.get(&OrganismId(1)), Some(&((2, 1), (3, 1))));
    assert_no_overlap(&sim);
}

#[test]
fn blocked_tail_blocks_entire_move_dependency_chain() {
    let cfg = categorical_config(7, 3);
    let mut sim = Simulation::new(cfg, 17).expect("simulation should initialize");
    configure_sim(
        &mut sim,
        vec![
            make_categorical_organism(0, 1, 1, FacingDirection::East, &[ActionType::Forward], 50),
            make_categorical_organism(1, 2, 1, FacingDirection::East, &[ActionType::Forward], 50),
            make_categorical_organism(2, 3, 1, FacingDirection::East, &[], 50),
        ],
    );

    let delta = tick_once(&mut sim);
    assert!(delta.moves.is_empty());
    assert_eq!((organism(&sim, 0).q, organism(&sim, 0).r), (1, 1));
    assert_eq!((organism(&sim, 1).q, organism(&sim, 1).r), (2, 1));
    assert_no_overlap(&sim);
}

#[test]
fn categorical_two_organism_swap_is_blocked() {
    let cfg = categorical_config(6, 2);
    let mut sim = Simulation::new(cfg, 18).expect("simulation should initialize");
    configure_sim(
        &mut sim,
        vec![
            make_categorical_organism(0, 1, 1, FacingDirection::East, &[ActionType::Forward], 50),
            make_categorical_organism(1, 2, 1, FacingDirection::West, &[ActionType::Forward], 50),
        ],
    );

    let moves = move_map(&tick_once(&mut sim));
    assert!(moves.is_empty());
    assert_no_overlap(&sim);
}

#[test]
fn categorical_three_organism_move_cycle_is_blocked() {
    let cfg = categorical_config(6, 3);
    let mut sim = Simulation::new(cfg, 19).expect("simulation should initialize");
    configure_sim(
        &mut sim,
        vec![
            make_categorical_organism(0, 1, 1, FacingDirection::East, &[ActionType::Forward], 50),
            make_categorical_organism(
                1,
                2,
                1,
                FacingDirection::SouthWest,
                &[ActionType::Forward],
                50,
            ),
            make_categorical_organism(
                2,
                1,
                2,
                FacingDirection::NorthWest,
                &[ActionType::Forward],
                50,
            ),
        ],
    );

    let moves = move_map(&tick_once(&mut sim));
    assert!(moves.is_empty());
    assert_no_overlap(&sim);
}

#[test]
fn move_resolution_blocks_wall_cells() {
    let cfg = categorical_config(5, 1);
    let mut sim = Simulation::new(cfg, 20).expect("simulation should initialize");
    let wall_idx = sim.cell_index(2, 1);
    sim.terrain_map[wall_idx] = true;
    configure_sim(
        &mut sim,
        vec![make_categorical_organism(
            0,
            1,
            1,
            FacingDirection::East,
            &[ActionType::Forward],
            50,
        )],
    );

    let delta = tick_once(&mut sim);
    assert!(delta.moves.is_empty());
    assert_eq!(sim.occupancy[wall_idx], Some(Occupant::Wall));
    assert_eq!((organism(&sim, 0).q, organism(&sim, 0).r), (1, 1));
}

#[test]
fn categorical_attack_records_escape_and_still_charges_attempt() {
    let cfg = competitive_config(6, 2);
    let mut sim = Simulation::new(cfg, 21).expect("simulation should initialize");
    configure_sim(
        &mut sim,
        vec![
            make_categorical_organism(0, 1, 1, FacingDirection::East, &[ActionType::Attack], 50),
            make_categorical_organism(1, 2, 1, FacingDirection::East, &[ActionType::Forward], 50),
        ],
    );

    let delta = tick_once(&mut sim);
    assert_eq!(
        move_map(&delta).get(&OrganismId(1)),
        Some(&((2, 1), (3, 1)))
    );
    assert_eq!(sim.attack_events_last_turn().len(), 1);
    let event = sim.attack_events_last_turn()[0];
    assert_eq!(event.outcome, AttackOutcome::TargetEvaded);
    assert_eq!(event.victim_id, Some(OrganismId(1)));
    assert_eq!(event.energy_transferred, 0);
    assert_eq!(event.attacker_energy_cost, 1);
    assert_eq!(organism(&sim, 0).energy, 48);
    assert_eq!(organism(&sim, 1).energy, 49);
}

#[test]
fn categorical_forward_does_not_also_attack_or_enter_an_occupied_snapshot_cell() {
    let cfg = categorical_config(7, 2);
    let mut sim = Simulation::new(cfg, 22).expect("simulation should initialize");
    configure_sim(
        &mut sim,
        vec![
            make_categorical_organism(
                0,
                1,
                1,
                FacingDirection::East,
                &[ActionType::Forward, ActionType::Attack],
                50,
            ),
            make_categorical_organism(1, 2, 1, FacingDirection::East, &[ActionType::Forward], 50),
        ],
    );

    let delta = tick_once(&mut sim);
    let moves = move_map(&delta);
    assert_eq!(moves.get(&OrganismId(0)), None);
    assert_eq!(moves.get(&OrganismId(1)), Some(&((2, 1), (3, 1))));
    assert!(sim.attack_events_last_turn().is_empty());
    assert_eq!(organism(&sim, 0).energy, 49);
    assert_eq!(organism(&sim, 1).energy, 49);
}

#[test]
fn categorical_turn_does_not_also_move_or_attack() {
    let cfg = categorical_config(7, 2);
    let mut sim = Simulation::new(cfg, 23).expect("simulation should initialize");
    configure_sim(
        &mut sim,
        vec![
            make_categorical_organism(
                0,
                1,
                1,
                FacingDirection::East,
                &[
                    ActionType::TurnRight,
                    ActionType::Forward,
                    ActionType::Attack,
                ],
                50,
            ),
            make_categorical_organism(1, 1, 3, FacingDirection::East, &[], 50),
        ],
    );

    let delta = tick_once(&mut sim);
    assert!(move_map(&delta).is_empty());
    assert_eq!(organism(&sim, 0).facing, FacingDirection::SouthEast);
    assert!(sim.attack_events_last_turn().is_empty());
    assert_eq!(organism(&sim, 0).energy, 49);
    assert_eq!(organism(&sim, 1).energy, 49);
}

#[test]
fn mutual_nonlethal_attacks_both_resolve_and_conserve_transfer_energy() {
    let cfg = competitive_config(6, 2);
    let mut sim = Simulation::new(cfg, 24).expect("simulation should initialize");
    configure_sim(
        &mut sim,
        vec![
            make_categorical_organism(0, 1, 1, FacingDirection::East, &[ActionType::Attack], 50),
            make_categorical_organism(1, 2, 1, FacingDirection::West, &[ActionType::Attack], 50),
        ],
    );

    let delta = tick_once(&mut sim);
    assert_eq!(delta.metrics.predations_last_turn, 2);
    assert_eq!(sim.attack_events_last_turn().len(), 2);
    assert!(sim
        .attack_events_last_turn()
        .iter()
        .all(|event| event.outcome == AttackOutcome::NonlethalHit));
    assert_eq!(organism(&sim, 0).energy, 48);
    assert_eq!(organism(&sim, 1).energy, 48);
    assert_eq!(
        delta.metrics.energy_ledger_last_turn.attack_transfer_energy,
        40.0
    );
    assert_eq!(
        delta.metrics.energy_ledger_last_turn.attack_attempt_cost,
        2.0
    );
    assert_eq!(delta.metrics.energy_ledger_last_turn.total_residual, 0.0);
}

#[test]
fn lethal_lower_id_attack_cancels_victims_queued_attack() {
    let cfg = competitive_config(6, 2);
    let mut sim = Simulation::new(cfg, 25).expect("simulation should initialize");
    configure_sim(
        &mut sim,
        vec![
            make_categorical_organism(0, 1, 1, FacingDirection::East, &[ActionType::Attack], 50),
            make_categorical_organism(1, 2, 1, FacingDirection::West, &[ActionType::Attack], 20),
        ],
    );

    let delta = tick_once(&mut sim);
    assert_eq!(sim.attack_events_last_turn().len(), 1);
    assert_eq!(
        sim.attack_events_last_turn()[0].outcome,
        AttackOutcome::Killed
    );
    assert_eq!(delta.metrics.predations_last_turn, 1);
    assert_eq!(sim.organisms.len(), 1);
    assert_eq!(organism(&sim, 0).energy, 68);
    assert_eq!(
        delta.metrics.energy_ledger_last_turn.attack_attempt_cost,
        1.0
    );
    assert_eq!(delta.metrics.energy_ledger_last_turn.total_residual, 0.0);
}

#[test]
fn multiple_attackers_share_remaining_victim_energy_in_id_order() {
    let cfg = competitive_config(7, 4);
    let mut sim = Simulation::new(cfg, 26).expect("simulation should initialize");
    configure_sim(
        &mut sim,
        vec![
            make_categorical_organism(0, 1, 2, FacingDirection::East, &[ActionType::Attack], 50),
            make_categorical_organism(
                1,
                2,
                1,
                FacingDirection::SouthEast,
                &[ActionType::Attack],
                50,
            ),
            make_categorical_organism(
                2,
                3,
                1,
                FacingDirection::SouthWest,
                &[ActionType::Attack],
                50,
            ),
            make_categorical_organism(3, 2, 2, FacingDirection::East, &[], 30),
        ],
    );

    let delta = tick_once(&mut sim);
    let events = sim.attack_events_last_turn();
    assert_eq!(events.len(), 3);
    assert_eq!(events[0].outcome, AttackOutcome::NonlethalHit);
    assert_eq!(events[0].energy_transferred, 20);
    assert_eq!(events[1].outcome, AttackOutcome::Killed);
    assert_eq!(events[1].energy_transferred, 10);
    assert_eq!(events[2].outcome, AttackOutcome::NoOrganismTarget);
    assert_eq!(events[2].energy_transferred, 0);
    assert_eq!(events[2].attacker_energy_cost, 1);
    assert_eq!(delta.metrics.predations_last_turn, 2);
    assert_eq!(organism(&sim, 0).energy, 68);
    assert_eq!(organism(&sim, 1).energy, 58);
    assert_eq!(organism(&sim, 2).energy, 48);
    assert_eq!(
        delta.metrics.energy_ledger_last_turn.attack_transfer_energy,
        30.0
    );
    assert_eq!(
        delta.metrics.energy_ledger_last_turn.attack_attempt_cost,
        3.0
    );
    assert_eq!(delta.metrics.energy_ledger_last_turn.total_residual, 0.0);
}

#[test]
fn categorical_collision_winner_moves_without_an_additional_attack() {
    let cfg = categorical_config(7, 3);
    let mut sim = Simulation::new(cfg, 27).expect("simulation should initialize");
    configure_sim(
        &mut sim,
        vec![
            make_categorical_organism(
                0,
                1,
                2,
                FacingDirection::East,
                &[ActionType::Forward, ActionType::Attack],
                50,
            ),
            make_categorical_organism(
                1,
                2,
                1,
                FacingDirection::SouthEast,
                &[ActionType::Forward, ActionType::Attack],
                50,
            ),
            make_categorical_organism(2, 3, 2, FacingDirection::East, &[], 50),
        ],
    );

    let delta = tick_once(&mut sim);
    let moves = move_map(&delta);
    assert_eq!(moves.len(), 1);
    assert_eq!(moves.get(&OrganismId(0)), Some(&((1, 2), (2, 2))));
    let events = sim.attack_events_last_turn();
    assert!(events.is_empty());
    assert_eq!(organism(&sim, 0).energy, 49);
    assert_eq!(organism(&sim, 1).energy, 49);
    assert_eq!(organism(&sim, 2).energy, 49);
    assert_no_overlap(&sim);
}
