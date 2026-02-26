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
fn food_regrows_from_scheduled_event() {
    let mut cfg = test_config(5, 1);
    cfg.food_regrowth_interval = 5;
    cfg.food_regrowth_jitter = 0;
    cfg.food_fertility_threshold = 0.0;

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

    assert!(sim.foods.is_empty());
    let regrow_idx = sim.cell_index(4, 4);
    sim.schedule_food_regrowth(regrow_idx);

    let mut spawned_turn = None;
    for _ in 0..8 {
        let delta = tick_once(&mut sim);
        if delta
            .food_spawned
            .iter()
            .any(|food| (food.q, food.r) == (4, 4))
        {
            spawned_turn = Some(delta.turn);
            break;
        }
    }
    assert!(
        spawned_turn.is_some(),
        "scheduled regrowth event should spawn food on the target fertile tile",
    );
    assert_eq!(spawned_turn, Some(6));
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
    let cfg = test_config(5, 1);
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
fn disabling_runtime_plasticity_keeps_weights_traces_and_dopamine_static() {
    let mut cfg = test_config(5, 1);
    cfg.runtime_plasticity_enabled = false;
    let mut sim = Simulation::new(cfg, 302).expect("simulation should initialize");

    let mut organism = make_organism(
        0,
        1,
        1,
        FacingDirection::East,
        false,
        false,
        false,
        0.9,
        100.0,
    );
    organism.genome.hebb_eta_gain = 0.5;
    organism.genome.synapse_prune_threshold = 0.5;
    organism.genome.age_of_maturity = 0;
    for synapse in &mut organism.brain.inter[0].synapses {
        synapse.weight = 0.2;
        synapse.eligibility = 0.0;
        synapse.pending_coactivation = 0.0;
    }
    let initial_weights: Vec<f32> = organism.brain.inter[0]
        .synapses
        .iter()
        .map(|synapse| synapse.weight)
        .collect();
    let initial_synapse_count = organism.brain.synapse_count;
    let initial_energy_prev = organism.energy_prev;
    configure_sim(&mut sim, vec![organism]);

    let _ = tick_once(&mut sim);
    let organism = sim
        .organisms
        .iter()
        .find(|item| item.id == OrganismId(0))
        .expect("organism should exist");

    let after_weights: Vec<f32> = organism.brain.inter[0]
        .synapses
        .iter()
        .map(|synapse| synapse.weight)
        .collect();
    assert_eq!(after_weights, initial_weights);
    assert_eq!(organism.brain.synapse_count, initial_synapse_count);
    assert!(
        organism.brain.inter[0]
            .synapses
            .iter()
            .all(|synapse| synapse.eligibility == 0.0 && synapse.pending_coactivation == 0.0),
        "plasticity-off mode should not compute eligibility or pending coactivations"
    );
    assert_eq!(organism.dopamine, 0.0);
    assert_eq!(organism.energy_prev, initial_energy_prev);
}
