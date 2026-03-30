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
    let mut cfg = test_config(20, 1);
    cfg.food_regrowth_interval = 5;
    cfg.food_regrowth_jitter = 0;

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

    sim.initialize_food_ecology();
    assert!(sim.foods.is_empty());
    let regrow_idx = sim.cell_index(4, 4);
    sim.food_fertility[regrow_idx] = true;
    let width = sim.config.world_width as usize;
    let regrow_q = (regrow_idx % width) as i32;
    let regrow_r = (regrow_idx / width) as i32;
    sim.schedule_food_regrowth(regrow_idx);

    let mut spawned_turn = None;
    for _ in 0..8 {
        let delta = tick_once(&mut sim);
        if delta
            .food_spawned
            .iter()
            .any(|food| (food.q, food.r) == (regrow_q, regrow_r))
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
fn eat_only_interacts_with_food() {
    let cfg = test_config(5, 2);
    let mut sim = Simulation::new(cfg, 204).expect("simulation should initialize");
    configure_sim(
        &mut sim,
        vec![
            make_single_action_organism(0, 1, 1, FacingDirection::East, ActionType::Eat, 0.9, 50.0),
            make_single_action_organism(1, 2, 1, FacingDirection::East, ActionType::Idle, 0.1, 50.0),
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
            make_single_action_organism(1, 2, 1, FacingDirection::East, ActionType::Idle, 0.1, 50.0),
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
    assert!(sim.organisms.iter().all(|organism| organism.id != OrganismId(1)));
    assert!(
        delta.food_spawned
            .iter()
            .any(|food| food.kind == sim_types::FoodKind::Corpse && (food.q, food.r) == (2, 1))
    );
}

#[test]
fn health_regenerates_slowly_over_time() {
    let cfg = test_config(5, 1);
    let mut sim = Simulation::new(cfg, 207).expect("simulation should initialize");
    let mut organism =
        make_single_action_organism(0, 1, 1, FacingDirection::East, ActionType::Idle, 0.9, 50.0);
    organism.health = 40.0;
    organism.max_health = 50.0;
    configure_sim(&mut sim, vec![organism]);

    let _ = tick_once(&mut sim);
    let organism = sim
        .organisms
        .iter()
        .find(|organism| organism.id == OrganismId(0))
        .expect("organism should survive");
    assert!(organism.health > 40.0);
    assert!(organism.health < organism.max_health);
}

#[test]
fn corpse_eating_returns_more_energy_than_plant_eating() {
    let cfg = test_config(5, 1);

    let plant_energy = {
        let mut sim = Simulation::new(cfg.clone(), 208).expect("simulation should initialize");
        configure_sim(
            &mut sim,
            vec![make_single_action_organism(
                0,
                1,
                1,
                FacingDirection::East,
                ActionType::Eat,
                0.9,
                50.0,
            )],
        );
        sim.foods.push(sim_types::FoodState {
            id: sim_types::FoodId(0),
            q: 2,
            r: 1,
            energy: 100.0,
            kind: sim_types::FoodKind::Plant,
        });
        let food_idx = sim.cell_index(2, 1);
        sim.occupancy[food_idx] = Some(Occupant::Food(sim_types::FoodId(0)));
        sim.next_food_id = 1;
        let _ = tick_once(&mut sim);
        sim.organisms[0].energy
    };

    let corpse_energy = {
        let mut sim = Simulation::new(cfg, 209).expect("simulation should initialize");
        configure_sim(
            &mut sim,
            vec![make_single_action_organism(
                0,
                1,
                1,
                FacingDirection::East,
                ActionType::Eat,
                0.9,
                50.0,
            )],
        );
        sim.foods.push(sim_types::FoodState {
            id: sim_types::FoodId(0),
            q: 2,
            r: 1,
            energy: 100.0,
            kind: sim_types::FoodKind::Corpse,
        });
        let food_idx = sim.cell_index(2, 1);
        sim.occupancy[food_idx] = Some(Occupant::Food(sim_types::FoodId(0)));
        sim.next_food_id = 1;
        let _ = tick_once(&mut sim);
        sim.organisms[0].energy
    };

    assert!(corpse_energy > plant_energy);
    assert!((corpse_energy - plant_energy) >= 50.0);
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
