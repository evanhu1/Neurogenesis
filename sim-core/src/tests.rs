use super::*;
use crate::brain::{
    look_sensor_value, make_action_neuron, make_sensory_neuron, reset_brain_runtime_state,
};
use crate::turn::facing_after_turn;
use sim_protocol::{
    ActionType, BrainState, FacingDirection, InterNeuronState, NeuronId, NeuronState, NeuronType,
    SensoryReceptorType, SynapseEdge,
};
use std::collections::{HashMap, HashSet};

fn test_config(world_width: u32, num_organisms: u32) -> WorldConfig {
    WorldConfig {
        world_width,
        num_organisms,
        num_neurons: 1,
        num_synapses: 0,
        turns_to_starve: 10,
        mutation_chance: 0.0,
        ..WorldConfig::default()
    }
}

fn forced_brain(
    wants_move: bool,
    turn_left: bool,
    turn_right: bool,
    confidence: f32,
) -> BrainState {
    let sensory = vec![make_sensory_neuron(0, SensoryReceptorType::Look)];
    let inter_id = NeuronId(1000);
    let inter_bias = 1.0;
    let inter_synapses = vec![
        SynapseEdge {
            post_neuron_id: NeuronId(2000),
            weight: if wants_move { 8.0 } else { -8.0 },
        },
        SynapseEdge {
            post_neuron_id: NeuronId(2001),
            weight: if turn_left { 8.0 } else { -8.0 },
        },
        SynapseEdge {
            post_neuron_id: NeuronId(2002),
            weight: if turn_right { 8.0 } else { -8.0 },
        },
    ];
    let inter = vec![InterNeuronState {
        neuron: NeuronState {
            neuron_id: inter_id,
            neuron_type: NeuronType::Inter,
            bias: inter_bias,
            activation: confidence,
            is_active: confidence > 0.0,
            parent_ids: Vec::new(),
        },
        synapses: inter_synapses,
    }];
    let mut action = vec![
        make_action_neuron(2000, ActionType::MoveForward),
        make_action_neuron(2001, ActionType::TurnLeft),
        make_action_neuron(2002, ActionType::TurnRight),
    ];
    for action_neuron in &mut action {
        action_neuron.neuron.parent_ids = vec![inter_id];
    }

    BrainState {
        sensory,
        inter,
        action,
        synapse_count: 3,
    }
}

fn make_organism(
    id: u64,
    q: i32,
    r: i32,
    facing: FacingDirection,
    wants_move: bool,
    turn_left: bool,
    turn_right: bool,
    confidence: f32,
    turns_since_last_consumption: u32,
) -> OrganismState {
    OrganismState {
        id: OrganismId(id),
        q,
        r,
        age_turns: 0,
        facing,
        turns_since_last_consumption,
        consumptions_count: 0,
        brain: forced_brain(wants_move, turn_left, turn_right, confidence),
    }
}

fn configure_sim(sim: &mut Simulation, mut organisms: Vec<OrganismState>) {
    organisms.sort_by_key(|organism| organism.id);
    sim.organisms = organisms;
    sim.next_organism_id = sim
        .organisms
        .iter()
        .map(|organism| organism.id.0)
        .max()
        .map_or(0, |max_id| max_id + 1);
    sim.occupancy = vec![None; world_capacity(sim.config.world_width)];
    for organism in &sim.organisms {
        let idx = sim
            .cell_index(organism.q, organism.r)
            .expect("test organism should be in bounds");
        assert!(
            sim.occupancy[idx].is_none(),
            "test setup should not overlap"
        );
        sim.occupancy[idx] = Some(organism.id);
    }
    sim.turn = 0;
    sim.metrics = MetricsSnapshot::default();
    sim.refresh_population_metrics();
}

fn tick_once(sim: &mut Simulation) -> TickDelta {
    sim.step_n(1).into_iter().next().expect("exactly one delta")
}

fn move_map(delta: &TickDelta) -> HashMap<OrganismId, ((i32, i32), (i32, i32))> {
    delta
        .moves
        .iter()
        .map(|movement| (movement.id, (movement.from, movement.to)))
        .collect()
}

fn assert_no_overlap(sim: &Simulation) {
    let mut seen = HashSet::new();
    for organism in &sim.organisms {
        assert!(
            seen.insert((organism.q, organism.r)),
            "organisms should not overlap",
        );
        let idx = sim
            .cell_index(organism.q, organism.r)
            .expect("organism should remain in bounds");
        assert_eq!(sim.occupancy[idx], Some(organism.id));
    }
    assert_eq!(sim.organisms.len(), sim.occupancy.iter().flatten().count());
}

#[test]
fn deterministic_seed() {
    let cfg = WorldConfig::default();
    let mut a = Simulation::new(cfg.clone(), 42).expect("simulation A should initialize");
    let mut b = Simulation::new(cfg, 42).expect("simulation B should initialize");
    a.step_n(30);
    b.step_n(30);
    assert_eq!(
        compare_snapshots(&a.snapshot(), &b.snapshot()),
        Ordering::Equal
    );
}

#[test]
fn fitness_stats_track_mean_median_and_max() {
    let cfg = test_config(4, 4);
    let mut sim = Simulation::new(cfg, 99).expect("simulation should initialize");

    configure_sim(
        &mut sim,
        vec![
            OrganismState {
                age_turns: 2,
                ..make_organism(0, 0, 0, FacingDirection::East, false, false, false, 0.1, 0)
            },
            OrganismState {
                age_turns: 4,
                ..make_organism(1, 1, 0, FacingDirection::East, false, false, false, 0.1, 0)
            },
            OrganismState {
                age_turns: 6,
                ..make_organism(2, 2, 0, FacingDirection::East, false, false, false, 0.1, 0)
            },
            OrganismState {
                age_turns: 10,
                ..make_organism(3, 3, 0, FacingDirection::East, false, false, false, 0.1, 0)
            },
        ],
    );

    let _ = tick_once(&mut sim);
    let fitness = &sim.metrics().fitness;
    assert_eq!(fitness.max_fitness, 11);
    assert!((fitness.mean_fitness - 6.5).abs() < f64::EPSILON);
    assert!((fitness.median_fitness - 6.0).abs() < f64::EPSILON);
}

#[test]
fn different_seed_changes_state() {
    let cfg = WorldConfig::default();
    let mut a = Simulation::new(cfg.clone(), 42).expect("simulation A should initialize");
    let mut b = Simulation::new(cfg, 43).expect("simulation B should initialize");
    a.step_n(10);
    b.step_n(10);
    assert_ne!(
        compare_snapshots(&a.snapshot(), &b.snapshot()),
        Ordering::Equal
    );
}

#[test]
fn config_validation_rejects_zero_world_width() {
    let cfg = WorldConfig {
        world_width: 0,
        ..WorldConfig::default()
    };
    let err = Simulation::new(cfg, 1).expect_err("expected invalid config error");
    assert!(err.to_string().contains("world_width"));
}

#[test]
fn config_validation_rejects_zero_mutation_operations() {
    let cfg = WorldConfig {
        mutation_operations: 0,
        ..WorldConfig::default()
    };
    let err = Simulation::new(cfg, 1).expect_err("expected invalid config error");
    assert!(err.to_string().contains("mutation_operations"));
}

#[test]
fn population_is_capped_by_world_capacity_without_overlap() {
    let cfg = WorldConfig {
        world_width: 3,
        num_organisms: 20,
        num_neurons: 0,
        num_synapses: 0,
        ..WorldConfig::default()
    };
    let sim = Simulation::new(cfg, 3).expect("simulation should initialize");
    assert_eq!(sim.organisms.len(), 9);
    assert_eq!(
        sim.occupancy.iter().filter(|cell| cell.is_some()).count(),
        9
    );
}

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
        sim.occupancy[idx] = Some(org.id);
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
fn move_into_cell_vacated_same_turn_succeeds() {
    let cfg = test_config(5, 2);
    let mut sim = Simulation::new(cfg, 11).expect("simulation should initialize");
    configure_sim(
        &mut sim,
        vec![
            make_organism(0, 1, 1, FacingDirection::East, true, false, false, 0.8, 0),
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
        ],
    );

    let delta = tick_once(&mut sim);
    let moves = move_map(&delta);
    assert_eq!(moves.len(), 2);
    assert_eq!(moves.get(&OrganismId(0)), Some(&((1, 1), (2, 1))));
    assert_eq!(moves.get(&OrganismId(1)), Some(&((2, 1), (2, 2))));
    assert_eq!(delta.metrics.consumptions_last_turn, 0);
}

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
    assert_eq!(moves.get(&OrganismId(0)), Some(&((1, 1), (2, 1))));
    assert_eq!(moves.get(&OrganismId(1)), Some(&((2, 1), (1, 1))));
    assert_eq!(delta.metrics.consumptions_last_turn, 0);
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
    assert_eq!(moves.get(&OrganismId(0)), Some(&((1, 1), (2, 1))));
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
    assert_eq!(moves.len(), 3);
    assert_eq!(moves.get(&OrganismId(0)), Some(&((1, 1), (1, 2))));
    assert_eq!(moves.get(&OrganismId(2)), Some(&((1, 2), (2, 1))));
    assert_eq!(moves.get(&OrganismId(1)), Some(&((2, 1), (1, 1))));
    assert_eq!(delta.metrics.consumptions_last_turn, 0);
}

#[test]
fn contested_occupied_target_where_occupant_remains_uses_consume_path() {
    let cfg = test_config(5, 3);
    let mut sim = Simulation::new(cfg, 17).expect("simulation should initialize");
    configure_sim(
        &mut sim,
        vec![
            make_organism(0, 1, 1, FacingDirection::East, true, false, false, 0.9, 0),
            make_organism(1, 2, 1, FacingDirection::West, false, false, false, 0.1, 0),
            make_organism(
                2,
                1,
                2,
                FacingDirection::NorthEast,
                true,
                false,
                false,
                0.2,
                0,
            ),
        ],
    );

    let delta = tick_once(&mut sim);
    let moves = move_map(&delta);
    assert_eq!(moves.len(), 1);
    assert_eq!(moves.get(&OrganismId(0)), Some(&((1, 1), (2, 1))));
    assert_eq!(delta.metrics.consumptions_last_turn, 1);
    assert!(sim
        .organisms
        .iter()
        .all(|organism| organism.id != OrganismId(1)));
}

#[test]
fn starvation_and_reproduction_interact_in_same_turn() {
    let mut cfg = test_config(6, 4);
    cfg.turns_to_starve = 2;

    let mut sim = Simulation::new(cfg, 18).expect("simulation should initialize");
    configure_sim(
        &mut sim,
        vec![
            make_organism(0, 1, 1, FacingDirection::East, true, false, false, 0.9, 1),
            make_organism(1, 2, 1, FacingDirection::West, false, false, false, 0.1, 0),
            make_organism(2, 0, 0, FacingDirection::East, false, false, false, 0.2, 1),
            make_organism(3, 4, 4, FacingDirection::West, false, false, false, 0.2, 0),
        ],
    );

    let delta = tick_once(&mut sim);
    assert_eq!(delta.metrics.consumptions_last_turn, 1);
    assert_eq!(delta.metrics.starvations_last_turn, 1);
    assert_eq!(delta.metrics.births_last_turn, 2);
    assert_eq!(
        delta
            .removed_positions
            .iter()
            .map(|entry| entry.id)
            .collect::<Vec<_>>(),
        vec![OrganismId(1), OrganismId(2)]
    );
    assert_eq!(delta.spawned.len(), 2);
    assert_eq!(sim.organisms.len(), 4);
    let consumer = sim
        .organisms
        .iter()
        .find(|organism| organism.id == OrganismId(0))
        .expect("consumer should survive");
    assert_eq!(consumer.turns_since_last_consumption, 0);
}

#[test]
fn spawn_queue_order_is_deterministic_under_limited_space() {
    let cfg = test_config(2, 3);
    let mut sim = Simulation::new(cfg, 19).expect("simulation should initialize");
    configure_sim(
        &mut sim,
        vec![
            make_organism(
                0,
                0,
                0,
                FacingDirection::NorthEast,
                true,
                false,
                false,
                0.9,
                0,
            ),
            make_organism(1, 1, 0, FacingDirection::West, false, false, false, 0.1, 0),
            make_organism(2, 0, 1, FacingDirection::East, false, false, false, 0.1, 0),
        ],
    );
    let parent_brain = sim.organisms[0].brain.clone();

    let spawned = sim.resolve_spawn_requests(&[
        SpawnRequest {
            kind: SpawnRequestKind::Reproduction {
                parent: OrganismId(0),
            },
        },
        SpawnRequest {
            kind: SpawnRequestKind::StarvationReplacement,
        },
    ]);

    assert_eq!(spawned.len(), 1);
    let child = sim
        .organisms
        .iter()
        .find(|organism| organism.id == OrganismId(3))
        .expect("first spawn request should consume final empty slot");
    assert_eq!((child.q, child.r), (1, 1));
    assert_ne!(child.brain, parent_brain);
}

#[test]
fn reproduction_offspring_brain_runtime_state_is_reset() {
    let mut cfg = test_config(8, 1);
    cfg.mutation_chance = 0.0;
    let mut sim = Simulation::new(cfg, 31).expect("simulation should initialize");
    configure_sim(
        &mut sim,
        vec![make_organism(
            0,
            3,
            3,
            FacingDirection::East,
            true,
            true,
            false,
            0.8,
            0,
        )],
    );

    let parent = sim
        .organisms
        .iter_mut()
        .find(|organism| organism.id == OrganismId(0))
        .expect("parent should exist");
    parent.brain.sensory[0].neuron.activation = 1.0;
    parent.brain.sensory[0].neuron.is_active = true;
    parent.brain.action[0].neuron.activation = 0.91;
    parent.brain.action[0].neuron.is_active = true;
    parent.brain.action[1].neuron.activation = 0.73;
    parent.brain.action[1].neuron.is_active = true;
    let parent_brain = parent.brain.clone();

    let spawned = sim.resolve_spawn_requests(&[SpawnRequest {
        kind: SpawnRequestKind::Reproduction {
            parent: OrganismId(0),
        },
    }]);

    assert_eq!(spawned.len(), 1);
    let child = &spawned[0];
    let mut expected_child_brain = parent_brain;
    reset_brain_runtime_state(&mut expected_child_brain);
    assert_eq!(child.brain, expected_child_brain);
}

#[test]
fn starvation_spawn_respects_center_min_fraction_when_space_exists() {
    let mut cfg = test_config(31, 1);
    cfg.center_spawn_min_fraction = 0.40;
    cfg.center_spawn_max_fraction = 0.60;
    let mut sim = Simulation::new(cfg.clone(), 29).expect("simulation should initialize");
    configure_sim(&mut sim, vec![]);

    let spawned = sim.resolve_spawn_requests(&[SpawnRequest {
        kind: SpawnRequestKind::StarvationReplacement,
    }]);

    assert_eq!(spawned.len(), 1);
    let child = &spawned[0];
    let center = (cfg.world_width as f64 - 1.0) / 2.0;
    let distance = ((child.q as f64 - center).powi(2) + (child.r as f64 - center).powi(2)).sqrt();
    let min_radius = cfg.world_width as f64 * f64::from(cfg.center_spawn_min_fraction) / 2.0;
    assert!(
        distance + 0.75 >= min_radius,
        "expected spawn distance {distance} to respect min radius {min_radius}"
    );
}

#[test]
fn reproduction_spawn_is_opposite_of_parent_facing() {
    let mut cfg = test_config(40, 1);
    cfg.center_spawn_min_fraction = 0.45;
    cfg.center_spawn_max_fraction = 0.55;
    cfg.mutation_chance = 0.0;
    let mut sim = Simulation::new(cfg, 30).expect("simulation should initialize");
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
            0.2,
            0,
        )],
    );

    let spawned = sim.resolve_spawn_requests(&[SpawnRequest {
        kind: SpawnRequestKind::Reproduction {
            parent: OrganismId(0),
        },
    }]);
    assert_eq!(spawned.len(), 1);
    let child = &spawned[0];
    assert_eq!((child.q, child.r), (0, 1));
}

#[test]
fn no_overlap_invariant_holds_after_mixed_turn() {
    let mut cfg = test_config(6, 4);
    cfg.turns_to_starve = 2;

    let mut sim = Simulation::new(cfg, 20).expect("simulation should initialize");
    configure_sim(
        &mut sim,
        vec![
            make_organism(0, 1, 1, FacingDirection::East, true, false, false, 0.9, 1),
            make_organism(1, 2, 1, FacingDirection::West, false, false, false, 0.1, 0),
            make_organism(2, 0, 0, FacingDirection::East, false, false, false, 0.2, 1),
            make_organism(3, 4, 4, FacingDirection::West, true, false, false, 0.4, 0),
        ],
    );

    let _ = tick_once(&mut sim);
    assert_no_overlap(&sim);
}

#[test]
fn targeted_complex_resolution_snapshot_is_deterministic() {
    let mut cfg = test_config(6, 4);
    cfg.turns_to_starve = 3;

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
