use super::support::*;
use super::*;
use crate::brain::{ACTION_ID_BASE, INTER_ID_BASE};
use crate::grid::rotate_left;

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

    let spawned = sim.resolve_spawn_requests(&[
        reproduction_request_at(&sim, OrganismId(0), 1, 1),
        reproduction_request_at(&sim, OrganismId(1), 1, 1),
    ]);

    assert_eq!(spawned.len(), 1);
    let child = sim
        .organisms
        .iter()
        .find(|organism| organism.id == OrganismId(3))
        .expect("first spawn request should consume final empty slot");
    assert_eq!((child.q, child.r), (1, 1));
    assert_eq!(child.generation, 1);
}

#[test]
fn reproduction_offspring_behavior_starts_from_genome_not_parent_runtime_state() {
    let cfg = test_config(8, 1);
    let mut sim = Simulation::new(cfg, 31).expect("simulation should initialize");
    let mut parent =
        make_single_action_organism(0, 3, 3, FacingDirection::East, ActionType::Idle, 0.8, 500.0);
    parent.genome.inter_biases = vec![0.9];
    parent.genome.inter_log_time_constants = vec![0.0];
    parent.genome.edges = ActionType::ALL
        .iter()
        .copied()
        .enumerate()
        .map(|(idx, action_type)| SynapseEdge {
            pre_neuron_id: NeuronId(INTER_ID_BASE),
            post_neuron_id: NeuronId(ACTION_ID_BASE + idx as u32),
            weight: if action_type == ActionType::TurnLeft {
                8.0
            } else {
                -8.0
            },
            eligibility: 0.0,
            pending_coactivation: 0.0,
        })
        .collect();
    parent.genome.num_synapses = parent.genome.edges.len() as u32;
    parent.brain.inter[0].neuron.activation = 1.0;
    parent.brain.inter[0].state = 1.0;
    for action_neuron in &mut parent.brain.action {
        action_neuron.logit = if action_neuron.action_type == ActionType::Forward {
            10.0
        } else {
            -10.0
        };
    }
    configure_sim(&mut sim, vec![parent]);

    let spawned =
        sim.resolve_spawn_requests(&[reproduction_request_from_parent(&sim, OrganismId(0))]);

    assert_eq!(spawned.len(), 1);
    let child_id = spawned[0].id;
    let initial_facing = spawned[0].facing;

    let delta = tick_once(&mut sim);
    assert!(
        delta.moves.iter().all(|movement| movement.id != child_id),
        "child should turn in place on its first tick",
    );

    let child = sim
        .organisms
        .iter()
        .find(|organism| organism.id == child_id)
        .expect("spawned child should still exist");
    assert_eq!(child.last_action_taken, ActionType::TurnLeft);
    assert_eq!(child.facing, rotate_left(initial_facing));
}
