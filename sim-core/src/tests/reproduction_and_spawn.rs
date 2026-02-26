use super::support::*;
use super::*;

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
fn reproduction_offspring_brain_runtime_state_is_reset() {
    let cfg = test_config(8, 1);
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

    let spawned =
        sim.resolve_spawn_requests(&[reproduction_request_from_parent(&sim, OrganismId(0))]);

    assert_eq!(spawned.len(), 1);
    let child = &spawned[0];
    assert!(child
        .brain
        .sensory
        .iter()
        .all(|n| n.neuron.activation == 0.0));
    assert!(child.brain.inter.iter().all(|n| n.neuron.activation == 0.0));
    assert!(child
        .brain
        .action
        .iter()
        .all(|n| n.neuron.activation == 0.0));
    assert_eq!(child.facing, FacingDirection::West);
}

#[test]
fn reproduction_offspring_brain_topology_matches_child_genome_edges() {
    let cfg = test_config(8, 1);
    let mut sim = Simulation::new(cfg, 83).expect("simulation should initialize");
    let mut parent = make_organism(0, 3, 3, FacingDirection::East, true, true, false, 0.8, 0);
    parent.genome.num_synapses = 2;
    parent.genome.edges = vec![
        SynapseEdge {
            pre_neuron_id: NeuronId(0),
            post_neuron_id: NeuronId(1000),
            weight: 0.2,
            eligibility: 0.0,
            pending_coactivation: 0.0,
        },
        SynapseEdge {
            pre_neuron_id: NeuronId(1000),
            post_neuron_id: NeuronId(2000),
            weight: 0.3,
            eligibility: 0.0,
            pending_coactivation: 0.0,
        },
    ];
    configure_sim(&mut sim, vec![parent]);

    let spawned =
        sim.resolve_spawn_requests(&[reproduction_request_from_parent(&sim, OrganismId(0))]);
    assert_eq!(spawned.len(), 1);

    let child = &spawned[0];
    assert_eq!(child.brain.synapse_count, child.genome.edges.len() as u32);

    let mut brain_edges = Vec::new();
    for sensory in &child.brain.sensory {
        for edge in &sensory.synapses {
            brain_edges.push((edge.pre_neuron_id, edge.post_neuron_id, edge.weight));
        }
    }
    for inter in &child.brain.inter {
        for edge in &inter.synapses {
            brain_edges.push((edge.pre_neuron_id, edge.post_neuron_id, edge.weight));
        }
    }
    brain_edges.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));

    let mut genome_edges: Vec<_> = child
        .genome
        .edges
        .iter()
        .map(|edge| (edge.pre_neuron_id, edge.post_neuron_id, edge.weight))
        .collect();
    genome_edges.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));

    assert_eq!(brain_edges, genome_edges);
}
