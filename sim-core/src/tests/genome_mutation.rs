use super::support::test_genome;
use super::*;
use crate::brain::{action_index, ACTION_ID_BASE, ENERGY_SENSORY_ID, INTER_ID_BASE};
use crate::genome::{mutate_add_neuron_split_edge, mutate_remove_neuron};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

#[test]
fn add_neuron_split_edge_inserts_interneuron_between_existing_endpoints() {
    let mut genome = test_genome();
    genome.num_neurons = 0;
    genome.num_synapses = 1;
    genome.inter_biases.clear();
    genome.inter_log_time_constants.clear();
    genome.inter_locations.clear();
    genome.edges = vec![SynapseEdge {
        pre_neuron_id: NeuronId(ENERGY_SENSORY_ID),
        post_neuron_id: NeuronId(ACTION_ID_BASE + action_index(ActionType::Forward) as u32),
        weight: 0.75,
        eligibility: 0.0,
        pending_coactivation: 0.0,
    }];

    let mut rng = ChaCha8Rng::seed_from_u64(7);
    mutate_add_neuron_split_edge(&mut genome, &mut rng);

    assert_eq!(genome.num_neurons, 1);
    assert_eq!(genome.inter_biases.len(), 1);
    assert_eq!(genome.inter_log_time_constants.len(), 1);
    assert_eq!(genome.inter_locations.len(), 1);
    assert_eq!(genome.num_synapses, 2);
    assert_eq!(genome.edges.len(), 2);

    let new_inter_id = NeuronId(INTER_ID_BASE);
    assert!(genome.edges.iter().any(|edge| {
        edge.pre_neuron_id == NeuronId(ENERGY_SENSORY_ID) && edge.post_neuron_id == new_inter_id
    }));
    assert!(genome.edges.iter().any(|edge| {
        edge.pre_neuron_id == new_inter_id
            && edge.post_neuron_id
                == NeuronId(ACTION_ID_BASE + action_index(ActionType::Forward) as u32)
    }));
}

#[test]
fn remove_neuron_drops_incident_edges_and_reindexes_remaining_inters() {
    let mut genome = test_genome();
    genome.num_neurons = 2;
    genome.num_synapses = 4;
    genome.inter_biases = vec![0.1, -0.2];
    genome.inter_log_time_constants = vec![0.0, 0.1];
    genome.inter_locations = vec![
        BrainLocation { x: 1.0, y: 1.0 },
        BrainLocation { x: 2.0, y: 2.0 },
    ];
    genome.edges = vec![
        SynapseEdge {
            pre_neuron_id: NeuronId(ENERGY_SENSORY_ID),
            post_neuron_id: NeuronId(INTER_ID_BASE),
            weight: 0.5,
            eligibility: 0.0,
            pending_coactivation: 0.0,
        },
        SynapseEdge {
            pre_neuron_id: NeuronId(INTER_ID_BASE),
            post_neuron_id: NeuronId(ACTION_ID_BASE + action_index(ActionType::Forward) as u32),
            weight: 0.6,
            eligibility: 0.0,
            pending_coactivation: 0.0,
        },
        SynapseEdge {
            pre_neuron_id: NeuronId(ENERGY_SENSORY_ID),
            post_neuron_id: NeuronId(INTER_ID_BASE + 1),
            weight: 0.7,
            eligibility: 0.0,
            pending_coactivation: 0.0,
        },
        SynapseEdge {
            pre_neuron_id: NeuronId(INTER_ID_BASE + 1),
            post_neuron_id: NeuronId(ACTION_ID_BASE + action_index(ActionType::Eat) as u32),
            weight: 0.8,
            eligibility: 0.0,
            pending_coactivation: 0.0,
        },
    ];

    let mut rng = ChaCha8Rng::seed_from_u64(0);
    mutate_remove_neuron(&mut genome, &mut rng);

    assert_eq!(genome.num_neurons, 1);
    assert_eq!(genome.inter_biases.len(), 1);
    assert_eq!(genome.inter_log_time_constants.len(), 1);
    assert_eq!(genome.inter_locations.len(), 1);
    assert_eq!(genome.num_synapses, genome.edges.len() as u32);
    assert!(genome.edges.iter().all(|edge| {
        edge.pre_neuron_id.0 != INTER_ID_BASE + 1 && edge.post_neuron_id.0 != INTER_ID_BASE + 1
    }));
}
