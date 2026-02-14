use super::support::test_genome;
use super::*;
use crate::brain::{
    action_index, apply_runtime_plasticity, evaluate_brain, express_genome, BrainScratch,
};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

fn loc(x: f32, y: f32) -> BrainLocation {
    BrainLocation { x, y }
}

#[test]
fn express_genome_samples_fixed_synapse_count_deterministically_for_seed() {
    let mut genome = test_genome();
    genome.num_neurons = 4;
    genome.num_synapses = 20;
    genome.inter_biases = vec![0.1; 8];
    genome.inter_log_taus = vec![0.0; 8];
    genome.interneuron_types = vec![InterNeuronType::Excitatory; 8];
    genome.inter_locations = (0..8).map(|i| loc(i as f32, 10.0 - i as f32)).collect();
    genome.sensory_locations = vec![loc(0.0, 0.0), loc(1.0, 0.0), loc(2.0, 0.0)];
    genome.action_locations = vec![
        loc(8.0, 1.0),
        loc(8.0, 2.0),
        loc(8.0, 3.0),
        loc(8.0, 4.0),
        loc(8.0, 5.0),
    ];

    let mut rng_a = ChaCha8Rng::seed_from_u64(11);
    let mut rng_b = ChaCha8Rng::seed_from_u64(11);
    let brain_a = express_genome(&genome, &mut rng_a);
    let brain_b = express_genome(&genome, &mut rng_b);

    assert_eq!(brain_a.synapse_count, 20);
    assert_eq!(brain_a.synapse_count, brain_b.synapse_count);
    assert_eq!(brain_a.sensory, brain_b.sensory);
    assert_eq!(brain_a.inter, brain_b.inter);
    assert_eq!(brain_a.action, brain_b.action);

    assert_eq!(brain_a.sensory[0].neuron.x, 0.0);
    assert_eq!(brain_a.sensory[0].neuron.y, 0.0);
    assert_eq!(brain_a.action[4].neuron.x, 8.0);
    assert_eq!(brain_a.action[4].neuron.y, 5.0);
}

#[test]
fn express_genome_respects_dale_signs_for_inter_outgoing_synapses() {
    let mut genome = test_genome();
    genome.num_neurons = 2;
    genome.num_synapses = 200;
    genome.inter_biases = vec![0.0, 0.0];
    genome.inter_log_taus = vec![0.0, 0.0];
    genome.interneuron_types = vec![InterNeuronType::Excitatory, InterNeuronType::Inhibitory];
    genome.inter_locations = vec![loc(5.0, 5.0), loc(5.5, 5.5)];

    let mut rng = ChaCha8Rng::seed_from_u64(1234);
    let brain = express_genome(&genome, &mut rng);

    let excitatory = &brain.inter[0];
    let inhibitory = &brain.inter[1];
    assert!(!excitatory.synapses.is_empty());
    assert!(!inhibitory.synapses.is_empty());
    assert!(excitatory.synapses.iter().all(|edge| edge.weight > 0.0));
    assert!(inhibitory.synapses.iter().all(|edge| edge.weight < 0.0));
}

#[test]
fn action_biases_drive_actions_without_incoming_synapses() {
    let mut genome = test_genome();
    genome.num_neurons = 0;
    genome.num_synapses = 0;
    genome.action_biases = vec![5.0, 0.0, 0.0, 6.0, 0.0];
    genome.inter_biases.clear();
    genome.inter_log_taus.clear();
    genome.interneuron_types.clear();
    genome.inter_locations.clear();

    let mut rng = ChaCha8Rng::seed_from_u64(5);
    let brain = express_genome(&genome, &mut rng);
    let mut organism = OrganismState {
        id: OrganismId(0),
        species_id: SpeciesId(0),
        q: 0,
        r: 0,
        age_turns: 0,
        facing: FacingDirection::East,
        energy: 10.0,
        consumptions_count: 0,
        reproductions_count: 0,
        brain,
        genome,
    };

    let occupancy = vec![None; 9];
    let mut scratch = BrainScratch::new();
    let vision_distance = organism.genome.vision_distance;
    let eval = evaluate_brain(&mut organism, 3, &occupancy, vision_distance, &mut scratch);

    assert!(eval.resolved_actions.wants_move);
    assert!(eval.resolved_actions.wants_reproduce);
    assert!(!eval.resolved_actions.wants_consume);
}

#[test]
fn runtime_plasticity_updates_weights_and_preserves_sign() {
    let mut genome = test_genome();
    genome.num_neurons = 0;
    genome.num_synapses = 0;
    genome.hebb_eta_baseline = 0.1;
    genome.hebb_eta_gain = 0.0;
    genome.synapse_prune_threshold = 0.0;

    let mut sensory = vec![make_sensory_neuron(
        2,
        SensoryReceptor::Energy,
        loc(1.0, 1.0),
    )];
    sensory[0].synapses.push(SynapseEdge {
        pre_neuron_id: NeuronId(2),
        post_neuron_id: NeuronId(2000),
        weight: 0.2,
        eligibility: 0.0,
    });
    let mut action = vec![
        make_action_neuron(2000, ActionType::MoveForward, 0.0, loc(2.0, 1.0)),
        make_action_neuron(2001, ActionType::Turn, 0.0, loc(2.0, 2.0)),
        make_action_neuron(2002, ActionType::Consume, 0.0, loc(2.0, 3.0)),
        make_action_neuron(2003, ActionType::Reproduce, 0.0, loc(2.0, 4.0)),
        make_action_neuron(2004, ActionType::Dopamine, 0.0, loc(2.0, 5.0)),
    ];
    action[action_index(ActionType::MoveForward)]
        .neuron
        .parent_ids = vec![NeuronId(2)];
    let brain = BrainState {
        sensory,
        inter: vec![],
        action,
        synapse_count: 1,
    };

    let mut organism = OrganismState {
        id: OrganismId(0),
        species_id: SpeciesId(0),
        q: 0,
        r: 0,
        age_turns: 50,
        facing: FacingDirection::East,
        energy: 100.0,
        consumptions_count: 0,
        reproductions_count: 0,
        brain,
        genome,
    };

    let occupancy = vec![None; 9];
    let mut scratch = BrainScratch::new();
    let vision_distance = organism.genome.vision_distance;
    let _ = evaluate_brain(&mut organism, 3, &occupancy, vision_distance, &mut scratch);

    let before = organism.brain.sensory[0].synapses[0].weight;
    apply_runtime_plasticity(&mut organism, &mut scratch);
    let after = organism.brain.sensory[0].synapses[0].weight;

    assert_ne!(before, after);
    assert!(after > 0.0);
}
