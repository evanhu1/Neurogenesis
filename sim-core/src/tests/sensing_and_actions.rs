use super::support::test_genome;
use super::*;
use crate::brain::{
    action_index, apply_runtime_plasticity, evaluate_brain, express_genome, scan_rays,
    BrainScratch, ACTION_COUNT_U32, ACTION_ID_BASE, INTER_ID_BASE, SENSORY_COUNT,
};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

fn loc(x: f32, y: f32) -> BrainLocation {
    BrainLocation { x, y }
}

fn post_location(brain: &BrainState, post_id: NeuronId) -> (f32, f32) {
    if let Some(inter) = brain.inter.iter().find(|n| n.neuron.neuron_id == post_id) {
        return (inter.neuron.x, inter.neuron.y);
    }
    if let Some(action) = brain.action.iter().find(|n| n.neuron.neuron_id == post_id) {
        return (action.neuron.x, action.neuron.y);
    }
    panic!("missing post neuron for synapse target {:?}", post_id);
}

fn mean_synapse_distance(brain: &BrainState) -> f32 {
    let mut total = 0.0;
    let mut count = 0usize;

    for sensory in &brain.sensory {
        let pre_x = sensory.neuron.x;
        let pre_y = sensory.neuron.y;
        for edge in &sensory.synapses {
            let (post_x, post_y) = post_location(brain, edge.post_neuron_id);
            total += ((pre_x - post_x).powi(2) + (pre_y - post_y).powi(2)).sqrt();
            count += 1;
        }
    }

    for inter in &brain.inter {
        let pre_x = inter.neuron.x;
        let pre_y = inter.neuron.y;
        for edge in &inter.synapses {
            let (post_x, post_y) = post_location(brain, edge.post_neuron_id);
            total += ((pre_x - post_x).powi(2) + (pre_y - post_y).powi(2)).sqrt();
            count += 1;
        }
    }

    if count == 0 {
        0.0
    } else {
        total / count as f32
    }
}

fn dense_edges(
    num_neurons: u32,
    inter_types: &[InterNeuronType],
    target: usize,
) -> Vec<SynapseEdge> {
    let mut edges = Vec::new();

    for pre_sensory in 0..SENSORY_COUNT {
        for post_inter in 0..num_neurons {
            edges.push(SynapseEdge {
                pre_neuron_id: NeuronId(pre_sensory),
                post_neuron_id: NeuronId(INTER_ID_BASE + post_inter),
                weight: 0.25,
                eligibility: 0.0,
            });
            if edges.len() == target {
                return edges;
            }
        }
        for post_action in 0..ACTION_COUNT_U32 {
            edges.push(SynapseEdge {
                pre_neuron_id: NeuronId(pre_sensory),
                post_neuron_id: NeuronId(ACTION_ID_BASE + post_action),
                weight: 0.25,
                eligibility: 0.0,
            });
            if edges.len() == target {
                return edges;
            }
        }
    }

    for pre_inter in 0..num_neurons {
        let pre_id = NeuronId(INTER_ID_BASE + pre_inter);
        let sign = match inter_types
            .get(pre_inter as usize)
            .copied()
            .unwrap_or(InterNeuronType::Excitatory)
        {
            InterNeuronType::Excitatory => 1.0,
            InterNeuronType::Inhibitory => -1.0,
        };
        for post_inter in 0..num_neurons {
            if pre_inter == post_inter {
                continue;
            }
            edges.push(SynapseEdge {
                pre_neuron_id: pre_id,
                post_neuron_id: NeuronId(INTER_ID_BASE + post_inter),
                weight: 0.25 * sign,
                eligibility: 0.0,
            });
            if edges.len() == target {
                return edges;
            }
        }
        for post_action in 0..ACTION_COUNT_U32 {
            edges.push(SynapseEdge {
                pre_neuron_id: pre_id,
                post_neuron_id: NeuronId(ACTION_ID_BASE + post_action),
                weight: 0.25 * sign,
                eligibility: 0.0,
            });
            if edges.len() == target {
                return edges;
            }
        }
    }

    edges
}

#[test]
fn express_genome_uses_stored_synapse_topology() {
    let mut genome = test_genome();
    genome.num_neurons = 4;
    genome.inter_biases = vec![0.1; 4];
    genome.inter_log_taus = vec![0.0; 4];
    genome.interneuron_types = vec![InterNeuronType::Excitatory; 4];
    genome.inter_locations = (0..4).map(|i| loc(i as f32, 10.0 - i as f32)).collect();
    genome.sensory_locations = vec![loc(0.0, 0.0); SENSORY_COUNT as usize];
    genome.action_locations = (0..ActionType::ALL.len())
        .map(|i| loc(8.0, 1.0 + i as f32))
        .collect();
    genome.edges = dense_edges(genome.num_neurons, &genome.interneuron_types, 20);
    genome.num_synapses = genome.edges.len() as u32;

    let mut rng_a = ChaCha8Rng::seed_from_u64(11);
    let mut rng_b = ChaCha8Rng::seed_from_u64(19);
    let brain_a = express_genome(&genome, &mut rng_a);
    let brain_b = express_genome(&genome, &mut rng_b);

    assert_eq!(brain_a.synapse_count, 20);
    assert_eq!(brain_a.synapse_count, brain_b.synapse_count);
    assert_eq!(brain_a.sensory, brain_b.sensory);
    assert_eq!(brain_a.inter, brain_b.inter);
    assert_eq!(brain_a.action, brain_b.action);

    assert_eq!(brain_a.sensory[0].neuron.x, 0.0);
    assert_eq!(brain_a.sensory[0].neuron.y, 0.0);
    let reproduce_idx = action_index(ActionType::Reproduce);
    assert_eq!(brain_a.action[reproduce_idx].neuron.x, 8.0);
    assert_eq!(brain_a.action[reproduce_idx].neuron.y, 8.0);
}

#[test]
fn synapse_addition_uses_spatial_prior() {
    let mut genome_template = test_genome();
    genome_template.num_neurons = 12;
    genome_template.num_synapses = 8;
    genome_template.inter_biases = vec![0.0; 12];
    genome_template.inter_log_taus = vec![0.0; 12];
    genome_template.interneuron_types = vec![InterNeuronType::Excitatory; 12];
    genome_template.sensory_locations = vec![loc(0.0, 0.0); SENSORY_COUNT as usize];
    genome_template.inter_locations = (0..12).map(|i| loc(1.0 + i as f32 * 0.7, 5.0)).collect();
    genome_template.action_locations = (0..ActionType::ALL.len())
        .map(|i| loc(1.0 + i as f32 * 0.7, 9.0))
        .collect();
    genome_template.mutation_rate_num_synapses = 0.0;
    genome_template.mutation_rate_neuron_location = 0.0;

    let mut local_distance_sum = 0.0;
    let mut global_distance_sum = 0.0;
    for seed in 0..32_u64 {
        let mut local_genome = genome_template.clone();
        local_genome.spatial_prior_sigma = 0.25;
        local_genome.edges.clear();
        let mut global_genome = genome_template.clone();
        global_genome.spatial_prior_sigma = 100.0;
        global_genome.edges.clear();

        let mut local_rng = ChaCha8Rng::seed_from_u64(10_000 + seed);
        let mut global_rng = ChaCha8Rng::seed_from_u64(10_000 + seed);
        crate::genome::mutate_genome(&mut local_genome, &mut local_rng);
        crate::genome::mutate_genome(&mut global_genome, &mut global_rng);
        assert!(local_genome.num_synapses > 0);
        assert!(global_genome.num_synapses > 0);

        let local_brain = express_genome(&local_genome, &mut local_rng);
        let global_brain = express_genome(&global_genome, &mut global_rng);

        local_distance_sum += mean_synapse_distance(&local_brain);
        global_distance_sum += mean_synapse_distance(&global_brain);
    }

    let local_mean_distance = local_distance_sum / 32.0;
    let global_mean_distance = global_distance_sum / 32.0;
    assert!(
        local_mean_distance + 0.05 < global_mean_distance,
        "expected stronger local prior to shorten connections; local_mean={local_mean_distance:.3}, global_mean={global_mean_distance:.3}"
    );
}

#[test]
fn express_genome_respects_dale_signs_for_inter_outgoing_synapses() {
    let mut genome = test_genome();
    genome.num_neurons = 2;
    genome.inter_biases = vec![0.0, 0.0];
    genome.inter_log_taus = vec![0.0, 0.0];
    genome.interneuron_types = vec![InterNeuronType::Excitatory, InterNeuronType::Inhibitory];
    genome.inter_locations = vec![loc(5.0, 5.0), loc(5.5, 5.5)];
    genome.edges = vec![
        SynapseEdge {
            pre_neuron_id: NeuronId(INTER_ID_BASE),
            post_neuron_id: NeuronId(ACTION_ID_BASE),
            weight: 0.3,
            eligibility: 0.0,
        },
        SynapseEdge {
            pre_neuron_id: NeuronId(INTER_ID_BASE),
            post_neuron_id: NeuronId(INTER_ID_BASE + 1),
            weight: 0.2,
            eligibility: 0.0,
        },
        SynapseEdge {
            pre_neuron_id: NeuronId(INTER_ID_BASE + 1),
            post_neuron_id: NeuronId(ACTION_ID_BASE + 1),
            weight: 0.4,
            eligibility: 0.0,
        },
        SynapseEdge {
            pre_neuron_id: NeuronId(INTER_ID_BASE + 1),
            post_neuron_id: NeuronId(INTER_ID_BASE),
            weight: 0.4,
            eligibility: 0.0,
        },
    ];
    genome.num_synapses = genome.edges.len() as u32;

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
    genome.action_biases = vec![0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 6.0];
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

    assert_eq!(eval.resolved_actions.selected_action, ActionType::Reproduce);
}

#[test]
fn scan_rays_stops_at_wall_occluders() {
    let world_width = 6;
    let center_idx = 1 * world_width + 1;
    let wall_idx = 1 * world_width + 2;
    let food_idx = 1 * world_width + 3;
    let organism_idx = 1 * world_width + 4;
    let mut occupancy = vec![None; world_width * world_width];
    occupancy[center_idx] = Some(Occupant::Organism(OrganismId(0)));
    occupancy[wall_idx] = Some(Occupant::Wall);
    occupancy[food_idx] = Some(Occupant::Food(FoodId(1)));
    occupancy[organism_idx] = Some(Occupant::Organism(OrganismId(2)));

    let scans = scan_rays(
        (1, 1),
        FacingDirection::East,
        OrganismId(0),
        world_width as i32,
        &occupancy,
        5,
    );
    assert!(
        scans[2].is_none(),
        "the center ray (offset 0) should be occluded by wall",
    );
}

#[test]
fn runtime_plasticity_updates_weights_and_preserves_sign() {
    let mut genome = test_genome();
    genome.num_neurons = 0;
    genome.num_synapses = 0;
    genome.hebb_eta_baseline = 0.1;
    genome.hebb_eta_gain = 0.0;
    genome.synapse_prune_threshold = 0.0;

    let energy_id = SENSORY_COUNT - 1;
    let mut sensory = vec![make_sensory_neuron(
        energy_id,
        SensoryReceptor::Energy,
        loc(1.0, 1.0),
    )];
    sensory[0].synapses.push(SynapseEdge {
        pre_neuron_id: NeuronId(energy_id),
        post_neuron_id: NeuronId(2000),
        weight: 0.2,
        eligibility: 0.0,
    });
    let mut action: Vec<_> = ActionType::ALL
        .into_iter()
        .enumerate()
        .map(|(idx, action_type)| {
            make_action_neuron(
                2000 + idx as u32,
                action_type,
                0.0,
                loc(2.0, 1.0 + idx as f32),
            )
        })
        .collect();
    action[action_index(ActionType::Forward)].neuron.parent_ids = vec![NeuronId(energy_id)];
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
