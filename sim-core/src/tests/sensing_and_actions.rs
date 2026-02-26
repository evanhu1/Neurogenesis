use super::support::test_genome;
use super::*;
use crate::brain::{
    action_index, evaluate_brain, express_genome, scan_rays, ActionSelectionPolicy, BrainScratch,
    ACTION_COUNT_U32, ACTION_ID_BASE, INTER_ID_BASE, SENSORY_COUNT,
};
use crate::genome::{BRAIN_SPACE_MAX, BRAIN_SPACE_MIN};
use crate::plasticity::{apply_runtime_weight_updates, compute_pending_coactivations};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

fn loc(x: f32, y: f32) -> BrainLocation {
    BrainLocation { x, y }
}

fn deterministic_action_policy() -> ActionSelectionPolicy {
    ActionSelectionPolicy { temperature: 0.5 }
}

fn simple_action_bias_organism(
    forward_bias: f32,
    reproduce_bias: f32,
    energy: f32,
) -> OrganismState {
    let mut genome = test_genome();
    genome.num_neurons = 0;
    genome.num_synapses = 0;
    genome.action_biases = vec![0.0; ActionType::ALL.len()];
    genome.action_biases[action_index(ActionType::Forward)] = forward_bias;
    genome.action_biases[action_index(ActionType::Reproduce)] = reproduce_bias;
    genome.inter_biases.clear();
    genome.inter_log_time_constants.clear();
    genome.inter_locations.clear();

    let mut rng = ChaCha8Rng::seed_from_u64(55);
    let brain = express_genome(&genome, &mut rng);
    OrganismState {
        id: OrganismId(0),
        species_id: SpeciesId(0),
        q: 0,
        r: 0,
        generation: 0,
        age_turns: 0,
        facing: FacingDirection::East,
        energy,
        energy_prev: energy,
        dopamine: 0.0,
        consumptions_count: 0,
        reproductions_count: 0,
        last_action_taken: ActionType::Idle,
        brain,
        genome,
    }
}

fn dense_edges(num_neurons: u32, target: usize) -> Vec<SynapseEdge> {
    let mut edges = Vec::new();

    for pre_sensory in 0..SENSORY_COUNT {
        for post_inter in 0..num_neurons {
            edges.push(SynapseEdge {
                pre_neuron_id: NeuronId(pre_sensory),
                post_neuron_id: NeuronId(INTER_ID_BASE + post_inter),
                weight: 0.25,
                eligibility: 0.0,
                pending_coactivation: 0.0,
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
                pending_coactivation: 0.0,
            });
            if edges.len() == target {
                return edges;
            }
        }
    }

    for pre_inter in 0..num_neurons {
        let pre_id = NeuronId(INTER_ID_BASE + pre_inter);
        for post_inter in 0..num_neurons {
            if pre_inter == post_inter {
                continue;
            }
            edges.push(SynapseEdge {
                pre_neuron_id: pre_id,
                post_neuron_id: NeuronId(INTER_ID_BASE + post_inter),
                weight: 0.25,
                eligibility: 0.0,
                pending_coactivation: 0.0,
            });
            if edges.len() == target {
                return edges;
            }
        }
        for post_action in 0..ACTION_COUNT_U32 {
            edges.push(SynapseEdge {
                pre_neuron_id: pre_id,
                post_neuron_id: NeuronId(ACTION_ID_BASE + post_action),
                weight: 0.25,
                eligibility: 0.0,
                pending_coactivation: 0.0,
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
    genome.inter_log_time_constants = vec![0.0; 4];
    genome.inter_locations = (0..4).map(|i| loc(i as f32, 10.0 - i as f32)).collect();
    genome.sensory_locations = vec![loc(0.0, 0.0); SENSORY_COUNT as usize];
    genome.action_locations = (0..ActionType::ALL.len())
        .map(|i| loc(8.0, 1.0 + i as f32))
        .collect();
    genome.edges = dense_edges(genome.num_neurons, 20);
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

    assert_eq!(brain_a.sensory[0].neuron.x, BRAIN_SPACE_MIN);
    assert_eq!(
        brain_a.sensory[0].neuron.y,
        0.5 * (BRAIN_SPACE_MIN + BRAIN_SPACE_MAX)
    );
    let reproduce_idx = action_index(ActionType::Reproduce);
    assert_eq!(brain_a.action[reproduce_idx].neuron.x, BRAIN_SPACE_MAX);
    assert_eq!(
        brain_a.action[reproduce_idx].neuron.y,
        0.5 * (BRAIN_SPACE_MIN + BRAIN_SPACE_MAX)
    );
}

#[test]
fn mutate_genome_adds_synapses_when_below_target() {
    let mut genome_template = test_genome();
    genome_template.num_neurons = 12;
    genome_template.num_synapses = 8;
    genome_template.inter_biases = vec![0.0; 12];
    genome_template.inter_log_time_constants = vec![0.0; 12];
    genome_template.sensory_locations = vec![loc(0.0, 0.0); SENSORY_COUNT as usize];
    genome_template.inter_locations = (0..12).map(|i| loc(1.0 + i as f32 * 0.7, 5.0)).collect();
    genome_template.action_locations = (0..ActionType::ALL.len())
        .map(|i| loc(1.0 + i as f32 * 0.7, 9.0))
        .collect();
    genome_template.mutation_rate_neuron_location = 0.0;

    let mut genome = genome_template.clone();
    genome.edges.clear();
    let mut rng = ChaCha8Rng::seed_from_u64(10_000);
    crate::genome::mutate_genome(&mut genome, 1.0, &mut rng);

    assert_eq!(genome.num_synapses, 8);
    assert_eq!(genome.edges.len(), 8);
}

#[test]
fn action_biases_drive_actions_without_incoming_synapses() {
    let mut organism = simple_action_bias_organism(5.0, 6.0, 10.0);

    let occupancy = vec![None; 9];
    let mut scratch = BrainScratch::new();
    let vision_distance = organism.genome.vision_distance;
    let eval = evaluate_brain(
        &mut organism,
        3,
        &occupancy,
        vision_distance,
        deterministic_action_policy(),
        0.5,
        &mut scratch,
    );

    assert_eq!(eval.resolved_actions.selected_action, ActionType::Reproduce);
}

#[test]
fn stochastic_action_selection_is_seed_deterministic() {
    let mut organism_a = simple_action_bias_organism(0.8, 0.7, 10.0);
    let mut organism_b = organism_a.clone();
    let occupancy = vec![None; 9];
    let mut scratch_a = BrainScratch::new();
    let mut scratch_b = BrainScratch::new();
    let vision_distance_a = organism_a.genome.vision_distance;
    let vision_distance_b = organism_b.genome.vision_distance;
    let policy = ActionSelectionPolicy { temperature: 1.5 };

    let eval_a = evaluate_brain(
        &mut organism_a,
        3,
        &occupancy,
        vision_distance_a,
        policy,
        0.25,
        &mut scratch_a,
    );
    let eval_b = evaluate_brain(
        &mut organism_b,
        3,
        &occupancy,
        vision_distance_b,
        policy,
        0.25,
        &mut scratch_b,
    );

    assert_eq!(eval_a.action_logits, eval_b.action_logits);
    assert_eq!(eval_a.resolved_actions, eval_b.resolved_actions);
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
    let center_ray = scans[2].as_ref().expect("center ray should detect wall");
    assert_eq!(center_ray.target, EntityType::Wall);
}

#[test]
fn runtime_plasticity_updates_weights() {
    let mut genome = test_genome();
    genome.num_neurons = 0;
    genome.num_synapses = 0;
    genome.hebb_eta_gain = 0.1;
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
        pending_coactivation: 0.0,
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
        generation: 0,
        age_turns: 50,
        facing: FacingDirection::East,
        energy: 100.0,
        energy_prev: 100.0,
        dopamine: 0.0,
        consumptions_count: 0,
        reproductions_count: 0,
        last_action_taken: ActionType::Idle,
        brain,
        genome,
    };

    let occupancy = vec![None; 9];
    let mut scratch = BrainScratch::new();
    let vision_distance = organism.genome.vision_distance;
    let _ = evaluate_brain(
        &mut organism,
        3,
        &occupancy,
        vision_distance,
        deterministic_action_policy(),
        0.5,
        &mut scratch,
    );

    let before = organism.brain.sensory[0].synapses[0].weight;
    compute_pending_coactivations(&mut organism, &mut scratch);
    apply_runtime_weight_updates(&mut organism, 0.0);
    let after = organism.brain.sensory[0].synapses[0].weight;

    assert_ne!(before, after);
}

#[test]
fn runtime_plasticity_neutralizes_passive_metabolism_for_dopamine() {
    let mut genome = test_genome();
    genome.num_neurons = 0;
    genome.num_synapses = 0;
    genome.hebb_eta_gain = 0.1;
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
        pending_coactivation: 0.0,
    });
    let action: Vec<_> = ActionType::ALL
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
        generation: 0,
        age_turns: 50,
        facing: FacingDirection::East,
        energy: 99.0,
        energy_prev: 100.0,
        dopamine: 0.0,
        consumptions_count: 0,
        reproductions_count: 0,
        last_action_taken: ActionType::Idle,
        brain,
        genome,
    };

    let occupancy = vec![None; 9];
    let mut scratch = BrainScratch::new();
    let vision_distance = organism.genome.vision_distance;
    let _ = evaluate_brain(
        &mut organism,
        3,
        &occupancy,
        vision_distance,
        deterministic_action_policy(),
        0.5,
        &mut scratch,
    );

    let before = organism.brain.sensory[0].synapses[0].weight;
    // Passive drain baseline (1.0) exactly cancels the raw -1.0 energy delta.
    compute_pending_coactivations(&mut organism, &mut scratch);
    apply_runtime_weight_updates(&mut organism, 1.0);
    let after = organism.brain.sensory[0].synapses[0].weight;

    let expected = before * (1.0 - 0.001);
    assert!((after - expected).abs() < 1.0e-6);
}

#[test]
fn inter_recurrent_eligibility_uses_prev_inter_pre_signal_only_for_inter_targets() {
    let mut genome = test_genome();
    genome.num_neurons = 2;
    genome.num_synapses = 0;
    genome.eligibility_retention = 0.0;
    genome.hebb_eta_gain = 0.1;

    let inter0 = InterNeuronState {
        neuron: NeuronState {
            neuron_id: NeuronId(INTER_ID_BASE),
            neuron_type: NeuronType::Inter,
            bias: 0.0,
            x: 0.0,
            y: 0.0,
            activation: 0.8,
            parent_ids: Vec::new(),
        },
        alpha: 1.0,
        synapses: vec![
            SynapseEdge {
                pre_neuron_id: NeuronId(INTER_ID_BASE),
                post_neuron_id: NeuronId(INTER_ID_BASE + 1),
                weight: 1.0,
                eligibility: 0.0,
                pending_coactivation: 0.0,
            },
            SynapseEdge {
                pre_neuron_id: NeuronId(INTER_ID_BASE),
                post_neuron_id: NeuronId(ACTION_ID_BASE),
                weight: 1.0,
                eligibility: 0.0,
                pending_coactivation: 0.0,
            },
        ],
    };
    let inter1 = InterNeuronState {
        neuron: NeuronState {
            neuron_id: NeuronId(INTER_ID_BASE + 1),
            neuron_type: NeuronType::Inter,
            bias: 0.0,
            x: 1.0,
            y: 0.0,
            activation: 0.0,
            parent_ids: Vec::new(),
        },
        alpha: 1.0,
        synapses: Vec::new(),
    };
    let action: Vec<_> = ActionType::ALL
        .into_iter()
        .enumerate()
        .map(|(idx, action_type)| {
            make_action_neuron(
                ACTION_ID_BASE + idx as u32,
                action_type,
                if idx == 0 { 1.5 } else { 0.0 },
                loc(2.0, idx as f32),
            )
        })
        .collect();
    let brain = BrainState {
        sensory: vec![],
        inter: vec![inter0, inter1],
        action,
        synapse_count: 2,
    };
    let mut organism = OrganismState {
        id: OrganismId(0),
        species_id: SpeciesId(0),
        q: 0,
        r: 0,
        generation: 0,
        age_turns: 0,
        facing: FacingDirection::East,
        energy: 100.0,
        energy_prev: 100.0,
        dopamine: 0.0,
        consumptions_count: 0,
        reproductions_count: 0,
        last_action_taken: ActionType::Idle,
        brain,
        genome,
    };

    let occupancy = vec![None; 9];
    let mut scratch = BrainScratch::new();
    let vision_distance = organism.genome.vision_distance;
    let _ = evaluate_brain(
        &mut organism,
        3,
        &occupancy,
        vision_distance,
        deterministic_action_policy(),
        0.5,
        &mut scratch,
    );
    compute_pending_coactivations(&mut organism, &mut scratch);

    let inter1_current = organism.brain.inter[1].neuron.activation;
    let recurrent_pending = organism.brain.inter[0].synapses[0].pending_coactivation;
    let action_pending = organism.brain.inter[0].synapses[1].pending_coactivation;

    let expected_recurrent = 0.8 * inter1_current;
    assert!((recurrent_pending - expected_recurrent).abs() < 1.0e-6);
    assert!(action_pending.abs() < 1.0e-6);
}

#[test]
fn action_target_eligibility_uses_centered_logits_not_sigmoid_activation() {
    let mut genome = test_genome();
    genome.num_neurons = 0;
    genome.num_synapses = 0;
    genome.eligibility_retention = 0.0;
    genome.hebb_eta_gain = 0.1;
    genome.starting_energy = 250.0;

    let energy_id = SENSORY_COUNT - 1;
    let mut sensory = vec![make_sensory_neuron(
        energy_id,
        SensoryReceptor::Energy,
        loc(1.0, 1.0),
    )];
    sensory[0].synapses.push(SynapseEdge {
        pre_neuron_id: NeuronId(energy_id),
        post_neuron_id: NeuronId(ACTION_ID_BASE),
        weight: 1.0,
        eligibility: 0.0,
        pending_coactivation: 0.0,
    });
    let action: Vec<_> = ActionType::ALL
        .into_iter()
        .enumerate()
        .map(|(idx, action_type)| {
            make_action_neuron(
                ACTION_ID_BASE + idx as u32,
                action_type,
                if idx == 0 { 2.0 } else { 0.0 },
                loc(2.0, 1.0 + idx as f32),
            )
        })
        .collect();
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
        generation: 0,
        age_turns: 0,
        facing: FacingDirection::East,
        energy: 250.0,
        energy_prev: 250.0,
        dopamine: 0.0,
        consumptions_count: 0,
        reproductions_count: 0,
        last_action_taken: ActionType::Idle,
        brain,
        genome,
    };

    let occupancy = vec![None; 9];
    let mut scratch = BrainScratch::new();
    let vision_distance = organism.genome.vision_distance;
    let eval = evaluate_brain(
        &mut organism,
        3,
        &occupancy,
        vision_distance,
        deterministic_action_policy(),
        0.5,
        &mut scratch,
    );
    compute_pending_coactivations(&mut organism, &mut scratch);

    let sensory_activation = organism.brain.sensory[0].neuron.activation;
    let edge_pending = organism.brain.sensory[0].synapses[0].pending_coactivation;
    let logit0 = eval.action_logits[0];
    let logit_mean = eval.action_logits.iter().sum::<f32>() / eval.action_logits.len() as f32;
    let expected_centered_logit = sensory_activation * (logit0 - logit_mean);
    let sigmoid_expected = sensory_activation * (1.0 / (1.0 + (-logit0).exp()));

    assert!((edge_pending - expected_centered_logit).abs() < 1.0e-6);
    assert!((edge_pending - sigmoid_expected).abs() > 1.0e-3);
}

#[test]
fn energy_sensor_clamps_and_scales_with_starting_energy() {
    let mut genome = test_genome();
    genome.starting_energy = 250.0;

    let energy_id = SENSORY_COUNT - 1;
    let sensory = vec![make_sensory_neuron(
        energy_id,
        SensoryReceptor::Energy,
        loc(1.0, 1.0),
    )];
    let action: Vec<_> = ActionType::ALL
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
    let brain = BrainState {
        sensory,
        inter: vec![],
        action,
        synapse_count: 0,
    };

    let mut organism = OrganismState {
        id: OrganismId(0),
        species_id: SpeciesId(0),
        q: 0,
        r: 0,
        generation: 0,
        age_turns: 0,
        facing: FacingDirection::East,
        energy: 250.0,
        energy_prev: 250.0,
        dopamine: 0.0,
        consumptions_count: 0,
        reproductions_count: 0,
        last_action_taken: ActionType::Idle,
        brain,
        genome,
    };

    let occupancy = vec![None; 9];
    let mut scratch = BrainScratch::new();
    let vision_distance = organism.genome.vision_distance;

    let _ = evaluate_brain(
        &mut organism,
        3,
        &occupancy,
        vision_distance,
        deterministic_action_policy(),
        0.5,
        &mut scratch,
    );
    assert_eq!(organism.brain.sensory[0].neuron.activation, 0.5);

    organism.energy = 1_000_000.0;
    let _ = evaluate_brain(
        &mut organism,
        3,
        &occupancy,
        vision_distance,
        deterministic_action_policy(),
        0.5,
        &mut scratch,
    );
    assert!(organism.brain.sensory[0].neuron.activation > 0.999_999);

    organism.energy = 0.0;
    let _ = evaluate_brain(
        &mut organism,
        3,
        &occupancy,
        vision_distance,
        deterministic_action_policy(),
        0.5,
        &mut scratch,
    );
    assert_eq!(organism.brain.sensory[0].neuron.activation, 0.0);

    organism.energy = 28.0;
    let _ = evaluate_brain(
        &mut organism,
        3,
        &occupancy,
        vision_distance,
        deterministic_action_policy(),
        0.5,
        &mut scratch,
    );
    assert!(organism.brain.sensory[0].neuron.activation < 0.05);
}
