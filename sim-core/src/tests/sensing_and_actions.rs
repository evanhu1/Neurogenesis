use super::support::test_genome;
use super::*;
use crate::brain::{
    action_index, evaluate_brain, express_genome, scan_rays, select_action_from_logits,
    BrainScratch, ACTION_COUNT, ACTION_COUNT_U32, ACTION_ID_BASE, CONTACT_SENSORY_ID,
    DAMAGE_SENSORY_ID, ENERGY_SENSORY_ID, EXPLICIT_IDLE_LOGIT_BIAS, INTER_ID_BASE, SENSORY_COUNT,
};
use crate::genome::{BRAIN_SPACE_MAX, BRAIN_SPACE_MIN};
use crate::plasticity::{apply_runtime_weight_updates, compute_pending_coactivations};

fn loc(x: f32, y: f32) -> BrainLocation {
    BrainLocation { x, y }
}

fn deterministic_action_policy() -> f32 {
    0.5
}

fn no_spikes(cell_count: usize) -> Vec<bool> {
    vec![false; cell_count]
}

fn simple_weighted_action_organism(
    forward_weight: f32,
    reproduce_weight: f32,
    energy: f32,
) -> OrganismState {
    let mut genome = test_genome();
    genome.num_neurons = 0;
    genome.num_synapses = 2;
    genome.inter_biases.clear();
    genome.inter_log_time_constants.clear();
    genome.inter_locations.clear();
    genome.edges = vec![
        SynapseEdge {
            pre_neuron_id: NeuronId(ENERGY_SENSORY_ID),
            post_neuron_id: NeuronId(ACTION_ID_BASE + action_index(ActionType::Forward) as u32),
            weight: forward_weight,
            eligibility: 0.0,
            pending_coactivation: 0.0,
        },
        SynapseEdge {
            pre_neuron_id: NeuronId(ENERGY_SENSORY_ID),
            post_neuron_id: NeuronId(ACTION_ID_BASE + action_index(ActionType::Reproduce) as u32),
            weight: reproduce_weight,
            eligibility: 0.0,
            pending_coactivation: 0.0,
        },
    ];

    let brain = express_genome(&genome);
    OrganismState {
        id: OrganismId(0),
        species_id: sim_types::SpeciesId(0),
        q: 0,
        r: 0,
        generation: 0,
        age_turns: 0,
        facing: FacingDirection::East,
        energy,
        health: energy.max(1.0),
        max_health: energy.max(1.0),
        energy_prev: energy,
        dopamine: 0.0,
        damage_taken_last_turn: 0.0,
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

    let brain_a = express_genome(&genome);
    let brain_b = express_genome(&genome);

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
    assert_eq!(brain_a.action[reproduce_idx].x, BRAIN_SPACE_MAX);
    assert_eq!(
        brain_a.action[reproduce_idx].y,
        0.5 * (BRAIN_SPACE_MIN + BRAIN_SPACE_MAX)
    );
}

#[test]
fn stochastic_action_selection_is_seed_deterministic() {
    let mut organism_a = simple_weighted_action_organism(0.8, 0.7, 10.0);
    let mut organism_b = organism_a.clone();
    let occupancy = vec![None; 9];
    let spike_map = no_spikes(occupancy.len());
    let mut scratch_a = BrainScratch::new();
    let mut scratch_b = BrainScratch::new();
    let vision_distance_a = organism_a.genome.vision_distance;
    let vision_distance_b = organism_b.genome.vision_distance;
    let action_temperature = 1.5;

    let eval_a = evaluate_brain(
        &mut organism_a,
        3,
        &occupancy,
        &spike_map,
        vision_distance_a,
        action_temperature,
        0.25,
        &mut scratch_a,
    );
    let eval_b = evaluate_brain(
        &mut organism_b,
        3,
        &occupancy,
        &spike_map,
        vision_distance_b,
        action_temperature,
        0.25,
        &mut scratch_b,
    );

    assert_eq!(eval_a.action_logits, eval_b.action_logits);
    assert_eq!(eval_a.selected_action, eval_b.selected_action);
}

#[test]
fn explicit_idle_bias_softmax_is_deterministic_for_equal_logits() {
    let logits = [0.25; ACTION_COUNT];
    let selected_a = select_action_from_logits(logits, EXPLICIT_IDLE_LOGIT_BIAS, 1.0, 0.42);
    let selected_b = select_action_from_logits(logits, EXPLICIT_IDLE_LOGIT_BIAS, 1.0, 0.42);

    assert_eq!(selected_a, selected_b);
}

#[test]
fn equal_action_logits_prefer_real_actions_before_idle_tail() {
    let logits = [0.0; ACTION_COUNT];

    let early_sample = select_action_from_logits(logits, EXPLICIT_IDLE_LOGIT_BIAS, 1.0, 0.10);
    let late_sample = select_action_from_logits(logits, EXPLICIT_IDLE_LOGIT_BIAS, 1.0, 0.99);

    assert_ne!(early_sample, ActionType::Idle);
    assert_eq!(late_sample, ActionType::Idle);
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
    let spike_map = no_spikes(occupancy.len());

    let scans = scan_rays(
        (1, 1),
        FacingDirection::East,
        OrganismId(0),
        world_width as i32,
        &occupancy,
        &spike_map,
        5,
    );
    let center_ray_idx = SensoryReceptor::LOOK_RAY_OFFSETS
        .iter()
        .position(|offset| *offset == 0)
        .expect("center ray offset should exist");
    let center_ray = scans[center_ray_idx];
    assert!(center_ray.wall_signal > 0.0);
    assert_eq!(center_ray.food_signal, 0.0);
    assert_eq!(center_ray.organism_signal, 0.0);
    assert_eq!(center_ray.spike_signal, 0.0);
}

#[test]
fn scan_rays_detect_empty_spike_tiles() {
    let world_width = 6;
    let center_idx = 1 * world_width + 1;
    let spike_idx = 1 * world_width + 2;
    let mut occupancy = vec![None; world_width * world_width];
    let mut spike_map = no_spikes(occupancy.len());
    occupancy[center_idx] = Some(Occupant::Organism(OrganismId(0)));
    spike_map[spike_idx] = true;

    let scans = scan_rays(
        (1, 1),
        FacingDirection::East,
        OrganismId(0),
        world_width as i32,
        &occupancy,
        &spike_map,
        5,
    );
    let center_ray_idx = SensoryReceptor::LOOK_RAY_OFFSETS
        .iter()
        .position(|offset| *offset == 0)
        .expect("center ray offset should exist");
    let center_ray = scans[center_ray_idx];
    assert!(center_ray.spike_signal > 0.0);
    assert_eq!(center_ray.food_signal, 0.0);
    assert_eq!(center_ray.organism_signal, 0.0);
    assert_eq!(center_ray.wall_signal, 0.0);
}

#[test]
fn scan_rays_report_food_and_spikes_for_shared_cell() {
    let world_width = 6;
    let center_idx = 1 * world_width + 1;
    let target_idx = 1 * world_width + 2;
    let mut occupancy = vec![None; world_width * world_width];
    let mut spike_map = no_spikes(occupancy.len());
    occupancy[center_idx] = Some(Occupant::Organism(OrganismId(0)));
    occupancy[target_idx] = Some(Occupant::Food(FoodId(1)));
    spike_map[target_idx] = true;

    let scans = scan_rays(
        (1, 1),
        FacingDirection::East,
        OrganismId(0),
        world_width as i32,
        &occupancy,
        &spike_map,
        5,
    );
    let center_ray_idx = SensoryReceptor::LOOK_RAY_OFFSETS
        .iter()
        .position(|offset| *offset == 0)
        .expect("center ray offset should exist");
    let center_ray = scans[center_ray_idx];
    assert!(center_ray.food_signal > 0.0);
    assert!(center_ray.spike_signal > 0.0);
    assert_eq!(center_ray.organism_signal, 0.0);
    assert_eq!(center_ray.wall_signal, 0.0);
}

#[test]
fn contact_and_damage_sensors_encode_local_state() {
    let mut genome = test_genome();
    genome.num_neurons = 0;
    genome.num_synapses = 0;

    let sensory = vec![
        make_sensory_neuron(
            CONTACT_SENSORY_ID,
            SensoryReceptor::ContactAhead,
            loc(0.0, 0.0),
        ),
        make_sensory_neuron(DAMAGE_SENSORY_ID, SensoryReceptor::Damage, loc(0.0, 1.0)),
        make_sensory_neuron(ENERGY_SENSORY_ID, SensoryReceptor::Energy, loc(0.0, 2.0)),
    ];
    let action: Vec<_> = ActionType::ALL
        .into_iter()
        .copied()
        .enumerate()
        .map(|(idx, action_type)| {
            make_action_neuron(
                ACTION_ID_BASE + idx as u32,
                action_type,
                loc(2.0, idx as f32),
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
        species_id: sim_types::SpeciesId(0),
        q: 1,
        r: 1,
        generation: 0,
        age_turns: 0,
        facing: FacingDirection::East,
        energy: 100.0,
        health: 100.0,
        max_health: 100.0,
        energy_prev: 100.0,
        dopamine: 0.0,
        damage_taken_last_turn: 25.0,
        consumptions_count: 0,
        reproductions_count: 0,
        last_action_taken: ActionType::Idle,
        brain,
        genome,
    };

    let mut occupancy = vec![None; 9];
    occupancy[1 * 3 + 2] = Some(Occupant::Wall);
    let spike_map = no_spikes(occupancy.len());
    let mut scratch = BrainScratch::new();
    let vision_distance = organism.genome.vision_distance;
    let _ = evaluate_brain(
        &mut organism,
        3,
        &occupancy,
        &spike_map,
        vision_distance,
        deterministic_action_policy(),
        0.5,
        &mut scratch,
    );

    assert_eq!(organism.brain.sensory[0].neuron.activation, 1.0);
    assert!(organism.brain.sensory[1].neuron.activation > 0.0);
}

#[test]
fn runtime_plasticity_updates_weights() {
    let mut genome = test_genome();
    genome.num_neurons = 0;
    genome.num_synapses = 0;
    genome.plasticity_start_age = 0;
    genome.juvenile_eta_scale = 1.0;
    genome.max_weight_delta_per_tick = 1.0;
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
        .copied()
        .enumerate()
        .map(|(idx, action_type)| {
            make_action_neuron(2000 + idx as u32, action_type, loc(2.0, 1.0 + idx as f32))
        })
        .collect();
    action[action_index(ActionType::Forward)].parent_ids = vec![NeuronId(energy_id)];
    let brain = BrainState {
        sensory,
        inter: vec![],
        action,
        synapse_count: 1,
    };

    let mut organism = OrganismState {
        id: OrganismId(0),
        species_id: sim_types::SpeciesId(0),
        q: 0,
        r: 0,
        generation: 0,
        age_turns: 50,
        facing: FacingDirection::East,
        energy: 100.0,
        health: 100.0,
        max_health: 100.0,
        energy_prev: 100.0,
        dopamine: 0.0,
        damage_taken_last_turn: 0.0,
        consumptions_count: 0,
        reproductions_count: 0,
        last_action_taken: ActionType::Idle,
        brain,
        genome,
    };

    let occupancy = vec![None; 9];
    let spike_map = no_spikes(occupancy.len());
    let mut scratch = BrainScratch::new();
    let vision_distance = organism.genome.vision_distance;
    let _ = evaluate_brain(
        &mut organism,
        3,
        &occupancy,
        &spike_map,
        vision_distance,
        deterministic_action_policy(),
        0.5,
        &mut scratch,
    );

    let before = organism.brain.sensory[0].synapses[0].weight;
    compute_pending_coactivations(&mut organism, &mut scratch);
    apply_runtime_weight_updates(
        &mut organism,
        RewardLedger {
            food_consumed_energy: 2.0,
            ..RewardLedger::default()
        },
    );
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
        .copied()
        .enumerate()
        .map(|(idx, action_type)| {
            make_action_neuron(2000 + idx as u32, action_type, loc(2.0, 1.0 + idx as f32))
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
        species_id: sim_types::SpeciesId(0),
        q: 0,
        r: 0,
        generation: 0,
        age_turns: 50,
        facing: FacingDirection::East,
        energy: 99.0,
        health: 100.0,
        max_health: 100.0,
        energy_prev: 100.0,
        dopamine: 0.0,
        damage_taken_last_turn: 0.0,
        consumptions_count: 0,
        reproductions_count: 0,
        last_action_taken: ActionType::Idle,
        brain,
        genome,
    };

    let occupancy = vec![None; 9];
    let spike_map = no_spikes(occupancy.len());
    let mut scratch = BrainScratch::new();
    let vision_distance = organism.genome.vision_distance;
    let _ = evaluate_brain(
        &mut organism,
        3,
        &occupancy,
        &spike_map,
        vision_distance,
        deterministic_action_policy(),
        0.5,
        &mut scratch,
    );

    let before = organism.brain.sensory[0].synapses[0].weight;
    // Passive drain baseline (1.0) exactly cancels the raw -1.0 energy delta.
    compute_pending_coactivations(&mut organism, &mut scratch);
    apply_runtime_weight_updates(&mut organism, RewardLedger::default());
    let after = organism.brain.sensory[0].synapses[0].weight;

    let expected = before * (1.0 - 0.001);
    assert!((after - expected).abs() < 1.0e-6);
}

#[test]
fn juvenile_plasticity_updates_weights_before_maturity() {
    let mut genome = test_genome();
    genome.num_neurons = 0;
    genome.num_synapses = 0;
    genome.age_of_maturity = 50;
    genome.plasticity_start_age = 0;
    genome.juvenile_eta_scale = 0.5;
    genome.max_weight_delta_per_tick = 1.0;
    genome.hebb_eta_gain = 0.2;
    genome.synapse_prune_threshold = 0.0;

    let energy_id = SENSORY_COUNT - 1;
    let mut sensory = vec![make_sensory_neuron(
        energy_id,
        SensoryReceptor::Energy,
        loc(1.0, 1.0),
    )];
    sensory[0].synapses.push(SynapseEdge {
        pre_neuron_id: NeuronId(energy_id),
        post_neuron_id: NeuronId(ACTION_ID_BASE + action_index(ActionType::Forward) as u32),
        weight: 0.2,
        eligibility: 0.0,
        pending_coactivation: 0.0,
    });
    let action: Vec<_> = ActionType::ALL
        .into_iter()
        .copied()
        .enumerate()
        .map(|(idx, action_type)| {
            make_action_neuron(2000 + idx as u32, action_type, loc(2.0, 1.0 + idx as f32))
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
        species_id: sim_types::SpeciesId(0),
        q: 0,
        r: 0,
        generation: 0,
        age_turns: 0,
        facing: FacingDirection::East,
        energy: 100.0,
        health: 100.0,
        max_health: 100.0,
        energy_prev: 100.0,
        dopamine: 0.0,
        damage_taken_last_turn: 0.0,
        consumptions_count: 0,
        reproductions_count: 0,
        last_action_taken: ActionType::Idle,
        brain,
        genome,
    };

    let occupancy = vec![None; 9];
    let spike_map = no_spikes(occupancy.len());
    let mut scratch = BrainScratch::new();
    let vision_distance = organism.genome.vision_distance;
    let _ = evaluate_brain(
        &mut organism,
        3,
        &occupancy,
        &spike_map,
        vision_distance,
        deterministic_action_policy(),
        0.5,
        &mut scratch,
    );

    let before = organism.brain.sensory[0].synapses[0].weight;
    compute_pending_coactivations(&mut organism, &mut scratch);
    apply_runtime_weight_updates(
        &mut organism,
        RewardLedger {
            food_consumed_energy: 10.0,
            ..RewardLedger::default()
        },
    );
    let after = organism.brain.sensory[0].synapses[0].weight;

    assert_ne!(before, after);
}

#[test]
fn max_weight_delta_per_tick_caps_plastic_updates() {
    let mut genome = test_genome();
    genome.num_neurons = 0;
    genome.num_synapses = 0;
    genome.plasticity_start_age = 0;
    genome.juvenile_eta_scale = 1.0;
    genome.max_weight_delta_per_tick = 0.01;
    genome.hebb_eta_gain = 10.0;
    genome.synapse_prune_threshold = 0.0;

    let energy_id = SENSORY_COUNT - 1;
    let mut sensory = vec![make_sensory_neuron(
        energy_id,
        SensoryReceptor::Energy,
        loc(1.0, 1.0),
    )];
    sensory[0].synapses.push(SynapseEdge {
        pre_neuron_id: NeuronId(energy_id),
        post_neuron_id: NeuronId(ACTION_ID_BASE + action_index(ActionType::Forward) as u32),
        weight: 0.2,
        eligibility: 1.0,
        pending_coactivation: 0.0,
    });
    let action: Vec<_> = ActionType::ALL
        .into_iter()
        .copied()
        .enumerate()
        .map(|(idx, action_type)| {
            make_action_neuron(2000 + idx as u32, action_type, loc(2.0, 1.0 + idx as f32))
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
        species_id: sim_types::SpeciesId(0),
        q: 0,
        r: 0,
        generation: 0,
        age_turns: 0,
        facing: FacingDirection::East,
        energy: 100.0,
        health: 100.0,
        max_health: 100.0,
        energy_prev: 100.0,
        dopamine: 0.0,
        damage_taken_last_turn: 0.0,
        consumptions_count: 0,
        reproductions_count: 0,
        last_action_taken: ActionType::Idle,
        brain,
        genome,
    };

    let before = organism.brain.sensory[0].synapses[0].weight;
    apply_runtime_weight_updates(
        &mut organism,
        RewardLedger {
            food_consumed_energy: 100.0,
            ..RewardLedger::default()
        },
    );
    let after = organism.brain.sensory[0].synapses[0].weight;

    assert!((after - before).abs() <= 0.01 + 1.0e-6);
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
        state: 0.8_f32.atanh(),
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
        state: 0.0,
        alpha: 1.0,
        synapses: Vec::new(),
    };
    let action: Vec<_> = ActionType::ALL
        .into_iter()
        .copied()
        .enumerate()
        .map(|(idx, action_type)| {
            make_action_neuron(
                ACTION_ID_BASE + idx as u32,
                action_type,
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
        species_id: sim_types::SpeciesId(0),
        q: 0,
        r: 0,
        generation: 0,
        age_turns: 0,
        facing: FacingDirection::East,
        energy: 100.0,
        health: 100.0,
        max_health: 100.0,
        energy_prev: 100.0,
        dopamine: 0.0,
        damage_taken_last_turn: 0.0,
        consumptions_count: 0,
        reproductions_count: 0,
        last_action_taken: ActionType::Idle,
        brain,
        genome,
    };

    let occupancy = vec![None; 9];
    let spike_map = no_spikes(occupancy.len());
    let mut scratch = BrainScratch::new();
    let vision_distance = organism.genome.vision_distance;
    let _ = evaluate_brain(
        &mut organism,
        3,
        &occupancy,
        &spike_map,
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
fn action_target_eligibility_only_credits_the_executed_action() {
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
        post_neuron_id: NeuronId(ACTION_ID_BASE + action_index(ActionType::Forward) as u32),
        weight: 1.0,
        eligibility: 0.0,
        pending_coactivation: 0.0,
    });
    sensory[0].synapses.push(SynapseEdge {
        pre_neuron_id: NeuronId(energy_id),
        post_neuron_id: NeuronId(ACTION_ID_BASE + action_index(ActionType::Reproduce) as u32),
        weight: -1.0,
        eligibility: 0.0,
        pending_coactivation: 0.0,
    });
    let action: Vec<_> = ActionType::ALL
        .into_iter()
        .copied()
        .enumerate()
        .map(|(idx, action_type)| {
            make_action_neuron(
                ACTION_ID_BASE + idx as u32,
                action_type,
                loc(2.0, 1.0 + idx as f32),
            )
        })
        .collect();
    let brain = BrainState {
        sensory,
        inter: vec![],
        action,
        synapse_count: 2,
    };
    let mut organism = OrganismState {
        id: OrganismId(0),
        species_id: sim_types::SpeciesId(0),
        q: 0,
        r: 0,
        generation: 0,
        age_turns: 0,
        facing: FacingDirection::East,
        energy: 250.0,
        health: 250.0,
        max_health: 250.0,
        energy_prev: 250.0,
        dopamine: 0.0,
        damage_taken_last_turn: 0.0,
        consumptions_count: 0,
        reproductions_count: 0,
        last_action_taken: ActionType::Idle,
        brain,
        genome,
    };

    let occupancy = vec![None; 9];
    let spike_map = no_spikes(occupancy.len());
    let mut scratch = BrainScratch::new();
    let vision_distance = organism.genome.vision_distance;
    let _ = evaluate_brain(
        &mut organism,
        3,
        &occupancy,
        &spike_map,
        vision_distance,
        deterministic_action_policy(),
        0.5,
        &mut scratch,
    );
    compute_pending_coactivations(&mut organism, &mut scratch);

    let sensory_activation = organism.brain.sensory[0].neuron.activation;
    let executed_pending = organism.brain.sensory[0].synapses[0].pending_coactivation;
    let unexecuted_pending = organism.brain.sensory[0].synapses[1].pending_coactivation;

    assert!(executed_pending > 0.0);
    assert!(executed_pending <= sensory_activation);
    assert_eq!(unexecuted_pending, 0.0);
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
        .copied()
        .enumerate()
        .map(|(idx, action_type)| {
            make_action_neuron(2000 + idx as u32, action_type, loc(2.0, 1.0 + idx as f32))
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
        species_id: sim_types::SpeciesId(0),
        q: 0,
        r: 0,
        generation: 0,
        age_turns: 0,
        facing: FacingDirection::East,
        energy: 250.0,
        health: 250.0,
        max_health: 250.0,
        energy_prev: 250.0,
        dopamine: 0.0,
        damage_taken_last_turn: 0.0,
        consumptions_count: 0,
        reproductions_count: 0,
        last_action_taken: ActionType::Idle,
        brain,
        genome,
    };

    let occupancy = vec![None; 9];
    let spike_map = no_spikes(occupancy.len());
    let mut scratch = BrainScratch::new();
    let vision_distance = organism.genome.vision_distance;

    let _ = evaluate_brain(
        &mut organism,
        3,
        &occupancy,
        &spike_map,
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
        &spike_map,
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
        &spike_map,
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
        &spike_map,
        vision_distance,
        deterministic_action_policy(),
        0.5,
        &mut scratch,
    );
    assert!(organism.brain.sensory[0].neuron.activation < 0.05);
}
