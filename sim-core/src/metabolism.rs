use crate::genome::distance_sq_between_locations;
use crate::topology::{action_array_index, inter_index};
use sim_types::{BrainLocation, BrainState, NeuronId, OrganismState, SynapseEdge, WorldConfig};

// Even tightly clustered wiring should still cost something, but much less than
// a spread-out circuit.
const LOCAL_SYNAPSE_METABOLIC_COST_FLOOR: f32 = 0.25;
// A roughly 3-unit latent-space jump adds one full metabolism unit; because the
// formula uses squared distance, longer-range links get expensive quickly.
const SPATIAL_SYNAPSE_METABOLIC_RADIUS_SQ: f32 = 9.0;

pub(crate) fn refresh_organism_base_metabolic_cost(organism: &mut OrganismState) {
    organism.base_metabolic_cost = organism_base_metabolic_cost(organism);
}

pub(crate) fn organism_passive_metabolic_energy_cost(
    config: &WorldConfig,
    organism: &OrganismState,
) -> f32 {
    config.passive_metabolism_cost_per_unit * organism.base_metabolic_cost
}

fn organism_base_metabolic_cost(organism: &OrganismState) -> f32 {
    let inter_neuron_count = organism.genome.topology.num_neurons as f32;
    let sensory_neuron_count = organism.brain.sensory.len() as f32;
    let synapse_spatial_cost_units = organism_synapse_spatial_metabolic_cost_units(&organism.brain);
    let vision_distance_cost_units = organism.genome.topology.vision_distance as f32 / 3.0;
    inter_neuron_count
        + sensory_neuron_count
        + synapse_spatial_cost_units
        + vision_distance_cost_units
}

fn organism_synapse_spatial_metabolic_cost_units(brain: &BrainState) -> f32 {
    let sensory_cost = brain
        .sensory
        .iter()
        .map(|sensory| {
            synapse_group_spatial_metabolic_cost_units(
                &sensory.synapses,
                BrainLocation {
                    x: sensory.neuron.x,
                    y: sensory.neuron.y,
                },
                brain,
            )
        })
        .sum::<f32>();
    let inter_cost = brain
        .inter
        .iter()
        .map(|inter| {
            synapse_group_spatial_metabolic_cost_units(
                &inter.synapses,
                BrainLocation {
                    x: inter.neuron.x,
                    y: inter.neuron.y,
                },
                brain,
            )
        })
        .sum::<f32>();
    sensory_cost + inter_cost
}

fn synapse_group_spatial_metabolic_cost_units(
    synapses: &[SynapseEdge],
    pre_location: BrainLocation,
    brain: &BrainState,
) -> f32 {
    synapses
        .iter()
        .filter_map(|edge| {
            post_neuron_location(brain, edge.post_neuron_id).map(|post_location| {
                synapse_spatial_metabolic_cost_units(distance_sq_between_locations(
                    pre_location,
                    post_location,
                ))
            })
        })
        .sum()
}

fn post_neuron_location(brain: &BrainState, neuron_id: NeuronId) -> Option<BrainLocation> {
    if let Some(idx) = inter_index(neuron_id, brain.inter.len()) {
        let neuron = brain.inter.get(idx)?;
        return Some(BrainLocation {
            x: neuron.neuron.x,
            y: neuron.neuron.y,
        });
    }

    let idx = action_array_index(neuron_id)?;
    let neuron = brain.action.get(idx)?;
    Some(BrainLocation {
        x: neuron.x,
        y: neuron.y,
    })
}

fn synapse_spatial_metabolic_cost_units(distance_sq: f32) -> f32 {
    LOCAL_SYNAPSE_METABOLIC_COST_FLOOR + distance_sq / SPATIAL_SYNAPSE_METABOLIC_RADIUS_SQ
}

#[cfg(test)]
mod tests {
    use super::*;
    use sim_types::{
        ActionNeuronState, ActionType, FacingDirection, InterNeuronState, NeuronState, NeuronType,
        OrganismGenome, OrganismId, SensoryNeuronState, SensoryReceptor, SpeciesId,
    };

    fn metabolism_test_organism() -> OrganismState {
        let mut organism = OrganismState {
            id: OrganismId(0),
            species_id: SpeciesId(0),
            q: 0,
            r: 0,
            generation: 0,
            age_turns: 0,
            facing: FacingDirection::East,
            energy: 100.0,
            health: 100.0,
            max_health: 100.0,
            energy_prev: 100.0,
            health_prev: 100.0,
            dopamine: 0.0,
            value_prev: 0.0,
            value_prev_inter_activations: Vec::new(),
            damage_taken_last_turn: 0.0,
            contingent_action_wasted_last_turn: false,
            is_gestating: false,
            consumptions_count: 0,
            plant_consumptions_count: 0,
            prey_consumptions_count: 0,
            reproductions_count: 0,
            last_action_taken: ActionType::Idle,
            base_metabolic_cost: 0.0,
            #[cfg(feature = "instrumentation")]
            instrumentation: Default::default(),
            brain: BrainState {
                sensory: vec![SensoryNeuronState {
                    neuron: NeuronState {
                        neuron_id: NeuronId(0),
                        neuron_type: NeuronType::Sensory,
                        bias: 0.0,
                        x: 0.0,
                        y: 0.0,
                        activation: 0.0,
                        parent_ids: Vec::new(),
                    },
                    receptor: SensoryReceptor::ContactAhead,
                    synapses: Vec::new(),
                }],
                inter: vec![InterNeuronState {
                    neuron: NeuronState {
                        neuron_id: NeuronId(1000),
                        neuron_type: NeuronType::Inter,
                        bias: 0.0,
                        x: 1.0,
                        y: 1.0,
                        activation: 0.0,
                        parent_ids: Vec::new(),
                    },
                    state: 0.0,
                    alpha: 1.0,
                    synapses: Vec::new(),
                    action_synapse_start: 0,
                }],
                action: vec![ActionNeuronState {
                    neuron_id: NeuronId(2000),
                    x: 1.0,
                    y: 1.0,
                    logit: 0.0,
                    parent_ids: Vec::new(),
                    action_type: ActionType::Forward,
                }],
                synapse_count: 0,
                value_weights: Vec::new(),
                sensory_mean_activation: Vec::new(),
                sensory_mean_initialized: Vec::new(),
                inter_mean_activation: Vec::new(),
                inter_mean_initialized: Vec::new(),
            },
            genome: {
                let mut genome = OrganismGenome::test_fixture();
                genome.topology.vision_distance = 3;
                genome.brain.sensory_locations = vec![BrainLocation { x: 0.0, y: 0.0 }; genome.brain.sensory_locations.len()];
                genome.brain.inter_locations = vec![BrainLocation { x: 1.0, y: 1.0 }];
                genome.brain.action_locations = vec![BrainLocation { x: 1.0, y: 1.0 }; genome.brain.action_locations.len()];
                genome
            },
        };
        refresh_organism_base_metabolic_cost(&mut organism);
        organism
    }

    #[test]
    fn long_range_synapses_cost_more_than_local_clusters() {
        let mut local = metabolism_test_organism();
        local.brain.sensory[0].synapses = vec![
            SynapseEdge {
                pre_neuron_id: NeuronId(0),
                post_neuron_id: NeuronId(1000),
                weight: 1.0,
                eligibility: 0.0,
                pending_coactivation: 0.0,
            },
            SynapseEdge {
                pre_neuron_id: NeuronId(0),
                post_neuron_id: NeuronId(2000),
                weight: 1.0,
                eligibility: 0.0,
                pending_coactivation: 0.0,
            },
        ];
        local.brain.synapse_count = local.brain.sensory[0].synapses.len() as u32;
        refresh_organism_base_metabolic_cost(&mut local);

        let mut long_range = local.clone();
        long_range.brain.inter[0].neuron.x = 8.0;
        long_range.brain.inter[0].neuron.y = 8.0;
        long_range.brain.action[0].x = 9.0;
        long_range.brain.action[0].y = 9.0;
        refresh_organism_base_metabolic_cost(&mut long_range);

        assert!(
            long_range.base_metabolic_cost > local.base_metabolic_cost + 10.0,
            "long-range wiring should carry a much heavier metabolic penalty \
             (local={}, long_range={})",
            local.base_metabolic_cost,
            long_range.base_metabolic_cost
        );
    }

    #[test]
    fn pruned_synapses_stop_contributing_to_metabolic_cost() {
        let mut organism = metabolism_test_organism();
        organism.genome.topology.num_synapses = 1;
        organism.genome.brain.edges = vec![SynapseEdge {
            pre_neuron_id: NeuronId(0),
            post_neuron_id: NeuronId(2000),
            weight: 1.0,
            eligibility: 0.0,
            pending_coactivation: 0.0,
        }];
        refresh_organism_base_metabolic_cost(&mut organism);

        let baseline_cost = organism.base_metabolic_cost;

        organism.brain.sensory[0].synapses = vec![SynapseEdge {
            pre_neuron_id: NeuronId(0),
            post_neuron_id: NeuronId(2000),
            weight: 1.0,
            eligibility: 0.0,
            pending_coactivation: 0.0,
        }];
        organism.brain.synapse_count = 1;
        refresh_organism_base_metabolic_cost(&mut organism);

        assert!(
            organism.base_metabolic_cost > baseline_cost,
            "only active runtime synapses should contribute to metabolic cost \
             (baseline={baseline_cost}, active={})",
            organism.base_metabolic_cost
        );
    }
}
