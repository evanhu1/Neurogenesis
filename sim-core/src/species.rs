use crate::{validate_species_config, Simulation};
use rand::Rng;
use sim_protocol::{ActionType, SensoryReceptorType, SpeciesConfig, SpeciesId};

const MAX_MUTATED_INTER_NEURONS: u32 = 256;
const MAX_SPECIES_MUTATIONS: u32 = 8;
const MUTATION_CHANCE_STEP: f32 = 0.01;

impl Simulation {
    pub(crate) fn initialize_species_registry_from_seed(&mut self) {
        self.species_registry.clear();
        self.next_species_id = 0;
        let species_id = self.alloc_species_id();
        debug_assert_eq!(species_id, SpeciesId(0));
        self.species_registry
            .insert(species_id, self.config.seed_species_config.clone());
    }

    pub(crate) fn alloc_species_id(&mut self) -> SpeciesId {
        let id = SpeciesId(self.next_species_id);
        self.next_species_id = self.next_species_id.saturating_add(1);
        id
    }

    pub(crate) fn species_config(&self, species_id: SpeciesId) -> Option<&SpeciesConfig> {
        self.species_registry.get(&species_id)
    }

    pub(crate) fn species_id_for_reproduction(
        &mut self,
        parent_species_id: SpeciesId,
    ) -> SpeciesId {
        let Some(parent_species_config) = self.species_config(parent_species_id).cloned() else {
            return parent_species_id;
        };

        let speciation_chance = parent_species_config.mutation_chance.clamp(0.0, 1.0);
        if self.rng.random::<f32>() >= speciation_chance {
            return parent_species_id;
        }

        let mutated_species = self.mutate_species(&parent_species_config);
        let new_species_id = self.alloc_species_id();
        self.species_registry
            .insert(new_species_id, mutated_species);
        new_species_id
    }

    pub(crate) fn mutate_species(&mut self, species_config: &SpeciesConfig) -> SpeciesConfig {
        let mut mutated = species_config.clone();
        let mutation_rate = species_config.mutation_chance.clamp(0.0, 1.0);
        let mutation_count = sample_species_mutation_count(mutation_rate, &mut self.rng);

        for _ in 0..mutation_count {
            mutate_random_species_trait(&mut mutated, &mut self.rng);
        }

        normalize_species_config(&mut mutated);
        if mutated == *species_config {
            let nudged = (mutated.mutation_chance + MUTATION_CHANCE_STEP).clamp(0.0, 1.0);
            mutated.mutation_chance = if nudged > mutated.mutation_chance {
                nudged
            } else {
                (mutated.mutation_chance - MUTATION_CHANCE_STEP).clamp(0.0, 1.0)
            };
        }

        debug_assert!(validate_species_config(&mutated).is_ok());
        mutated
    }
}

fn sample_species_mutation_count<R: Rng + ?Sized>(mutation_rate: f32, rng: &mut R) -> u32 {
    let mut count = 1_u32;
    while count < MAX_SPECIES_MUTATIONS && rng.random::<f32>() < mutation_rate {
        count += 1;
    }
    count
}

fn mutate_random_species_trait<R: Rng + ?Sized>(species: &mut SpeciesConfig, rng: &mut R) {
    match rng.random_range(0..4) {
        0 => {
            let max_neurons = species.max_num_neurons.min(MAX_MUTATED_INTER_NEURONS);
            species.num_neurons = mutate_step_u32(species.num_neurons, 0, max_neurons, rng);
            species.num_synapses = species
                .num_synapses
                .min(max_synapses_for_inter_count(species.num_neurons));
        }
        1 => {
            let min_neurons = species.num_neurons.min(MAX_MUTATED_INTER_NEURONS);
            species.max_num_neurons = mutate_step_u32(
                species.max_num_neurons,
                min_neurons,
                MAX_MUTATED_INTER_NEURONS,
                rng,
            );
        }
        2 => {
            let max_synapses = max_synapses_for_inter_count(species.num_neurons);
            species.num_synapses = mutate_step_u32(species.num_synapses, 0, max_synapses, rng);
        }
        _ => {
            let delta = if rng.random::<bool>() {
                MUTATION_CHANCE_STEP
            } else {
                -MUTATION_CHANCE_STEP
            };
            species.mutation_chance = (species.mutation_chance + delta).clamp(0.0, 1.0);
        }
    }
}

fn mutate_step_u32<R: Rng + ?Sized>(value: u32, min: u32, max: u32, rng: &mut R) -> u32 {
    if min >= max {
        return min;
    }
    if value <= min {
        return min.saturating_add(1).min(max);
    }
    if value >= max {
        return max.saturating_sub(1).max(min);
    }
    if rng.random::<bool>() {
        value.saturating_add(1).min(max)
    } else {
        value.saturating_sub(1).max(min)
    }
}

fn normalize_species_config(mutated: &mut SpeciesConfig) {
    mutated.num_neurons = mutated.num_neurons.min(MAX_MUTATED_INTER_NEURONS);
    mutated.max_num_neurons = mutated
        .max_num_neurons
        .clamp(mutated.num_neurons, MAX_MUTATED_INTER_NEURONS);
    mutated.num_synapses = mutated
        .num_synapses
        .min(max_synapses_for_inter_count(mutated.num_neurons));

    if !mutated.mutation_chance.is_finite() {
        mutated.mutation_chance = 0.0;
    }
    mutated.mutation_chance = mutated.mutation_chance.clamp(0.0, 1.0);
}

fn max_synapses_for_inter_count(inter_count: u32) -> u32 {
    let sensory_count = SensoryReceptorType::ALL.len() as u32;
    let action_count = ActionType::ALL.len() as u32;
    let pre_count = sensory_count + inter_count;
    let post_count = inter_count + action_count;
    pre_count
        .saturating_mul(post_count)
        .saturating_sub(inter_count)
}
