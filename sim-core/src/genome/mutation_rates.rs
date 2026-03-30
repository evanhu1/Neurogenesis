use super::*;

const MUTATION_RATE_ADAPTATION_TIME_CONSTANT: f32 = 0.25;
const MUTATION_RATE_MIN: f32 = 1.0e-4;
const MUTATION_RATE_MAX: f32 = 0.5;
const MUTATION_RATE_LATENT_MIN: f32 = -8.0;
const MUTATION_RATE_LATENT_MAX: f32 = 8.0;
const MUTATION_RATE_LOGIT_EPSILON: f32 = 1.0e-6;

pub(super) struct EffectiveMutationRates {
    pub(super) age_of_maturity: f32,
    pub(super) vision_distance: f32,
    pub(super) inter_bias: f32,
    pub(super) inter_update_rate: f32,
    pub(super) eligibility_retention: f32,
    pub(super) synapse_prune_threshold: f32,
    pub(super) neuron_location: f32,
    pub(super) synapse_weight_perturbation: f32,
    pub(super) add_synapse: f32,
    pub(super) remove_synapse: f32,
    pub(super) remove_neuron: f32,
    pub(super) add_neuron_split_edge: f32,
}

pub(super) fn mutate_mutation_rate_genes<R: Rng + ?Sized>(
    genome: &mut OrganismGenome,
    rng: &mut R,
) {
    let mut rates = [
        genome.mutation_rate_age_of_maturity,
        genome.mutation_rate_vision_distance,
        genome.mutation_rate_inter_bias,
        genome.mutation_rate_inter_update_rate,
        genome.mutation_rate_eligibility_retention,
        genome.mutation_rate_synapse_prune_threshold,
        genome.mutation_rate_neuron_location,
        genome.mutation_rate_synapse_weight_perturbation,
        genome.mutation_rate_add_synapse,
        genome.mutation_rate_remove_synapse,
        genome.mutation_rate_remove_neuron,
        genome.mutation_rate_add_neuron_split_edge,
    ];
    let shared_normal = standard_normal(rng) * MUTATION_RATE_ADAPTATION_TIME_CONSTANT;

    for rate in &mut rates {
        let mut latent = mutation_rate_to_latent(*rate);
        let gene_normal = standard_normal(rng) * MUTATION_RATE_ADAPTATION_TIME_CONSTANT;
        latent = (latent + shared_normal + gene_normal)
            .clamp(MUTATION_RATE_LATENT_MIN, MUTATION_RATE_LATENT_MAX);
        *rate = mutation_rate_from_latent(latent);
    }

    genome.mutation_rate_age_of_maturity = rates[0];
    genome.mutation_rate_vision_distance = rates[1];
    genome.mutation_rate_inter_bias = rates[2];
    genome.mutation_rate_inter_update_rate = rates[3];
    genome.mutation_rate_eligibility_retention = rates[4];
    genome.mutation_rate_synapse_prune_threshold = rates[5];
    genome.mutation_rate_neuron_location = rates[6];
    genome.mutation_rate_synapse_weight_perturbation = rates[7];
    genome.mutation_rate_add_synapse = rates[8];
    genome.mutation_rate_remove_synapse = rates[9];
    genome.mutation_rate_remove_neuron = rates[10];
    genome.mutation_rate_add_neuron_split_edge = rates[11];
}

pub(super) fn effective_mutation_rates(
    genome: &OrganismGenome,
    global_mutation_rate_modifier: f32,
) -> EffectiveMutationRates {
    EffectiveMutationRates {
        age_of_maturity: effective_mutation_rate(
            genome.mutation_rate_age_of_maturity,
            global_mutation_rate_modifier,
        ),
        vision_distance: effective_mutation_rate(
            genome.mutation_rate_vision_distance,
            global_mutation_rate_modifier,
        ),
        inter_bias: effective_mutation_rate(
            genome.mutation_rate_inter_bias,
            global_mutation_rate_modifier,
        ),
        inter_update_rate: effective_mutation_rate(
            genome.mutation_rate_inter_update_rate,
            global_mutation_rate_modifier,
        ),
        eligibility_retention: effective_mutation_rate(
            genome.mutation_rate_eligibility_retention,
            global_mutation_rate_modifier,
        ),
        synapse_prune_threshold: effective_mutation_rate(
            genome.mutation_rate_synapse_prune_threshold,
            global_mutation_rate_modifier,
        ),
        neuron_location: effective_mutation_rate(
            genome.mutation_rate_neuron_location,
            global_mutation_rate_modifier,
        ),
        synapse_weight_perturbation: effective_mutation_rate(
            genome.mutation_rate_synapse_weight_perturbation,
            global_mutation_rate_modifier,
        ),
        add_synapse: effective_mutation_rate(
            genome.mutation_rate_add_synapse,
            global_mutation_rate_modifier,
        ),
        remove_synapse: effective_mutation_rate(
            genome.mutation_rate_remove_synapse,
            global_mutation_rate_modifier,
        ),
        remove_neuron: effective_mutation_rate(
            genome.mutation_rate_remove_neuron,
            global_mutation_rate_modifier,
        ),
        add_neuron_split_edge: effective_mutation_rate(
            genome.mutation_rate_add_neuron_split_edge,
            global_mutation_rate_modifier,
        ),
    }
}

fn clamp_mutation_rate(rate: f32) -> f32 {
    rate.clamp(MUTATION_RATE_MIN, MUTATION_RATE_MAX)
}

fn mutation_rate_to_latent(rate: f32) -> f32 {
    let clamped_rate = clamp_mutation_rate(rate);
    let span = (MUTATION_RATE_MAX - MUTATION_RATE_MIN).max(f32::MIN_POSITIVE);
    let normalized = ((clamped_rate - MUTATION_RATE_MIN) / span).clamp(
        MUTATION_RATE_LOGIT_EPSILON,
        1.0 - MUTATION_RATE_LOGIT_EPSILON,
    );
    (normalized / (1.0 - normalized)).ln()
}

fn mutation_rate_from_latent(latent: f32) -> f32 {
    let clamped_latent = latent.clamp(MUTATION_RATE_LATENT_MIN, MUTATION_RATE_LATENT_MAX);
    let sigmoid = 1.0 / (1.0 + (-clamped_latent).exp());
    let rate = MUTATION_RATE_MIN + sigmoid * (MUTATION_RATE_MAX - MUTATION_RATE_MIN);
    clamp_mutation_rate(rate)
}

fn effective_mutation_rate(rate: f32, global_mutation_rate_modifier: f32) -> f32 {
    (rate * global_mutation_rate_modifier).clamp(0.0, MUTATION_RATE_MAX)
}
