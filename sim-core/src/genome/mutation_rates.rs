use super::*;

const META_MUTATION_STEP_STDDEV: f32 = 0.12;
const META_MUTATION_EXPLORATION_STDDEV: f32 = 0.35;
const META_MUTATION_EXPLORATION_PROBABILITY: f32 = 0.1;
const META_MUTATION_BASELINE_PULL: f32 = 0.15;
const META_MUTATION_SECOND_GENE_PROBABILITY: f32 = 0.35;
const META_MUTATION_THIRD_GENE_PROBABILITY: f32 = 0.1;
const MUTATION_RATE_BASELINE_FLOOR_FRACTION: f32 = 0.05;
const MUTATION_RATE_GENE_COUNT: usize = 15;
const MUTATION_RATE_MIN: f32 = 1.0e-4;
const MUTATION_RATE_MAX: f32 = 0.5;
const MUTATION_RATE_LATENT_MIN: f32 = -8.0;
const MUTATION_RATE_LATENT_MAX: f32 = 8.0;
const MUTATION_RATE_LOGIT_EPSILON: f32 = 1.0e-6;

pub(super) struct EffectiveMutationRates {
    pub(super) age_of_maturity: f32,
    pub(super) gestation_ticks: f32,
    pub(super) max_organism_age: f32,
    pub(super) vision_distance: f32,
    pub(super) max_health: f32,
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
    seed_genome_config: &SeedGenomeConfig,
    rng: &mut R,
) {
    let mut rates = mutation_rate_gene_values(genome);
    let baseline_rates = seed_mutation_rate_values(seed_genome_config);
    let mutations_to_apply = 1
        + usize::from(rng.random::<f32>() < META_MUTATION_SECOND_GENE_PROBABILITY)
        + usize::from(rng.random::<f32>() < META_MUTATION_THIRD_GENE_PROBABILITY);
    let mut touched = [false; MUTATION_RATE_GENE_COUNT];

    for _ in 0..mutations_to_apply {
        let Some(idx) = random_untouched_index(&touched, rng) else {
            break;
        };
        touched[idx] = true;
        rates[idx] = mutate_single_mutation_rate(rates[idx], baseline_rates[idx], rng);
    }

    apply_mutation_rate_gene_values(genome, rates);
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
        gestation_ticks: effective_mutation_rate(
            genome.mutation_rate_gestation_ticks,
            global_mutation_rate_modifier,
        ),
        max_organism_age: effective_mutation_rate(
            genome.mutation_rate_max_organism_age,
            global_mutation_rate_modifier,
        ),
        vision_distance: effective_mutation_rate(
            genome.mutation_rate_vision_distance,
            global_mutation_rate_modifier,
        ),
        max_health: effective_mutation_rate(
            genome.mutation_rate_max_health,
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

fn mutation_rate_gene_values(genome: &OrganismGenome) -> [f32; MUTATION_RATE_GENE_COUNT] {
    [
        genome.mutation_rate_age_of_maturity,
        genome.mutation_rate_gestation_ticks,
        genome.mutation_rate_max_organism_age,
        genome.mutation_rate_vision_distance,
        genome.mutation_rate_max_health,
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
    ]
}

fn seed_mutation_rate_values(config: &SeedGenomeConfig) -> [f32; MUTATION_RATE_GENE_COUNT] {
    [
        config.mutation_rate_age_of_maturity,
        config.mutation_rate_gestation_ticks,
        config.mutation_rate_max_organism_age,
        config.mutation_rate_vision_distance,
        config.mutation_rate_max_health,
        config.mutation_rate_inter_bias,
        config.mutation_rate_inter_update_rate,
        config.mutation_rate_eligibility_retention,
        config.mutation_rate_synapse_prune_threshold,
        config.mutation_rate_neuron_location,
        config.mutation_rate_synapse_weight_perturbation,
        config.mutation_rate_add_synapse,
        config.mutation_rate_remove_synapse,
        config.mutation_rate_remove_neuron,
        config.mutation_rate_add_neuron_split_edge,
    ]
}

fn apply_mutation_rate_gene_values(
    genome: &mut OrganismGenome,
    rates: [f32; MUTATION_RATE_GENE_COUNT],
) {
    genome.mutation_rate_age_of_maturity = rates[0];
    genome.mutation_rate_gestation_ticks = rates[1];
    genome.mutation_rate_max_organism_age = rates[2];
    genome.mutation_rate_vision_distance = rates[3];
    genome.mutation_rate_max_health = rates[4];
    genome.mutation_rate_inter_bias = rates[5];
    genome.mutation_rate_inter_update_rate = rates[6];
    genome.mutation_rate_eligibility_retention = rates[7];
    genome.mutation_rate_synapse_prune_threshold = rates[8];
    genome.mutation_rate_neuron_location = rates[9];
    genome.mutation_rate_synapse_weight_perturbation = rates[10];
    genome.mutation_rate_add_synapse = rates[11];
    genome.mutation_rate_remove_synapse = rates[12];
    genome.mutation_rate_remove_neuron = rates[13];
    genome.mutation_rate_add_neuron_split_edge = rates[14];
}

fn random_untouched_index<R: Rng + ?Sized>(
    touched: &[bool; MUTATION_RATE_GENE_COUNT],
    rng: &mut R,
) -> Option<usize> {
    let remaining = touched.iter().filter(|&&was_touched| !was_touched).count();
    if remaining == 0 {
        return None;
    }

    let target = rng.random_range(0..remaining);
    let mut seen = 0;
    for (idx, was_touched) in touched.iter().enumerate() {
        if *was_touched {
            continue;
        }
        if seen == target {
            return Some(idx);
        }
        seen += 1;
    }

    None
}

fn mutate_single_mutation_rate<R: Rng + ?Sized>(
    current_rate: f32,
    baseline_rate: f32,
    rng: &mut R,
) -> f32 {
    let current_latent = mutation_rate_to_latent(current_rate);
    let baseline_latent = mutation_rate_to_latent(baseline_rate);
    let noise_scale = if rng.random::<f32>() < META_MUTATION_EXPLORATION_PROBABILITY {
        META_MUTATION_EXPLORATION_STDDEV
    } else {
        META_MUTATION_STEP_STDDEV
    };
    let latent = current_latent
        + (baseline_latent - current_latent) * META_MUTATION_BASELINE_PULL
        + standard_normal(rng) * noise_scale;
    let floor = exploration_floor_rate(baseline_rate);
    mutation_rate_from_latent(latent).max(floor)
}

fn exploration_floor_rate(baseline_rate: f32) -> f32 {
    clamp_mutation_rate(clamp_mutation_rate(baseline_rate) * MUTATION_RATE_BASELINE_FLOOR_FRACTION)
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
