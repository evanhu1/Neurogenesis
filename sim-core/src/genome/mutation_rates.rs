use super::*;

const META_MUTATION_STEP_STDDEV: f32 = 0.12;
const META_MUTATION_EXPLORATION_STDDEV: f32 = 0.35;
const META_MUTATION_EXPLORATION_PROBABILITY: f32 = 0.1;
const META_MUTATION_BASELINE_PULL: f32 = 0.15;
const META_MUTATION_SECOND_GENE_PROBABILITY: f32 = 0.35;
const META_MUTATION_THIRD_GENE_PROBABILITY: f32 = 0.1;
const MUTATION_RATE_BASELINE_FLOOR_FRACTION: f32 = 0.05;
const MUTATION_RATE_GENE_COUNT: usize = 16;
const MUTATION_RATE_MIN: f32 = 1.0e-4;
const MUTATION_RATE_MAX: f32 = 0.5;
const MUTATION_RATE_LOGIT_EPSILON: f32 = 1.0e-6;
/// Latent clamp implied by `MUTATION_RATE_LOGIT_EPSILON` (the single source
/// of truth): `logit(1 - ε) = ln((1 - ε) / ε) ≈ 13.815510` for ε = 1e-6,
/// rounded up at f32 precision so `mutation_rate_to_latent` /
/// `mutation_rate_from_latent` round-trip at the `MUTATION_RATE_MIN`/`MAX`
/// boundaries instead of snapping inward on the first meta-mutation.
const MUTATION_RATE_LATENT_MAX: f32 = 13.815_511;
const MUTATION_RATE_LATENT_MIN: f32 = -MUTATION_RATE_LATENT_MAX;

/// Generates the `EffectiveMutationRates` struct and all gene-value
/// extraction/application functions from a single field list. Adding or
/// removing a mutation-rate gene only requires updating this invocation;
/// the macro guarantees that the struct, the genome-to-array extraction,
/// the config-to-array extraction, the array-to-genome application, and
/// the effective-rate construction all stay in sync. The array size is
/// cross-checked against `MUTATION_RATE_GENE_COUNT` at compile time.
macro_rules! define_mutation_rate_ops {
    ( $( $gene:ident : $seed_field:ident ),+ $(,)? ) => {
        pub(super) struct EffectiveMutationRates {
            $(pub(super) $gene: f32,)+
        }

        pub(super) fn effective_mutation_rates(
            genome: &OrganismGenome,
            global_mutation_rate_modifier: f32,
        ) -> EffectiveMutationRates {
            EffectiveMutationRates {
                $($gene: effective_mutation_rate(
                    genome.mutation_rates.$gene,
                    global_mutation_rate_modifier,
                ),)+
            }
        }

        fn mutation_rate_gene_values(genome: &OrganismGenome) -> [f32; MUTATION_RATE_GENE_COUNT] {
            [$(genome.mutation_rates.$gene),+]
        }

        fn seed_mutation_rate_values(config: &SeedGenomeConfig) -> [f32; MUTATION_RATE_GENE_COUNT] {
            [$(config.$seed_field),+]
        }

        fn apply_mutation_rate_gene_values(
            genome: &mut OrganismGenome,
            rates: [f32; MUTATION_RATE_GENE_COUNT],
        ) {
            let mut _i = 0;
            $(
                genome.mutation_rates.$gene = rates[_i];
                _i += 1;
            )+
        }
    };
}

define_mutation_rate_ops! {
    age_of_maturity:             mutation_rate_age_of_maturity,
    gestation_ticks:             mutation_rate_gestation_ticks,
    max_organism_age:            mutation_rate_max_organism_age,
    vision_distance:             mutation_rate_vision_distance,
    hebb_eta_gain:               mutation_rate_hebb_eta_gain,
    juvenile_eta_scale:          mutation_rate_juvenile_eta_scale,
    inter_bias:                  mutation_rate_inter_bias,
    inter_update_rate:           mutation_rate_inter_update_rate,
    eligibility_retention:       mutation_rate_eligibility_retention,
    synapse_prune_threshold:     mutation_rate_synapse_prune_threshold,
    synapse_weight_perturbation: mutation_rate_synapse_weight_perturbation,
    add_synapse:                 mutation_rate_add_synapse,
    remove_synapse:              mutation_rate_remove_synapse,
    remove_neuron:               mutation_rate_remove_neuron,
    add_neuron_split_edge:       mutation_rate_add_neuron_split_edge,
    max_weight_delta_per_tick:   mutation_rate_max_weight_delta_per_tick,
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

fn clamp_mutation_rate(rate: f32) -> f32 {
    rate.clamp(MUTATION_RATE_MIN, MUTATION_RATE_MAX)
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
    // Zero is absorbing: a mutation-rate gene inherited as exactly 0.0 is
    // exempt from meta-mutation and stays 0.0 forever (logit(0) is -inf
    // anyway, and the exploration floor would otherwise resurrect it).
    // Zeroing a rate in the seed config therefore hard-disables that
    // operator for the lineage. The early return draws no RNG; the skip
    // condition is a pure function of the inherited genome, so determinism
    // is preserved.
    if current_rate == 0.0 {
        return 0.0;
    }
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

pub(super) fn effective_mutation_rate(rate: f32, global_mutation_rate_modifier: f32) -> f32 {
    (rate * global_mutation_rate_modifier).clamp(0.0, MUTATION_RATE_MAX)
}
