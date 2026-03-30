use super::*;

pub(crate) fn mutate_random_neuron_location<R: Rng + ?Sized>(
    genome: &mut OrganismGenome,
    rng: &mut R,
) {
    let enabled_inter = genome.num_neurons as usize;
    let total = SENSORY_COUNT as usize + enabled_inter + ACTION_COUNT;
    if total == 0 {
        return;
    }

    let idx = rng.random_range(0..total);
    let location = if idx < SENSORY_COUNT as usize {
        genome.sensory_locations.get_mut(idx)
    } else if idx < SENSORY_COUNT as usize + enabled_inter {
        genome
            .inter_locations
            .get_mut(idx.saturating_sub(SENSORY_COUNT as usize))
    } else {
        genome
            .action_locations
            .get_mut(idx.saturating_sub(SENSORY_COUNT as usize + enabled_inter))
    };

    if let Some(location) = location {
        location.x = perturb_clamped(
            location.x,
            LOCATION_PERTURBATION_STDDEV,
            BRAIN_SPACE_MIN,
            BRAIN_SPACE_MAX,
            rng,
        );
        location.y = perturb_clamped(
            location.y,
            LOCATION_PERTURBATION_STDDEV,
            BRAIN_SPACE_MIN,
            BRAIN_SPACE_MAX,
            rng,
        );
    }
}

pub(crate) fn mutate_synapse_weights<R: Rng + ?Sized>(genome: &mut OrganismGenome, rng: &mut R) {
    mutate_many_or_one(
        &mut genome.edges,
        SYNAPSE_WEIGHT_PERTURB_EDGE_RATE,
        rng,
        |edge, rng| {
            if rng.random::<f32>() < SYNAPSE_WEIGHT_REPLACEMENT_RATE {
                edge.weight = super::spatial_prior::sample_lognormal_weight(rng);
            } else {
                let magnitude_scale =
                    (SYNAPSE_WEIGHT_PERTURBATION_STDDEV * standard_normal(rng)).exp();
                edge.weight = constrain_weight(edge.weight * magnitude_scale);
            }
        },
    );
}

pub(crate) fn mutate_inter_biases<R: Rng + ?Sized>(genome: &mut OrganismGenome, rng: &mut R) {
    mutate_many_or_one(
        &mut genome.inter_biases,
        INTER_BIAS_PERTURB_NEURON_RATE,
        rng,
        |bias, rng| {
            *bias = perturb_clamped(*bias, BIAS_PERTURBATION_STDDEV, -BIAS_MAX, BIAS_MAX, rng);
        },
    );
}

pub(crate) fn mutate_inter_update_rates<R: Rng + ?Sized>(genome: &mut OrganismGenome, rng: &mut R) {
    mutate_many_or_one(
        &mut genome.inter_log_time_constants,
        INTER_UPDATE_RATE_PERTURB_NEURON_RATE,
        rng,
        |log_tau, rng| {
            *log_tau = perturb_clamped(
                *log_tau,
                INTER_LOG_TIME_CONSTANT_PERTURBATION_STDDEV,
                INTER_LOG_TIME_CONSTANT_MIN,
                INTER_LOG_TIME_CONSTANT_MAX,
                rng,
            );
        },
    );
}

pub(crate) fn inter_alpha_from_log_time_constant(log_time_constant: f32) -> f32 {
    let clamped_log_time_constant =
        log_time_constant.clamp(INTER_LOG_TIME_CONSTANT_MIN, INTER_LOG_TIME_CONSTANT_MAX);
    let time_constant = clamped_log_time_constant
        .exp()
        .clamp(INTER_TIME_CONSTANT_MIN, INTER_TIME_CONSTANT_MAX);
    1.0 - (-1.0 / time_constant).exp()
}
