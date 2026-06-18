use sim_types::{OrganismState, WorldConfig};

// Kleiber's law: basal metabolic rate scales as mass^0.75. Sub-linear scaling
// means larger bodies pay more in absolute terms but less per unit of mass,
// which prevents large-body phenotypes from being strictly uninhabitable.
const BODY_MASS_METABOLIC_EXPONENT: f32 = 0.75;

pub(crate) fn refresh_organism_base_metabolic_cost(
    organism: &mut OrganismState,
    body_mass_metabolic_cost_coeff: f32,
) {
    organism.base_metabolic_cost =
        organism_base_metabolic_cost(organism, body_mass_metabolic_cost_coeff);
}

pub(crate) fn organism_passive_metabolic_energy_cost(
    config: &WorldConfig,
    organism: &OrganismState,
) -> f32 {
    // Homeostatic metabolism: low-energy organisms downregulate their passive
    // burn so they get more ticks to find food before starving. The factor is
    // 1.0 above a small energy threshold and ramps linearly down to a 0.5 floor
    // as energy approaches 0. Pure deterministic function of energy (no RNG).
    const HOMEOSTATIC_THRESHOLD: f32 = 5.0;
    let factor = if organism.energy >= HOMEOSTATIC_THRESHOLD {
        1.0
    } else {
        0.5 + 0.5 * (organism.energy.max(0.0) / HOMEOSTATIC_THRESHOLD)
    };
    config.passive_metabolism_cost_per_unit * organism.base_metabolic_cost * factor
}

fn organism_base_metabolic_cost(
    organism: &OrganismState,
    body_mass_metabolic_cost_coeff: f32,
) -> f32 {
    let inter_neuron_count = organism.brain.inter.len() as f32;
    let sensory_neuron_count = organism.brain.sensory.len() as f32;
    let vision_distance_cost_units = organism.genome.topology.vision_distance as f32 / 3.0;
    let body_mass_cost_units =
        body_mass_metabolic_cost_units(organism.max_health, body_mass_metabolic_cost_coeff);
    inter_neuron_count + sensory_neuron_count + vision_distance_cost_units + body_mass_cost_units
}

fn body_mass_metabolic_cost_units(max_health: f32, coeff: f32) -> f32 {
    if coeff <= 0.0 {
        return 0.0;
    }
    coeff * max_health.max(0.0).powf(BODY_MASS_METABOLIC_EXPONENT)
}
