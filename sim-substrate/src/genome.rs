//! The complete heritable specification of one organism: a CPPN that develops
//! into the brain + interface, plus a header of direct scalars for everything
//! the CPPN does not paint (global plasticity, lifecycle, develop params,
//! per-operator mutation rates, and morphology).

use crate::cppn::CppnGenome;
use serde::{Deserialize, Serialize};

/// Global plasticity constants — the same fields and clamp bands as today's
/// `PlasticityGenes`. The CPPN's per-edge `PLR` scales `hebb_eta_gain`
/// multiplicatively (hybrid adaptive-HyperNEAT).
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct PlasticityGenes {
    pub hebb_eta_gain: f32,
    pub juvenile_eta_scale: f32,
    pub eligibility_retention: f32,
    pub max_weight_delta_per_tick: f32,
    pub synapse_prune_threshold: f32,
}

impl PlasticityGenes {
    pub const HEBB_ETA_GAIN_MAX: f32 = 0.2;
    pub const JUVENILE_ETA_SCALE_MAX: f32 = 4.0;
    pub const MAX_WEIGHT_DELTA_MIN: f32 = 0.005;
    pub const MAX_WEIGHT_DELTA_MAX: f32 = 0.5;
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct LifecycleGenes {
    pub age_of_maturity: u32,
    pub gestation_ticks: u8,
    pub max_organism_age: u32,
}

/// Bounds + budgets `develop()` enforces per birth.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct DevelopGenes {
    pub aff_threshold: f32,
    pub leo_threshold: f32,
    pub weight_scale: f32,
    /// Quadtree subdivision cap (bounded per birth; unbounded across lineages).
    pub quadtree_depth: u8,
    /// Variance above which a quadtree cell subdivides.
    pub variance_threshold: f32,
    pub max_neurons: u32,
    pub max_edges: u32,
}

/// Per-operator mutation rates — an evolvable strategy vector meta-mutated in
/// logit space (baseline-pull + zero-absorbing), mirroring `mutation_rates.rs`.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct MutationRates {
    // Header scalar rates.
    pub age_of_maturity: f32,
    pub gestation_ticks: f32,
    pub max_organism_age: f32,
    pub hebb_eta_gain: f32,
    pub juvenile_eta_scale: f32,
    pub eligibility_retention: f32,
    pub synapse_prune_threshold: f32,
    pub max_weight_delta_per_tick: f32,
    pub morphology: f32,
    // CPPN structural rates.
    pub cppn_weight_perturb: f32,
    pub cppn_add_conn: f32,
    pub cppn_add_node: f32,
    pub cppn_toggle_enable: f32,
    pub cppn_mutate_activation: f32,
    pub cppn_perturb_bias: f32,
}

impl MutationRates {
    pub const COUNT: usize = 15;

    pub fn as_array(&self) -> [f32; Self::COUNT] {
        [
            self.age_of_maturity,
            self.gestation_ticks,
            self.max_organism_age,
            self.hebb_eta_gain,
            self.juvenile_eta_scale,
            self.eligibility_retention,
            self.synapse_prune_threshold,
            self.max_weight_delta_per_tick,
            self.morphology,
            self.cppn_weight_perturb,
            self.cppn_add_conn,
            self.cppn_add_node,
            self.cppn_toggle_enable,
            self.cppn_mutate_activation,
            self.cppn_perturb_bias,
        ]
    }

    pub fn from_array(a: [f32; Self::COUNT]) -> Self {
        MutationRates {
            age_of_maturity: a[0],
            gestation_ticks: a[1],
            max_organism_age: a[2],
            hebb_eta_gain: a[3],
            juvenile_eta_scale: a[4],
            eligibility_retention: a[5],
            synapse_prune_threshold: a[6],
            max_weight_delta_per_tick: a[7],
            morphology: a[8],
            cppn_weight_perturb: a[9],
            cppn_add_conn: a[10],
            cppn_add_node: a[11],
            cppn_toggle_enable: a[12],
            cppn_mutate_activation: a[13],
            cppn_perturb_bias: a[14],
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeaderGenes {
    pub plasticity: PlasticityGenes,
    pub lifecycle: LifecycleGenes,
    pub develop: DevelopGenes,
    pub mutation_rates: MutationRates,
    /// Normalized `[0,1]` scalars aligned to the environment's morphology schema
    /// (`SubstrateCatalog::morphology`). Denormalized by the environment.
    pub morphology: Vec<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Genome {
    pub cppn: CppnGenome,
    pub header: HeaderGenes,
}

impl Genome {
    /// Canonicalize the CPPN so the bincode form is bit-stable.
    pub fn canonicalize(&mut self) {
        self.cppn.canonicalize();
    }

    /// Flat "bitstring" form. Callers should `canonicalize()` first for a
    /// bit-stable, cache-keyable result.
    pub fn to_bytes(&self) -> Vec<u8> {
        bincode::serialize(self).expect("genome serialize")
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Genome, bincode::Error> {
        bincode::deserialize(bytes)
    }
}
