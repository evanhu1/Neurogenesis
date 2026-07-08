//! The three evolutionary operators. `mutate` mirrors the frozen, append-only
//! gate sequence of `sim-core/src/genome/mod.rs::mutate_genome`; `crossover` is
//! NEAT innovation-aligned; `reproduce` is sexual (`crossover` then `mutate`).
//! Meta-mutation of the rate block reuses the logit-space / baseline-pull /
//! zero-absorbing machinery from `mutation_rates.rs`.

use crate::cppn::{Activation, CppnConnGene, CppnGenome, CppnNodeGene, NodeKind};
use crate::genome::{Genome, MutationRates};
use rand::Rng;
use rand_distr::{Distribution, StandardNormal};

// ---- scalar perturbation helpers (ported) ---------------------------------

fn standard_normal<R: Rng + ?Sized>(rng: &mut R) -> f32 {
    StandardNormal.sample(rng)
}

fn perturb_clamped<R: Rng + ?Sized>(v: f32, stddev: f32, min: f32, max: f32, rng: &mut R) -> f32 {
    (v + standard_normal(rng) * stddev).clamp(min, max)
}

fn perturb_multiplicative_f32<R: Rng + ?Sized>(
    v: f32,
    log_stddev: f32,
    min: f32,
    max: f32,
    rng: &mut R,
) -> f32 {
    let scale = (log_stddev * standard_normal(rng)).exp();
    (v * scale).clamp(min, max)
}

fn perturb_multiplicative_u32<R: Rng + ?Sized>(
    v: u32,
    log_stddev: f32,
    min: u32,
    max: u32,
    rng: &mut R,
) -> u32 {
    if min >= max {
        return min;
    }
    let scale = (log_stddev * standard_normal(rng)).exp();
    let scaled = ((v as f32) * scale).round().clamp(min as f32, max as f32) as u32;
    if scaled != v {
        return scaled;
    }
    step_u32(v, min, max, rng)
}

fn step_u32<R: Rng + ?Sized>(v: u32, min: u32, max: u32, rng: &mut R) -> u32 {
    if min >= max {
        return min;
    }
    if v <= min {
        return min.saturating_add(1).min(max);
    }
    if v >= max {
        return max.saturating_sub(1).max(min);
    }
    if rng.random::<bool>() {
        v.saturating_add(1).min(max)
    } else {
        v.saturating_sub(1).max(min)
    }
}

// ---- clamp bands (ported) --------------------------------------------------

const LARGE_UNBOUNDED_LOG_STDDEV: f32 = 0.1;
const MIN_AGE_OF_MATURITY: u32 = 0;
const MAX_AGE_OF_MATURITY: u32 = 10_000;
const MIN_GESTATION_TICKS: u32 = 0;
const MAX_GESTATION_TICKS: u32 = 10;
const MIN_MAX_ORGANISM_AGE: u32 = 1;
const MAX_MAX_ORGANISM_AGE: u32 = 100_000;

// ---- mutation-rate meta-mutation (logit-space, ported) --------------------

const MUTATION_RATE_MIN: f32 = 1.0e-4;
const MUTATION_RATE_MAX: f32 = 0.5;
const MUTATION_RATE_LATENT_MAX: f32 = 13.815_511;
const META_MUTATION_BASELINE_PULL: f32 = 0.15;
const META_NOISE_SCALE: f32 = 0.12;
const META_EXPLORE_PROB: f32 = 0.1;
const META_EXPLORE_NOISE_SCALE: f32 = 0.35;

fn rate_to_latent(rate: f32) -> f32 {
    let clamped = rate.clamp(MUTATION_RATE_MIN, MUTATION_RATE_MAX);
    let normalized = (clamped - MUTATION_RATE_MIN) / (MUTATION_RATE_MAX - MUTATION_RATE_MIN);
    let eps = 1.0e-6;
    let p = normalized.clamp(eps, 1.0 - eps);
    (p / (1.0 - p)).ln().clamp(-MUTATION_RATE_LATENT_MAX, MUTATION_RATE_LATENT_MAX)
}

fn rate_from_latent(latent: f32) -> f32 {
    let l = latent.clamp(-MUTATION_RATE_LATENT_MAX, MUTATION_RATE_LATENT_MAX);
    let p = 1.0 / (1.0 + (-l).exp());
    MUTATION_RATE_MIN + p * (MUTATION_RATE_MAX - MUTATION_RATE_MIN)
}

fn mutate_single_rate<R: Rng + ?Sized>(current: f32, baseline: f32, rng: &mut R) -> f32 {
    // Zero-absorbing: a gene turned fully off stays off and draws no RNG.
    if current == 0.0 {
        return 0.0;
    }
    let noise_scale = if rng.random::<f32>() < META_EXPLORE_PROB {
        META_EXPLORE_NOISE_SCALE
    } else {
        META_NOISE_SCALE
    };
    let current_latent = rate_to_latent(current);
    let baseline_latent = rate_to_latent(baseline);
    let pulled = current_latent + (baseline_latent - current_latent) * META_MUTATION_BASELINE_PULL;
    let latent = pulled + standard_normal(rng) * noise_scale;
    rate_from_latent(latent)
}

/// Mutate 1–3 of the rate genes each call.
fn mutate_mutation_rates<R: Rng + ?Sized>(
    rates: &mut MutationRates,
    baseline: &MutationRates,
    rng: &mut R,
) {
    let mut arr = rates.as_array();
    let base = baseline.as_array();
    let count = 1 + rng.random_range(0..3usize);
    for _ in 0..count {
        let idx = rng.random_range(0..MutationRates::COUNT);
        arr[idx] = mutate_single_rate(arr[idx], base[idx], rng);
    }
    *rates = MutationRates::from_array(arr);
}

// ---- CPPN structural operators --------------------------------------------

fn cppn_has_conn(cppn: &CppnGenome, from: u64, to: u64) -> bool {
    cppn.conns.iter().any(|c| c.from == from && c.to == to)
}

/// DFS ancestor check: would adding `from -> to` create a cycle? (i.e. is
/// `from` reachable from `to` over enabled edges?)
fn would_create_cycle(cppn: &CppnGenome, from: u64, to: u64) -> bool {
    let mut stack = vec![to];
    let mut seen = std::collections::HashSet::new();
    while let Some(node) = stack.pop() {
        if node == from {
            return true;
        }
        if !seen.insert(node) {
            continue;
        }
        for c in &cppn.conns {
            if c.enabled && c.from == node {
                stack.push(c.to);
            }
        }
    }
    false
}

fn mutate_add_conn<R: Rng + ?Sized>(cppn: &mut CppnGenome, rng: &mut R) {
    if cppn.nodes.len() < 2 {
        return;
    }
    for _ in 0..8 {
        let from = cppn.nodes[rng.random_range(0..cppn.nodes.len())];
        let to = cppn.nodes[rng.random_range(0..cppn.nodes.len())];
        if from.id == to.id
            || matches!(to.kind, NodeKind::Input | NodeKind::Bias)
            || matches!(from.kind, NodeKind::Output)
        {
            continue;
        }
        if cppn_has_conn(cppn, from.id, to.id) || would_create_cycle(cppn, from.id, to.id) {
            continue;
        }
        cppn.conns.push(CppnConnGene {
            innovation: CppnGenome::conn_innovation(from.id, to.id),
            from: from.id,
            to: to.id,
            weight: standard_normal(rng),
            enabled: true,
        });
        return;
    }
}

fn mutate_add_node<R: Rng + ?Sized>(cppn: &mut CppnGenome, rng: &mut R) {
    let enabled: Vec<usize> = cppn
        .conns
        .iter()
        .enumerate()
        .filter(|(_, c)| c.enabled)
        .map(|(i, _)| i)
        .collect();
    if enabled.is_empty() {
        return;
    }
    let ci = enabled[rng.random_range(0..enabled.len())];
    let conn = cppn.conns[ci];
    cppn.conns[ci].enabled = false;
    let new_id = CppnGenome::split_node_id(conn.innovation);
    // Guard against re-splitting the same homologous connection twice.
    if cppn.nodes.iter().any(|n| n.id == new_id) {
        cppn.conns[ci].enabled = true;
        return;
    }
    let activation = Activation::ALL[rng.random_range(0..Activation::ALL.len())];
    cppn.nodes.push(CppnNodeGene {
        id: new_id,
        kind: NodeKind::Hidden,
        activation,
        bias: 0.0,
        io_slot: u16::MAX,
    });
    cppn.conns.push(CppnConnGene {
        innovation: CppnGenome::conn_innovation(conn.from, new_id),
        from: conn.from,
        to: new_id,
        weight: 1.0,
        enabled: true,
    });
    cppn.conns.push(CppnConnGene {
        innovation: CppnGenome::conn_innovation(new_id, conn.to),
        from: new_id,
        to: conn.to,
        weight: conn.weight,
        enabled: true,
    });
}

fn mutate_cppn_weights<R: Rng + ?Sized>(cppn: &mut CppnGenome, rng: &mut R) {
    if cppn.conns.is_empty() {
        return;
    }
    let mut any = false;
    for c in &mut cppn.conns {
        if rng.random::<f32>() < 0.8 {
            any = true;
            if rng.random::<f32>() < 0.1 {
                c.weight = standard_normal(rng);
            } else {
                c.weight += standard_normal(rng) * 0.15;
            }
        }
    }
    if !any {
        let idx = rng.random_range(0..cppn.conns.len());
        cppn.conns[idx].weight += standard_normal(rng) * 0.15;
    }
}

fn mutate_toggle_enable<R: Rng + ?Sized>(cppn: &mut CppnGenome, rng: &mut R) {
    if cppn.conns.is_empty() {
        return;
    }
    let idx = rng.random_range(0..cppn.conns.len());
    cppn.conns[idx].enabled = !cppn.conns[idx].enabled;
}

fn mutate_activation<R: Rng + ?Sized>(cppn: &mut CppnGenome, rng: &mut R) {
    let hidden_out: Vec<usize> = cppn
        .nodes
        .iter()
        .enumerate()
        .filter(|(_, n)| matches!(n.kind, NodeKind::Hidden | NodeKind::Output))
        .map(|(i, _)| i)
        .collect();
    if hidden_out.is_empty() {
        return;
    }
    let idx = hidden_out[rng.random_range(0..hidden_out.len())];
    cppn.nodes[idx].activation = Activation::ALL[rng.random_range(0..Activation::ALL.len())];
}

fn mutate_cppn_bias<R: Rng + ?Sized>(cppn: &mut CppnGenome, rng: &mut R) {
    let hidden_out: Vec<usize> = cppn
        .nodes
        .iter()
        .enumerate()
        .filter(|(_, n)| matches!(n.kind, NodeKind::Hidden | NodeKind::Output))
        .map(|(i, _)| i)
        .collect();
    if hidden_out.is_empty() {
        return;
    }
    let idx = hidden_out[rng.random_range(0..hidden_out.len())];
    cppn.nodes[idx].bias = perturb_clamped(cppn.nodes[idx].bias, 0.15, -3.0, 3.0, rng);
}

// ---- the operators ---------------------------------------------------------

/// Context for a mutation pass.
pub struct MutateCtx<'a> {
    pub global_mutation_rate_modifier: f32,
    pub meta_mutation_enabled: bool,
    /// Baseline (seed) rates for the meta-mutation baseline-pull.
    pub baseline_rates: &'a MutationRates,
}

#[inline]
fn effective(rate: f32, modifier: f32) -> f32 {
    (rate * modifier).clamp(0.0, 1.0)
}

/// Frozen, append-only gate sequence. New gates must be appended before the
/// meta-mutation step to preserve the RNG-draw prefix of existing operators.
pub fn mutate<R: Rng + ?Sized>(genome: &mut Genome, ctx: &MutateCtx, rng: &mut R) {
    let m = ctx.global_mutation_rate_modifier;
    let rates = genome.header.mutation_rates;

    if rng.random::<f32>() < effective(rates.age_of_maturity, m) {
        genome.header.lifecycle.age_of_maturity = perturb_multiplicative_u32(
            genome.header.lifecycle.age_of_maturity,
            LARGE_UNBOUNDED_LOG_STDDEV,
            MIN_AGE_OF_MATURITY,
            MAX_AGE_OF_MATURITY,
            rng,
        );
    }
    if rng.random::<f32>() < effective(rates.gestation_ticks, m) {
        genome.header.lifecycle.gestation_ticks = step_u32(
            u32::from(genome.header.lifecycle.gestation_ticks),
            MIN_GESTATION_TICKS,
            MAX_GESTATION_TICKS,
            rng,
        ) as u8;
    }
    if rng.random::<f32>() < effective(rates.max_organism_age, m) {
        genome.header.lifecycle.max_organism_age = perturb_multiplicative_u32(
            genome.header.lifecycle.max_organism_age,
            LARGE_UNBOUNDED_LOG_STDDEV,
            MIN_MAX_ORGANISM_AGE,
            MAX_MAX_ORGANISM_AGE,
            rng,
        );
    }
    if rng.random::<f32>() < effective(rates.hebb_eta_gain, m) {
        genome.header.plasticity.hebb_eta_gain = perturb_clamped(
            genome.header.plasticity.hebb_eta_gain,
            0.005,
            0.0,
            crate::genome::PlasticityGenes::HEBB_ETA_GAIN_MAX,
            rng,
        );
    }
    if rng.random::<f32>() < effective(rates.juvenile_eta_scale, m) {
        genome.header.plasticity.juvenile_eta_scale = perturb_clamped(
            genome.header.plasticity.juvenile_eta_scale,
            0.25,
            0.0,
            crate::genome::PlasticityGenes::JUVENILE_ETA_SCALE_MAX,
            rng,
        );
    }
    if rng.random::<f32>() < effective(rates.eligibility_retention, m) {
        genome.header.plasticity.eligibility_retention =
            perturb_clamped(genome.header.plasticity.eligibility_retention, 0.05, 0.0, 1.0, rng);
    }
    if rng.random::<f32>() < effective(rates.synapse_prune_threshold, m) {
        genome.header.plasticity.synapse_prune_threshold =
            perturb_clamped(genome.header.plasticity.synapse_prune_threshold, 0.02, 0.0, 1.0, rng);
    }
    // Morphology block — each scalar gated on the same rate.
    if rng.random::<f32>() < effective(rates.morphology, m) {
        for v in &mut genome.header.morphology {
            if rng.random::<f32>() < 0.5 {
                *v = perturb_clamped(*v, 0.1, 0.0, 1.0, rng);
            }
        }
    }
    // CPPN structural ops.
    if rng.random::<f32>() < effective(rates.cppn_weight_perturb, m) {
        mutate_cppn_weights(&mut genome.cppn, rng);
    }
    if rng.random::<f32>() < effective(rates.cppn_add_conn, m) {
        mutate_add_conn(&mut genome.cppn, rng);
    }
    if rng.random::<f32>() < effective(rates.cppn_add_node, m) {
        mutate_add_node(&mut genome.cppn, rng);
    }
    if rng.random::<f32>() < effective(rates.cppn_toggle_enable, m) {
        mutate_toggle_enable(&mut genome.cppn, rng);
    }
    if rng.random::<f32>() < effective(rates.cppn_mutate_activation, m) {
        mutate_activation(&mut genome.cppn, rng);
    }
    if rng.random::<f32>() < effective(rates.cppn_perturb_bias, m) {
        mutate_cppn_bias(&mut genome.cppn, rng);
    }
    // Appended-last gate: max_weight_delta_per_tick.
    if rng.random::<f32>() < effective(rates.max_weight_delta_per_tick, m) {
        genome.header.plasticity.max_weight_delta_per_tick = perturb_multiplicative_f32(
            genome.header.plasticity.max_weight_delta_per_tick,
            LARGE_UNBOUNDED_LOG_STDDEV,
            crate::genome::PlasticityGenes::MAX_WEIGHT_DELTA_MIN,
            crate::genome::PlasticityGenes::MAX_WEIGHT_DELTA_MAX,
            rng,
        );
    }

    if ctx.meta_mutation_enabled {
        let mut rates = genome.header.mutation_rates;
        mutate_mutation_rates(&mut rates, ctx.baseline_rates, rng);
        genome.header.mutation_rates = rates;
    }

    genome.canonicalize();
}

/// NEAT innovation-aligned crossover. `a` is the fitter parent (disjoint/excess
/// genes flow from it). One deterministic offspring.
pub fn crossover<R: Rng + ?Sized>(a: &Genome, b: &Genome, rng: &mut R) -> Genome {
    let mut node_map: std::collections::BTreeMap<u64, CppnNodeGene> =
        a.cppn.nodes.iter().map(|n| (n.id, *n)).collect();
    for n in &b.cppn.nodes {
        node_map.entry(n.id).or_insert(*n);
    }

    let b_conns: std::collections::HashMap<u64, &CppnConnGene> =
        b.cppn.conns.iter().map(|c| (c.innovation, c)).collect();

    let mut conns: Vec<CppnConnGene> = Vec::with_capacity(a.cppn.conns.len());
    for ca in &a.cppn.conns {
        if let Some(cb) = b_conns.get(&ca.innovation) {
            // Matching gene: coin-flip the weight/enabled from either parent.
            if rng.random::<bool>() {
                conns.push(*ca);
            } else {
                conns.push(**cb);
            }
        } else {
            // Disjoint/excess: from the fitter parent `a`.
            conns.push(*ca);
        }
    }

    let mut kept_nodes: Vec<CppnNodeGene> = node_map
        .values()
        .filter(|n| {
            matches!(n.kind, NodeKind::Input | NodeKind::Output | NodeKind::Bias)
                || conns.iter().any(|c| c.from == n.id || c.to == n.id)
        })
        .copied()
        .collect();
    kept_nodes.sort_by_key(|n| n.id);

    let header = crossover_header(a, b, rng);
    let mut child = Genome {
        cppn: CppnGenome {
            nodes: kept_nodes,
            conns,
        },
        header,
    };
    child.canonicalize();
    child
}

fn crossover_header<R: Rng + ?Sized>(
    a: &Genome,
    b: &Genome,
    rng: &mut R,
) -> crate::genome::HeaderGenes {
    let pick = |x: f32, y: f32, rng: &mut R| if rng.random::<bool>() { x } else { y };
    let ha = &a.header;
    let hb = &b.header;
    let mut header = ha.clone();
    header.plasticity.hebb_eta_gain =
        pick(ha.plasticity.hebb_eta_gain, hb.plasticity.hebb_eta_gain, rng);
    header.plasticity.juvenile_eta_scale =
        pick(ha.plasticity.juvenile_eta_scale, hb.plasticity.juvenile_eta_scale, rng);
    header.plasticity.eligibility_retention =
        pick(ha.plasticity.eligibility_retention, hb.plasticity.eligibility_retention, rng);
    header.plasticity.max_weight_delta_per_tick = pick(
        ha.plasticity.max_weight_delta_per_tick,
        hb.plasticity.max_weight_delta_per_tick,
        rng,
    );
    header.plasticity.synapse_prune_threshold =
        pick(ha.plasticity.synapse_prune_threshold, hb.plasticity.synapse_prune_threshold, rng);
    if rng.random::<bool>() {
        header.lifecycle = hb.lifecycle;
    }
    for (i, v) in header.morphology.iter_mut().enumerate() {
        if let Some(bv) = hb.morphology.get(i) {
            *v = pick(*v, *bv, rng);
        }
    }
    header
}

/// Sexual reproduction: crossover then mutate. `a` should be the fitter parent.
pub fn reproduce<R: Rng + ?Sized>(a: &Genome, b: &Genome, ctx: &MutateCtx, rng: &mut R) -> Genome {
    let mut child = crossover(a, b, rng);
    mutate(&mut child, ctx, rng);
    child
}
