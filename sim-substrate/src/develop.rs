//! ES-HyperNEAT expression: genotype → phenotype. Pure and RNG-free, so it is
//! cacheable by CPPN content hash and identical on every thread.
//!
//! Pipeline (plan §develop):
//!   1. compile CPPN
//!   2. interface selection via AFF self-probe (+ viability floor)
//!   3. hidden-neuron discovery via a bounded adaptive quadtree
//!   4. connections via LEO gate; weight from W, plasticity_scale from PLR
//!   5. neuron bias/α from NBIAS/LTC; prune dangling; enforce hard rails

use crate::brain::{
    constrain_weight, fast_tanh, inter_alpha_from_log_time_constant, BrainNet, Edge, Neuron,
    NeuronKind,
};
use crate::catalog::{plane, ActionLayout, Coord, ObsLayout, SubstrateCatalog};
use crate::cppn::{out, CompiledCppn, CPPN_INPUTS};
use crate::genome::Genome;

/// Absolute safety rails, independent of the (evolvable) header. Even a
/// pathological genome cannot exceed these.
#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub struct DevelopConfig {
    pub hard_max_neurons: u32,
    pub hard_max_edges: u32,
}

impl Default for DevelopConfig {
    fn default() -> Self {
        DevelopConfig {
            hard_max_neurons: 256,
            hard_max_edges: 4096,
        }
    }
}

pub struct Phenotype {
    pub brain: BrainNet,
    pub obs_layout: ObsLayout,
    pub action_layout: ActionLayout,
    /// Denormalized morphology values, aligned to the catalog morphology schema.
    pub morphology_values: Vec<f32>,
}

/// Build the fixed 11-input probe vector for a directed substrate query.
#[inline]
fn probe(src: Coord, tgt: Coord) -> [f32; CPPN_INPUTS] {
    let dx = tgt.x - src.x;
    let dy = tgt.y - src.y;
    let dz = tgt.z - src.z;
    let dist = (dx * dx + dy * dy + dz * dz).sqrt();
    [src.x, src.y, src.z, tgt.x, tgt.y, tgt.z, dx, dy, dz, dist, 1.0]
}

#[inline]
fn plr_to_scale(plr: f32) -> f32 {
    // Map the raw PLR output to a non-negative multiplier in [0, 2], centered on
    // 1.0 so `PLR == 0` reproduces the global rule.
    ((fast_tanh(plr) * 0.5 + 0.5) * 2.0).clamp(0.0, 2.0)
}

pub fn develop(genome: &Genome, catalog: &SubstrateCatalog, cfg: &DevelopConfig) -> Phenotype {
    let compiled = genome.cppn.compile();
    let dev = &genome.header.develop;

    // Fallback: a cyclic/empty CPPN cannot express; hand back a minimal viable
    // brain (one sensor, one actuator, no hidden) so no organism is degenerate.
    let Some(cppn) = compiled else {
        return minimal_viable(genome, catalog);
    };

    // ---- 2. interface selection --------------------------------------------
    let mut sensor_aff: Vec<(usize, f32)> = catalog
        .sensors
        .iter()
        .enumerate()
        .map(|(i, s)| (i, cppn.evaluate(&probe(s.coord, s.coord))[out::AFF]))
        .collect();
    let mut actuator_aff: Vec<(usize, f32)> = catalog
        .actuators
        .iter()
        .enumerate()
        .map(|(i, a)| (i, cppn.evaluate(&probe(a.coord, a.coord))[out::AFF]))
        .collect();

    let mut expressed_sensors: Vec<usize> = sensor_aff
        .iter()
        .filter(|(_, aff)| *aff > dev.aff_threshold)
        .map(|(i, _)| *i)
        .collect();
    if expressed_sensors.is_empty() && !sensor_aff.is_empty() {
        sensor_aff.sort_by(|a, b| b.1.total_cmp(&a.1));
        expressed_sensors.push(sensor_aff[0].0);
    }
    let mut expressed_actuators: Vec<usize> = actuator_aff
        .iter()
        .filter(|(_, aff)| *aff > dev.aff_threshold)
        .map(|(i, _)| *i)
        .collect();
    if expressed_actuators.is_empty() && !actuator_aff.is_empty() {
        actuator_aff.sort_by(|a, b| b.1.total_cmp(&a.1));
        expressed_actuators.push(actuator_aff[0].0);
    }
    expressed_sensors.sort_unstable();
    expressed_actuators.sort_unstable();

    // Build obs layout + input neurons (one input neuron per obs slot).
    let mut obs_layout = ObsLayout::default();
    let mut input_coords: Vec<Coord> = Vec::new();
    for &si in &expressed_sensors {
        let spec = &catalog.sensors[si];
        obs_layout.sensor_indices.push(si);
        obs_layout.offsets.push(obs_layout.len);
        for slot in 0..spec.arity.max(1) {
            // Fan multi-arity sensors out along x so each obs slot has a
            // distinct substrate coordinate.
            let jitter = slot as f32 * 0.01;
            input_coords.push(Coord::new(spec.coord.x + jitter, spec.coord.y, spec.coord.z));
            obs_layout.len += 1;
        }
    }

    let mut action_layout = ActionLayout::default();
    let mut output_coords: Vec<Coord> = Vec::new();
    for &ai in &expressed_actuators {
        let spec = &catalog.actuators[ai];
        action_layout.actuator_indices.push(ai);
        output_coords.push(spec.coord);
        action_layout.len += 1;
    }

    // ---- 3. hidden-neuron discovery (bounded adaptive quadtree) ------------
    let neuron_budget = (dev.max_neurons).min(cfg.hard_max_neurons) as usize;
    let hidden_budget = neuron_budget.saturating_sub(input_coords.len() + output_coords.len());
    let hidden_coords = discover_hidden(&cppn, dev, hidden_budget);

    // ---- assemble neuron list: [inputs][hidden][outputs] -------------------
    let input_count = input_coords.len() as u32;
    let hidden_count = hidden_coords.len() as u32;
    let output_count = output_coords.len() as u32;

    let mut coords: Vec<Coord> =
        Vec::with_capacity(input_coords.len() + hidden_coords.len() + output_coords.len());
    coords.extend_from_slice(&input_coords);
    coords.extend_from_slice(&hidden_coords);
    coords.extend_from_slice(&output_coords);

    let mut neurons: Vec<Neuron> = Vec::with_capacity(coords.len());
    for (idx, coord) in coords.iter().enumerate() {
        let kind = if (idx as u32) < input_count {
            NeuronKind::Input
        } else if (idx as u32) < input_count + hidden_count {
            NeuronKind::Hidden
        } else {
            NeuronKind::Output
        };
        let selfq = cppn.evaluate(&probe(*coord, *coord));
        let bias = (selfq[out::NBIAS]).clamp(-1.0, 1.0);
        let alpha = inter_alpha_from_log_time_constant(selfq[out::LTC]);
        neurons.push(Neuron {
            kind,
            bias: if kind == NeuronKind::Input { 0.0 } else { bias },
            alpha,
            state: 0.0,
            activation: 0.0,
            mean_activation: 0.0,
        });
    }

    // ---- 4. connections (LEO gate) ----------------------------------------
    let edge_budget = dev.max_edges.min(cfg.hard_max_edges) as usize;
    let hidden_start = input_count as usize;
    let output_start = (input_count + hidden_count) as usize;
    let n = coords.len();
    let mut edges: Vec<Edge> = Vec::new();

    let try_edge = |edges: &mut Vec<Edge>, from: usize, to: usize| {
        if edges.len() >= edge_budget {
            return;
        }
        let q = cppn.evaluate(&probe(coords[from], coords[to]));
        if q[out::LEO] > dev.leo_threshold {
            edges.push(Edge {
                from: from as u32,
                to: to as u32,
                weight: constrain_weight(q[out::W] * dev.weight_scale),
                plasticity_scale: plr_to_scale(q[out::PLR]),
                eligibility: 0.0,
                pending: 0.0,
            });
        }
    };

    // input/hidden -> hidden
    for to in hidden_start..output_start {
        for from in 0..output_start {
            if from == to {
                continue;
            }
            try_edge(&mut edges, from, to);
        }
    }
    // input/hidden -> output
    for to in output_start..n {
        for from in 0..output_start {
            try_edge(&mut edges, from, to);
        }
    }

    // ---- 5. prune dangling hidden neurons ----------------------------------
    let (neurons, edges, input_count, hidden_count, output_count) =
        prune_dangling(neurons, edges, input_count, hidden_count, output_count);

    let brain = BrainNet::new(neurons, edges, input_count, hidden_count, output_count);
    debug_assert!(brain.edges.len() <= edge_budget);

    if brain.output_count == 0 {
        // Degenerate: no actuator survived pruning. Fall back to a viable brain.
        return minimal_viable(genome, catalog);
    }

    Phenotype {
        brain,
        obs_layout,
        action_layout,
        morphology_values: denormalize_morphology(genome, catalog),
    }
}

/// Adaptive quadtree over the hidden plane. Cells whose CPPN-`W` output varies
/// strongly across their corners subdivide (up to `depth`); the centers of
/// high-variance leaves become hidden neurons, capped at `budget`.
fn discover_hidden(
    cppn: &CompiledCppn,
    dev: &crate::genome::DevelopGenes,
    budget: usize,
) -> Vec<Coord> {
    if budget == 0 {
        return Vec::new();
    }
    let z = plane::HIDDEN;
    let origin = Coord::new(0.0, 0.0, plane::EXTEROCEPTIVE);
    let sample =
        |x: f32, y: f32| -> f32 { cppn.evaluate(&probe(origin, Coord::new(x, y, z)))[out::W] };

    struct Cell {
        cx: f32,
        cy: f32,
        half: f32,
        depth: u8,
    }
    let mut stack = vec![Cell {
        cx: 0.0,
        cy: 0.0,
        half: 1.0,
        depth: 0,
    }];
    let mut out_coords: Vec<Coord> = Vec::new();
    while let Some(cell) = stack.pop() {
        if out_coords.len() >= budget {
            break;
        }
        let corners = [
            sample(cell.cx - cell.half, cell.cy - cell.half),
            sample(cell.cx + cell.half, cell.cy - cell.half),
            sample(cell.cx - cell.half, cell.cy + cell.half),
            sample(cell.cx + cell.half, cell.cy + cell.half),
        ];
        let mean = corners.iter().sum::<f32>() / 4.0;
        let var = corners.iter().map(|c| (c - mean) * (c - mean)).sum::<f32>() / 4.0;
        if cell.depth < dev.quadtree_depth && var > dev.variance_threshold {
            let h = cell.half * 0.5;
            for (sx, sy) in [(-1.0, -1.0), (1.0, -1.0), (-1.0, 1.0), (1.0, 1.0)] {
                stack.push(Cell {
                    cx: cell.cx + sx * h,
                    cy: cell.cy + sy * h,
                    half: h,
                    depth: cell.depth + 1,
                });
            }
        } else if mean.abs() > dev.leo_threshold {
            out_coords.push(Coord::new(cell.cx, cell.cy, z));
        }
    }
    out_coords.truncate(budget);
    out_coords
}

/// Iteratively drop hidden neurons that have no incoming or no outgoing edge,
/// then compact indices. Inputs and outputs are always retained.
fn prune_dangling(
    mut neurons: Vec<Neuron>,
    mut edges: Vec<Edge>,
    input_count: u32,
    mut hidden_count: u32,
    output_count: u32,
) -> (Vec<Neuron>, Vec<Edge>, u32, u32, u32) {
    loop {
        let n = neurons.len();
        let hidden_start = input_count as usize;
        let output_start = (input_count + hidden_count) as usize;
        let mut has_in = vec![false; n];
        let mut has_out = vec![false; n];
        for e in &edges {
            has_out[e.from as usize] = true;
            has_in[e.to as usize] = true;
        }
        let dead: Vec<usize> = (hidden_start..output_start)
            .filter(|&i| !has_in[i] || !has_out[i])
            .collect();
        if dead.is_empty() {
            break;
        }
        let dead_set: std::collections::HashSet<usize> = dead.into_iter().collect();
        let mut remap = vec![u32::MAX; n];
        let mut new_neurons: Vec<Neuron> = Vec::with_capacity(n - dead_set.len());
        for (i, neuron) in neurons.into_iter().enumerate() {
            if dead_set.contains(&i) {
                continue;
            }
            remap[i] = new_neurons.len() as u32;
            new_neurons.push(neuron);
        }
        neurons = new_neurons;
        hidden_count -= dead_set.len() as u32;
        edges = edges
            .into_iter()
            .filter(|e| remap[e.from as usize] != u32::MAX && remap[e.to as usize] != u32::MAX)
            .map(|mut e| {
                e.from = remap[e.from as usize];
                e.to = remap[e.to as usize];
                e
            })
            .collect();
    }
    (neurons, edges, input_count, hidden_count, output_count)
}

fn denormalize_morphology(genome: &Genome, catalog: &SubstrateCatalog) -> Vec<f32> {
    catalog
        .morphology
        .iter()
        .enumerate()
        .map(|(i, param)| {
            let normalized = genome.header.morphology.get(i).copied().unwrap_or(
                if (param.max - param.min).abs() < f32::EPSILON {
                    0.0
                } else {
                    (param.default - param.min) / (param.max - param.min)
                },
            );
            param.denormalize(normalized)
        })
        .collect()
}

/// One-sensor / one-actuator brain used when the CPPN cannot express a viable
/// interface. Guarantees no organism is ever degenerate.
fn minimal_viable(genome: &Genome, catalog: &SubstrateCatalog) -> Phenotype {
    let mut obs_layout = ObsLayout::default();
    let mut neurons: Vec<Neuron> = Vec::new();
    if let Some((si, _spec)) = catalog.sensors.iter().enumerate().next() {
        obs_layout.sensor_indices.push(si);
        obs_layout.offsets.push(0);
        obs_layout.len = 1;
        neurons.push(Neuron {
            kind: NeuronKind::Input,
            bias: 0.0,
            alpha: 1.0,
            state: 0.0,
            activation: 0.0,
            mean_activation: 0.0,
        });
    }
    let mut action_layout = ActionLayout::default();
    if let Some((ai, _)) = catalog.actuators.iter().enumerate().next() {
        action_layout.actuator_indices.push(ai);
        action_layout.len = 1;
        neurons.push(Neuron {
            kind: NeuronKind::Output,
            bias: 0.0,
            alpha: 1.0,
            state: 0.0,
            activation: 0.0,
            mean_activation: 0.0,
        });
    }
    let input_count = obs_layout.len as u32;
    let output_count = action_layout.len as u32;
    let mut edges = Vec::new();
    if input_count == 1 && output_count == 1 {
        edges.push(Edge {
            from: 0,
            to: 1,
            weight: constrain_weight(0.5),
            plasticity_scale: 1.0,
            eligibility: 0.0,
            pending: 0.0,
        });
    }
    let brain = BrainNet::new(neurons, edges, input_count, 0, output_count);
    Phenotype {
        brain,
        obs_layout,
        action_layout,
        morphology_values: denormalize_morphology(genome, catalog),
    }
}
