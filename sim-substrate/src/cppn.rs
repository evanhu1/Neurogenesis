//! Compositional Pattern Producing Network — the compact, indirectly-encoding
//! genotype that `develop()` expands into a phenotype brain.
//!
//! Node/connection identity is by 64-bit **structural hash**, not a global
//! innovation counter, so crossover alignment is deterministic under parallel,
//! continuous-birth reproduction (plan risk #1).

use crate::rng::{hash_conn, hash_node};
use serde::{Deserialize, Serialize};

/// Fixed CPPN input signature: two 3-D substrate coordinates, their component
/// deltas, euclidean distance, and a constant bias lane. The `z` axis is the
/// **functional plane** — exteroceptive sensors, interoceptive sensors, hidden
/// neurons, and actuators each live on their own plane so the CPPN can key on
/// function without semantically-unrelated inputs colliding in geometry
/// (HyperNEAT fracture mitigation, plan risk #2).
pub const CPPN_INPUTS: usize = 11;
/// Fixed CPPN output signature: weight, link-expression, neuron bias, log time
/// constant, affordance, and per-edge plasticity-rate scale.
pub const CPPN_OUTPUTS: usize = 6;

/// Output-vector indices.
pub mod out {
    pub const W: usize = 0;
    pub const LEO: usize = 1;
    pub const NBIAS: usize = 2;
    pub const LTC: usize = 3;
    pub const AFF: usize = 4;
    pub const PLR: usize = 5;
}

/// Activation functions available to CPPN nodes. The classic HyperNEAT palette:
/// a mix of symmetric, periodic, and monotone functions gives the CPPN its
/// regularity-generating power.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Activation {
    Tanh,
    Sigmoid,
    Gaussian,
    Sin,
    Abs,
    Linear,
}

impl Activation {
    pub const ALL: [Activation; 6] = [
        Activation::Tanh,
        Activation::Sigmoid,
        Activation::Gaussian,
        Activation::Sin,
        Activation::Abs,
        Activation::Linear,
    ];

    #[inline]
    pub fn apply(self, x: f32) -> f32 {
        match self {
            Activation::Tanh => crate::brain::fast_tanh(x),
            Activation::Sigmoid => 1.0 / (1.0 + (-x).exp()),
            Activation::Gaussian => (-(x * x)).exp(),
            Activation::Sin => x.sin(),
            Activation::Abs => x.abs(),
            Activation::Linear => x,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NodeKind {
    Input,
    Output,
    Hidden,
    Bias,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct CppnNodeGene {
    pub id: u64,
    pub kind: NodeKind,
    pub activation: Activation,
    pub bias: f32,
    /// Input/output nodes carry a fixed slot index into the I/O vectors; hidden
    /// nodes leave it at `u16::MAX`.
    pub io_slot: u16,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct CppnConnGene {
    pub innovation: u64,
    pub from: u64,
    pub to: u64,
    pub weight: f32,
    pub enabled: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CppnGenome {
    pub nodes: Vec<CppnNodeGene>,
    pub conns: Vec<CppnConnGene>,
}

impl CppnGenome {
    /// Put the genome in canonical form: nodes by id, connections by innovation.
    /// Canonical order makes the bincode form bit-stable and keys the develop
    /// cache, and it is the merge order crossover walks.
    pub fn canonicalize(&mut self) {
        self.nodes.sort_by_key(|n| n.id);
        self.nodes.dedup_by_key(|n| n.id);
        self.conns.sort_by_key(|c| c.innovation);
        self.conns.dedup_by_key(|c| c.innovation);
    }

    pub fn conn_innovation(from: u64, to: u64) -> u64 {
        hash_conn(from, to)
    }

    pub fn split_node_id(parent_conn_innovation: u64) -> u64 {
        hash_node(parent_conn_innovation)
    }

    /// Compile into an evaluatable, topologically-sorted network. Returns `None`
    /// only if the graph has a cycle among enabled connections (CPPNs are DAGs;
    /// the operators never introduce a cycle, so this is a hard invariant).
    pub fn compile(&self) -> Option<CompiledCppn> {
        CompiledCppn::from_genome(self)
    }
}

/// A CPPN flattened for fast, allocation-light evaluation. Nodes are stored in
/// topological order; each carries its incoming edges so a single forward sweep
/// produces every output.
pub struct CompiledCppn {
    order: Vec<CompiledNode>,
    input_slot_to_idx: [usize; CPPN_INPUTS],
    output_slot_to_idx: [Option<usize>; CPPN_OUTPUTS],
    scratch: std::cell::RefCell<Vec<f32>>,
    /// Content hash keying the develop cache and dedup.
    pub content_hash: u64,
}

struct CompiledNode {
    kind: NodeKind,
    activation: Activation,
    bias: f32,
    io_slot: u16,
    incoming: Vec<(usize, f32)>, // (source index in `order`, weight)
}

impl CompiledCppn {
    fn from_genome(genome: &CppnGenome) -> Option<Self> {
        let n = genome.nodes.len();
        let id_to_pos: std::collections::HashMap<u64, usize> =
            genome.nodes.iter().enumerate().map(|(i, node)| (node.id, i)).collect();

        // Adjacency + in-degree over enabled connections only.
        let mut adj: Vec<Vec<(usize, f32)>> = vec![Vec::new(); n];
        let mut incoming: Vec<Vec<(usize, f32)>> = vec![Vec::new(); n];
        let mut in_degree = vec![0usize; n];
        for c in &genome.conns {
            if !c.enabled {
                continue;
            }
            let (Some(&f), Some(&t)) = (id_to_pos.get(&c.from), id_to_pos.get(&c.to)) else {
                continue;
            };
            adj[f].push((t, c.weight));
            incoming[t].push((f, c.weight));
            in_degree[t] += 1;
        }

        // Kahn topological sort.
        let mut ready: Vec<usize> = (0..n).filter(|&i| in_degree[i] == 0).collect();
        ready.sort_unstable(); // determinism
        let mut topo: Vec<usize> = Vec::with_capacity(n);
        let mut in_degree_work = in_degree.clone();
        while let Some(node) = ready.pop() {
            topo.push(node);
            for &(t, _) in &adj[node] {
                in_degree_work[t] -= 1;
                if in_degree_work[t] == 0 {
                    ready.push(t);
                }
            }
        }
        if topo.len() != n {
            return None; // cycle
        }

        // Remap: position in `order` == position in `topo`.
        let mut pos_to_order = vec![usize::MAX; n];
        for (order_idx, &genome_pos) in topo.iter().enumerate() {
            pos_to_order[genome_pos] = order_idx;
        }

        let mut order: Vec<CompiledNode> = Vec::with_capacity(n);
        let mut input_slot_to_idx = [usize::MAX; CPPN_INPUTS];
        let mut output_slot_to_idx: [Option<usize>; CPPN_OUTPUTS] = [None; CPPN_OUTPUTS];
        for &genome_pos in &topo {
            let node = &genome.nodes[genome_pos];
            let incoming_mapped: Vec<(usize, f32)> = incoming[genome_pos]
                .iter()
                .map(|&(src_pos, w)| (pos_to_order[src_pos], w))
                .collect();
            let order_idx = order.len();
            match node.kind {
                NodeKind::Input => {
                    if (node.io_slot as usize) < CPPN_INPUTS {
                        input_slot_to_idx[node.io_slot as usize] = order_idx;
                    }
                }
                NodeKind::Output => {
                    if (node.io_slot as usize) < CPPN_OUTPUTS {
                        output_slot_to_idx[node.io_slot as usize] = Some(order_idx);
                    }
                }
                _ => {}
            }
            order.push(CompiledNode {
                kind: node.kind,
                activation: node.activation,
                bias: node.bias,
                io_slot: node.io_slot,
                incoming: incoming_mapped,
            });
        }

        let content_hash = content_hash(genome);
        Some(CompiledCppn {
            scratch: std::cell::RefCell::new(vec![0.0; order.len()]),
            order,
            input_slot_to_idx,
            output_slot_to_idx,
            content_hash,
        })
    }

    /// Evaluate the CPPN for a fixed input vector, writing all outputs.
    pub fn evaluate(&self, inputs: &[f32; CPPN_INPUTS]) -> [f32; CPPN_OUTPUTS] {
        let mut values = self.scratch.borrow_mut();
        for (idx, node) in self.order.iter().enumerate() {
            let raw = match node.kind {
                NodeKind::Bias => 1.0,
                NodeKind::Input => {
                    let slot = node.io_slot as usize;
                    if slot < CPPN_INPUTS {
                        inputs[slot]
                    } else {
                        0.0
                    }
                }
                _ => {
                    let mut sum = node.bias;
                    for &(src, w) in &node.incoming {
                        sum += values[src] * w;
                    }
                    node.activation.apply(sum)
                }
            };
            values[idx] = raw;
        }

        let mut out = [0.0f32; CPPN_OUTPUTS];
        for (slot, maybe_idx) in self.output_slot_to_idx.iter().enumerate() {
            if let Some(idx) = maybe_idx {
                out[slot] = values[*idx];
            }
        }
        out
    }

    pub fn input_present(&self, slot: usize) -> bool {
        slot < CPPN_INPUTS && self.input_slot_to_idx[slot] != usize::MAX
    }
}

/// Order-independent content hash over the canonical structure of the genome.
fn content_hash(genome: &CppnGenome) -> u64 {
    let mut h = 0xF00D_BABE_1234_5678u64;
    for node in &genome.nodes {
        h = crate::rng::mix_u64(
            h ^ node.id
                ^ ((node.activation as u64) << 3)
                ^ (node.bias.to_bits() as u64).rotate_left(11)
                ^ ((node.kind as u64) << 1),
        );
    }
    for c in &genome.conns {
        if !c.enabled {
            continue;
        }
        h = crate::rng::mix_u64(h ^ c.innovation ^ (c.weight.to_bits() as u64).rotate_left(19));
    }
    h
}
