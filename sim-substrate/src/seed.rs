//! Hand-authored seed genome: a small "primordial" CPPN plus sane header
//! defaults. The seed paints a viable default interface + wiring so founder
//! organisms are never degenerate (plan risk #3/#4).

use crate::catalog::SubstrateCatalog;
use crate::cppn::{Activation, CppnConnGene, CppnGenome, CppnNodeGene, NodeKind, CPPN_INPUTS};
use crate::genome::{
    DevelopGenes, Genome, HeaderGenes, LifecycleGenes, MutationRates, PlasticityGenes,
};

/// Node ids for the fixed I/O layer. Inputs occupy `0..CPPN_INPUTS`, outputs
/// occupy `100..100+CPPN_OUTPUTS`, a bias node sits at `200`.
const INPUT_ID_BASE: u64 = 0;
const OUTPUT_ID_BASE: u64 = 100;
const BIAS_ID: u64 = 200;

/// Default per-operator mutation rates (the meta-mutation baseline).
pub fn default_mutation_rates() -> MutationRates {
    MutationRates {
        age_of_maturity: 0.1,
        gestation_ticks: 0.05,
        max_organism_age: 0.1,
        hebb_eta_gain: 0.1,
        juvenile_eta_scale: 0.1,
        eligibility_retention: 0.1,
        synapse_prune_threshold: 0.1,
        max_weight_delta_per_tick: 0.05,
        morphology: 0.1,
        cppn_weight_perturb: 0.5,
        cppn_add_conn: 0.08,
        cppn_add_node: 0.04,
        cppn_toggle_enable: 0.02,
        cppn_mutate_activation: 0.05,
        cppn_perturb_bias: 0.1,
    }
}

/// A minimal CPPN that expresses a broad interface (AFF biased positive) and a
/// distance-decaying weight field, so founders develop a working brain.
pub fn primordial_cppn() -> CppnGenome {
    let mut nodes: Vec<CppnNodeGene> = Vec::new();
    for slot in 0..CPPN_INPUTS {
        nodes.push(CppnNodeGene {
            id: INPUT_ID_BASE + slot as u64,
            kind: NodeKind::Input,
            activation: Activation::Linear,
            bias: 0.0,
            io_slot: slot as u16,
        });
    }
    // Output activations chosen so raw outputs land in useful ranges.
    let output_acts = [
        Activation::Tanh,   // W
        Activation::Tanh,   // LEO
        Activation::Tanh,   // NBIAS
        Activation::Linear, // LTC
        Activation::Linear, // AFF (bias positive so most affordances express)
        Activation::Tanh,   // PLR
    ];
    for (slot, &activation) in output_acts.iter().enumerate() {
        nodes.push(CppnNodeGene {
            id: OUTPUT_ID_BASE + slot as u64,
            kind: NodeKind::Output,
            activation,
            // Positive AFF bias => broad interface; slightly-positive LEO bias
            // => some default connectivity.
            bias: match slot {
                crate::cppn::out::AFF => 0.6,
                crate::cppn::out::LEO => 0.1,
                _ => 0.0,
            },
            io_slot: slot as u16,
        });
    }
    nodes.push(CppnNodeGene {
        id: BIAS_ID,
        kind: NodeKind::Bias,
        activation: Activation::Linear,
        bias: 0.0,
        io_slot: u16::MAX,
    });

    // Seed connections: distance (input slot 9) inhibits LEO and W (short-range
    // wiring); bias feeds AFF. Innovation ids are structural hashes so they
    // align under crossover.
    let dist_id = INPUT_ID_BASE + 9;
    let leo_id = OUTPUT_ID_BASE + crate::cppn::out::LEO as u64;
    let w_id = OUTPUT_ID_BASE + crate::cppn::out::W as u64;
    let aff_id = OUTPUT_ID_BASE + crate::cppn::out::AFF as u64;
    let mut conns = Vec::new();
    let push = |from: u64, to: u64, weight: f32, conns: &mut Vec<CppnConnGene>| {
        conns.push(CppnConnGene {
            innovation: CppnGenome::conn_innovation(from, to),
            from,
            to,
            weight,
            enabled: true,
        });
    };
    push(dist_id, leo_id, -1.5, &mut conns);
    push(dist_id, w_id, -0.8, &mut conns);
    push(BIAS_ID, aff_id, 0.4, &mut conns);

    let mut cppn = CppnGenome { nodes, conns };
    cppn.canonicalize();
    cppn
}

/// A full seed genome for the given catalog (morphology defaults come from the
/// catalog schema).
pub fn seed_genome(catalog: &SubstrateCatalog) -> Genome {
    let header = HeaderGenes {
        plasticity: PlasticityGenes {
            hebb_eta_gain: 0.02,
            juvenile_eta_scale: 2.0,
            eligibility_retention: 0.9,
            max_weight_delta_per_tick: 0.05,
            synapse_prune_threshold: 0.02,
        },
        lifecycle: LifecycleGenes {
            age_of_maturity: 20,
            gestation_ticks: 2,
            max_organism_age: 2000,
        },
        develop: DevelopGenes {
            aff_threshold: 0.0,
            leo_threshold: 0.0,
            weight_scale: 1.5,
            quadtree_depth: 3,
            variance_threshold: 0.02,
            max_neurons: 48,
            max_edges: 512,
        },
        mutation_rates: default_mutation_rates(),
        morphology: catalog.morphology_defaults(),
    };
    let mut genome = Genome {
        cppn: primordial_cppn(),
        header,
    };
    genome.canonicalize();
    genome
}
