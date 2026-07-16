use super::sanitization::is_valid_synapse_pair;
use super::*;

#[derive(Clone, Copy)]
struct Candidate {
    priority: f32,
    innovation: InnovationId,
    pre: GeneNodeId,
    post: GeneNodeId,
    timing: SynapseTiming,
}

pub(super) fn add_synapse_genes<R: Rng + ?Sized>(
    genome: &mut OrganismGenome,
    add_count: usize,
    predation_enabled: bool,
    rng: &mut R,
) {
    if add_count == 0 {
        return;
    }

    debug_assert!(genome
        .brain
        .edges
        .windows(2)
        .all(|w| w[0].innovation < w[1].innovation));

    if add_count == 1 {
        let mut best: Option<Candidate> = None;
        for_each_candidate(genome, predation_enabled, rng, |candidate| {
            if best
                .as_ref()
                .is_none_or(|current| candidate_cmp(&candidate, current) == Ordering::Less)
            {
                best = Some(candidate);
            }
        });
        if let Some(candidate) = best {
            insert_or_reenable(genome, candidate, rng);
        }
        return;
    }

    let mut weighted_candidates = Vec::new();
    for_each_candidate(genome, predation_enabled, rng, |candidate| {
        weighted_candidates.push(candidate)
    });

    let selected: &mut [Candidate] = if add_count < weighted_candidates.len() {
        let (top, _, _) = weighted_candidates.select_nth_unstable_by(add_count, candidate_cmp);
        top
    } else {
        &mut weighted_candidates
    };
    selected.sort_unstable_by(candidate_cmp);

    for candidate in selected.iter().copied() {
        insert_or_reenable(genome, candidate, rng);
    }
}

fn insert_or_reenable<R: Rng + ?Sized>(
    genome: &mut OrganismGenome,
    candidate: Candidate,
    rng: &mut R,
) {
    if super::connection_would_create_cycle(genome, candidate.pre, candidate.post, candidate.timing)
    {
        return;
    }
    match genome
        .brain
        .edges
        .binary_search_by_key(&candidate.innovation, |edge| edge.innovation)
    {
        Ok(index) => {
            let edge = &mut genome.brain.edges[index];
            if (edge.pre_node_id, edge.post_node_id, edge.timing)
                != (candidate.pre, candidate.post, candidate.timing)
            {
                // A true innovation collision is not an existing copy of this
                // connection. Reject it in release rather than re-enabling an
                // unrelated structural locus.
                return;
            }
            edge.enabled = true;
        }
        Err(index) => genome.brain.edges.insert(
            index,
            SynapseGene {
                innovation: candidate.innovation,
                pre_node_id: candidate.pre,
                post_node_id: candidate.post,
                timing: candidate.timing,
                weight: sample_synapse_weight(INITIAL_SYNAPSE_EXCITATORY_PROBABILITY, rng),
                enabled: true,
            },
        ),
    }
}

fn candidate_cmp(a: &Candidate, b: &Candidate) -> Ordering {
    a.priority
        .total_cmp(&b.priority)
        .then_with(|| a.innovation.cmp(&b.innovation))
        .then_with(|| a.pre.cmp(&b.pre))
        .then_with(|| a.post.cmp(&b.post))
        .then_with(|| a.timing.cmp(&b.timing))
}

/// Enumerate every valid missing or disabled structural connection. Disabled
/// genes retain their innovation identity and can therefore be re-enabled
/// without manufacturing a second historical marker for the same endpoints.
fn for_each_candidate<R: Rng + ?Sized>(
    genome: &OrganismGenome,
    predation_enabled: bool,
    rng: &mut R,
    mut visit: impl FnMut(Candidate),
) {
    let all_presynaptic = SensoryReceptor::active(predation_enabled)
        .filter_map(SensoryReceptor::neuron_id)
        .map(|id| sensory_gene_node_id(id.0))
        .chain(genome.brain.hidden_nodes.iter().map(|node| node.id));

    for pre in all_presynaptic {
        for post in post_ids(genome, predation_enabled) {
            consider_candidate(
                genome,
                pre,
                post,
                SynapseTiming::CurrentTick,
                rng,
                &mut visit,
            );
        }
    }
    for pre in genome.brain.hidden_nodes.iter().map(|node| node.id) {
        for post in genome.brain.hidden_nodes.iter().map(|node| node.id) {
            consider_candidate(
                genome,
                pre,
                post,
                SynapseTiming::PreviousTick,
                rng,
                &mut visit,
            );
        }
    }
}

fn consider_candidate<R: Rng + ?Sized>(
    genome: &OrganismGenome,
    pre: GeneNodeId,
    post: GeneNodeId,
    timing: SynapseTiming,
    rng: &mut R,
    visit: &mut impl FnMut(Candidate),
) {
    if super::connection_would_create_cycle(genome, pre, post, timing) {
        return;
    }
    debug_assert!(is_valid_synapse_pair(genome, pre, post, timing));
    let innovation = connection_innovation_id(pre, post, timing);
    let existing = genome
        .brain
        .edges
        .binary_search_by_key(&innovation, |edge| edge.innovation)
        .ok()
        .map(|index| &genome.brain.edges[index]);
    if let Some(edge) = existing {
        // Same identity but different structural triple is a hash collision.
        if (edge.pre_node_id, edge.post_node_id, edge.timing) != (pre, post, timing) || edge.enabled
        {
            return;
        }
    }
    visit(Candidate {
        priority: uniform_priority(rng),
        innovation,
        pre,
        post,
        timing,
    });
}

fn post_ids(
    genome: &OrganismGenome,
    predation_enabled: bool,
) -> impl Iterator<Item = GeneNodeId> + '_ {
    genome.brain.hidden_nodes.iter().map(|node| node.id).chain(
        ActionType::active(predation_enabled)
            .filter_map(ActionType::neuron_id)
            .map(|id| action_gene_node_id((id.0 - ACTION_ID_BASE) as usize)),
    )
}

fn uniform_priority<R: Rng + ?Sized>(rng: &mut R) -> f32 {
    let u = rng.random::<f32>().max(f32::MIN_POSITIVE);
    -u.ln()
}

fn sample_synapse_weight<R: Rng + ?Sized>(excitatory_probability: f32, rng: &mut R) -> f32 {
    let z = standard_normal(rng);
    let magnitude = (SYNAPSE_WEIGHT_LOG_NORMAL_MU + SYNAPSE_WEIGHT_LOG_NORMAL_SIGMA * z)
        .exp()
        .clamp(SYNAPSE_STRENGTH_MIN, SYNAPSE_STRENGTH_MAX);
    if rng.random::<f32>() < excitatory_probability {
        magnitude
    } else {
        -magnitude
    }
}
