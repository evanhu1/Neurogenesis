//! Non-diluting controller capacity for expanding-task experiments.
//!
//! A solved controller is sealed as an immutable prefix. Later search starts
//! from an exact clone and may add a residual graph around it, but projection
//! restores every sealed parameter and removes any new edge that could write
//! into a sealed hidden node. The old recurrent computation therefore remains
//! structurally isolated while new nodes can read it and add action residuals.
//!
//! This is deliberately only the capacity half of a progressive algorithm.
//! A residual can still change the selected action, and (when runtime
//! plasticity is enabled) can indirectly change lifetime weight trajectories.
//! The task archive must therefore replay every accepted task through
//! [`enforce_retention`] before accepting a new controller.

use ring::digest::{digest, SHA256};
use serde::{Deserialize, Serialize};
use sim_types::{GeneNodeId, InnovationId, OrganismGenome, SynapseGene};
use std::collections::{BTreeMap, BTreeSet};
use thiserror::Error;

type EdgeIdentitySets = (BTreeSet<InnovationId>, BTreeSet<(GeneNodeId, GeneNodeId)>);

/// An accepted controller whose parameters and recurrent hidden computation
/// form the immutable prefix of every later candidate.
#[derive(Debug, Clone)]
pub struct ProtectedResidual {
    sealed: OrganismGenome,
    protected_nodes: BTreeSet<GeneNodeId>,
    protected_innovations: BTreeSet<InnovationId>,
    protected_endpoints: BTreeSet<(GeneNodeId, GeneNodeId)>,
}

impl ProtectedResidual {
    /// Seal an accepted genome. Duplicate structural identities are rejected
    /// rather than made ambiguous; externally loaded genomes should pass the
    /// normal simulation intake sanitizer before becoming an archive entry.
    pub fn seal(genome: &OrganismGenome) -> Result<Self, ProtectedResidualError> {
        let protected_nodes = unique_nodes(genome)?;
        let (protected_innovations, protected_endpoints) = unique_edges(genome)?;
        Ok(Self {
            sealed: genome.clone(),
            protected_nodes,
            protected_innovations,
            protected_endpoints,
        })
    }

    /// Start a new search stage without changing the genotype at all. Two
    /// worlds built from the sealed genome and this candidate are therefore
    /// byte-identical for the same config, seed, and tick count.
    pub fn seed_extension(&self) -> OrganismGenome {
        self.sealed.clone()
    }

    /// Deterministic causal knockout for the current search stage: discard all
    /// residual additions and materialize the exact last accepted controller.
    pub fn knockout_extension(&self) -> OrganismGenome {
        self.sealed.clone()
    }

    pub fn sealed_genome(&self) -> &OrganismGenome {
        &self.sealed
    }

    pub fn is_protected_node(&self, id: GeneNodeId) -> bool {
        self.protected_nodes.contains(&id)
    }

    pub fn is_protected_innovation(&self, id: InnovationId) -> bool {
        self.protected_innovations.contains(&id)
    }

    /// Deterministically project a crossover/mutation result back into the
    /// legal residual search space.
    ///
    /// - all scalar genes, action biases, sealed hidden-node parameters, and
    ///   sealed connection genes are restored exactly;
    /// - structural aliases of sealed connections are removed;
    /// - new inputs to sealed hidden nodes are removed, so the extension cannot
    ///   rewrite the old recurrent computation;
    /// - new nodes and residual paths into actions remain available.
    pub fn project(&self, candidate: &mut OrganismGenome) {
        candidate.topology = self.sealed.topology.clone();
        candidate.lifecycle = self.sealed.lifecycle.clone();
        candidate.plasticity = self.sealed.plasticity.clone();
        candidate.brain.action_biases = self.sealed.brain.action_biases.clone();

        candidate
            .brain
            .hidden_nodes
            .retain(|node| !self.protected_nodes.contains(&node.id));
        candidate
            .brain
            .hidden_nodes
            .extend(self.sealed.brain.hidden_nodes.iter().copied());
        candidate.brain.hidden_nodes.sort_unstable_by(|a, b| {
            a.id.cmp(&b.id)
                .then_with(|| a.bias.total_cmp(&b.bias))
                .then_with(|| a.log_time_constant.total_cmp(&b.log_time_constant))
        });
        candidate.brain.hidden_nodes.dedup_by_key(|node| node.id);

        candidate.brain.edges.retain(|edge| {
            !self.protected_innovations.contains(&edge.innovation)
                && !self
                    .protected_endpoints
                    .contains(&(edge.pre_node_id, edge.post_node_id))
                && !self.protected_nodes.contains(&edge.post_node_id)
        });
        candidate
            .brain
            .edges
            .extend(self.sealed.brain.edges.iter().copied());
        sort_edges(&mut candidate.brain.edges);
    }

    /// Verify that a candidate is in the legal residual search space. This is
    /// an integrity check, not a substitute for replaying the retention suite.
    pub fn verify(&self, candidate: &OrganismGenome) -> Result<(), ProtectedResidualError> {
        if candidate.topology != self.sealed.topology {
            return Err(ProtectedResidualError::ScalarDrift("topology"));
        }
        if candidate.lifecycle != self.sealed.lifecycle {
            return Err(ProtectedResidualError::ScalarDrift("lifecycle"));
        }
        if candidate.plasticity != self.sealed.plasticity {
            return Err(ProtectedResidualError::ScalarDrift("plasticity"));
        }
        if candidate.brain.action_biases != self.sealed.brain.action_biases {
            return Err(ProtectedResidualError::ScalarDrift("action_biases"));
        }

        let mut candidate_nodes: BTreeMap<GeneNodeId, Vec<_>> = BTreeMap::new();
        for node in &candidate.brain.hidden_nodes {
            candidate_nodes.entry(node.id).or_default().push(node);
        }
        for expected in &self.sealed.brain.hidden_nodes {
            let copies = candidate_nodes.get(&expected.id).map(Vec::as_slice);
            if copies != Some(&[expected]) {
                return Err(ProtectedResidualError::ProtectedNodeChanged(expected.id));
            }
        }

        let mut candidate_edges: BTreeMap<InnovationId, Vec<_>> = BTreeMap::new();
        for edge in &candidate.brain.edges {
            candidate_edges
                .entry(edge.innovation)
                .or_default()
                .push(edge);
        }
        for expected in &self.sealed.brain.edges {
            let copies = candidate_edges.get(&expected.innovation).map(Vec::as_slice);
            if copies != Some(&[expected]) {
                return Err(ProtectedResidualError::ProtectedEdgeChanged(
                    expected.innovation,
                ));
            }
        }

        for edge in &candidate.brain.edges {
            if self.protected_innovations.contains(&edge.innovation) {
                continue;
            }
            if self
                .protected_endpoints
                .contains(&(edge.pre_node_id, edge.post_node_id))
            {
                return Err(ProtectedResidualError::ProtectedEndpointAliased {
                    pre: edge.pre_node_id,
                    post: edge.post_node_id,
                });
            }
            if self.protected_nodes.contains(&edge.post_node_id) {
                return Err(ProtectedResidualError::NewInputToProtectedNode {
                    innovation: edge.innovation,
                    post: edge.post_node_id,
                });
            }
        }
        Ok(())
    }
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum ProtectedResidualError {
    #[error("sealed genome contains duplicate hidden-node identity {0:?}")]
    DuplicateNode(GeneNodeId),
    #[error("sealed genome contains duplicate innovation identity {0:?}")]
    DuplicateInnovation(InnovationId),
    #[error("sealed genome contains duplicate connection endpoints {pre:?}->{post:?}")]
    DuplicateEndpoints { pre: GeneNodeId, post: GeneNodeId },
    #[error("protected scalar field changed: {0}")]
    ScalarDrift(&'static str),
    #[error("protected hidden node changed or was duplicated: {0:?}")]
    ProtectedNodeChanged(GeneNodeId),
    #[error("protected connection changed or was duplicated: {0:?}")]
    ProtectedEdgeChanged(InnovationId),
    #[error("new connection aliases protected endpoints {pre:?}->{post:?}")]
    ProtectedEndpointAliased { pre: GeneNodeId, post: GeneNodeId },
    #[error("new connection {innovation:?} writes into protected hidden node {post:?}")]
    NewInputToProtectedNode {
        innovation: InnovationId,
        post: GeneNodeId,
    },
}

/// SHA-256 of a canonically ordered, CBOR-serialized artifact. Callers must use
/// ordered containers (`Vec`, `BTreeMap`, sorted fact rows) inside the value;
/// unordered maps do not define a reproducible byte representation.
pub fn exact_fingerprint<T: Serialize>(value: &T) -> Result<String, FingerprintError> {
    let mut bytes = Vec::new();
    ciborium::into_writer(value, &mut bytes)
        .map_err(|error| FingerprintError(error.to_string()))?;
    let bytes = digest(&SHA256, &bytes);
    let mut hex = String::with_capacity(bytes.as_ref().len() * 2);
    for byte in bytes.as_ref() {
        use std::fmt::Write as _;
        write!(&mut hex, "{byte:02x}").expect("writing to a String cannot fail");
    }
    Ok(hex)
}

#[derive(Debug, Error)]
#[error("cannot fingerprint artifact: {0}")]
pub struct FingerprintError(String);

/// Durable checkpoint for every accepted task, not merely the immediate
/// predecessor. The accepted behavior fingerprint vector is ordered exactly
/// like `trial_seeds` and makes archive replay independently auditable.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub struct RetentionRequirementHeader {
    pub task_id: u64,
    pub minimum_passes: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct TaskCheckpoint {
    #[serde(flatten)]
    pub requirement: RetentionRequirementHeader,
    pub task_fingerprint: String,
    pub accepted_controller_fingerprint: String,
    pub trial_seeds: Vec<u64>,
    pub accepted_passes: u32,
    pub accepted_behavior_fingerprints: Vec<String>,
}

/// Exact replay of either an archived accepted controller or the current
/// candidate on one historical task and its frozen trial suite.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct TaskReplay {
    pub task_id: u64,
    pub task_fingerprint: String,
    pub controller_fingerprint: String,
    pub trial_seeds: Vec<u64>,
    pub passes: u32,
    pub behavior_fingerprints: Vec<String>,
}

/// Evidence required before a residual controller can become the next sealed
/// checkpoint. Both vectors must cover the complete archive exactly once.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct AllHistoryRetention {
    pub candidate_controller_fingerprint: String,
    pub ancestor_replays: Vec<TaskReplay>,
    pub candidate_replays: Vec<TaskReplay>,
}

/// Fail closed unless every archived controller still reproduces its exact
/// checkpoint trace and the candidate meets every historical pass threshold.
pub fn enforce_retention(
    checkpoints: &[TaskCheckpoint],
    evidence: &AllHistoryRetention,
) -> Result<(), RetentionError> {
    require_fingerprint(
        "candidate_controller_fingerprint",
        &evidence.candidate_controller_fingerprint,
    )?;

    let checkpoint_map = unique_checkpoints(checkpoints)?;
    let mut ancestor_map = unique_replays("ancestor", &evidence.ancestor_replays)?;
    let mut candidate_map = unique_replays("candidate", &evidence.candidate_replays)?;

    for (task_id, checkpoint) in checkpoint_map {
        validate_checkpoint(checkpoint)?;
        let ancestor = ancestor_map
            .remove(&task_id)
            .ok_or(RetentionError::MissingReplay {
                kind: "ancestor",
                task_id,
            })?;
        let candidate = candidate_map
            .remove(&task_id)
            .ok_or(RetentionError::MissingReplay {
                kind: "candidate",
                task_id,
            })?;

        validate_replay_shape(ancestor)?;
        validate_replay_shape(candidate)?;
        if ancestor.task_fingerprint != checkpoint.task_fingerprint
            || ancestor.controller_fingerprint != checkpoint.accepted_controller_fingerprint
            || ancestor.trial_seeds != checkpoint.trial_seeds
            || ancestor.passes != checkpoint.accepted_passes
            || ancestor.behavior_fingerprints != checkpoint.accepted_behavior_fingerprints
        {
            return Err(RetentionError::AncestorReplayMismatch(task_id));
        }
        if candidate.task_fingerprint != checkpoint.task_fingerprint
            || candidate.controller_fingerprint != evidence.candidate_controller_fingerprint
            || candidate.trial_seeds != checkpoint.trial_seeds
        {
            return Err(RetentionError::CandidateReplayMismatch(task_id));
        }
        if candidate.passes < checkpoint.requirement.minimum_passes {
            return Err(RetentionError::Forgotten {
                task_id,
                required: checkpoint.requirement.minimum_passes,
                observed: candidate.passes,
            });
        }
    }

    if let Some((&task_id, _)) = ancestor_map.first_key_value() {
        return Err(RetentionError::UnexpectedReplay {
            kind: "ancestor",
            task_id,
        });
    }
    if let Some((&task_id, _)) = candidate_map.first_key_value() {
        return Err(RetentionError::UnexpectedReplay {
            kind: "candidate",
            task_id,
        });
    }
    Ok(())
}

/// Paired causal evidence for the just-added residual. The knockout controller
/// must fingerprint to the previous sealed checkpoint, the trial contexts must
/// be unique, and the enabled controller must clear an explicit capability gap
/// in addition to changing its exact per-seed behavior traces.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ExtensionEffectEvidence {
    pub task_id: u64,
    pub task_fingerprint: String,
    pub trial_seeds: Vec<u64>,
    pub enabled_controller_fingerprint: String,
    pub knockout_controller_fingerprint: String,
    pub enabled_passes: u32,
    pub knockout_passes: u32,
    pub minimum_enabled_passes: u32,
    pub maximum_knockout_passes: u32,
    pub enabled_behavior_fingerprints: Vec<String>,
    pub knockout_behavior_fingerprints: Vec<String>,
}

pub fn verify_extension_effect(
    sealed_controller_fingerprint: &str,
    evidence: &ExtensionEffectEvidence,
) -> Result<(), ExtensionEffectError> {
    if !is_fingerprint(sealed_controller_fingerprint)
        || !is_fingerprint(&evidence.task_fingerprint)
        || !is_fingerprint(&evidence.enabled_controller_fingerprint)
        || !is_fingerprint(&evidence.knockout_controller_fingerprint)
    {
        return Err(ExtensionEffectError::MalformedFingerprint);
    }
    if evidence.knockout_controller_fingerprint != sealed_controller_fingerprint {
        return Err(ExtensionEffectError::WrongKnockoutController);
    }
    if evidence.enabled_controller_fingerprint == evidence.knockout_controller_fingerprint {
        return Err(ExtensionEffectError::NoGenotypeChange);
    }
    let trials = evidence.trial_seeds.len();
    if trials == 0
        || !trial_seeds_are_unique(&evidence.trial_seeds)
        || evidence.enabled_behavior_fingerprints.len() != trials
        || evidence.knockout_behavior_fingerprints.len() != trials
        || evidence.enabled_passes as usize > trials
        || evidence.knockout_passes as usize > trials
        || evidence.minimum_enabled_passes as usize > trials
        || evidence.maximum_knockout_passes as usize > trials
        || evidence.minimum_enabled_passes <= evidence.maximum_knockout_passes
    {
        return Err(ExtensionEffectError::MalformedTrialPanel);
    }
    for fingerprint in evidence
        .enabled_behavior_fingerprints
        .iter()
        .chain(&evidence.knockout_behavior_fingerprints)
    {
        if !is_fingerprint(fingerprint) {
            return Err(ExtensionEffectError::MalformedFingerprint);
        }
    }
    if evidence.enabled_behavior_fingerprints == evidence.knockout_behavior_fingerprints {
        return Err(ExtensionEffectError::NoBehavioralEffect);
    }
    if evidence.enabled_passes < evidence.minimum_enabled_passes
        || evidence.knockout_passes > evidence.maximum_knockout_passes
        || evidence.enabled_passes <= evidence.knockout_passes
    {
        return Err(ExtensionEffectError::NoCapabilityGain {
            enabled: evidence.enabled_passes,
            knockout: evidence.knockout_passes,
        });
    }
    Ok(())
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum ExtensionEffectError {
    #[error("extension knockout does not materialize the sealed controller")]
    WrongKnockoutController,
    #[error("enabled extension and knockout have the same controller fingerprint")]
    NoGenotypeChange,
    #[error("extension-effect trial panel is empty or dimensionally inconsistent")]
    MalformedTrialPanel,
    #[error("extension-effect evidence contains a malformed fingerprint")]
    MalformedFingerprint,
    #[error("enabling the residual changes no per-seed behavior trace")]
    NoBehavioralEffect,
    #[error("residual fails capability thresholds: enabled {enabled}, knockout {knockout}")]
    NoCapabilityGain { enabled: u32, knockout: u32 },
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum RetentionError {
    #[error("duplicate checkpoint for task {0}")]
    DuplicateCheckpoint(u64),
    #[error("duplicate {kind} replay for task {task_id}")]
    DuplicateReplay { kind: &'static str, task_id: u64 },
    #[error("{field} is not a lowercase SHA-256 fingerprint")]
    MalformedFingerprint { field: &'static str },
    #[error("checkpoint for task {0} is dimensionally inconsistent")]
    InvalidCheckpoint(u64),
    #[error("replay for task {0} is dimensionally inconsistent")]
    InvalidReplay(u64),
    #[error("{kind} replay missing for task {task_id}")]
    MissingReplay { kind: &'static str, task_id: u64 },
    #[error("unexpected {kind} replay for task {task_id}")]
    UnexpectedReplay { kind: &'static str, task_id: u64 },
    #[error("accepted controller did not exactly reproduce checkpoint task {0}")]
    AncestorReplayMismatch(u64),
    #[error("candidate replay contract mismatch for task {0}")]
    CandidateReplayMismatch(u64),
    #[error("task {task_id} forgotten: required {required} passes, observed {observed}")]
    Forgotten {
        task_id: u64,
        required: u32,
        observed: u32,
    },
}

fn unique_checkpoints(
    checkpoints: &[TaskCheckpoint],
) -> Result<BTreeMap<u64, &TaskCheckpoint>, RetentionError> {
    let mut map = BTreeMap::new();
    for checkpoint in checkpoints {
        let task_id = checkpoint.requirement.task_id;
        if map.insert(task_id, checkpoint).is_some() {
            return Err(RetentionError::DuplicateCheckpoint(task_id));
        }
    }
    Ok(map)
}

fn unique_replays<'a>(
    kind: &'static str,
    replays: &'a [TaskReplay],
) -> Result<BTreeMap<u64, &'a TaskReplay>, RetentionError> {
    let mut map = BTreeMap::new();
    for replay in replays {
        if map.insert(replay.task_id, replay).is_some() {
            return Err(RetentionError::DuplicateReplay {
                kind,
                task_id: replay.task_id,
            });
        }
    }
    Ok(map)
}

fn validate_checkpoint(checkpoint: &TaskCheckpoint) -> Result<(), RetentionError> {
    let task_id = checkpoint.requirement.task_id;
    let trials = checkpoint.trial_seeds.len();
    if trials == 0
        || !trial_seeds_are_unique(&checkpoint.trial_seeds)
        || checkpoint.accepted_behavior_fingerprints.len() != trials
        || checkpoint.accepted_passes as usize > trials
        || checkpoint.requirement.minimum_passes as usize > trials
    {
        return Err(RetentionError::InvalidCheckpoint(task_id));
    }
    require_fingerprint("task_fingerprint", &checkpoint.task_fingerprint)?;
    require_fingerprint(
        "accepted_controller_fingerprint",
        &checkpoint.accepted_controller_fingerprint,
    )?;
    for fingerprint in &checkpoint.accepted_behavior_fingerprints {
        require_fingerprint("accepted_behavior_fingerprint", fingerprint)?;
    }
    Ok(())
}

fn validate_replay_shape(replay: &TaskReplay) -> Result<(), RetentionError> {
    if replay.trial_seeds.is_empty()
        || !trial_seeds_are_unique(&replay.trial_seeds)
        || replay.behavior_fingerprints.len() != replay.trial_seeds.len()
        || replay.passes as usize > replay.trial_seeds.len()
    {
        return Err(RetentionError::InvalidReplay(replay.task_id));
    }
    require_fingerprint("task_fingerprint", &replay.task_fingerprint)?;
    require_fingerprint("controller_fingerprint", &replay.controller_fingerprint)?;
    for fingerprint in &replay.behavior_fingerprints {
        require_fingerprint("behavior_fingerprint", fingerprint)?;
    }
    Ok(())
}

fn require_fingerprint(field: &'static str, value: &str) -> Result<(), RetentionError> {
    if is_fingerprint(value) {
        Ok(())
    } else {
        Err(RetentionError::MalformedFingerprint { field })
    }
}

fn is_fingerprint(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn trial_seeds_are_unique(seeds: &[u64]) -> bool {
    seeds.iter().copied().collect::<BTreeSet<_>>().len() == seeds.len()
}

fn unique_nodes(genome: &OrganismGenome) -> Result<BTreeSet<GeneNodeId>, ProtectedResidualError> {
    let mut ids = BTreeSet::new();
    for node in &genome.brain.hidden_nodes {
        if !ids.insert(node.id) {
            return Err(ProtectedResidualError::DuplicateNode(node.id));
        }
    }
    Ok(ids)
}

fn unique_edges(genome: &OrganismGenome) -> Result<EdgeIdentitySets, ProtectedResidualError> {
    let mut innovations = BTreeSet::new();
    let mut endpoints = BTreeSet::new();
    for edge in &genome.brain.edges {
        if !innovations.insert(edge.innovation) {
            return Err(ProtectedResidualError::DuplicateInnovation(edge.innovation));
        }
        if !endpoints.insert((edge.pre_node_id, edge.post_node_id)) {
            return Err(ProtectedResidualError::DuplicateEndpoints {
                pre: edge.pre_node_id,
                post: edge.post_node_id,
            });
        }
    }
    Ok((innovations, endpoints))
}

fn sort_edges(edges: &mut [SynapseGene]) {
    edges.sort_unstable_by(|a, b| {
        a.innovation
            .cmp(&b.innovation)
            .then_with(|| a.pre_node_id.cmp(&b.pre_node_id))
            .then_with(|| a.post_node_id.cmp(&b.post_node_id))
            .then_with(|| b.enabled.cmp(&a.enabled))
            .then_with(|| a.weight.total_cmp(&b.weight))
    });
}
