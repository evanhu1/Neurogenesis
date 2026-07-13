//! Persistent public proof-carrying energy caches (PPEC).
//!
//! This module owns a nonblocking serialized artifact overlay and a generic
//! queued interaction channel. Plant consumption remains the sole funding
//! path: an enabled cache receives a configured fraction of the plant energy,
//! and a correct public two-bit response transfers that stored energy to any
//! adjacent organism. Programs and challenges are fully public; neither creator
//! IDs nor hidden evaluator labels participate in the response function.

mod experiment;

pub use experiment::*;

use crate::{grid::hex_neighbor, Simulation};
use anyhow::{bail, Result};
use sim_types::{
    ArtifactCacheState, ArtifactId, ArtifactInteractionEvent, ArtifactInteractionOutcome,
    ArtifactInteractionRequest, FacingDirection, OrganismGenome, OrganismId, PublicProtocolProgram,
};

const PROTOCOL_DOMAIN: u64 = 0x5050_4543_5052_4f54;
const CHALLENGE_DOMAIN: u64 = 0x5050_4543_4348_414c;

#[derive(Debug, Default)]
pub(crate) struct ArtifactEnergyFlow {
    pub(crate) protocol_interaction_cost_energy: f64,
    pub(crate) artifact_release_debit: f64,
    pub(crate) artifact_release_credit: f64,
    pub(crate) artifact_release_loss: f64,
}

impl Simulation {
    /// Persistent nonblocking cache overlay, maintained in ascending ID order.
    pub fn artifact_caches(&self) -> &[ArtifactCacheState] {
        &self.artifact_caches
    }

    /// Ordered committed interaction facts for the most recently completed tick.
    pub fn artifact_events_last_turn(&self) -> &[ArtifactInteractionEvent] {
        &self.artifact_events_last_turn
    }

    /// Queue a two-bit response through the generic public interaction path.
    /// Resolution occurs after ordinary movement/food interaction in the next
    /// canonical tick and is ordered by `(artifact, organism, request ordinal)`.
    pub fn queue_artifact_interaction(
        &mut self,
        organism_id: OrganismId,
        artifact_id: ArtifactId,
        response_bits: u8,
    ) -> u64 {
        let request_ordinal = self.next_artifact_interaction_ordinal;
        self.next_artifact_interaction_ordinal =
            self.next_artifact_interaction_ordinal.saturating_add(1);
        self.pending_artifact_interactions
            .push(ArtifactInteractionRequest {
                request_ordinal,
                organism_id,
                artifact_id,
                response_bits: response_bits & 0b11,
            });
        request_ordinal
    }

    pub(crate) fn clear_artifact_events_last_turn(&mut self) {
        self.artifact_events_last_turn.clear();
    }

    /// Split a just-consumed plant's existing positive energy into a persistent
    /// cache. Returns the exact allocation so the caller can credit only the
    /// complementary amount to the consumer.
    pub(crate) fn create_cache_from_plant_transfer(
        &mut self,
        creator_idx: usize,
        q: i32,
        r: i32,
        consumed_energy: f32,
    ) -> f32 {
        if !self.config.protocol_cache_enabled || self.config.cache_energy_fraction == 0.0 {
            return 0.0;
        }
        let allocation = consumed_energy.max(0.0) * self.config.cache_energy_fraction;
        if allocation <= 0.0 {
            return 0.0;
        }
        assert!(
            allocation.is_finite(),
            "PPEC cache allocation is nonfinite at turn {}",
            self.turn.saturating_add(1)
        );
        let creator = &self.organisms[creator_idx];
        let public_program = public_protocol_for_genome(&creator.genome);
        let owner_protocol_fingerprint = public_protocol_fingerprint(&public_program);
        let id = ArtifactId(self.next_artifact_id);
        self.next_artifact_id = self.next_artifact_id.saturating_add(1);
        let challenge_bits =
            deterministic_public_challenge(self.seed, id, 0, q, r, &public_program);
        self.artifact_caches.push(ArtifactCacheState {
            id,
            q,
            r,
            energy: allocation,
            creator_id: creator.id,
            owner_protocol_fingerprint,
            public_program,
            challenge_bits,
            attempt_ordinal: 0,
            created_turn: self.turn.saturating_add(1),
        });
        allocation
    }

    pub(crate) fn resolve_pending_artifact_interactions(
        &mut self,
        eligible_artifact_id_frontier: u64,
    ) -> ArtifactEnergyFlow {
        let mut requests = std::mem::take(&mut self.pending_artifact_interactions);
        requests.sort_unstable_by_key(|request| {
            (
                request.artifact_id,
                request.organism_id,
                request.request_ordinal,
            )
        });
        let mut flow = ArtifactEnergyFlow::default();
        let mut claimed_artifacts: std::collections::BTreeMap<ArtifactId, ArtifactCacheState> =
            std::collections::BTreeMap::new();
        let interaction_cost = self.config.protocol_interaction_energy_cost;
        let release_efficiency = self.config.cache_release_efficiency;

        for request in requests {
            let organism_idx = self
                .organisms
                .binary_search_by_key(&request.organism_id, |organism| organism.id)
                .ok();
            if let Some(artifact) = claimed_artifacts.get(&request.artifact_id) {
                let organism_energy = organism_idx.map_or(0.0, |idx| self.organisms[idx].energy);
                let cache_energy_after = self
                    .artifact_caches
                    .binary_search_by_key(&request.artifact_id, |cache| cache.id)
                    .ok()
                    .map_or(0.0, |idx| self.artifact_caches[idx].energy);
                let own_protocol = organism_idx.map(|idx| {
                    let actor_protocol = public_protocol_for_genome(&self.organisms[idx].genome);
                    public_protocol_fingerprint(&actor_protocol)
                        == artifact.owner_protocol_fingerprint
                });
                self.artifact_events_last_turn
                    .push(ArtifactInteractionEvent {
                        turn: self.turn.saturating_add(1),
                        request_ordinal: request.request_ordinal,
                        organism_id: request.organism_id,
                        artifact_id: request.artifact_id,
                        owner_protocol_fingerprint: Some(
                            artifact.owner_protocol_fingerprint.clone(),
                        ),
                        challenge_bits: artifact.challenge_bits.clone(),
                        response_bits: request.response_bits,
                        expected_response_bits: Some(
                            evaluate_public_protocol(
                                &artifact.public_program,
                                &artifact.challenge_bits,
                            )
                            .expect("validated artifact protocol must evaluate"),
                        ),
                        own_protocol,
                        outcome: ArtifactInteractionOutcome::ContendedThisTick,
                        cache_energy_before: artifact.energy,
                        cache_energy_after,
                        organism_energy_before: organism_energy,
                        organism_energy_after: organism_energy,
                        released_energy: 0.0,
                        release_loss: 0.0,
                    });
                continue;
            }
            let artifact_idx = self
                .artifact_caches
                .binary_search_by_key(&request.artifact_id, |artifact| artifact.id)
                .ok();
            let organism_energy = organism_idx.map_or(0.0, |idx| self.organisms[idx].energy);

            let (Some(organism_idx), Some(artifact_idx)) = (organism_idx, artifact_idx) else {
                self.artifact_events_last_turn
                    .push(ArtifactInteractionEvent {
                        turn: self.turn.saturating_add(1),
                        request_ordinal: request.request_ordinal,
                        organism_id: request.organism_id,
                        artifact_id: request.artifact_id,
                        owner_protocol_fingerprint: artifact_idx.map(|idx| {
                            self.artifact_caches[idx].owner_protocol_fingerprint.clone()
                        }),
                        challenge_bits: artifact_idx.map_or_else(Vec::new, |idx| {
                            self.artifact_caches[idx].challenge_bits.clone()
                        }),
                        response_bits: request.response_bits,
                        expected_response_bits: artifact_idx.map(|idx| {
                            evaluate_public_protocol(
                                &self.artifact_caches[idx].public_program,
                                &self.artifact_caches[idx].challenge_bits,
                            )
                            .expect("validated artifact protocol must evaluate")
                        }),
                        own_protocol: None,
                        outcome: if organism_idx.is_none() {
                            ArtifactInteractionOutcome::MissingOrganism
                        } else {
                            ArtifactInteractionOutcome::MissingArtifact
                        },
                        cache_energy_before: artifact_idx
                            .map_or(0.0, |idx| self.artifact_caches[idx].energy),
                        cache_energy_after: artifact_idx
                            .map_or(0.0, |idx| self.artifact_caches[idx].energy),
                        organism_energy_before: organism_energy,
                        organism_energy_after: organism_energy,
                        released_energy: 0.0,
                        release_loss: 0.0,
                    });
                continue;
            };

            let organism_position = (
                self.organisms[organism_idx].q,
                self.organisms[organism_idx].r,
            );
            let artifact_position = (
                self.artifact_caches[artifact_idx].q,
                self.artifact_caches[artifact_idx].r,
            );
            let adjacent = FacingDirection::ALL.iter().copied().any(|direction| {
                hex_neighbor(organism_position, direction, self.config.world_width as i32)
                    == artifact_position
            });
            let artifact = &self.artifact_caches[artifact_idx];
            let expected_response =
                evaluate_public_protocol(&artifact.public_program, &artifact.challenge_bits)
                    .expect("validated artifact protocol must evaluate");
            let actor_protocol = public_protocol_for_genome(&self.organisms[organism_idx].genome);
            let own_protocol =
                public_protocol_fingerprint(&actor_protocol) == artifact.owner_protocol_fingerprint;
            let challenge_bits = artifact.challenge_bits.clone();
            let owner_protocol_fingerprint = artifact.owner_protocol_fingerprint.clone();
            let cache_energy_before = artifact.energy;
            let organism_energy_before = self.organisms[organism_idx].energy;

            // Requests are allowed to name absent/future IDs, but an artifact
            // created by plant consumption during this tick was not public at
            // tick start. It becomes eligible only for a request queued into a
            // later tick, preventing prediction of the next ID from opening a
            // cache in the same commit that funded it.
            if request.artifact_id.0 >= eligible_artifact_id_frontier {
                self.artifact_events_last_turn
                    .push(ArtifactInteractionEvent {
                        turn: self.turn.saturating_add(1),
                        request_ordinal: request.request_ordinal,
                        organism_id: request.organism_id,
                        artifact_id: request.artifact_id,
                        owner_protocol_fingerprint: Some(owner_protocol_fingerprint),
                        challenge_bits,
                        response_bits: request.response_bits,
                        expected_response_bits: Some(expected_response),
                        own_protocol: Some(own_protocol),
                        outcome: ArtifactInteractionOutcome::CreatedThisTick,
                        cache_energy_before,
                        cache_energy_after: cache_energy_before,
                        organism_energy_before,
                        organism_energy_after: organism_energy_before,
                        released_energy: 0.0,
                        release_loss: 0.0,
                    });
                continue;
            }

            if !adjacent {
                self.artifact_events_last_turn
                    .push(ArtifactInteractionEvent {
                        turn: self.turn.saturating_add(1),
                        request_ordinal: request.request_ordinal,
                        organism_id: request.organism_id,
                        artifact_id: request.artifact_id,
                        owner_protocol_fingerprint: Some(owner_protocol_fingerprint),
                        challenge_bits,
                        response_bits: request.response_bits,
                        expected_response_bits: Some(expected_response),
                        own_protocol: Some(own_protocol),
                        outcome: ArtifactInteractionOutcome::NotAdjacent,
                        cache_energy_before,
                        cache_energy_after: cache_energy_before,
                        organism_energy_before,
                        organism_energy_after: organism_energy_before,
                        released_energy: 0.0,
                        release_loss: 0.0,
                    });
                continue;
            }

            // Protocol interactions are a new energy path and may not borrow
            // through zero. Exact equality is eligible: paying the cost leaves
            // zero before the response result is applied.
            if organism_energy_before < interaction_cost {
                self.artifact_events_last_turn
                    .push(ArtifactInteractionEvent {
                        turn: self.turn.saturating_add(1),
                        request_ordinal: request.request_ordinal,
                        organism_id: request.organism_id,
                        artifact_id: request.artifact_id,
                        owner_protocol_fingerprint: Some(owner_protocol_fingerprint),
                        challenge_bits,
                        response_bits: request.response_bits,
                        expected_response_bits: Some(expected_response),
                        own_protocol: Some(own_protocol),
                        outcome: ArtifactInteractionOutcome::InsufficientEnergy,
                        cache_energy_before,
                        cache_energy_after: cache_energy_before,
                        organism_energy_before,
                        organism_energy_after: organism_energy_before,
                        released_energy: 0.0,
                        release_loss: 0.0,
                    });
                continue;
            }

            // Only a real attempt can claim the artifact's per-tick slot.
            // Missing actors, future artifacts, and nonadjacent requests must
            // not let an early stable ID deny a later valid organism.
            claimed_artifacts.insert(
                request.artifact_id,
                self.artifact_caches[artifact_idx].clone(),
            );
            let organism = &mut self.organisms[organism_idx];
            let before_cost = organism.energy;
            organism.energy -= interaction_cost;
            assert!(
                organism.energy.is_finite(),
                "PPEC interaction cost produced nonfinite organism energy"
            );
            flow.protocol_interaction_cost_energy +=
                f64::from(before_cost) - f64::from(organism.energy);

            if request.response_bits != expected_response {
                let artifact = &mut self.artifact_caches[artifact_idx];
                artifact.attempt_ordinal = artifact.attempt_ordinal.saturating_add(1);
                let mut next_challenge = deterministic_public_challenge(
                    self.seed,
                    artifact.id,
                    artifact.attempt_ordinal,
                    artifact.q,
                    artifact.r,
                    &artifact.public_program,
                );
                // A wrong attempt promises a rotated public challenge. With a
                // short bit-vector, an independent deterministic hash can
                // collide with the previous challenge often enough to make
                // that promise false. Resolve that collision canonically by
                // flipping one public bit selected from the same state.
                if next_challenge == challenge_bits {
                    let flip_index = (mix64(
                        self.seed
                            ^ artifact.id.0.rotate_left(17)
                            ^ artifact.attempt_ordinal.rotate_left(41),
                    ) % next_challenge.len() as u64) as usize;
                    next_challenge[flip_index] ^= 1;
                }
                artifact.challenge_bits = next_challenge;
                self.artifact_events_last_turn
                    .push(ArtifactInteractionEvent {
                        turn: self.turn.saturating_add(1),
                        request_ordinal: request.request_ordinal,
                        organism_id: request.organism_id,
                        artifact_id: request.artifact_id,
                        owner_protocol_fingerprint: Some(owner_protocol_fingerprint),
                        challenge_bits,
                        response_bits: request.response_bits,
                        expected_response_bits: Some(expected_response),
                        own_protocol: Some(own_protocol),
                        outcome: ArtifactInteractionOutcome::WrongResponse,
                        cache_energy_before,
                        cache_energy_after: cache_energy_before,
                        organism_energy_before,
                        organism_energy_after: organism.energy,
                        released_energy: 0.0,
                        release_loss: 0.0,
                    });
                continue;
            }

            let credited = cache_energy_before * release_efficiency;
            let release_loss = cache_energy_before - credited;
            organism.energy += credited;
            assert!(
                organism.energy.is_finite() && credited.is_finite() && release_loss.is_finite(),
                "PPEC cache release produced nonfinite energy"
            );
            flow.artifact_release_debit += f64::from(cache_energy_before);
            flow.artifact_release_credit += f64::from(credited);
            flow.artifact_release_loss += f64::from(release_loss);
            self.artifact_caches.remove(artifact_idx);
            self.artifact_events_last_turn
                .push(ArtifactInteractionEvent {
                    turn: self.turn.saturating_add(1),
                    request_ordinal: request.request_ordinal,
                    organism_id: request.organism_id,
                    artifact_id: request.artifact_id,
                    owner_protocol_fingerprint: Some(owner_protocol_fingerprint),
                    challenge_bits,
                    response_bits: request.response_bits,
                    expected_response_bits: Some(expected_response),
                    own_protocol: Some(own_protocol),
                    outcome: if credited > 0.0 {
                        ArtifactInteractionOutcome::Released
                    } else {
                        ArtifactInteractionOutcome::AcceptedNoPayoff
                    },
                    cache_energy_before,
                    cache_energy_after: 0.0,
                    organism_energy_before,
                    organism_energy_after: organism.energy,
                    released_energy: credited,
                    release_loss,
                });
        }

        flow
    }
}

/// Deterministic, variable-length public protocol derived from the existing
/// structural genome. This avoids a second hidden identity channel: consumers
/// receive these exact bytes, and structural brain growth lengthens the public
/// program rather than selecting a fixed recipe ID.
pub fn public_protocol_for_genome(genome: &OrganismGenome) -> PublicProtocolProgram {
    let mut opcodes = Vec::with_capacity(
        32 + genome.brain.hidden_nodes.len() * 20 + genome.brain.edges.len() * 33,
    );
    opcodes.extend_from_slice(&[0x50, 0x50, 0x45, 0x43, 0x01]);
    opcodes.extend_from_slice(&genome.topology.vision_distance.to_le_bytes());
    opcodes.extend_from_slice(&genome.lifecycle.age_of_maturity.to_le_bytes());
    opcodes.push(genome.lifecycle.gestation_ticks);
    opcodes.extend_from_slice(&genome.lifecycle.max_organism_age.to_le_bytes());
    for node in &genome.brain.hidden_nodes {
        opcodes.extend_from_slice(&node.id.0.to_le_bytes());
        opcodes.extend_from_slice(&node.bias.to_bits().to_le_bytes());
        opcodes.extend_from_slice(&node.log_time_constant.to_bits().to_le_bytes());
    }
    for edge in &genome.brain.edges {
        opcodes.extend_from_slice(&edge.innovation.0.to_le_bytes());
        opcodes.extend_from_slice(&edge.pre_node_id.0.to_le_bytes());
        opcodes.extend_from_slice(&edge.post_node_id.0.to_le_bytes());
        opcodes.extend_from_slice(&edge.weight.to_bits().to_le_bytes());
        opcodes.push(u8::from(edge.enabled));
    }
    for bias in &genome.brain.action_biases {
        opcodes.extend_from_slice(&bias.to_bits().to_le_bytes());
    }
    let structural_width = 2usize
        .saturating_add(genome.brain.hidden_nodes.len())
        .saturating_add(
            genome
                .brain
                .edges
                .iter()
                .filter(|edge| edge.enabled)
                .count(),
        );
    let input_arity = structural_width.clamp(2, u16::MAX as usize) as u16;
    PublicProtocolProgram {
        input_arity,
        opcodes,
    }
}

pub fn public_protocol_fingerprint(program: &PublicProtocolProgram) -> String {
    let mut left = mix64(PROTOCOL_DOMAIN ^ u64::from(program.input_arity));
    let mut right = mix64(!PROTOCOL_DOMAIN ^ program.opcodes.len() as u64);
    for (index, &opcode) in program.opcodes.iter().enumerate() {
        left = mix64(left ^ u64::from(opcode) ^ (index as u64).rotate_left(17));
        right = mix64(right ^ u64::from(opcode).rotate_left((index % 63) as u32));
    }
    format!("{left:016x}{right:016x}")
}

/// Evaluate the Stage-0 public byte program as an explicit acyclic NAND
/// circuit. Challenge bits are the initial wires; each opcode deterministically
/// selects two currently available wires and appends one NAND output. The final
/// two gates are the public two-bit response. No world identity or evaluator
/// state is consulted.
pub fn evaluate_public_protocol(
    program: &PublicProtocolProgram,
    challenge_bits: &[u8],
) -> Result<u8> {
    if program.input_arity < 2
        || challenge_bits.len() != usize::from(program.input_arity)
        || challenge_bits.iter().any(|bit| *bit > 1)
        || program.opcodes.len() < 2
    {
        bail!("invalid public protocol program/challenge");
    }
    let mut wires = Vec::with_capacity(challenge_bits.len() + program.opcodes.len());
    wires.extend(challenge_bits.iter().map(|bit| *bit != 0));
    for (index, &opcode) in program.opcodes.iter().enumerate() {
        let wire_count = wires.len();
        let left_index = (usize::from(opcode) ^ index.wrapping_mul(13)) % wire_count;
        let right_index =
            (usize::from(opcode >> 2) ^ (index.wrapping_mul(7).wrapping_add(1))) % wire_count;
        wires.push(!(wires[left_index] && wires[right_index]));
    }
    let last = wires.len() - 1;
    Ok((u8::from(wires[last - 1]) << 1) | u8::from(wires[last]))
}

pub fn permuted_public_program(program: &PublicProtocolProgram) -> PublicProtocolProgram {
    let mut permuted = program.clone();
    permuted.opcodes.reverse();
    for opcode in &mut permuted.opcodes {
        *opcode = opcode.rotate_left(1) ^ 0xa5;
    }
    permuted
}

pub fn permuted_public_challenge(challenge_bits: &[u8]) -> Vec<u8> {
    challenge_bits.iter().map(|bit| bit ^ 1).collect()
}

fn deterministic_public_challenge(
    seed: u64,
    artifact_id: ArtifactId,
    attempt_ordinal: u64,
    q: i32,
    r: i32,
    program: &PublicProtocolProgram,
) -> Vec<u8> {
    let mut state = mix64(
        seed ^ CHALLENGE_DOMAIN
            ^ artifact_id.0.rotate_left(17)
            ^ attempt_ordinal.rotate_left(41)
            ^ (q as u32 as u64).rotate_left(7)
            ^ (r as u32 as u64).rotate_left(29),
    );
    let mut bits = Vec::with_capacity(usize::from(program.input_arity));
    for index in 0..program.input_arity {
        state = mix64(state ^ u64::from(index).wrapping_mul(0x9e37_79b9_7f4a_7c15));
        bits.push((state & 1) as u8);
    }
    bits
}

fn mix64(mut value: u64) -> u64 {
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}
