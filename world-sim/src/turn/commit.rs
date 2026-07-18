use super::*;

struct CommitPhaseContext<'a> {
    sim: &'a mut Simulation,
    intents: &'a [OrganismIntent],
    resolutions: &'a [MoveResolution],
    world_width_usize: usize,
    dead_organisms: Vec<bool>,
    result: CommitResult,
}

impl Simulation {
    pub(super) fn commit_phase(
        &mut self,
        world_width: i32,
        intents: &[OrganismIntent],
        resolutions: &[MoveResolution],
    ) -> CommitResult {
        CommitPhaseContext::new(self, world_width, intents, resolutions).run()
    }
}

impl<'a> CommitPhaseContext<'a> {
    fn new(
        sim: &'a mut Simulation,
        world_width: i32,
        intents: &'a [OrganismIntent],
        resolutions: &'a [MoveResolution],
    ) -> Self {
        let org_count = sim.organisms.len();
        let mut dead_organisms = std::mem::take(&mut sim.turn_scratch.dead_organisms);
        dead_organisms.clear();
        dead_organisms.resize(org_count, false);
        Self {
            sim,
            intents,
            resolutions,
            world_width_usize: world_width as usize,
            dead_organisms,
            result: CommitResult::default(),
        }
    }

    fn run(mut self) -> CommitResult {
        self.apply_facing();
        self.apply_moves();
        self.resolve_interactions();
        self.finalize();
        self.result
    }

    fn apply_facing(&mut self) {
        for (idx, intent) in self.intents.iter().enumerate() {
            debug_assert_eq!(intent.idx, idx);
            let organism = &mut self.sim.organisms[idx];
            if organism.facing != intent.facing_after_actions {
                self.result.facing_updates.push(OrganismFacing {
                    id: organism.id,
                    facing: intent.facing_after_actions,
                });
            }
            organism.facing = intent.facing_after_actions;
            self.result.actions_applied += u64::from(intent.command_count);
        }
    }

    fn apply_moves(&mut self) {
        // Clear every source first, then fill every destination. This is
        // observably identical for categorical moves (all targets began empty)
        // and makes treatment moves into simultaneously vacated cells atomic.
        for resolution in self.resolutions {
            let from_idx =
                resolution.from.1 as usize * self.world_width_usize + resolution.from.0 as usize;
            debug_assert_eq!(
                self.sim.occupancy[from_idx],
                Some(Occupant::Organism(resolution.actor_id))
            );
            self.sim.occupancy[from_idx] = None;
        }
        for resolution in self.resolutions {
            let to_idx =
                resolution.to.1 as usize * self.world_width_usize + resolution.to.0 as usize;
            debug_assert_eq!(self.sim.occupancy[to_idx], None);
            self.sim.occupancy[to_idx] = Some(Occupant::Organism(resolution.actor_id));
            let organism = &mut self.sim.organisms[resolution.actor_idx];
            organism.q = resolution.to.0;
            organism.r = resolution.to.1;
            #[cfg(feature = "instrumentation")]
            self.sim
                .mark_action_succeeded(resolution.actor_idx, ActionType::Forward);
        }
    }

    fn resolve_interactions(&mut self) {
        for (idx, intent) in self.intents.iter().enumerate() {
            if !intent.wants_attack || self.dead_organisms[idx] {
                continue;
            }
            // A single action per tick means an attacker never moved this tick,
            // so the target computed at intent-build time is still correct.
            let Some((target_q, target_r)) = intent.interaction_target else {
                continue;
            };
            let target_idx = target_r as usize * self.world_width_usize + target_q as usize;

            if intent.wants_attack && self.sim.config.predation_enabled {
                let attacker_energy_cost = self.charge_attack_attempt(idx);
                let event = if self.dead_organisms[idx] {
                    AttackEvent {
                        turn: self.sim.turn.saturating_add(1),
                        attacker_id: intent.id,
                        attacker_species_id: self.sim.organisms[idx].species_id,
                        victim_id: None,
                        victim_species_id: None,
                        outcome: AttackOutcome::InsufficientEnergy,
                        victim_energy_before: 0,
                        victim_energy_after: 0,
                        energy_transferred: 0,
                        attacker_energy_cost,
                    }
                } else {
                    match self.sim.occupancy[target_idx] {
                        Some(Occupant::Organism(prey_id)) => self.resolve_attack_transfer(
                            idx,
                            prey_id,
                            target_idx,
                            attacker_energy_cost,
                        ),
                        None | Some(Occupant::Wall) => {
                            let evaded = intent.snapshot_attack_target.and_then(|victim_id| {
                                organism_index_by_id(&self.sim.organisms, victim_id)
                                    .filter(|victim_idx| !self.dead_organisms[*victim_idx])
                                    .map(|victim_idx| {
                                        (victim_id, self.sim.organisms[victim_idx].species_id)
                                    })
                            });
                            AttackEvent {
                                turn: self.sim.turn.saturating_add(1),
                                attacker_id: intent.id,
                                attacker_species_id: self.sim.organisms[idx].species_id,
                                victim_id: evaded.map(|value| value.0),
                                victim_species_id: evaded.map(|value| value.1),
                                outcome: if evaded.is_some() {
                                    AttackOutcome::TargetEvaded
                                } else {
                                    AttackOutcome::NoOrganismTarget
                                },
                                victim_energy_before: 0,
                                victim_energy_after: 0,
                                energy_transferred: 0,
                                attacker_energy_cost,
                            }
                        }
                    }
                };
                self.result.attack_events.push(event);
            }
        }
    }

    /// Pay the cost of emitting an attack before looking at its target. An
    /// organism that cannot pay the full cost spends its remaining energy,
    /// dies, and cannot recover by receiving a transfer from that attempt.
    fn charge_attack_attempt(&mut self, attacker_idx: usize) -> u32 {
        let configured_cost = self.sim.config.attack_attempt_cost;
        let attacker = &mut self.sim.organisms[attacker_idx];
        let paid = configured_cost.min(attacker.energy);
        attacker.energy -= paid;
        attacker.energy_flow_last_tick =
            add_signed_energy_flow(attacker.energy_flow_last_tick, paid, false);
        self.result.attack_attempt_cost += f64::from(paid);

        if attacker.energy == 0 {
            let attacker_id = attacker.id;
            let position = (attacker.q, attacker.r);
            let cell_idx = attacker.r as usize * self.world_width_usize + attacker.q as usize;
            self.kill_organism(attacker_idx, attacker_id, cell_idx, position);
        }
        paid
    }

    fn resolve_attack_transfer(
        &mut self,
        predator_idx: usize,
        prey_id: OrganismId,
        target_idx: usize,
        attacker_energy_cost: u32,
    ) -> AttackEvent {
        let attacker_id = self.sim.organisms[predator_idx].id;
        let attacker_species_id = self.sim.organisms[predator_idx].species_id;
        let base_event = |outcome, victim_species_id| AttackEvent {
            turn: self.sim.turn.saturating_add(1),
            attacker_id,
            attacker_species_id,
            victim_id: Some(prey_id),
            victim_species_id,
            outcome,
            victim_energy_before: 0,
            victim_energy_after: 0,
            energy_transferred: 0,
            attacker_energy_cost,
        };
        let Some(prey_idx) = organism_index_by_id(&self.sim.organisms, prey_id) else {
            return base_event(AttackOutcome::NoOrganismTarget, None);
        };
        let victim_species_id = self.sim.organisms[prey_idx].species_id;
        if predator_idx == prey_idx || self.dead_organisms[prey_idx] {
            return base_event(AttackOutcome::NoOrganismTarget, Some(victim_species_id));
        }
        if let Some(pool_count) = self.sim.cross_pool_predation_pool_count {
            let predator_pool = self.sim.organisms[predator_idx].species_id.0 as usize % pool_count;
            let prey_pool = self.sim.organisms[prey_idx].species_id.0 as usize % pool_count;
            if predator_pool == prey_pool {
                return base_event(AttackOutcome::SamePoolBlocked, Some(victim_species_id));
            }
        }

        let attack_energy_transfer = self.sim.config.attack_energy_transfer;
        let prey = &mut self.sim.organisms[prey_idx];
        let victim_energy_before = prey.energy;
        let energy_transferred = attack_energy_transfer.min(victim_energy_before);
        prey.energy -= energy_transferred;
        prey.energy_flow_last_tick =
            add_signed_energy_flow(prey.energy_flow_last_tick, energy_transferred, false);
        let killed = prey.energy == 0;
        let prey_q = prey.q;
        let prey_r = prey.r;
        let event = AttackEvent {
            turn: self.sim.turn.saturating_add(1),
            attacker_id,
            attacker_species_id,
            victim_id: Some(prey_id),
            victim_species_id: Some(victim_species_id),
            outcome: if killed {
                AttackOutcome::Killed
            } else {
                AttackOutcome::NonlethalHit
            },
            victim_energy_before,
            victim_energy_after: prey.energy,
            energy_transferred,
            attacker_energy_cost,
        };

        #[cfg(feature = "instrumentation")]
        self.sim
            .mark_action_succeeded(predator_idx, ActionType::Attack);

        if energy_transferred > 0 {
            let predator = &mut self.sim.organisms[predator_idx];
            predator.energy = predator
                .energy
                .checked_add(energy_transferred)
                .expect("attack energy credit overflow");
            predator.energy_flow_last_tick =
                add_signed_energy_flow(predator.energy_flow_last_tick, energy_transferred, true);
            predator.successful_attacks_count = predator.successful_attacks_count.saturating_add(1);
            self.result.predations += 1;
            self.result.attack_transfer_energy += f64::from(energy_transferred);
            #[cfg(feature = "instrumentation")]
            {
                let predator = &self.sim.organisms[predator_idx];
                self.sim
                    .record_successful_attacks(predator_idx, predator.successful_attacks_count);
            }
        }
        if killed {
            self.kill_organism(prey_idx, prey_id, target_idx, (prey_q, prey_r));
        }
        event
    }

    fn kill_organism(
        &mut self,
        organism_idx: usize,
        organism_id: OrganismId,
        cell_idx: usize,
        position: (i32, i32),
    ) {
        if self.dead_organisms[organism_idx] {
            return;
        }

        self.dead_organisms[organism_idx] = true;
        self.result.removed_positions.push(RemovedEntityPosition {
            entity_id: EntityId::Organism(organism_id),
            q: position.0,
            r: position.1,
        });
        self.sim.occupancy[cell_idx] = None;
    }

    fn finalize(&mut self) {
        self.sim.compact_organism_state(&self.dead_organisms);

        self.result.moves = self
            .resolutions
            .iter()
            .map(|resolution| OrganismMove {
                id: resolution.actor_id,
                from: resolution.from,
                to: resolution.to,
            })
            .collect();

        // Return the scratch buffers to the simulation for reuse next tick.
        self.sim.turn_scratch.dead_organisms = std::mem::take(&mut self.dead_organisms);
    }
}

fn add_signed_energy_flow(current: i32, amount: u32, positive: bool) -> i32 {
    let signed = i32::try_from(amount).unwrap_or(i32::MAX);
    if positive {
        current.saturating_add(signed)
    } else {
        current.saturating_sub(signed)
    }
}
