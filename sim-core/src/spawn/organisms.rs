use super::*;
use crate::metabolism::refresh_organism_base_metabolic_cost;

/// Rejection-sampling attempt budget per requested periodic-injection spawn.
/// 64 attempts per spawn keeps the expected shortfall negligible down to
/// ~10% open-cell density while bounding work in crowded worlds, where fewer
/// than the configured count may be queued.
const INJECTION_SAMPLE_ATTEMPTS_PER_SPAWN: usize = 64;

impl Simulation {
    /// Resolves queued spawn requests, returning outcomes aligned 1:1 with the
    /// drained queue order: `Some(organism)` for each request that spawned and
    /// `None` for each request whose target cell was occupied. Callers that
    /// attribute children to parents positionally rely on this alignment.
    pub(crate) fn resolve_spawn_requests(
        &mut self,
        queue: &mut Vec<SpawnRequest>,
    ) -> Vec<Option<OrganismState>> {
        let mut spawned = Vec::with_capacity(queue.len());
        for request in queue.drain(..) {
            let organism = match request {
                SpawnRequest::Reproduction(mut reproduction) => {
                    mutate_genome(
                        &mut reproduction.parent_genome,
                        &self.config.seed_genome_config,
                        self.config.global_mutation_rate_modifier,
                        self.config.meta_mutation_enabled,
                        &mut self.rng,
                    );
                    self.build_organism(
                        reproduction.parent_genome,
                        OrganismSpawnParams {
                            species_id: Some(reproduction.parent_species_id),
                            q: reproduction.q,
                            r: reproduction.r,
                            generation: reproduction.parent_generation.saturating_add(1),
                            facing: opposite_direction(reproduction.parent_facing),
                            starting_energy_override: Some(reproduction.offspring_starting_energy),
                        },
                    )
                }
                SpawnRequest::PeriodicInjection(injection) => {
                    let genome =
                        generate_seed_genome(&self.config.seed_genome_config, &mut self.rng);
                    let facing = self.random_facing();
                    self.build_organism(
                        genome,
                        OrganismSpawnParams {
                            species_id: None,
                            q: injection.q,
                            r: injection.r,
                            generation: 0,
                            facing,
                            starting_energy_override: None,
                        },
                    )
                }
            };

            // Move the organism into the grid; only successful adds pay for a
            // clone (taken from the inserted element) for the returned vec.
            let added = self.add_organism(organism);
            spawned.push(added.then(|| {
                self.organisms
                    .last()
                    .expect("add_organism pushed the organism on success")
                    .clone()
            }));
        }

        spawned
    }

    pub(crate) fn enqueue_periodic_injections(&mut self, queue: &mut Vec<SpawnRequest>) {
        let interval = self.config.periodic_injection_interval_turns as u64;
        let count = self.config.periodic_injection_count as usize;
        if interval == 0 || count == 0 {
            return;
        }

        let next_turn = self.turn.saturating_add(1);
        if !next_turn.is_multiple_of(interval) {
            return;
        }

        let width = self.config.world_width as usize;
        // Cells already targeted by queued spawn requests (plus the cells
        // picked below), kept as a small sorted vec probed via binary search
        // instead of a world-capacity bool buffer.
        let mut reserved: Vec<usize> = queue
            .iter()
            .map(|request| {
                let (q, r) = request.target_position();
                r as usize * width + q as usize
            })
            .collect();
        reserved.sort_unstable();

        // Deterministic rejection sampling: draw uniformly random cell
        // indices from the simulation RNG and keep the first `count` distinct
        // cells that are unoccupied and unreserved. This replaces the former
        // full-grid scan + partial_shuffle (which allocated a world-capacity
        // position vec); the RNG draw sequence differs from that scheme but
        // is still a pure function of (config, seed, sim state), so fixed
        // config + seed yields identical results.
        let capacity = self.occupancy.len();
        let max_attempts = count.saturating_mul(INJECTION_SAMPLE_ATTEMPTS_PER_SPAWN);
        let mut picked = 0usize;
        for _ in 0..max_attempts {
            if picked == count {
                break;
            }
            let idx = self.rng.random_range(0..capacity);
            if self.occupancy[idx].is_some() {
                continue;
            }
            let Err(insert_at) = reserved.binary_search(&idx) else {
                continue;
            };
            reserved.insert(insert_at, idx);
            picked += 1;
            queue.push(SpawnRequest::PeriodicInjection(PeriodicInjectionSpawn {
                q: (idx % width) as i32,
                r: (idx / width) as i32,
            }));
        }
    }

    pub(crate) fn spawn_initial_population(&mut self) {
        let mut open_positions = self.empty_positions();
        open_positions.shuffle(&mut self.rng);
        let initial_population = (self.config.num_organisms as usize).min(open_positions.len());

        for _ in 0..initial_population {
            let (q, r) = open_positions
                .pop()
                .expect("initial population requires at least one unique cell per organism");
            let genome = if self.champion_pool.is_empty() {
                generate_seed_genome(&self.config.seed_genome_config, &mut self.rng)
            } else {
                let idx = self.rng.random_range(0..self.champion_pool.len());
                self.champion_pool[idx].clone()
            };
            let facing = self.random_facing();
            let organism = self.build_organism(
                genome,
                OrganismSpawnParams {
                    species_id: None,
                    q,
                    r,
                    generation: 0,
                    facing,
                    starting_energy_override: None,
                },
            );
            let added = self.add_organism(organism);
            debug_assert!(added);
        }
    }

    fn build_organism(
        &mut self,
        mut genome: OrganismGenome,
        params: OrganismSpawnParams,
    ) -> OrganismState {
        genome.topology.vision_distance = genome
            .topology
            .vision_distance
            .clamp(MIN_MUTATED_VISION_DISTANCE, MAX_MUTATED_VISION_DISTANCE);
        genome.lifecycle.body_color = genome.lifecycle.body_color.clamped();
        let id = self.alloc_organism_id();
        // Founders (no inherited species) take their own ID as species ID.
        let species_id = params.species_id.unwrap_or(SpeciesId(id.0));
        let max_health = sim_types::offspring_transfer_energy(genome.lifecycle.gestation_ticks);
        let starting_energy = params.starting_energy_override.unwrap_or(max_health);
        let mut organism = OrganismState {
            id,
            species_id,
            q: params.q,
            r: params.r,
            generation: params.generation,
            age_turns: 0,
            facing: params.facing,
            energy: starting_energy,
            health: max_health,
            max_health,
            energy_prev: starting_energy,
            health_prev: max_health,
            energy_at_last_sensing: starting_energy,
            dopamine: 0.0,
            value_prev: 0.0,
            reward_prev: 0.0,
            value_prev_feature_activations: Vec::new(),
            damage_taken_last_turn: 0.0,
            contingent_action_wasted_last_turn: false,
            is_gestating: false,
            consumptions_count: 0,
            plant_consumptions_count: 0,
            prey_consumptions_count: 0,
            reproductions_count: 0,
            last_action_taken: ActionType::Idle,
            base_metabolic_cost: 0.0,
            #[cfg(feature = "instrumentation")]
            instrumentation: Default::default(),
            brain: express_genome(&genome),
            genome,
        };
        refresh_organism_base_metabolic_cost(
            &mut organism,
            self.config.body_mass_metabolic_cost_coeff,
        );
        organism
    }

    fn random_facing(&mut self) -> FacingDirection {
        FacingDirection::ALL[self.rng.random_range(0..FacingDirection::ALL.len())]
    }

    fn alloc_organism_id(&mut self) -> OrganismId {
        let id = OrganismId(self.next_organism_id);
        self.next_organism_id += 1;
        id
    }

    fn empty_positions(&self) -> Vec<(i32, i32)> {
        let width = self.config.world_width as i32;
        let mut positions = Vec::new();
        for r in 0..width {
            for q in 0..width {
                if self.occupant_at(q, r).is_none() {
                    positions.push((q, r));
                }
            }
        }
        positions
    }
}
