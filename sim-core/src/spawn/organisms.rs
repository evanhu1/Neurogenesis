use super::*;

impl Simulation {
    pub(crate) fn resolve_spawn_requests(&mut self, queue: &[SpawnRequest]) -> Vec<OrganismState> {
        let mut spawned = Vec::new();
        for request in queue {
            let organism = match &request.kind {
                SpawnRequestKind::Reproduction(reproduction) => {
                    let mut child_genome = reproduction.parent_genome.clone();
                    mutate_genome(
                        &mut child_genome,
                        self.config.global_mutation_rate_modifier,
                        self.config.meta_mutation_enabled,
                        &mut self.rng,
                    );
                    self.build_organism(
                        child_genome,
                        OrganismSpawnParams {
                            species_id: reproduction.parent_species_id,
                            q: reproduction.q,
                            r: reproduction.r,
                            generation: reproduction.parent_generation.saturating_add(1),
                            facing: opposite_direction(reproduction.parent_facing),
                            starting_energy_override: Some(reproduction.offspring_starting_energy),
                        },
                    )
                }
                SpawnRequestKind::PeriodicInjection(injection) => {
                    let genome =
                        generate_seed_genome(&self.config.seed_genome_config, &mut self.rng);
                    let starting_energy = sim_types::offspring_transfer_energy(genome.gestation_ticks);
                    let species_id = founder_species_id(OrganismId(self.next_organism_id));
                    let facing = self.random_facing();
                    self.build_organism(
                        genome,
                        OrganismSpawnParams {
                            species_id,
                            q: injection.q,
                            r: injection.r,
                            generation: 0,
                            facing,
                            starting_energy_override: Some(starting_energy),
                        },
                    )
                }
            };

            if self.add_organism(organism.clone()) {
                spawned.push(organism);
            }
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
        if next_turn % interval != 0 {
            return;
        }

        let width = self.config.world_width as usize;
        let mut reserved = vec![false; self.occupancy.len()];
        for request in queue.iter() {
            let (q, r) = request.target_position();
            let idx = r as usize * width + q as usize;
            reserved[idx] = true;
        }

        let mut open_positions = Vec::new();
        for (idx, occupant) in self.occupancy.iter().enumerate() {
            if occupant.is_none() && !reserved[idx] {
                let q = (idx % width) as i32;
                let r = (idx / width) as i32;
                open_positions.push((q, r));
            }
        }
        open_positions.shuffle(&mut self.rng);

        for (q, r) in open_positions.into_iter().take(count) {
            queue.push(SpawnRequest {
                kind: SpawnRequestKind::PeriodicInjection(PeriodicInjectionSpawn { q, r }),
            });
        }
    }

    pub(crate) fn spawn_initial_population(&mut self) {
        let seed_config = self.config.seed_genome_config.clone();
        let champion_pool = self.champion_pool.clone();

        let mut open_positions = self.empty_positions();
        open_positions.shuffle(&mut self.rng);
        let initial_population = self.target_population().min(open_positions.len());

        for _ in 0..initial_population {
            let (q, r) = open_positions
                .pop()
                .expect("initial population requires at least one unique cell per organism");
            let genome = if champion_pool.is_empty() {
                generate_seed_genome(&seed_config, &mut self.rng)
            } else {
                let idx = self.rng.random_range(0..champion_pool.len());
                champion_pool[idx].clone()
            };
            let starting_energy = sim_types::offspring_transfer_energy(genome.gestation_ticks);
            let species_id = founder_species_id(OrganismId(self.next_organism_id));
            let facing = self.random_facing();
            let organism = self.build_organism(
                genome,
                OrganismSpawnParams {
                    species_id,
                    q,
                    r,
                    generation: 0,
                    facing,
                    starting_energy_override: Some(starting_energy),
                },
            );
            let added = self.add_organism(organism);
            debug_assert!(added);
        }
    }

    fn build_organism(
        &mut self,
        genome: OrganismGenome,
        params: OrganismSpawnParams,
    ) -> OrganismState {
        let id = self.alloc_organism_id();
        let starting_energy = params
            .starting_energy_override
            .unwrap_or_else(|| sim_types::offspring_transfer_energy(genome.gestation_ticks));
        let max_health = genome.max_health.max(1.0);
        OrganismState {
            id,
            species_id: params.species_id,
            q: params.q,
            r: params.r,
            generation: params.generation,
            age_turns: 0,
            facing: params.facing,
            energy: starting_energy,
            health: max_health,
            max_health,
            energy_prev: starting_energy,
            dopamine: 0.0,
            damage_taken_last_turn: 0.0,
            consumptions_count: 0,
            reproductions_count: 0,
            last_action_taken: ActionType::Idle,
            brain: express_genome(&genome),
            genome,
        }
    }

    fn random_facing(&mut self) -> FacingDirection {
        FacingDirection::ALL[self.rng.random_range(0..FacingDirection::ALL.len())]
    }

    fn alloc_organism_id(&mut self) -> OrganismId {
        let id = OrganismId(self.next_organism_id);
        self.next_organism_id += 1;
        id
    }

    fn target_population(&self) -> usize {
        let max_population = self.config.num_organisms as usize;
        let available_cells = if self.terrain_map.is_empty() {
            world_capacity(self.config.world_width)
        } else {
            self.terrain_map.iter().filter(|blocked| !**blocked).count()
        };
        max_population.min(available_cells)
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
