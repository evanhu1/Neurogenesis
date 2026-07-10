use super::*;
use crate::metabolism::refresh_organism_base_metabolic_cost;

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
                SpawnRequest::Reproduction(reproduction) => {
                    // Clonal birth: the child inherits the parent genome exactly.
                    // All genetic variation is owned by the NEAT outer loop.
                    self.build_organism(
                        reproduction.offspring_genome,
                        OrganismSpawnParams {
                            species_id: Some(reproduction.parent_species_id),
                            q: reproduction.q,
                            r: reproduction.r,
                            generation: reproduction.offspring_generation,
                            facing: opposite_direction(reproduction.parent_facing),
                            starting_energy_override: Some(reproduction.offspring_starting_energy),
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

    pub(crate) fn spawn_initial_population(&mut self) {
        let mut open_positions = self.empty_positions();
        open_positions.shuffle(&mut self.rng);
        let initial_population = (self.config.num_organisms as usize).min(open_positions.len());

        for _ in 0..initial_population {
            let (q, r) = open_positions
                .pop()
                .expect("initial population requires at least one unique cell per organism");
            // Founders draw from the champion pool when one is provided (NEAT
            // seeds a clonal colony from a single candidate; the server seeds a
            // diverse population from its pool), otherwise from a fresh seed
            // genome. In-world variation no longer exists — genetic diversity is
            // owned by the NEAT outer loop.
            let genome = if self.champion_pool.is_empty() {
                generate_seed_genome(
                    &self.config.seed_genome_config,
                    self.config.predation_enabled,
                    &mut self.rng,
                )
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
        restrict_predation_genes(&mut genome, self.config.predation_enabled);
        genome.topology.vision_distance = genome
            .topology
            .vision_distance
            .clamp(MIN_MUTATED_VISION_DISTANCE, MAX_MUTATED_VISION_DISTANCE);
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
            energy_at_last_sensing: starting_energy,
            damage_taken_last_turn: 0.0,
            is_gestating: false,
            consumptions_count: 0,
            plant_consumptions_count: 0,
            prey_consumptions_count: 0,
            reproductions_count: 0,
            last_action_taken: ActionType::Idle,
            base_metabolic_cost: 0.0,
            #[cfg(feature = "instrumentation")]
            instrumentation: Default::default(),
            brain: express_genome(&genome, self.config.predation_enabled),
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

