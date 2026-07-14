use super::*;

impl Simulation {
    pub(crate) fn spawn_initial_population(&mut self) {
        let mut open_positions = self.empty_positions();
        open_positions.shuffle(&mut self.rng);
        let initial_population = (self.config.num_organisms as usize).min(open_positions.len());

        for founder_slot in 0..initial_population {
            let (q, r) = open_positions
                .pop()
                .expect("initial population requires at least one unique cell per organism");
            // Founders draw from the champion pool when one is provided,
            // otherwise from a fresh seed genome. In-world variation no longer
            // exists — genetic diversity is owned by the NEAT outer loop.
            //
            // Pool assignment is round-robin (founder `k` uses pool entry
            // `k % len`) rather than random. This gives every pool genome an
            // even founder share AND makes lineage attributable without a new
            // field: a founder takes its own id as species_id and descendants
            // inherit it, so any organism's source pool entry is
            // `species_id % pool_len` — used by competitive NEAT evaluation. A
            // single-entry pool (the clonal colony) is unchanged.
            let genome = if self.champion_pool.is_empty() {
                generate_seed_genome(
                    &self.config.seed_genome_config,
                    self.config.predation_enabled,
                    &mut self.rng,
                )
            } else {
                self.champion_pool[founder_slot % self.champion_pool.len()].clone()
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
        let id = self.alloc_organism_id();
        // Founders (no inherited species) take their own ID as species ID.
        let species_id = params.species_id.unwrap_or(SpeciesId(id.0));
        let starting_energy = params
            .starting_energy_override
            .unwrap_or(self.config.starting_energy);
        OrganismState {
            id,
            species_id,
            q: params.q,
            r: params.r,
            generation: params.generation,
            age_turns: 0,
            facing: params.facing,
            energy: starting_energy,
            energy_at_last_sensing: starting_energy,
            energy_flow_last_tick: 0,
            consumptions_count: 0,
            plant_consumptions_count: 0,
            prey_consumptions_count: 0,
            last_action_taken: ActionType::Idle,
            #[cfg(feature = "instrumentation")]
            instrumentation: Default::default(),
            brain: express_genome(&genome, self.config.predation_enabled),
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
