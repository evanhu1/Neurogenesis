use super::*;
use crate::metabolism::refresh_organism_base_metabolic_cost;

impl Simulation {
    pub(crate) fn respawn_champion_pool_organism(
        &mut self,
        pool_index: usize,
        energy_fraction: f32,
        placement_version: u32,
    ) -> Option<RespawnedOpponent> {
        let genome = self.champion_pool.get(pool_index)?.clone();
        let capacity = self.occupancy.len();
        let event_key = respawn_mix64(
            self.seed
                ^ self.turn.rotate_left(17)
                ^ (pool_index as u64).rotate_left(31)
                ^ self.next_organism_id.rotate_left(43)
                ^ u64::from(placement_version),
        );
        let cell_idx = (0..capacity)
            .filter(|&index| self.occupancy[index].is_none() && !self.terrain_map[index])
            .min_by_key(|&index| respawn_mix64(event_key ^ index as u64))?;
        let width = self.config.world_width as usize;
        let q = (cell_idx % width) as i32;
        let r = (cell_idx / width) as i32;
        let facing = FacingDirection::ALL
            [(respawn_mix64(event_key ^ 0xface_cafe) as usize) % FacingDirection::ALL.len()];
        let max_health = sim_types::offspring_transfer_energy(genome.lifecycle.gestation_ticks);
        let starting_energy = max_health * energy_fraction.clamp(0.0, 1.0);
        let organism = self.build_organism(
            genome,
            OrganismSpawnParams {
                species_id: Some(SpeciesId(pool_index as u64)),
                q,
                r,
                generation: 0,
                facing,
                starting_energy_override: Some(starting_energy),
            },
        );
        let organism_id = organism.id;
        if !self.add_organism(organism) {
            return None;
        }
        self.refresh_population_metrics();
        Some(RespawnedOpponent {
            organism_id,
            pool_index,
            q,
            r,
            energy_injected: starting_energy,
        })
    }

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
            consumptions_count: 0,
            plant_consumptions_count: 0,
            prey_consumptions_count: 0,
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

pub(crate) struct RespawnedOpponent {
    pub(crate) organism_id: OrganismId,
    pub(crate) pool_index: usize,
    pub(crate) q: i32,
    pub(crate) r: i32,
    pub(crate) energy_injected: f32,
}

fn respawn_mix64(mut value: u64) -> u64 {
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}
