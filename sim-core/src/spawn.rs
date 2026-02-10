use crate::brain::reset_brain_runtime_state;
use crate::grid::opposite_direction;
use crate::Simulation;
use crate::{world_capacity, SpawnRequest, SpawnRequestKind};
use rand::seq::SliceRandom;
use rand::Rng;
use sim_protocol::{FacingDirection, OrganismId, OrganismState, SpeciesId};
use std::f64::consts::PI;

impl Simulation {
    pub(crate) fn resolve_spawn_requests(&mut self, queue: &[SpawnRequest]) -> Vec<OrganismState> {
        let mut spawned = Vec::new();
        for request in queue {
            let organism = match &request.kind {
                SpawnRequestKind::StarvationReplacement => {
                    let Some((q, r)) = self.sample_center_weighted_spawn_position() else {
                        continue;
                    };
                    let Some(species_id) = self.sample_random_species_id() else {
                        continue;
                    };
                    let Some(species_config) = self.species_config(species_id).cloned() else {
                        continue;
                    };
                    OrganismState {
                        id: self.alloc_organism_id(),
                        species_id,
                        q,
                        r,
                        age_turns: 0,
                        facing: self.random_facing(),
                        energy: self.config.starting_energy,
                        consumptions_count: 0,
                        reproductions_count: 0,
                        brain: self.generate_brain(&species_config),
                    }
                }
                SpawnRequestKind::Reproduction(reproduction) => {
                    let Some(species_config) =
                        self.species_config(reproduction.species_id).cloned()
                    else {
                        continue;
                    };

                    let mut brain = reproduction.parent_brain.clone();
                    self.mutate_brain(&mut brain, &species_config);
                    reset_brain_runtime_state(&mut brain);
                    OrganismState {
                        id: self.alloc_organism_id(),
                        species_id: reproduction.species_id,
                        q: reproduction.q,
                        r: reproduction.r,
                        age_turns: 0,
                        facing: opposite_direction(reproduction.parent_facing),
                        energy: self.config.starting_energy,
                        consumptions_count: 0,
                        reproductions_count: 0,
                        brain,
                    }
                }
            };

            if self.add_organism(organism.clone()) {
                spawned.push(organism);
            }
        }

        self.organisms.sort_by_key(|organism| organism.id);
        spawned
    }

    fn sample_random_species_id(&mut self) -> Option<SpeciesId> {
        if self.species_registry.is_empty() {
            return None;
        }
        let species_ids: Vec<SpeciesId> = self.species_registry.keys().copied().collect();
        Some(species_ids[self.rng.random_range(0..species_ids.len())])
    }

    fn sample_center_weighted_spawn_position(&mut self) -> Option<(i32, i32)> {
        if self.organisms.len() >= world_capacity(self.config.world_width) {
            return None;
        }

        let width = self.config.world_width as i32;
        let attempts = (world_capacity(self.config.world_width) * 4).max(64);
        let center = (width as f64 - 1.0) / 2.0;
        let min_radius = width as f64 * f64::from(self.config.center_spawn_min_fraction) / 2.0;
        let max_radius = width as f64 * f64::from(self.config.center_spawn_max_fraction) / 2.0;
        let radius_sigma = ((max_radius - min_radius) / 2.0).max(0.5);
        let radius_mean = (min_radius + max_radius) / 2.0;
        let min_radius_sq = min_radius * min_radius;
        let max_radius_sq = max_radius * max_radius;

        for _ in 0..attempts {
            let (z_radius, _) = self.sample_standard_normal_pair();
            let theta = self.rng.random_range(0.0..(2.0 * PI));
            let radius = (radius_mean + z_radius * radius_sigma).clamp(min_radius, max_radius);
            let q = (center + radius * theta.cos()).round() as i32;
            let r = (center + radius * theta.sin()).round() as i32;
            if !self.in_bounds(q, r) {
                continue;
            }
            let distance_sq = (q as f64 - center).powi(2) + (r as f64 - center).powi(2);
            if distance_sq < min_radius_sq || distance_sq > max_radius_sq {
                continue;
            }
            if self.occupant_at(q, r).is_none() {
                return Some((q, r));
            }
        }

        self.nearest_empty_to_point(center, center, Some((min_radius_sq, max_radius_sq)))
            .or_else(|| self.nearest_empty_to_point(center, center, None))
    }

    fn sample_standard_normal_pair(&mut self) -> (f64, f64) {
        let u1 = loop {
            let sample = self.rng.random::<f64>();
            if sample > f64::EPSILON {
                break sample;
            }
        };
        let u2 = self.rng.random::<f64>();

        let radius = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * PI * u2;
        (radius * theta.cos(), radius * theta.sin())
    }

    fn nearest_empty_to_point(
        &self,
        point_q: f64,
        point_r: f64,
        radius_sq_bounds: Option<(f64, f64)>,
    ) -> Option<(i32, i32)> {
        let width = self.config.world_width as i32;

        let mut best: Option<((i32, i32), f64)> = None;
        for r in 0..width {
            for q in 0..width {
                if self.occupant_at(q, r).is_some() {
                    continue;
                }
                let distance = (q as f64 - point_q).powi(2) + (r as f64 - point_r).powi(2);
                if let Some((min_radius_sq, max_radius_sq)) = radius_sq_bounds {
                    if distance < min_radius_sq || distance > max_radius_sq {
                        continue;
                    }
                }
                match best {
                    None => best = Some(((q, r), distance)),
                    Some(((best_q, best_r), best_distance))
                        if distance < best_distance
                            || ((distance - best_distance).abs() <= f64::EPSILON
                                && (r, q) < (best_r, best_q)) =>
                    {
                        best = Some(((q, r), distance));
                    }
                    _ => {}
                }
            }
        }

        best.map(|(position, _)| position)
    }

    pub(crate) fn spawn_initial_population(&mut self) {
        let Some((&seed_species_id, seed_species_config)) = self.species_registry.first_key_value()
        else {
            return;
        };
        let seed_species_config = seed_species_config.clone();

        let mut open_positions = self.empty_positions();
        open_positions.shuffle(&mut self.rng);

        for _ in 0..self.target_population() {
            let (q, r) = open_positions
                .pop()
                .expect("initial population requires at least one unique cell per organism");
            let id = self.alloc_organism_id();
            let brain = self.generate_brain(&seed_species_config);
            let facing = self.random_facing();
            let organism = OrganismState {
                id,
                species_id: seed_species_id,
                q,
                r,
                age_turns: 0,
                facing,
                energy: self.config.starting_energy,
                consumptions_count: 0,
                reproductions_count: 0,
                brain,
            };
            let added = self.add_organism(organism);
            debug_assert!(added);
        }

        self.organisms.sort_by_key(|organism| organism.id);
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
        (self.config.num_organisms as usize).min(world_capacity(self.config.world_width))
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
