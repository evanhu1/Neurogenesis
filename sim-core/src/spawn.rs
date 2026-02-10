use crate::Simulation;
use crate::{world_capacity, SpawnRequest, SpawnRequestKind};
use rand::seq::SliceRandom;
use rand::Rng;
use sim_protocol::{FacingDirection, OrganismId, OrganismState};
use std::f64::consts::PI;

impl Simulation {
    pub(crate) fn resolve_spawn_requests(&mut self, queue: &[SpawnRequest]) -> Vec<OrganismState> {
        let mut spawned = Vec::new();
        for request in queue {
            let Some((q, r)) = self.sample_center_weighted_spawn_position() else {
                continue;
            };

            let id = self.alloc_organism_id();
            let organism = match request.kind {
                SpawnRequestKind::StarvationReplacement => OrganismState {
                    id,
                    q,
                    r,
                    age_turns: 0,
                    facing: self.random_facing(),
                    turns_since_last_meal: 0,
                    meals_eaten: 0,
                    brain: self.generate_brain(),
                },
                SpawnRequestKind::Reproduction { parent } => {
                    let Some(parent_state) = self
                        .organisms
                        .iter()
                        .find(|organism| organism.id == parent)
                        .cloned()
                    else {
                        continue;
                    };

                    let mut brain = parent_state.brain;
                    self.mutate_brain(&mut brain);
                    OrganismState {
                        id,
                        q,
                        r,
                        age_turns: 0,
                        facing: parent_state.facing,
                        turns_since_last_meal: 0,
                        meals_eaten: 0,
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

    fn sample_center_weighted_spawn_position(&mut self) -> Option<(i32, i32)> {
        if self.organisms.len() >= world_capacity(self.config.world_width) {
            return None;
        }

        let width = self.config.world_width as i32;
        let attempts = (world_capacity(self.config.world_width) * 4).max(64);
        let center = (width as f64 - 1.0) / 2.0;
        let spread = (self.config.center_spawn_max_fraction
            - self.config.center_spawn_min_fraction)
            .abs()
            .max(0.05);
        let sigma = (width as f64 * f64::from(spread) / 2.0).max(0.5);

        for _ in 0..attempts {
            let (z_q, z_r) = self.sample_standard_normal_pair();
            let q = (center + z_q * sigma).round() as i32;
            let r = (center + z_r * sigma).round() as i32;
            if !self.in_bounds(q, r) {
                continue;
            }
            if self.occupant_at(q, r).is_none() {
                return Some((q, r));
            }
        }

        self.nearest_empty_to_center()
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

    fn nearest_empty_to_center(&self) -> Option<(i32, i32)> {
        let width = self.config.world_width as i32;
        let center = (width as f64 - 1.0) / 2.0;

        let mut best: Option<((i32, i32), f64)> = None;
        for r in 0..width {
            for q in 0..width {
                if self.occupant_at(q, r).is_some() {
                    continue;
                }
                let distance = (q as f64 - center).powi(2) + (r as f64 - center).powi(2);
                match best {
                    None => best = Some(((q, r), distance)),
                    Some(((best_q, best_r), best_distance))
                        if distance < best_distance
                            || (distance == best_distance && (r, q) < (best_r, best_q)) =>
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
        let mut open_positions = self.empty_positions();
        open_positions.shuffle(&mut self.rng);

        for _ in 0..self.target_population() {
            let (q, r) = open_positions
                .pop()
                .expect("initial population requires at least one unique cell per organism");
            let id = self.alloc_organism_id();
            let brain = self.generate_brain();
            let facing = self.random_facing();
            let organism = OrganismState {
                id,
                q,
                r,
                age_turns: 0,
                facing,
                turns_since_last_meal: 0,
                meals_eaten: 0,
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
