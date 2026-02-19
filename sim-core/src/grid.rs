use crate::Simulation;
use sim_types::FacingDirection;
#[cfg(test)]
use sim_types::FoodState;
use sim_types::{Occupant, OrganismState};

pub(crate) fn world_capacity(width: u32) -> usize {
    width as usize * width as usize
}

pub(crate) fn rotate_left(direction: FacingDirection) -> FacingDirection {
    match direction {
        FacingDirection::East => FacingDirection::NorthEast,
        FacingDirection::NorthEast => FacingDirection::NorthWest,
        FacingDirection::NorthWest => FacingDirection::West,
        FacingDirection::West => FacingDirection::SouthWest,
        FacingDirection::SouthWest => FacingDirection::SouthEast,
        FacingDirection::SouthEast => FacingDirection::East,
    }
}

pub(crate) fn rotate_right(direction: FacingDirection) -> FacingDirection {
    match direction {
        FacingDirection::East => FacingDirection::SouthEast,
        FacingDirection::SouthEast => FacingDirection::SouthWest,
        FacingDirection::SouthWest => FacingDirection::West,
        FacingDirection::West => FacingDirection::NorthWest,
        FacingDirection::NorthWest => FacingDirection::NorthEast,
        FacingDirection::NorthEast => FacingDirection::East,
    }
}

pub(crate) fn wrap_position(position: (i32, i32), world_width: i32) -> (i32, i32) {
    (
        position.0.rem_euclid(world_width),
        position.1.rem_euclid(world_width),
    )
}

pub(crate) fn hex_neighbor(
    position: (i32, i32),
    facing: FacingDirection,
    world_width: i32,
) -> (i32, i32) {
    let (q, r) = position;
    let neighbor = match facing {
        FacingDirection::East => (q + 1, r),
        FacingDirection::NorthEast => (q + 1, r - 1),
        FacingDirection::NorthWest => (q, r - 1),
        FacingDirection::West => (q - 1, r),
        FacingDirection::SouthWest => (q - 1, r + 1),
        FacingDirection::SouthEast => (q, r + 1),
    };
    wrap_position(neighbor, world_width)
}

pub(crate) fn opposite_direction(direction: FacingDirection) -> FacingDirection {
    match direction {
        FacingDirection::East => FacingDirection::West,
        FacingDirection::NorthEast => FacingDirection::SouthWest,
        FacingDirection::NorthWest => FacingDirection::SouthEast,
        FacingDirection::West => FacingDirection::East,
        FacingDirection::SouthWest => FacingDirection::NorthEast,
        FacingDirection::SouthEast => FacingDirection::NorthWest,
    }
}

impl Simulation {
    pub(crate) fn debug_assert_consistent_state(&self) {
        if cfg!(debug_assertions) {
            debug_assert!(
                self.organisms.windows(2).all(|w| w[0].id < w[1].id),
                "organisms must be sorted by id"
            );
            debug_assert!(
                self.foods.windows(2).all(|w| w[0].id < w[1].id),
                "foods must be sorted by id"
            );
            debug_assert_eq!(
                self.organisms.len()
                    + self.foods.len()
                    + self.terrain_map.iter().filter(|blocked| **blocked).count(),
                self.occupancy.iter().flatten().count(),
                "occupancy vector count should match total entity count",
            );
            for organism in &self.organisms {
                let expected = self.wrap_position(organism.q, organism.r);
                debug_assert_eq!(
                    (organism.q, organism.r),
                    expected,
                    "organism position must remain canonical",
                );
                let idx = self.cell_index(organism.q, organism.r);
                debug_assert_eq!(
                    self.occupancy[idx],
                    Some(Occupant::Organism(organism.id)),
                    "occupancy must point at organism occupying that cell",
                );
            }
            for food in &self.foods {
                let expected = self.wrap_position(food.q, food.r);
                debug_assert_eq!(
                    (food.q, food.r),
                    expected,
                    "food position must remain canonical",
                );
                let idx = self.cell_index(food.q, food.r);
                debug_assert_eq!(
                    self.occupancy[idx],
                    Some(Occupant::Food(food.id)),
                    "occupancy must point at food occupying that cell",
                );
            }
            for (idx, blocked) in self.terrain_map.iter().copied().enumerate() {
                if blocked {
                    debug_assert_eq!(
                        self.occupancy[idx],
                        Some(Occupant::Wall),
                        "occupancy must point at wall for blocked terrain cells",
                    );
                }
            }
        }
    }

    pub(crate) fn add_organism(&mut self, mut organism: OrganismState) -> bool {
        let (q, r) = self.wrap_position(organism.q, organism.r);
        let cell_idx = self.cell_index(q, r);
        if self.occupancy[cell_idx].is_some() {
            return false;
        }

        organism.q = q;
        organism.r = r;
        self.occupancy[cell_idx] = Some(Occupant::Organism(organism.id));
        self.organisms.push(organism);
        true
    }

    #[cfg(test)]
    pub(crate) fn add_food(&mut self, mut food: FoodState) -> bool {
        let (q, r) = self.wrap_position(food.q, food.r);
        let cell_idx = self.cell_index(q, r);
        if self.occupancy[cell_idx].is_some() {
            return false;
        }

        food.q = q;
        food.r = r;
        self.occupancy[cell_idx] = Some(Occupant::Food(food.id));
        self.foods.push(food);
        true
    }

    pub(crate) fn occupant_at(&self, q: i32, r: i32) -> Option<Occupant> {
        let idx = self.cell_index(q, r);
        self.occupancy[idx]
    }

    pub(crate) fn wrap_position(&self, q: i32, r: i32) -> (i32, i32) {
        let width = self.config.world_width as i32;
        wrap_position((q, r), width)
    }

    pub(crate) fn cell_index(&self, q: i32, r: i32) -> usize {
        let (q, r) = self.wrap_position(q, r);
        let width = self.config.world_width as usize;
        r as usize * width + q as usize
    }
}
