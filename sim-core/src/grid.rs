use crate::Simulation;
use sim_protocol::FacingDirection;
use sim_protocol::{OrganismId, OrganismState};

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

pub(crate) fn hex_neighbor(position: (i32, i32), facing: FacingDirection) -> (i32, i32) {
    let (q, r) = position;
    match facing {
        FacingDirection::East => (q + 1, r),
        FacingDirection::NorthEast => (q + 1, r - 1),
        FacingDirection::NorthWest => (q, r - 1),
        FacingDirection::West => (q - 1, r),
        FacingDirection::SouthWest => (q - 1, r + 1),
        FacingDirection::SouthEast => (q, r + 1),
    }
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
            debug_assert_eq!(
                self.organisms.len(),
                self.occupancy.iter().flatten().count(),
                "occupancy vector count should match organism count",
            );
            for organism in &self.organisms {
                let idx = self
                    .cell_index(organism.q, organism.r)
                    .expect("organism position must remain in bounds");
                debug_assert_eq!(
                    self.occupancy[idx],
                    Some(organism.id),
                    "occupancy must point at organism occupying that cell",
                );
            }
        }
    }

    pub(crate) fn add_organism(&mut self, organism: OrganismState) -> bool {
        let Some(cell_idx) = self.cell_index(organism.q, organism.r) else {
            return false;
        };
        if self.occupancy[cell_idx].is_some() {
            return false;
        }

        self.occupancy[cell_idx] = Some(organism.id);
        self.organisms.push(organism);
        true
    }

    pub(crate) fn rebuild_occupancy(&mut self) {
        self.occupancy.fill(None);
        for organism in &self.organisms {
            let idx = self
                .cell_index(organism.q, organism.r)
                .expect("organism must remain in bounds");
            debug_assert!(self.occupancy[idx].is_none());
            self.occupancy[idx] = Some(organism.id);
        }
    }

    pub(crate) fn occupant_at(&self, q: i32, r: i32) -> Option<OrganismId> {
        let idx = self.cell_index(q, r)?;
        self.occupancy[idx]
    }

    pub(crate) fn in_bounds(&self, q: i32, r: i32) -> bool {
        let width = self.config.world_width as i32;
        q >= 0 && r >= 0 && q < width && r < width
    }

    pub(crate) fn cell_index(&self, q: i32, r: i32) -> Option<usize> {
        if !self.in_bounds(q, r) {
            return None;
        }
        let width = self.config.world_width as usize;
        Some(r as usize * width + q as usize)
    }
}
