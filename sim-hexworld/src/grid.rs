//! Hex grid helpers, ported verbatim from `sim-core/src/grid.rs`. Axial `(q, r)`
//! coordinates on a toroidal grid; six facings.

use sim_types::FacingDirection;

pub fn rotate_left(d: FacingDirection) -> FacingDirection {
    match d {
        FacingDirection::East => FacingDirection::NorthEast,
        FacingDirection::NorthEast => FacingDirection::NorthWest,
        FacingDirection::NorthWest => FacingDirection::West,
        FacingDirection::West => FacingDirection::SouthWest,
        FacingDirection::SouthWest => FacingDirection::SouthEast,
        FacingDirection::SouthEast => FacingDirection::East,
    }
}

pub fn rotate_right(d: FacingDirection) -> FacingDirection {
    match d {
        FacingDirection::East => FacingDirection::SouthEast,
        FacingDirection::SouthEast => FacingDirection::SouthWest,
        FacingDirection::SouthWest => FacingDirection::West,
        FacingDirection::West => FacingDirection::NorthWest,
        FacingDirection::NorthWest => FacingDirection::NorthEast,
        FacingDirection::NorthEast => FacingDirection::East,
    }
}

pub fn rotate_by_steps(mut d: FacingDirection, steps: i8) -> FacingDirection {
    if steps >= 0 {
        for _ in 0..steps as u8 {
            d = rotate_right(d);
        }
    } else {
        for _ in 0..steps.unsigned_abs() {
            d = rotate_left(d);
        }
    }
    d
}

pub fn opposite_direction(d: FacingDirection) -> FacingDirection {
    match d {
        FacingDirection::East => FacingDirection::West,
        FacingDirection::NorthEast => FacingDirection::SouthWest,
        FacingDirection::NorthWest => FacingDirection::SouthEast,
        FacingDirection::West => FacingDirection::East,
        FacingDirection::SouthWest => FacingDirection::NorthEast,
        FacingDirection::SouthEast => FacingDirection::NorthWest,
    }
}

#[inline]
pub fn facing_delta(facing: FacingDirection) -> (i32, i32) {
    match facing {
        FacingDirection::East => (1, 0),
        FacingDirection::NorthEast => (1, -1),
        FacingDirection::NorthWest => (0, -1),
        FacingDirection::West => (-1, 0),
        FacingDirection::SouthWest => (-1, 1),
        FacingDirection::SouthEast => (0, 1),
    }
}

pub const ALL_FACINGS: [FacingDirection; 6] = [
    FacingDirection::East,
    FacingDirection::NorthEast,
    FacingDirection::NorthWest,
    FacingDirection::West,
    FacingDirection::SouthWest,
    FacingDirection::SouthEast,
];

#[inline]
pub fn wrap_coord(c: i32, width: i32) -> i32 {
    c.rem_euclid(width)
}

#[inline]
pub fn hex_neighbor(pos: (i32, i32), facing: FacingDirection, width: i32) -> (i32, i32) {
    let (dq, dr) = facing_delta(facing);
    (wrap_coord(pos.0 + dq, width), wrap_coord(pos.1 + dr, width))
}

#[inline]
pub fn cell_index(q: i32, r: i32, width: i32) -> usize {
    let (q, r) = (wrap_coord(q, width), wrap_coord(r, width));
    r as usize * width as usize + q as usize
}
