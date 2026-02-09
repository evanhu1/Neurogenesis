use sim_protocol::FacingDirection;

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
