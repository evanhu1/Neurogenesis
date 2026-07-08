//! The hex world's substrate catalog: the 18-receptor sensory interface and the
//! 6 actuators (from `sim-core`'s `SensoryReceptor` / `ActionType`, with solo
//! `Reproduce` replaced by embodied `Mate`), plus the morphology schema.
//!
//! Coordinates place vision receptors on the exteroceptive plane laid out by
//! ray-offset × channel, and the interoceptive scalars on their own plane, so
//! geometric proximity tracks functional similarity (fracture mitigation).

use sim_substrate::catalog::{plane, ActuatorSpec, Coord, MorphologyParam, SensorSpec};
use sim_substrate::SubstrateCatalog;

pub const RAY_OFFSETS: [i8; 3] = [-1, 0, 1];
/// Channel order: Red, Green, Blue, Shape.
pub const CHANNELS: [&str; 4] = ["r", "g", "b", "shape"];

/// Sensor keys, in the canonical obs order the catalog exposes.
pub fn build_catalog() -> SubstrateCatalog {
    let mut sensors: Vec<SensorSpec> = Vec::new();

    // 12 vision receptors: 3 ray offsets × 4 channels, on the exteroceptive plane.
    for (oi, off) in RAY_OFFSETS.iter().enumerate() {
        for (ci, ch) in CHANNELS.iter().enumerate() {
            let x = (oi as f32 - 1.0) * 0.6; // -0.6, 0, 0.6
            let y = (ci as f32 - 1.5) * 0.4; // spread channels on y
            sensors.push(SensorSpec {
                key: format!("vision.{off}.{ch}"),
                arity: 1,
                coord: Coord::new(x, y, plane::EXTEROCEPTIVE),
            });
        }
    }
    // Contact-ahead sits on the exteroceptive plane too, at its own spot.
    sensors.push(SensorSpec {
        key: "contact_ahead".into(),
        arity: 1,
        coord: Coord::new(0.0, 0.9, plane::EXTEROCEPTIVE),
    });
    // Interoceptive scalars on their own plane.
    for (i, key) in [
        "intero.energy",
        "intero.health",
        "intero.energy_delta",
        "intero.last_forward",
        "intero.last_eat",
    ]
    .iter()
    .enumerate()
    {
        let x = (i as f32 - 2.0) * 0.4;
        sensors.push(SensorSpec {
            key: (*key).into(),
            arity: 1,
            coord: Coord::new(x, 0.0, plane::INTEROCEPTIVE),
        });
    }

    // 6 actuators. Idle is implicit (softmax idle logit).
    let actuators: Vec<ActuatorSpec> = [
        ("turn_left", -1.0),
        ("turn_right", -0.6),
        ("forward", -0.2),
        ("eat", 0.2),
        ("attack", 0.6),
        ("mate", 1.0),
    ]
    .iter()
    .map(|(key, x)| ActuatorSpec {
        key: (*key).into(),
        coord: Coord::new(*x, 0.0, plane::ACTUATOR),
    })
    .collect();

    // Morphology: body color (RGB), vision distance, and body size. `size` is a
    // directly-evolvable morphology dial here (decoupled from gestation, which
    // the substrate header owns) so body scale has no blind spot.
    let morphology = vec![
        MorphologyParam { key: "body_r".into(), min: 0.0, max: 1.0, default: 0.5 },
        MorphologyParam { key: "body_g".into(), min: 0.0, max: 1.0, default: 0.5 },
        MorphologyParam { key: "body_b".into(), min: 0.0, max: 1.0, default: 0.5 },
        MorphologyParam { key: "vision_distance".into(), min: 1.0, max: 10.0, default: 4.0 },
        MorphologyParam { key: "size".into(), min: 100.0, max: 1100.0, default: 200.0 },
    ];

    SubstrateCatalog {
        sensors,
        actuators,
        morphology,
    }
}

// Morphology indices.
pub const M_BODY_R: usize = 0;
pub const M_BODY_G: usize = 1;
pub const M_BODY_B: usize = 2;
pub const M_VISION: usize = 3;
pub const M_SIZE: usize = 4;
