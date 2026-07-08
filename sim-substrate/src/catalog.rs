//! The environment's declaration of what an organism *could* sense, do, and be
//! shaped like. `develop()` reads the catalog to decide, per genome, which
//! affordances actually grow.
//!
//! Coordinates carry a `z` **functional plane** so that geometric proximity
//! tracks functional similarity (fracture mitigation, plan risk #2): put the
//! vision receptors on one plane, interoceptors on another, actuators on a
//! third. The CPPN then keys on function through geometry rather than in spite
//! of it.

use serde::{Deserialize, Serialize};

/// A point in the ES-HyperNEAT substrate. `x`/`y` place the neuron within its
/// plane; `z` selects the plane.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Coord {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl Coord {
    pub const fn new(x: f32, y: f32, z: f32) -> Self {
        Coord { x, y, z }
    }
}

/// Conventional plane coordinates. Keeping them apart on `z` is the whole point.
pub mod plane {
    pub const EXTEROCEPTIVE: f32 = -1.0;
    pub const INTEROCEPTIVE: f32 = -0.5;
    pub const HIDDEN: f32 = 0.0;
    pub const ACTUATOR: f32 = 1.0;
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SensorSpec {
    /// Stable string key (e.g. "vision.r.ahead", "intero.energy").
    pub key: String,
    /// How many obs-vector slots this sensor occupies (usually 1).
    pub arity: u16,
    /// Where the sensor's input neuron sits in the substrate.
    pub coord: Coord,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActuatorSpec {
    pub key: String,
    /// Where the actuator's output neuron sits in the substrate.
    pub coord: Coord,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MorphologyParam {
    pub key: String,
    pub min: f32,
    pub max: f32,
    pub default: f32,
}

impl MorphologyParam {
    /// Map a normalized `[0,1]` header scalar onto the real parameter range.
    pub fn denormalize(&self, normalized: f32) -> f32 {
        self.min + normalized.clamp(0.0, 1.0) * (self.max - self.min)
    }
}

/// The full interface an environment exposes to the substrate.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubstrateCatalog {
    pub sensors: Vec<SensorSpec>,
    pub actuators: Vec<ActuatorSpec>,
    pub morphology: Vec<MorphologyParam>,
}

impl SubstrateCatalog {
    pub fn morphology_defaults(&self) -> Vec<f32> {
        // Normalized defaults; we store the *normalized* value so mutation stays
        // uniform.
        self.morphology
            .iter()
            .map(|m| {
                if (m.max - m.min).abs() < f32::EPSILON {
                    0.0
                } else {
                    ((m.default - m.min) / (m.max - m.min)).clamp(0.0, 1.0)
                }
            })
            .collect()
    }
}

/// Which sensors expressed, and where each lands in the observation vector.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ObsLayout {
    /// Catalog sensor index for each expressed sensor, in obs-slot order.
    pub sensor_indices: Vec<usize>,
    /// Starting obs-vector offset for each expressed sensor (parallel to
    /// `sensor_indices`); `len` gives the total obs width.
    pub offsets: Vec<usize>,
    pub len: usize,
}

/// Which actuators expressed, and where each lands in the action vector.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ActionLayout {
    /// Catalog actuator index for each expressed actuator, in action-slot order.
    pub actuator_indices: Vec<usize>,
    pub len: usize,
}
