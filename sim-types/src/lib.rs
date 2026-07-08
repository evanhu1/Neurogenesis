//! `sim-types` — generic, environment-neutral domain types shared by the hex
//! world (and any future consumer). After the substrate redesign this crate no
//! longer holds any genome/brain/interface types (those live in
//! `sim-substrate`); it keeps only the small, reusable value types the hex
//! ecology needs: colors, visual properties, hex facings, food kinds, and the
//! deterministic helpers over them.

use serde::{Deserialize, Serialize};

pub const MAX_GESTATION_TICKS: u8 = 10;
pub const BASE_OFFSPRING_TRANSFER_ENERGY: f32 = 100.0;
pub const GESTATION_TRANSFER_ENERGY_STEP: f32 = 100.0;

pub const ORGANISM_VISUAL_OPACITY: f32 = 0.8;
pub const FOOD_VISUAL_OPACITY: f32 = 0.8;
pub const ORGANISM_SHAPE: f32 = 0.2;
pub const PLANT_SHAPE: f32 = 0.4;
pub const CORPSE_SHAPE: f32 = 0.6;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
pub enum FoodKind {
    #[default]
    Plant,
    Corpse,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum FacingDirection {
    East,
    NorthEast,
    NorthWest,
    West,
    SouthWest,
    SouthEast,
}

impl FacingDirection {
    pub const ALL: [FacingDirection; 6] = [
        FacingDirection::East,
        FacingDirection::NorthEast,
        FacingDirection::NorthWest,
        FacingDirection::West,
        FacingDirection::SouthWest,
        FacingDirection::SouthEast,
    ];
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Default)]
pub struct RgbColor {
    pub r: f32,
    pub g: f32,
    pub b: f32,
}

impl RgbColor {
    pub fn clamped(self) -> Self {
        Self {
            r: self.r.clamp(0.0, 1.0),
            g: self.g.clamp(0.0, 1.0),
            b: self.b.clamp(0.0, 1.0),
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Default)]
pub struct VisualProperties {
    pub r: f32,
    pub g: f32,
    pub b: f32,
    pub opacity: f32,
    pub shape: f32,
}

impl VisualProperties {
    pub fn clamped(self) -> Self {
        Self {
            r: self.r.clamp(0.0, 1.0),
            g: self.g.clamp(0.0, 1.0),
            b: self.b.clamp(0.0, 1.0),
            opacity: self.opacity.clamp(0.0, 1.0),
            shape: self.shape.clamp(0.0, 1.0),
        }
    }
}

/// Energy an organism invests per offspring, as a function of gestation length.
pub fn offspring_transfer_energy(gestation_ticks: u8) -> f32 {
    BASE_OFFSPRING_TRANSFER_ENERGY
        + GESTATION_TRANSFER_ENERGY_STEP * f32::from(gestation_ticks.min(MAX_GESTATION_TICKS))
}

/// Hue angle (radians) of an RGB color on the color wheel. Greys map to 0.
/// Pure and deterministic — used by the zero-sum social-color transfer.
pub fn color_hue(c: RgbColor) -> f32 {
    let two_r_minus_g_minus_b = 2.0 * c.r - c.g - c.b;
    let sqrt3_g_minus_b = 3.0_f32.sqrt() * (c.g - c.b);
    if two_r_minus_g_minus_b == 0.0 && sqrt3_g_minus_b == 0.0 {
        return 0.0;
    }
    sqrt3_g_minus_b.atan2(two_r_minus_g_minus_b)
}

pub fn organism_visual(color: RgbColor) -> VisualProperties {
    VisualProperties {
        r: color.r,
        g: color.g,
        b: color.b,
        opacity: ORGANISM_VISUAL_OPACITY,
        shape: ORGANISM_SHAPE,
    }
    .clamped()
}

pub fn food_visual(kind: FoodKind) -> VisualProperties {
    match kind {
        FoodKind::Plant => VisualProperties {
            r: 0.0,
            g: 1.0,
            b: 0.0,
            opacity: FOOD_VISUAL_OPACITY,
            shape: PLANT_SHAPE,
        },
        FoodKind::Corpse => VisualProperties {
            r: 0.95,
            g: 0.45,
            b: 0.10,
            opacity: FOOD_VISUAL_OPACITY,
            shape: CORPSE_SHAPE,
        },
    }
}
