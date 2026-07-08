//! Quality-Diversity champion archive (MAP-Elites). With no fitness function,
//! open-ended progress is measured by *behavioral coverage*: champions are
//! archived by a discretized behavior descriptor, one elite per cell. Coverage
//! and QD-score are the progress signals that replace a fitness curve.

use crate::genome::Genome;
use serde::{Deserialize, Serialize};

/// A behavior descriptor: a small vector of normalized `[0,1]` behavioral traits
/// (env-specific, e.g. diet ratio, brain size, exploration). The archive
/// discretizes each dimension into `resolution` bins.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BehaviorDescriptor {
    pub values: Vec<f32>,
}

impl BehaviorDescriptor {
    pub fn new(values: Vec<f32>) -> Self {
        BehaviorDescriptor { values }
    }

    fn cell(&self, resolution: usize) -> Vec<u16> {
        self.values
            .iter()
            .map(|v| {
                let clamped = v.clamp(0.0, 1.0 - f32::EPSILON);
                (clamped * resolution as f32) as u16
            })
            .collect()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArchiveEntry {
    pub genome: Genome,
    pub descriptor: BehaviorDescriptor,
    /// Local quality used to break ties within a cell (e.g. lifetime
    /// reproductions/energy). Higher wins.
    pub quality: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QdArchive {
    pub resolution: usize,
    cells: std::collections::BTreeMap<Vec<u16>, ArchiveEntry>,
}

impl QdArchive {
    pub fn new(resolution: usize) -> Self {
        QdArchive {
            resolution: resolution.max(1),
            cells: std::collections::BTreeMap::new(),
        }
    }

    /// Insert a candidate; keeps the higher-quality elite per cell. Returns
    /// `true` if it became (or replaced) the cell's elite.
    pub fn insert(&mut self, genome: Genome, descriptor: BehaviorDescriptor, quality: f32) -> bool {
        let cell = descriptor.cell(self.resolution);
        match self.cells.get(&cell) {
            Some(existing) if existing.quality >= quality => false,
            _ => {
                self.cells.insert(
                    cell,
                    ArchiveEntry {
                        genome,
                        descriptor,
                        quality,
                    },
                );
                true
            }
        }
    }

    /// Number of occupied cells.
    pub fn coverage(&self) -> usize {
        self.cells.len()
    }

    /// Sum of elite qualities across the archive — the standard QD-score.
    pub fn qd_score(&self) -> f32 {
        self.cells.values().map(|e| e.quality).sum()
    }

    pub fn entries(&self) -> impl Iterator<Item = &ArchiveEntry> {
        self.cells.values()
    }

    pub fn len(&self) -> usize {
        self.cells.len()
    }

    pub fn is_empty(&self) -> bool {
        self.cells.is_empty()
    }
}
