use crate::grid::rotate_by_steps;
use brain::VISION_RAY_COUNT;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};
use types::{Occupant, OrganismId, OrganismState, SensoryReceptor};

#[cfg(feature = "profiling")]
use crate::profiling::{self, BrainStage};
#[cfg(feature = "profiling")]
use std::time::Instant;

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub(crate) struct VisionSignal {
    pub(crate) proximity: f32,
    pub(crate) energy_affordance: f32,
}

#[cfg_attr(not(feature = "instrumentation"), allow(dead_code))]
#[derive(Clone, Copy, Default)]
pub(crate) struct ScanResult {
    pub(crate) signal: VisionSignal,
    pub(crate) food_visible: bool,
}

type RayScans = [ScanResult; VISION_RAY_COUNT];

/// Immutable `(origin cell, absolute direction, distance) -> cell index`
/// lookup. The table removes coordinate updates and toroidal modulo operations
/// from the per-organism sensing hot path. It contains no world state, so
/// sharing it across simulation clones is behaviorally inert.
#[derive(Debug, Clone, Default)]
pub(crate) struct VisionRayTable {
    world_width: u32,
    vision_range: u32,
    cell_indices: Arc<[u32]>,
}

type RayGeometryKey = (u32, u32);
type SharedRayIndices = Arc<[u32]>;
type RayTableCache = Mutex<HashMap<RayGeometryKey, SharedRayIndices>>;

static VISION_RAY_TABLE_CACHE: OnceLock<RayTableCache> = OnceLock::new();

impl VisionRayTable {
    pub(crate) fn new(world_width: u32, vision_range: u32) -> Self {
        let cache = VISION_RAY_TABLE_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
        let mut cache = cache.lock().expect("vision ray table cache lock poisoned");
        let cell_indices = cache
            .entry((world_width, vision_range))
            .or_insert_with(|| {
                let width = world_width as usize;
                let range = vision_range as usize;
                let capacity = width * width;
                let mut cell_indices = Vec::with_capacity(capacity * VISION_RAY_COUNT * range);

                for origin_idx in 0..capacity {
                    let origin_q = (origin_idx % width) as i32;
                    let origin_r = (origin_idx / width) as i32;
                    for &facing in types::FacingDirection::ALL {
                        let (dq, dr) = facing_delta(facing);
                        let mut q = origin_q;
                        let mut r = origin_r;
                        for _ in 0..range {
                            q = (q + dq).rem_euclid(world_width as i32);
                            r = (r + dr).rem_euclid(world_width as i32);
                            let cell_idx = r as usize * width + q as usize;
                            cell_indices.push(
                                u32::try_from(cell_idx).expect("world cell index must fit in u32"),
                            );
                        }
                    }
                }
                cell_indices.into()
            })
            .clone();

        Self {
            world_width,
            vision_range,
            cell_indices,
        }
    }

    #[inline(always)]
    fn ray(&self, position: (i32, i32), facing: types::FacingDirection) -> &[u32] {
        debug_assert_eq!(self.cell_indices.len(), self.expected_len());
        let width = self.world_width as usize;
        let origin_idx = position.1 as usize * width + position.0 as usize;
        let start =
            (origin_idx * VISION_RAY_COUNT + facing_index(facing)) * self.vision_range as usize;
        &self.cell_indices[start..start + self.vision_range as usize]
    }

    fn expected_len(&self) -> usize {
        self.world_width as usize
            * self.world_width as usize
            * VISION_RAY_COUNT
            * self.vision_range as usize
    }
}

#[derive(Clone, Copy)]
struct RaycastContext<'a> {
    organism_id: OrganismId,
    occupancy: &'a [Option<Occupant>],
    predation_enabled: bool,
}

pub(super) fn encode_sensory_inputs(
    organism: &mut OrganismState,
    ray_table: &VisionRayTable,
    occupancy: &[Option<Occupant>],
    starting_energy: u32,
    plant_energy: u32,
    predation_enabled: bool,
) -> RayScans {
    let energy = energy_signal(organism, starting_energy);
    let energy_flow = energy_flow_signal(organism, plant_energy);
    // Plasticity runs after action commit and uses this snapshot to measure the
    // action's within-tick energy consequence. Keep the encoded Energy sensor
    // based on the pre-action value above, then roll the persisted baseline
    // forward once per sensing pass.
    organism.energy_at_last_sensing = organism.energy;
    organism.energy_flow_last_tick = 0;
    #[cfg(feature = "profiling")]
    let scan_started = Instant::now();
    let ray_scans = scan_rays(
        (organism.q, organism.r),
        organism.facing,
        organism.id,
        ray_table,
        occupancy,
        predation_enabled,
    );
    #[cfg(feature = "profiling")]
    profiling::record_brain_stage(BrainStage::ScanAhead, scan_started.elapsed());

    #[cfg(feature = "profiling")]
    let encoding_started = Instant::now();
    for sensory_neuron in &mut organism.brain.sensory {
        sensory_neuron.neuron.activation = match sensory_neuron.receptor {
            SensoryReceptor::RayProximity { ray_offset } => {
                ray_signal(&ray_scans, ray_offset).proximity
            }
            SensoryReceptor::RayEnergyAffordance { ray_offset } => {
                ray_signal(&ray_scans, ray_offset).energy_affordance
            }
            SensoryReceptor::SelfEnergy => energy,
            SensoryReceptor::EnergyFlowLastTick => energy_flow,
        };
    }
    #[cfg(feature = "profiling")]
    profiling::record_brain_stage(BrainStage::SensoryEncoding, encoding_started.elapsed());

    ray_scans
}

fn ray_signal(ray_scans: &RayScans, ray_offset: i8) -> VisionSignal {
    let Some(ray_idx) = ray_offset_index(ray_offset) else {
        return VisionSignal::default();
    };
    ray_scans[ray_idx].signal
}

pub(crate) fn scan_rays(
    position: (i32, i32),
    facing: types::FacingDirection,
    organism_id: OrganismId,
    ray_table: &VisionRayTable,
    occupancy: &[Option<Occupant>],
    predation_enabled: bool,
) -> RayScans {
    let context = RaycastContext {
        organism_id,
        occupancy,
        predation_enabled,
    };
    std::array::from_fn(|idx| {
        let ray_facing = rotate_by_steps(facing, SensoryReceptor::RAY_OFFSETS[idx]);
        scan_ray(ray_table.ray(position, ray_facing), context)
    })
}

fn ray_offset_index(ray_offset: i8) -> Option<usize> {
    SensoryReceptor::RAY_OFFSETS
        .iter()
        .position(|offset| *offset == ray_offset)
}

fn energy_signal(organism: &OrganismState, starting_energy: u32) -> f32 {
    let energy = organism.energy as f32;
    energy / (energy + starting_energy as f32)
}

fn energy_flow_signal(organism: &OrganismState, plant_energy: u32) -> f32 {
    (organism.energy_flow_last_tick as f32 / plant_energy.max(1) as f32).clamp(-1.0, 1.0)
}

fn scan_ray(cell_indices: &[u32], context: RaycastContext<'_>) -> ScanResult {
    let max_dist = cell_indices.len().max(1) as u32;
    let inv_max_dist = 1.0 / max_dist as f32;

    for (distance_idx, &cell_idx) in cell_indices.iter().enumerate() {
        let distance = distance_idx as u32 + 1;
        let distance_signal = (max_dist - distance + 1) as f32 * inv_max_dist;
        let signal = match context.occupancy[cell_idx as usize] {
            Some(Occupant::Food(_)) => VisionSignal {
                proximity: distance_signal,
                energy_affordance: 1.0,
            },
            Some(Occupant::Organism(id)) if id != context.organism_id => VisionSignal {
                proximity: distance_signal,
                energy_affordance: if context.predation_enabled { -1.0 } else { 0.0 },
            },
            Some(Occupant::Wall) => VisionSignal {
                proximity: distance_signal,
                energy_affordance: 0.0,
            },
            Some(Occupant::Organism(_)) | None => continue,
        };
        return ScanResult {
            food_visible: signal.energy_affordance > 0.0,
            signal,
        };
    }

    ScanResult::default()
}

#[inline(always)]
fn facing_index(facing: types::FacingDirection) -> usize {
    use types::FacingDirection::*;
    match facing {
        East => 0,
        NorthEast => 1,
        NorthWest => 2,
        West => 3,
        SouthWest => 4,
        SouthEast => 5,
    }
}

#[inline(always)]
fn facing_delta(facing: types::FacingDirection) -> (i32, i32) {
    use types::FacingDirection::*;
    match facing {
        East => (1, 0),
        NorthEast => (1, -1),
        NorthWest => (0, -1),
        West => (-1, 0),
        SouthWest => (-1, 1),
        SouthEast => (0, 1),
    }
}

#[cfg(test)]
mod tests {
    use super::{energy_signal, scan_ray, RaycastContext, VisionRayTable};
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;
    use types::{
        BrainState, FacingDirection, FoodId, Occupant, OrganismId, OrganismState, SpeciesId,
    };

    fn test_organism() -> OrganismState {
        let config = config::default_world_config();
        let mut rng = ChaCha8Rng::seed_from_u64(7);
        let genome = brain::genome::generate_seed_genome(
            &config.seed_genome_config,
            config.predation_enabled,
            &mut rng,
        );
        OrganismState::new(
            OrganismId(1),
            SpeciesId(1),
            1,
            1,
            0,
            0,
            FacingDirection::East,
            config.starting_energy,
            0,
            0,
            0,
            types::ActionType::Idle,
            BrainState {
                sensory: Vec::new(),
                inter: Vec::new(),
                action: Vec::new(),
                synapse_count: 0,
                sensory_mean_activation: Vec::new(),
                inter_mean_activation: Vec::new(),
                action_mean_activation: Vec::new(),
                means_initialized: false,
            },
            genome,
        )
    }

    #[test]
    fn energy_signal_is_normalized_against_starting_energy() {
        let mut organism = test_organism();
        organism.energy = 250;
        assert_eq!(energy_signal(&organism, 250), 0.5);
        organism.energy = 0;
        assert_eq!(energy_signal(&organism, 250), 0.0);
    }

    #[test]
    fn scan_ray_reports_nearest_entity_type_and_distance() {
        let world_width = 5;
        let mut occupancy = vec![None; (world_width * world_width) as usize];
        occupancy[world_width as usize + 3] = Some(Occupant::Food(FoodId(0)));
        let ray_table = VisionRayTable::new(world_width, 4);
        let scan = scan_ray(
            ray_table.ray((1, 1), FacingDirection::East),
            RaycastContext {
                organism_id: OrganismId(1),
                occupancy: &occupancy,
                predation_enabled: true,
            },
        );
        assert!(scan.food_visible);
        assert_eq!(scan.signal.proximity, 0.75);
        assert_eq!(scan.signal.energy_affordance, 1.0);
    }
}
