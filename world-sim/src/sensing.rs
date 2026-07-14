use crate::grid::rotate_by_steps;
use brain::VISION_RAY_COUNT;
use types::{Occupant, OrganismId, OrganismState, SensoryReceptor};

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

#[derive(Clone, Copy)]
struct RaycastContext<'a> {
    organism_id: OrganismId,
    world_width: i32,
    occupancy: &'a [Option<Occupant>],
    vision_range: u32,
    predation_enabled: bool,
}

pub(super) fn encode_sensory_inputs(
    organism: &mut OrganismState,
    world_width: i32,
    occupancy: &[Option<Occupant>],
    vision_range: u32,
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
    let ray_scans = scan_rays(
        (organism.q, organism.r),
        organism.facing,
        organism.id,
        world_width,
        occupancy,
        vision_range,
        predation_enabled,
    );

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
    world_width: i32,
    occupancy: &[Option<Occupant>],
    vision_range: u32,
    predation_enabled: bool,
) -> RayScans {
    let context = RaycastContext {
        organism_id,
        world_width,
        occupancy,
        vision_range,
        predation_enabled,
    };
    std::array::from_fn(|idx| {
        scan_ray(
            position,
            rotate_by_steps(facing, SensoryReceptor::RAY_OFFSETS[idx]),
            context,
        )
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

fn scan_ray(
    position: (i32, i32),
    ray_facing: types::FacingDirection,
    context: RaycastContext<'_>,
) -> ScanResult {
    let max_dist = context.vision_range.max(1);
    let inv_max_dist = 1.0 / max_dist as f32;
    let width = context.world_width;
    let width_usize = width as usize;
    let (dq, dr) = facing_delta(ray_facing);
    let mut q = position.0;
    let mut r = position.1;

    for distance in 1..=max_dist {
        q = (q + dq).rem_euclid(width);
        r = (r + dr).rem_euclid(width);
        let idx = r as usize * width_usize + q as usize;
        let distance_signal = (max_dist - distance + 1) as f32 * inv_max_dist;
        let signal = match context.occupancy[idx] {
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
    use super::{energy_signal, scan_ray, RaycastContext};
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
        let scan = scan_ray(
            (1, 1),
            FacingDirection::East,
            RaycastContext {
                organism_id: OrganismId(1),
                world_width,
                occupancy: &occupancy,
                vision_range: 4,
                predation_enabled: true,
            },
        );
        assert!(scan.food_visible);
        assert_eq!(scan.signal.proximity, 0.75);
        assert_eq!(scan.signal.energy_affordance, 1.0);
    }
}
