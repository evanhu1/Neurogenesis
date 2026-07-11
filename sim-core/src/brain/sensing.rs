use super::*;
use crate::grid::{hex_neighbor, rotate_by_steps};

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub(crate) struct VisionSignal {
    pub(crate) food: f32,
    pub(crate) organism: f32,
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
    vision_distance: u32,
}

pub(super) fn encode_sensory_inputs(
    organism: &mut OrganismState,
    world_width: i32,
    occupancy: &[Option<Occupant>],
    vision_distance: u32,
) -> RayScans {
    let contact_ahead = contact_ahead_signal(
        (organism.q, organism.r),
        organism.facing,
        world_width,
        occupancy,
    );
    let energy = energy_signal(organism);
    let health = health_signal(organism);
    let ray_scans = scan_rays(
        (organism.q, organism.r),
        organism.facing,
        organism.id,
        world_width,
        occupancy,
        vision_distance,
    );

    for sensory_neuron in &mut organism.brain.sensory {
        sensory_neuron.neuron.activation = match sensory_neuron.receptor {
            SensoryReceptor::FoodRay { ray_offset } => ray_signal(&ray_scans, ray_offset).food,
            SensoryReceptor::ContactAhead => contact_ahead,
            SensoryReceptor::Energy => energy,
            SensoryReceptor::OrganismRay { ray_offset } => {
                ray_signal(&ray_scans, ray_offset).organism
            }
            SensoryReceptor::Health => health,
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
    facing: sim_types::FacingDirection,
    organism_id: OrganismId,
    world_width: i32,
    occupancy: &[Option<Occupant>],
    vision_distance: u32,
) -> RayScans {
    let context = RaycastContext {
        organism_id,
        world_width,
        occupancy,
        vision_distance,
    };
    std::array::from_fn(|idx| {
        scan_ray(
            position,
            rotate_by_steps(facing, SensoryReceptor::VISION_RAY_OFFSETS[idx]),
            context,
        )
    })
}

fn ray_offset_index(ray_offset: i8) -> Option<usize> {
    SensoryReceptor::VISION_RAY_OFFSETS
        .iter()
        .position(|offset| *offset == ray_offset)
}

fn contact_ahead_signal(
    position: (i32, i32),
    facing: sim_types::FacingDirection,
    world_width: i32,
    occupancy: &[Option<Occupant>],
) -> f32 {
    let ahead = hex_neighbor(position, facing, world_width);
    let idx = ahead.1 as usize * world_width as usize + ahead.0 as usize;
    f32::from(occupancy[idx].is_some())
}

fn energy_signal(organism: &OrganismState) -> f32 {
    let scale = organism.max_health.max(1.0);
    let energy = organism.energy.max(0.0);
    energy / (energy + scale)
}

fn health_signal(organism: &OrganismState) -> f32 {
    let max_health = organism.max_health.max(1.0);
    (organism.health / max_health).clamp(0.0, 1.0)
}

fn scan_ray(
    position: (i32, i32),
    ray_facing: sim_types::FacingDirection,
    context: RaycastContext<'_>,
) -> ScanResult {
    let max_dist = context.vision_distance.max(1);
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
                food: distance_signal,
                ..VisionSignal::default()
            },
            Some(Occupant::Organism(id)) if id != context.organism_id => VisionSignal {
                organism: distance_signal,
                ..VisionSignal::default()
            },
            Some(Occupant::Wall) => VisionSignal {
                ..VisionSignal::default()
            },
            Some(Occupant::Organism(_)) | None => continue,
        };
        return ScanResult {
            food_visible: signal.food > 0.0,
            signal,
        };
    }

    ScanResult::default()
}

#[inline(always)]
fn facing_delta(facing: sim_types::FacingDirection) -> (i32, i32) {
    use sim_types::FacingDirection::*;
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
    use super::{contact_ahead_signal, energy_signal, health_signal, scan_ray, RaycastContext};
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;
    use sim_types::{
        BrainState, FacingDirection, FoodId, Occupant, OrganismId, OrganismState, SpeciesId,
    };

    fn test_organism() -> OrganismState {
        let config = sim_config::default_world_config();
        let mut rng = ChaCha8Rng::seed_from_u64(7);
        let genome = crate::genome::generate_seed_genome(
            &config.seed_genome_config,
            config.predation_enabled,
            &mut rng,
        );
        let max_health = sim_types::offspring_transfer_energy(genome.lifecycle.gestation_ticks);
        OrganismState::new(
            OrganismId(1),
            SpeciesId(1),
            1,
            1,
            0,
            0,
            FacingDirection::East,
            max_health,
            max_health,
            max_health,
            0.0,
            0,
            0,
            0,
            sim_types::ActionType::Idle,
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
    fn contact_ahead_detects_occupants() {
        let world_width = 4;
        let mut occupancy = vec![None; (world_width * world_width) as usize];
        assert_eq!(
            contact_ahead_signal((1, 1), FacingDirection::East, world_width, &occupancy),
            0.0
        );
        occupancy[world_width as usize + 2] = Some(Occupant::Wall);
        assert_eq!(
            contact_ahead_signal((1, 1), FacingDirection::East, world_width, &occupancy),
            1.0
        );
    }

    #[test]
    fn health_and_energy_signals_are_bounded() {
        let mut organism = test_organism();
        organism.health = 25.0;
        organism.max_health = 100.0;
        organism.energy = 100.0;
        assert_eq!(health_signal(&organism), 0.25);
        assert_eq!(energy_signal(&organism), 0.5);
        organism.health = 150.0;
        organism.energy = -5.0;
        assert_eq!(health_signal(&organism), 1.0);
        assert_eq!(energy_signal(&organism), 0.0);
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
                vision_distance: 4,
            },
        );
        assert!(scan.food_visible);
        assert_eq!(scan.signal.food, 0.75);
        assert_eq!(scan.signal.organism, 0.0);
    }
}
