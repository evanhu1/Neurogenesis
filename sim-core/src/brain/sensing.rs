use super::*;

#[derive(Clone, Copy, Default)]
pub(crate) struct ScanResult {
    pub(crate) food_signal: f32,
    pub(crate) organism_signal: f32,
    pub(crate) wall_signal: f32,
    pub(crate) spike_signal: f32,
}

impl ScanResult {
    fn signal_for(self, look_target: EntityType) -> f32 {
        match look_target {
            EntityType::Food => self.food_signal,
            EntityType::Organism => self.organism_signal,
            EntityType::Wall => self.wall_signal,
            EntityType::Spikes => self.spike_signal,
        }
    }
}

type RayScans = [ScanResult; LOOK_RAY_COUNT];

pub(super) fn encode_sensory_inputs(
    organism: &mut OrganismState,
    world_width: i32,
    occupancy: &[Option<Occupant>],
    spike_map: &[bool],
    vision_distance: u32,
) -> RayScans {
    let ray_scans = scan_rays(
        (organism.q, organism.r),
        organism.facing,
        organism.id,
        world_width,
        occupancy,
        spike_map,
        vision_distance,
    );
    let contact_ahead = hex_neighbor((organism.q, organism.r), organism.facing, world_width);
    let contact_ahead_idx =
        contact_ahead.1 as usize * world_width as usize + contact_ahead.0 as usize;
    let contact_ahead_signal = occupancy[contact_ahead_idx].map_or(0.0, |_| 1.0);
    let energy_signal = energy_sensor_value(
        organism.energy,
        organism.genome.starting_energy.max(MIN_ENERGY_SENSOR_SCALE),
    );
    let damage_signal = (organism.damage_taken_last_turn
        / organism.genome.starting_energy.max(MIN_ENERGY_SENSOR_SCALE))
    .clamp(0.0, 1.0);

    for sensory in &mut organism.brain.sensory {
        sensory.neuron.activation = match &sensory.receptor {
            SensoryReceptor::LookRay {
                ray_offset,
                look_target,
            } => look_ray_signal(&ray_scans, *ray_offset, *look_target),
            SensoryReceptor::ContactAhead => contact_ahead_signal,
            SensoryReceptor::Damage => damage_signal,
            SensoryReceptor::Energy => energy_signal,
        };
    }

    ray_scans
}

pub(super) fn look_ray_signal(
    ray_scans: &RayScans,
    ray_offset: i8,
    look_target: EntityType,
) -> f32 {
    let Some(ray_idx) = ray_offset_index(ray_offset) else {
        return 0.0;
    };
    ray_scans[ray_idx].signal_for(look_target)
}

pub(crate) fn scan_rays(
    position: (i32, i32),
    facing: sim_types::FacingDirection,
    organism_id: OrganismId,
    world_width: i32,
    occupancy: &[Option<Occupant>],
    spike_map: &[bool],
    vision_distance: u32,
) -> RayScans {
    std::array::from_fn(|idx| {
        scan_ray(
            position,
            facing,
            SensoryReceptor::LOOK_RAY_OFFSETS[idx],
            organism_id,
            world_width,
            occupancy,
            spike_map,
            vision_distance,
        )
    })
}

fn energy_sensor_value(energy: f32, scale: f32) -> f32 {
    let safe_scale = scale.max(MIN_ENERGY_SENSOR_SCALE);
    let ratio = energy.max(0.0) / safe_scale;
    let curved = ratio.powf(ENERGY_SENSOR_CURVE_EXPONENT);
    curved / (1.0 + curved)
}

fn ray_offset_index(ray_offset: i8) -> Option<usize> {
    SensoryReceptor::LOOK_RAY_OFFSETS
        .iter()
        .position(|offset| *offset == ray_offset)
}

fn rotate_facing_by_offset(
    mut facing: sim_types::FacingDirection,
    ray_offset: i8,
) -> sim_types::FacingDirection {
    if ray_offset >= 0 {
        for _ in 0..u8::try_from(ray_offset).unwrap_or(0) {
            facing = rotate_right(facing);
        }
        return facing;
    }

    for _ in 0..ray_offset.unsigned_abs() {
        facing = rotate_left(facing);
    }
    facing
}

fn scan_ray(
    position: (i32, i32),
    facing: sim_types::FacingDirection,
    ray_offset: i8,
    organism_id: OrganismId,
    world_width: i32,
    occupancy: &[Option<Occupant>],
    spike_map: &[bool],
    vision_distance: u32,
) -> ScanResult {
    let ray_facing = rotate_facing_by_offset(facing, ray_offset);
    let max_dist = vision_distance.max(1);
    let mut current = position;
    for d in 1..=max_dist {
        current = hex_neighbor(current, ray_facing, world_width);
        let idx = current.1 as usize * world_width as usize + current.0 as usize;
        let signal = (max_dist - d + 1) as f32 / max_dist as f32;
        let mut hit = ScanResult::default();
        if spike_map[idx] {
            hit.spike_signal = signal;
        }
        match occupancy[idx] {
            Some(Occupant::Organism(id)) if id == organism_id => {}
            Some(Occupant::Food(_)) => {
                hit.food_signal = signal;
                return hit;
            }
            Some(Occupant::Organism(_)) => {
                hit.organism_signal = signal;
                return hit;
            }
            Some(Occupant::Wall) => {
                hit.wall_signal = signal;
                return hit;
            }
            None => {
                if hit.spike_signal > 0.0 {
                    return hit;
                }
            }
        }
    }
    ScanResult::default()
}
