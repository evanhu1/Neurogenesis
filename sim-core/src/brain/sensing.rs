use super::*;
use crate::grid::{hex_neighbor, rotate_by_steps};

#[derive(Clone, Copy, Default)]
pub(crate) struct ColorSignal {
    pub(crate) red: f32,
    pub(crate) green: f32,
    pub(crate) blue: f32,
}

impl ColorSignal {
    fn signal_for(self, channel: VisionChannel) -> f32 {
        match channel {
            VisionChannel::Red => self.red,
            VisionChannel::Green => self.green,
            VisionChannel::Blue => self.blue,
        }
    }

    fn clamped(self) -> Self {
        Self {
            red: self.red.clamp(0.0, 1.0),
            green: self.green.clamp(0.0, 1.0),
            blue: self.blue.clamp(0.0, 1.0),
        }
    }
}

#[cfg_attr(not(feature = "instrumentation"), allow(dead_code))]
#[derive(Clone, Copy, Default)]
pub(crate) struct ScanResult {
    pub(crate) color: ColorSignal,
    pub(crate) food_visible: bool,
}

type RayScans = [ScanResult; VISION_RAY_COUNT];

#[derive(Clone, Copy)]
struct RaycastContext<'a> {
    organism_id: OrganismId,
    world_width: i32,
    occupancy: &'a [Option<Occupant>],
    spike_map: &'a [bool],
    organism_colors: &'a [RgbColor],
    food_visuals: &'a [VisualProperties],
    vision_distance: u32,
}

pub(super) fn encode_sensory_inputs(
    organism: &mut OrganismState,
    world_width: i32,
    occupancy: &[Option<Occupant>],
    spike_map: &[bool],
    organism_colors: &[RgbColor],
    food_visuals: &[VisualProperties],
    vision_distance: u32,
) -> RayScans {
    let contact_ahead = contact_ahead_signal(
        (organism.q, organism.r),
        organism.facing,
        world_width,
        occupancy,
        spike_map,
    );
    let energy = energy_signal(organism);
    let health = health_signal(organism);
    let ray_scans = scan_rays(
        (organism.q, organism.r),
        organism.facing,
        organism.id,
        world_width,
        occupancy,
        spike_map,
        organism_colors,
        food_visuals,
        vision_distance,
    );

    for sensory_neuron in &mut organism.brain.sensory {
        sensory_neuron.neuron.activation = match sensory_neuron.receptor {
            SensoryReceptor::VisionRay {
                ray_offset,
                channel,
            } => vision_ray_signal(&ray_scans, ray_offset, channel),
            SensoryReceptor::ContactAhead => contact_ahead,
            SensoryReceptor::Energy => energy,
            SensoryReceptor::Health => health,
        };
    }

    ray_scans
}

pub(super) fn vision_ray_signal(
    ray_scans: &RayScans,
    ray_offset: i8,
    channel: VisionChannel,
) -> f32 {
    let Some(ray_idx) = ray_offset_index(ray_offset) else {
        return 0.0;
    };
    ray_scans[ray_idx].color.signal_for(channel)
}

pub(crate) fn scan_rays(
    position: (i32, i32),
    facing: sim_types::FacingDirection,
    organism_id: OrganismId,
    world_width: i32,
    occupancy: &[Option<Occupant>],
    spike_map: &[bool],
    organism_colors: &[RgbColor],
    food_visuals: &[VisualProperties],
    vision_distance: u32,
) -> RayScans {
    let context = RaycastContext {
        organism_id,
        world_width,
        occupancy,
        spike_map,
        organism_colors,
        food_visuals,
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
    spike_map: &[bool],
) -> f32 {
    let ahead = hex_neighbor(position, facing, world_width);
    let idx = ahead.1 as usize * world_width as usize + ahead.0 as usize;
    f32::from(spike_map[idx] || occupancy[idx].is_some())
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
    let mut current = position;
    let inv_max_dist = 1.0 / max_dist as f32;
    let mut color = ColorSignal::default();
    let mut food_visible = false;
    let mut remaining_visibility = 1.0_f32;

    for distance in 1..=max_dist {
        current = hex_neighbor(current, ray_facing, context.world_width);
        let idx = current.1 as usize * context.world_width as usize + current.0 as usize;
        let distance_signal = (max_dist - distance + 1) as f32 * inv_max_dist;

        if context.spike_map[idx] {
            accumulate_visual(
                &mut color,
                &mut remaining_visibility,
                sim_types::terrain_visual(sim_types::TerrainType::Spikes),
                distance_signal,
            );
        }

        match context.occupancy[idx] {
            Some(Occupant::Organism(id)) if id == context.organism_id => {}
            Some(Occupant::Food(id)) => {
                food_visible |= remaining_visibility > 0.0;
                accumulate_visual(
                    &mut color,
                    &mut remaining_visibility,
                    food_visual_for_id(context.food_visuals, id),
                    distance_signal,
                );
            }
            Some(Occupant::Organism(id)) => {
                accumulate_visual(
                    &mut color,
                    &mut remaining_visibility,
                    organism_visual_for_id(context.organism_colors, id),
                    distance_signal,
                );
            }
            Some(Occupant::Wall) => {
                accumulate_visual(
                    &mut color,
                    &mut remaining_visibility,
                    sim_types::terrain_visual(sim_types::TerrainType::Mountain),
                    distance_signal,
                );
            }
            None => {}
        }

        if remaining_visibility <= f32::EPSILON {
            break;
        }
    }

    ScanResult {
        color: color.clamped(),
        food_visible,
    }
}

fn accumulate_visual(
    color: &mut ColorSignal,
    remaining_visibility: &mut f32,
    visual: VisualProperties,
    distance_signal: f32,
) {
    if *remaining_visibility <= 0.0 {
        return;
    }

    let contribution = distance_signal * *remaining_visibility;
    color.red += visual.r * contribution;
    color.green += visual.g * contribution;
    color.blue += visual.b * contribution;
    *remaining_visibility *= 1.0 - visual.opacity.clamp(0.0, 1.0);
}

fn organism_visual_for_id(colors: &[RgbColor], id: OrganismId) -> VisualProperties {
    let color = colors.get(id.0 as usize).copied().unwrap_or_default();
    sim_types::organism_visual(color)
}

fn food_visual_for_id(visuals: &[VisualProperties], id: sim_types::FoodId) -> VisualProperties {
    visuals.get(id.0 as usize).copied().unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::{contact_ahead_signal, energy_signal, health_signal};
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;
    use sim_types::{BrainState, FacingDirection, OrganismId, OrganismState, SpeciesId};

    fn test_organism() -> OrganismState {
        let config = sim_config::default_world_config();
        let mut rng = ChaCha8Rng::seed_from_u64(7);
        let genome = crate::genome::generate_seed_genome(&config.seed_genome_config, &mut rng);
        let max_health = genome.max_health.max(1.0);
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
            max_health,
            0.0,
            0.0,
            false,
            0,
            0,
            0,
            0,
            sim_types::ActionType::Idle,
            BrainState {
                sensory: Vec::new(),
                inter: Vec::new(),
                action: Vec::new(),
                synapse_count: 0,
            },
            genome,
        )
    }

    #[test]
    fn contact_ahead_detects_spikes_and_occupants() {
        let world_width = 4;
        let mut occupancy = vec![None; (world_width * world_width) as usize];
        let mut spike_map = vec![false; occupancy.len()];

        assert_eq!(
            contact_ahead_signal(
                (1, 1),
                FacingDirection::East,
                world_width,
                &occupancy,
                &spike_map,
            ),
            0.0
        );

        occupancy[1 * world_width as usize + 2] = Some(sim_types::Occupant::Wall);
        assert_eq!(
            contact_ahead_signal(
                (1, 1),
                FacingDirection::East,
                world_width,
                &occupancy,
                &spike_map,
            ),
            1.0
        );

        occupancy[1 * world_width as usize + 2] = None;
        spike_map[1 * world_width as usize + 2] = true;
        assert_eq!(
            contact_ahead_signal(
                (1, 1),
                FacingDirection::East,
                world_width,
                &occupancy,
                &spike_map,
            ),
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
}
