use super::*;
use crate::grid::{hex_neighbor, rotate_by_steps};

#[derive(Debug, Clone, Copy, Default, PartialEq)]
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

    for distance in 1..=max_dist {
        current = hex_neighbor(current, ray_facing, context.world_width);
        let idx = current.1 as usize * context.world_width as usize + current.0 as usize;
        let distance_signal = (max_dist - distance + 1) as f32 * inv_max_dist;
        let mut hit = ScanResult::default();

        if context.spike_map[idx] {
            add_visual_signal(
                &mut hit.color,
                sim_types::terrain_visual(sim_types::TerrainType::Spikes),
                distance_signal,
            );
        }

        match context.occupancy[idx] {
            Some(Occupant::Organism(id)) if id == context.organism_id => {}
            Some(Occupant::Food(id)) => {
                hit.food_visible = true;
                add_visual_signal(
                    &mut hit.color,
                    food_visual_for_id(context.food_visuals, id),
                    distance_signal,
                );
                hit.color = hit.color.clamped();
                return hit;
            }
            Some(Occupant::Organism(id)) => {
                add_visual_signal(
                    &mut hit.color,
                    organism_visual_for_id(context.organism_colors, id),
                    distance_signal,
                );
                hit.color = hit.color.clamped();
                return hit;
            }
            Some(Occupant::Wall) => {
                add_visual_signal(
                    &mut hit.color,
                    sim_types::terrain_visual(sim_types::TerrainType::Mountain),
                    distance_signal,
                );
                hit.color = hit.color.clamped();
                return hit;
            }
            None => {
                if context.spike_map[idx] {
                    hit.color = hit.color.clamped();
                    return hit;
                }
            }
        }
    }

    ScanResult::default()
}

fn add_visual_signal(color: &mut ColorSignal, visual: VisualProperties, distance_signal: f32) {
    let contribution = distance_signal;
    color.red += visual.r * contribution;
    color.green += visual.g * contribution;
    color.blue += visual.b * contribution;
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
    use super::{
        contact_ahead_signal, energy_signal, health_signal, scan_ray, ColorSignal, RaycastContext,
    };
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;
    use sim_types::{
        food_visual, BrainState, FacingDirection, FoodId, Occupant, OrganismId, OrganismState,
        RgbColor, SpeciesId,
    };

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

    #[test]
    fn scan_ray_stops_at_first_visible_hit() {
        let world_width = 5;
        let mut occupancy = vec![None; (world_width * world_width) as usize];
        let spike_map = vec![false; occupancy.len()];
        occupancy[1 * world_width as usize + 2] = Some(Occupant::Food(FoodId(0)));
        occupancy[1 * world_width as usize + 3] = Some(Occupant::Organism(OrganismId(7)));
        let organism_colors = vec![
            RgbColor::default(),
            RgbColor::default(),
            RgbColor::default(),
            RgbColor::default(),
            RgbColor::default(),
            RgbColor::default(),
            RgbColor::default(),
            RgbColor {
                r: 1.0,
                g: 0.0,
                b: 0.0,
            },
        ];
        let food_visuals = vec![food_visual(sim_types::FoodKind::Plant)];

        let scan = scan_ray(
            (1, 1),
            FacingDirection::East,
            RaycastContext {
                organism_id: OrganismId(1),
                world_width,
                occupancy: &occupancy,
                spike_map: &spike_map,
                organism_colors: &organism_colors,
                food_visuals: &food_visuals,
                vision_distance: 4,
            },
        );

        assert!(scan.food_visible);
        assert_eq!(
            scan.color,
            ColorSignal {
                red: 0.0,
                green: 1.0,
                blue: 0.0,
            }
        );
    }

    #[test]
    fn scan_ray_spike_tile_blocks_farther_entities() {
        let world_width = 5;
        let mut occupancy = vec![None; (world_width * world_width) as usize];
        let mut spike_map = vec![false; occupancy.len()];
        spike_map[1 * world_width as usize + 2] = true;
        occupancy[1 * world_width as usize + 3] = Some(Occupant::Food(FoodId(0)));
        let food_visuals = vec![food_visual(sim_types::FoodKind::Plant)];

        let scan = scan_ray(
            (1, 1),
            FacingDirection::East,
            RaycastContext {
                organism_id: OrganismId(1),
                world_width,
                occupancy: &occupancy,
                spike_map: &spike_map,
                organism_colors: &[],
                food_visuals: &food_visuals,
                vision_distance: 4,
            },
        );

        assert!(!scan.food_visible);
        assert_eq!(
            scan.color,
            ColorSignal {
                red: 1.0,
                green: 0.0,
                blue: 0.0,
            }
        );
    }
}
