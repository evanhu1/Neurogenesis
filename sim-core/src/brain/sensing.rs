use super::*;
use crate::grid::{hex_neighbor, rotate_by_steps};

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub(crate) struct ColorSignal {
    pub(crate) red: f32,
    pub(crate) green: f32,
    pub(crate) blue: f32,
    pub(crate) shape: f32,
}

impl ColorSignal {
    fn signal_for(self, channel: VisionChannel) -> f32 {
        match channel {
            VisionChannel::Red => self.red,
            VisionChannel::Green => self.green,
            VisionChannel::Blue => self.blue,
            VisionChannel::Shape => self.shape,
        }
    }

    fn clamped(self) -> Self {
        Self {
            red: self.red.clamp(0.0, 1.0),
            green: self.green.clamp(0.0, 1.0),
            blue: self.blue.clamp(0.0, 1.0),
            shape: self.shape.clamp(0.0, 1.0),
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
    spike_visual_map: &'a [VisualProperties],
    visual_map: &'a [VisualProperties],
    vision_distance: u32,
}

#[allow(clippy::too_many_arguments)]
pub(super) fn encode_sensory_inputs(
    organism: &mut OrganismState,
    world_width: i32,
    occupancy: &[Option<Occupant>],
    spike_map: &[bool],
    spike_visual_map: &[VisualProperties],
    visual_map: &[VisualProperties],
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
    let energy_delta = energy_delta_signal(organism);
    // Re-stash at sensing time so the next EnergyDelta reading spans
    // sensing-to-sensing and can observe eat/attack/action-cost changes.
    organism.energy_at_last_sensing = organism.energy;
    let last_forward = f32::from(organism.last_action_taken == sim_types::ActionType::Forward);
    let last_eat = f32::from(organism.last_action_taken == sim_types::ActionType::Eat);
    let ray_scans = scan_rays(
        (organism.q, organism.r),
        organism.facing,
        organism.id,
        world_width,
        occupancy,
        spike_map,
        spike_visual_map,
        visual_map,
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
            SensoryReceptor::EnergyDelta => energy_delta,
            SensoryReceptor::LastActionForward => last_forward,
            SensoryReceptor::LastActionEat => last_eat,
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

#[allow(clippy::too_many_arguments)]
pub(crate) fn scan_rays(
    position: (i32, i32),
    facing: sim_types::FacingDirection,
    organism_id: OrganismId,
    world_width: i32,
    occupancy: &[Option<Occupant>],
    spike_map: &[bool],
    spike_visual_map: &[VisualProperties],
    visual_map: &[VisualProperties],
    vision_distance: u32,
) -> RayScans {
    let context = RaycastContext {
        organism_id,
        world_width,
        occupancy,
        spike_map,
        spike_visual_map,
        visual_map,
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

fn energy_delta_signal(organism: &OrganismState) -> f32 {
    // Signed [-1, 1] momentum signal via tanh on the sensing-to-sensing
    // energy delta scaled by max_health.
    let scale = organism.max_health.max(1.0);
    let delta = organism.energy - organism.energy_at_last_sensing;
    (delta / scale).tanh()
}

fn scan_ray(
    position: (i32, i32),
    ray_facing: sim_types::FacingDirection,
    context: RaycastContext<'_>,
) -> ScanResult {
    let max_dist = context.vision_distance.max(1);
    let inv_max_dist = 1.0 / max_dist as f32;
    let world_width = context.world_width;
    let world_width_usize = world_width as usize;
    let occupancy = context.occupancy;
    let spike_map = context.spike_map;
    let spike_visual_map = context.spike_visual_map;
    let visual_map = context.visual_map;
    let organism_id = context.organism_id;

    // Hex step deltas for the chosen facing — hoisted out of the loop.
    let (dq, dr) = facing_delta(ray_facing);
    let mut q = position.0;
    let mut r = position.1;
    let mut color = ColorSignal::default();
    let mut food_visible = false;
    let mut remaining_visibility = 1.0_f32;

    for distance in 1..=max_dist {
        q += dq;
        if q < 0 {
            q += world_width;
        } else if q >= world_width {
            q -= world_width;
        }
        r += dr;
        if r < 0 {
            r += world_width;
        } else if r >= world_width {
            r -= world_width;
        }
        let idx = r as usize * world_width_usize + q as usize;
        let distance_signal = (max_dist - distance + 1) as f32 * inv_max_dist;

        if spike_map[idx] {
            accumulate_visual(
                &mut color,
                &mut remaining_visibility,
                spike_visual_map[idx],
                distance_signal,
            );
        }

        match occupancy[idx] {
            Some(Occupant::Organism(id)) if id == organism_id => {}
            Some(Occupant::Food(_)) => {
                food_visible |= remaining_visibility > 0.0;
                accumulate_visual(
                    &mut color,
                    &mut remaining_visibility,
                    visual_map[idx],
                    distance_signal,
                );
            }
            Some(Occupant::Organism(_) | Occupant::Wall) => {
                accumulate_visual(
                    &mut color,
                    &mut remaining_visibility,
                    visual_map[idx],
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

/// Per-step (dq, dr) offset for a given facing direction. Inlining this lets
/// the optimizer hoist the offset out of the inner ray-walk loop instead of
/// matching on the facing every step.
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

fn accumulate_visual(
    color: &mut ColorSignal,
    remaining_visibility: &mut f32,
    visual: VisualProperties,
    distance_signal: f32,
) {
    if *remaining_visibility <= 0.0 {
        return;
    }

    let opacity = visual.opacity.clamp(0.0, 1.0);
    let contribution = opacity * distance_signal * *remaining_visibility;
    color.red += visual.r * contribution;
    color.green += visual.g * contribution;
    color.blue += visual.b * contribution;
    color.shape += visual.shape * contribution;
    *remaining_visibility *= 1.0 - opacity;
}

#[cfg(test)]
mod tests {
    use super::{
        contact_ahead_signal, energy_signal, health_signal, scan_ray, ColorSignal, RaycastContext,
    };
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;
    use sim_types::{
        BrainState, FacingDirection, FoodId, Occupant, OrganismId, OrganismState, SpeciesId,
        VisualProperties,
    };

    fn test_organism() -> OrganismState {
        let config = sim_config::default_world_config();
        let mut rng = ChaCha8Rng::seed_from_u64(7);
        let genome = crate::genome::generate_seed_genome(&config.seed_genome_config, &mut rng);
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
                sensory_mean_activation: Vec::new(),
                inter_mean_activation: Vec::new(),
                action_mean_activation: Vec::new(),
                means_initialized: false,
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

        occupancy[world_width as usize + 2] = Some(sim_types::Occupant::Wall);
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

        occupancy[world_width as usize + 2] = None;
        spike_map[world_width as usize + 2] = true;
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
    fn scan_ray_accumulates_translucent_hits_along_ray() {
        let world_width = 5;
        let mut occupancy = vec![None; (world_width * world_width) as usize];
        let spike_map = vec![false; occupancy.len()];
        let spike_visual_map = vec![VisualProperties::default(); occupancy.len()];
        let mut visual_map = vec![VisualProperties::default(); occupancy.len()];

        let food_0_visual = VisualProperties {
            r: 1.0,
            g: 0.0,
            b: 0.0,
            opacity: 0.5,
            shape: 0.0,
        };
        let food_1_visual = VisualProperties {
            r: 0.0,
            g: 0.0,
            b: 1.0,
            opacity: 0.25,
            shape: 0.0,
        };

        occupancy[world_width as usize + 2] = Some(Occupant::Food(FoodId(0)));
        visual_map[world_width as usize + 2] = food_0_visual;

        occupancy[world_width as usize + 3] = Some(Occupant::Food(FoodId(1)));
        visual_map[world_width as usize + 3] = food_1_visual;

        let scan = scan_ray(
            (1, 1),
            FacingDirection::East,
            RaycastContext {
                organism_id: OrganismId(1),
                world_width,
                occupancy: &occupancy,
                spike_map: &spike_map,
                spike_visual_map: &spike_visual_map,
                visual_map: &visual_map,
                vision_distance: 4,
            },
        );

        assert!(scan.food_visible);
        // Food(0) at distance 1: signal=1.0, vis=1.0, opacity=0.5 → red += 0.5, vis *= 0.5 → 0.5
        // Food(1) at distance 2: signal=0.75, vis=0.5, opacity=0.25 → blue += 0.09375, vis *= 0.75 → 0.375
        assert_eq!(
            scan.color,
            ColorSignal {
                red: 0.5,
                green: 0.0,
                blue: 0.09375,
                shape: 0.0,
            }
        );
    }
}
