use super::*;
use sim_config::terrain_generation_policy;

impl Simulation {
    pub(crate) fn initialize_terrain(&mut self) {
        let policy = terrain_generation_policy();
        let width = self.config.world_width;
        let terrain_seed = self.seed ^ policy.terrain_seed_mix;
        self.terrain_map = if (self.config.terrain_threshold as f64 - policy.default_threshold)
            .abs()
            > f64::EPSILON
        {
            build_noise_mask_with_threshold(
                width,
                width,
                self.config.terrain_noise_scale as f64,
                terrain_seed,
                self.config.terrain_threshold as f64,
            )
        } else {
            build_terrain_map(
                width,
                width,
                self.config.terrain_noise_scale as f64,
                terrain_seed,
            )
        };
        let spike_seed = self.seed ^ policy.spike_seed_mix;
        self.spike_map = build_random_mask_with_density(
            width,
            width,
            spike_seed,
            self.config.spike_density as f64,
        );
        for (idx, blocked) in self.terrain_map.iter().copied().enumerate() {
            if blocked {
                self.occupancy[idx] = Some(Occupant::Wall);
                self.spike_map[idx] = false;
            }
        }
    }
}

pub(crate) fn build_terrain_map(width: u32, height: u32, scale: f64, seed: u64) -> Vec<bool> {
    build_noise_mask_with_threshold(
        width,
        height,
        scale,
        seed,
        terrain_generation_policy().default_threshold,
    )
}

fn build_noise_mask_with_threshold(
    width: u32,
    height: u32,
    scale: f64,
    seed: u64,
    threshold: f64,
) -> Vec<bool> {
    let width = width as usize;
    let height = height as usize;
    let mut blocked = Vec::with_capacity(width * height);
    for r in 0..height {
        for q in 0..width {
            let x = q as f64 * scale;
            let y = r as f64 * scale;
            let value = fractal_perlin_2d(x, y, seed);
            let normalized = ((value + 1.0) * 0.5).clamp(0.0, 1.0);
            blocked.push(normalized > threshold);
        }
    }
    blocked
}

fn build_random_mask_with_density(width: u32, height: u32, seed: u64, density: f64) -> Vec<bool> {
    let width = width as usize;
    let height = height as usize;
    let density = density.clamp(0.0, 1.0);
    if density <= 0.0 {
        return vec![false; width * height];
    }
    if density >= 1.0 {
        return vec![true; width * height];
    }

    let mut blocked = Vec::with_capacity(width * height);
    for r in 0..height {
        for q in 0..width {
            let sample = hash_to_unit_interval(hash_2d(q as i64, r as i64, seed));
            blocked.push(sample < density);
        }
    }
    blocked
}

pub(super) fn fractal_perlin_2d(x: f64, y: f64, seed: u64) -> f64 {
    const OCTAVES: usize = 1;
    let mut amplitude = 1.0_f64;
    let mut frequency = 1.0_f64;
    let mut total = 0.0_f64;
    let mut weight = 0.0_f64;

    for octave in 0..OCTAVES {
        let octave_seed =
            seed.wrapping_add(0x9E37_79B9_7F4A_7C15_u64.wrapping_mul(octave as u64 + 1));
        total += amplitude * perlin_2d(x * frequency, y * frequency, octave_seed);
        weight += amplitude;
        amplitude *= 0.5;
        frequency *= 2.0;
    }

    if weight == 0.0 {
        0.0
    } else {
        total / weight
    }
}

fn perlin_2d(x: f64, y: f64, seed: u64) -> f64 {
    let x0 = x.floor() as i64;
    let y0 = y.floor() as i64;
    let x1 = x0 + 1;
    let y1 = y0 + 1;

    let dx = x - x0 as f64;
    let dy = y - y0 as f64;

    let n00 = grad(hash_2d(x0, y0, seed), dx, dy);
    let n10 = grad(hash_2d(x1, y0, seed), dx - 1.0, dy);
    let n01 = grad(hash_2d(x0, y1, seed), dx, dy - 1.0);
    let n11 = grad(hash_2d(x1, y1, seed), dx - 1.0, dy - 1.0);

    let u = fade(dx);
    let v = fade(dy);
    let nx0 = lerp(n00, n10, u);
    let nx1 = lerp(n01, n11, u);
    lerp(nx0, nx1, v)
}

pub(super) fn hash_2d(x: i64, y: i64, seed: u64) -> u64 {
    let mut z = seed
        ^ (x as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
        ^ (y as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z ^= z >> 30;
    z = z.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z ^= z >> 27;
    z = z.wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    z
}

fn hash_to_unit_interval(hash: u64) -> f64 {
    const HASH_TO_UNIT_F64_SCALE: f64 = 1.0 / ((1_u64 << 53) as f64);
    ((hash >> 11) as f64) * HASH_TO_UNIT_F64_SCALE
}

fn grad(hash: u64, x: f64, y: f64) -> f64 {
    match (hash & 7) as u8 {
        0 => x + y,
        1 => x - y,
        2 => -x + y,
        3 => -x - y,
        4 => x,
        5 => -x,
        6 => y,
        _ => -y,
    }
}

fn fade(t: f64) -> f64 {
    t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
}

fn lerp(a: f64, b: f64, t: f64) -> f64 {
    a + t * (b - a)
}
