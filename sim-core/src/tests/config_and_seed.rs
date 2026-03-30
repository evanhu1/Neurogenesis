use super::support::{stable_test_config, test_genome};
use super::*;

#[test]
fn terrain_map_is_seed_deterministic_and_threshold_controlled() {
    let mut low_threshold_cfg = stable_test_config();
    low_threshold_cfg.world_width = 48;
    low_threshold_cfg.num_organisms = 1;
    low_threshold_cfg.terrain_noise_scale = 0.02;
    low_threshold_cfg.terrain_threshold = 0.60;
    let mut high_threshold_cfg = low_threshold_cfg.clone();
    high_threshold_cfg.terrain_threshold = 0.95;

    let sim_a = Simulation::new(low_threshold_cfg.clone(), 2026).expect("simulation should init");
    let sim_b = Simulation::new(low_threshold_cfg, 2026).expect("simulation should init");
    let sim_c = Simulation::new(high_threshold_cfg, 2026).expect("simulation should init");
    let sim_other_seed = Simulation::new(stable_test_config_with_terrain(0.60), 2027)
        .expect("simulation should init");

    assert_eq!(sim_a.terrain_map, sim_b.terrain_map);
    assert_ne!(sim_a.terrain_map, sim_other_seed.terrain_map);

    let low_blocked = sim_a.terrain_map.iter().filter(|blocked| **blocked).count();
    let high_blocked = sim_c.terrain_map.iter().filter(|blocked| **blocked).count();
    assert!(
        low_blocked > high_blocked,
        "lower threshold should produce more blocked terrain: low={low_blocked} high={high_blocked}",
    );
}

#[test]
fn fertility_map_is_seed_deterministic_with_cell_jitter() {
    let mut cfg = stable_test_config();
    cfg.world_width = 48;
    cfg.num_organisms = 1;
    cfg.terrain_noise_scale = 0.02;
    cfg.terrain_threshold = 1.0;

    let sim_a = Simulation::new(cfg.clone(), 2026).expect("simulation should init");
    let sim_b = Simulation::new(cfg, 2026).expect("simulation should init");
    let sim_other_seed = Simulation::new(stable_test_config_with_terrain(1.0), 2027)
        .expect("simulation should init");

    assert_eq!(sim_a.food_fertility, sim_b.food_fertility);
    assert_ne!(sim_a.food_fertility, sim_other_seed.food_fertility);

    let fertile_count = sim_a
        .food_fertility
        .iter()
        .filter(|fertile| **fertile)
        .count();
    assert!(
        fertile_count > 0 && fertile_count < sim_a.food_fertility.len(),
        "fertility map should contain both fertile and infertile cells: fertile={fertile_count} total={}",
        sim_a.food_fertility.len(),
    );
}

#[test]
fn spike_map_is_seed_deterministic_and_density_controlled() {
    let mut low_density_cfg = stable_test_config();
    low_density_cfg.world_width = 48;
    low_density_cfg.num_organisms = 1;
    low_density_cfg.terrain_threshold = 1.0;
    low_density_cfg.spike_density = 0.05;
    let mut high_density_cfg = low_density_cfg.clone();
    high_density_cfg.spike_density = 0.25;

    let sim_a = Simulation::new(low_density_cfg.clone(), 2026).expect("simulation should init");
    let sim_b = Simulation::new(low_density_cfg, 2026).expect("simulation should init");
    let sim_c = Simulation::new(high_density_cfg, 2026).expect("simulation should init");

    let mut other_seed_cfg = stable_test_config();
    other_seed_cfg.world_width = 48;
    other_seed_cfg.num_organisms = 1;
    other_seed_cfg.terrain_threshold = 1.0;
    other_seed_cfg.spike_density = 0.05;
    let sim_other_seed = Simulation::new(other_seed_cfg, 2027).expect("simulation should init");

    assert_eq!(sim_a.spike_map, sim_b.spike_map);
    assert_ne!(sim_a.spike_map, sim_other_seed.spike_map);

    let low_spikes = sim_a.spike_map.iter().filter(|blocked| **blocked).count();
    let high_spikes = sim_c.spike_map.iter().filter(|blocked| **blocked).count();
    assert!(
        high_spikes > low_spikes,
        "higher density should produce more scattered spikes: low={low_spikes} high={high_spikes}",
    );
}

#[test]
fn stochastic_action_sampling_is_deterministic_for_repeated_runs() {
    let mut cfg = stable_test_config();
    cfg.world_width = 30;
    cfg.num_organisms = 120;
    cfg.action_temperature = 1.2;

    let mut run_a = Simulation::new(cfg.clone(), 2026).expect("simulation should initialize");
    let mut run_b = Simulation::new(cfg, 2026).expect("simulation should initialize");

    for _ in 0..25 {
        let _ = run_a.tick();
        let _ = run_b.tick();
    }

    assert_eq!(run_a.snapshot(), run_b.snapshot());
}

#[test]
fn champion_pool_bootstrap_is_seed_deterministic() {
    let mut cfg = stable_test_config();
    cfg.world_width = 24;
    cfg.num_organisms = 32;

    let mut champion_a = test_genome();
    champion_a.num_neurons = 2;
    champion_a.vision_distance = 7;
    champion_a.starting_energy = 321.0;
    champion_a.inter_biases = vec![0.1, -0.2];
    champion_a.inter_log_time_constants = vec![0.0, 0.2];
    champion_a.inter_locations = vec![
        BrainLocation { x: 1.0, y: 1.0 },
        BrainLocation { x: 2.0, y: 2.0 },
    ];

    let mut champion_b = champion_a.clone();
    champion_b.num_neurons = 3;
    champion_b.vision_distance = 9;
    champion_b.starting_energy = 654.0;
    champion_b.inter_biases = vec![0.3, -0.4, 0.5];
    champion_b.inter_log_time_constants = vec![0.0, 0.1, 0.2];
    champion_b.inter_locations = vec![
        BrainLocation { x: 3.0, y: 3.0 },
        BrainLocation { x: 4.0, y: 4.0 },
        BrainLocation { x: 5.0, y: 5.0 },
    ];

    let champion_pool = vec![champion_a.clone(), champion_b.clone()];
    let run_a = Simulation::new_with_champion_pool(cfg.clone(), 2026, champion_pool.clone())
        .expect("simulation should initialize");
    let run_b = Simulation::new_with_champion_pool(cfg, 2026, champion_pool)
        .expect("simulation should initialize");

    assert_eq!(run_a.snapshot(), run_b.snapshot());
    assert!(run_a
        .organisms()
        .iter()
        .all(|organism| { organism.genome == champion_a || organism.genome == champion_b }));
}

#[test]
fn reset_preserves_champion_pool_bootstrap_behavior() {
    let mut cfg = stable_test_config();
    cfg.world_width = 18;
    cfg.num_organisms = 16;

    let mut champion = test_genome();
    champion.num_neurons = 4;
    champion.vision_distance = 11;
    champion.starting_energy = 777.0;
    champion.inter_biases = vec![0.0, 0.1, 0.2, 0.3];
    champion.inter_log_time_constants = vec![0.0, 0.1, 0.2, 0.3];
    champion.inter_locations = vec![
        BrainLocation { x: 1.0, y: 1.0 },
        BrainLocation { x: 2.0, y: 2.0 },
        BrainLocation { x: 3.0, y: 3.0 },
        BrainLocation { x: 4.0, y: 4.0 },
    ];

    let mut sim = Simulation::new_with_champion_pool(cfg, 99, vec![champion.clone()])
        .expect("simulation should initialize");
    let initial_snapshot = sim.snapshot();

    sim.advance_n(10);
    sim.reset(Some(99));

    assert_eq!(sim.snapshot(), initial_snapshot);
    assert!(sim
        .organisms()
        .iter()
        .all(|organism| organism.genome == champion));
}

fn stable_test_config_with_terrain(terrain_threshold: f32) -> WorldConfig {
    let mut cfg = stable_test_config();
    cfg.world_width = 48;
    cfg.num_organisms = 1;
    cfg.terrain_noise_scale = 0.02;
    cfg.terrain_threshold = terrain_threshold;
    cfg
}
