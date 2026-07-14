use super::support::{stable_test_config, test_genome};
use super::*;

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
fn champion_pool_init_reset_and_replay_are_deterministic() {
    let mut cfg = stable_test_config();
    cfg.world_width = 24;
    cfg.num_organisms = 32;

    let mut champion_a = test_genome();
    champion_a.lifecycle.plasticity_maturity_ticks = 7;
    champion_a.brain.hidden_nodes = vec![
        HiddenNodeGene {
            id: seed_hidden_gene_node_id(0),
            bias: 0.1,
            log_time_constant: 0.0,
        },
        HiddenNodeGene {
            id: seed_hidden_gene_node_id(1),
            bias: -0.2,
            log_time_constant: 0.2,
        },
    ];

    let mut champion_b = champion_a.clone();
    champion_b.lifecycle.plasticity_maturity_ticks = 9;
    champion_b.brain.hidden_nodes = [(0.3, 0.0), (-0.4, 0.1), (0.5, 0.2)]
        .into_iter()
        .enumerate()
        .map(|(index, (bias, log_time_constant))| HiddenNodeGene {
            id: seed_hidden_gene_node_id(index as u32),
            bias,
            log_time_constant,
        })
        .collect();

    let champion_pool = vec![champion_a.clone(), champion_b.clone()];
    let mut run = Simulation::new_with_champion_pool(cfg.clone(), 2026, champion_pool.clone())
        .expect("simulation should initialize");
    let mut replay = Simulation::new_with_champion_pool(cfg, 2026, champion_pool)
        .expect("simulation should initialize");
    let initial_snapshot = run.snapshot();

    assert_eq!(initial_snapshot, replay.snapshot());
    assert!(run
        .organisms()
        .iter()
        .all(|organism| { organism.genome == champion_a || organism.genome == champion_b }));

    run.advance_n(12);
    replay.advance_n(12);
    let replay_snapshot = run.snapshot();
    assert_eq!(replay_snapshot, replay.snapshot());

    run.reset(Some(2026));
    assert_eq!(run.snapshot(), initial_snapshot);

    run.advance_n(12);
    assert_eq!(run.snapshot(), replay_snapshot);
}

#[test]
fn reset_preserves_champion_pool_bootstrap_behavior() {
    let mut cfg = stable_test_config();
    cfg.world_width = 18;
    cfg.num_organisms = 16;

    let mut champion = test_genome();
    champion.lifecycle.plasticity_maturity_ticks = 10;
    champion.brain.hidden_nodes = [0.0, 0.1, 0.2, 0.3]
        .into_iter()
        .enumerate()
        .map(|(index, value)| HiddenNodeGene {
            id: seed_hidden_gene_node_id(index as u32),
            bias: value,
            log_time_constant: value,
        })
        .collect();

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

#[test]
fn higher_food_tile_fraction_increases_initial_plant_supply() {
    let mut dense_cfg = stable_test_config();
    dense_cfg.world_width = 48;
    dense_cfg.num_organisms = 1;
    dense_cfg.food_tile_fraction = 0.6;

    let mut sparse_cfg = dense_cfg.clone();
    sparse_cfg.food_tile_fraction = 0.1;

    let dense = Simulation::new(dense_cfg, 2026).expect("simulation should initialize");
    let sparse = Simulation::new(sparse_cfg, 2026).expect("simulation should initialize");

    let dense_food = dense.snapshot().foods.len();
    let sparse_food = sparse.snapshot().foods.len();

    assert!(
        sparse_food < dense_food,
        "lowering the food tile fraction should reduce initial plant supply: dense={dense_food} sparse={sparse_food}"
    );
}
