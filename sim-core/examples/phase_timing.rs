use sim_core::Simulation;
use sim_types::{SeedGenomeConfig, WorldConfig};
use std::time::Instant;

fn stable_perf_config() -> WorldConfig {
    WorldConfig {
        world_width: 100,
        num_organisms: 2_000,
        periodic_injection_interval_turns: 0,
        periodic_injection_count: 0,
        food_energy: 50.0,
        passive_metabolism_cost_per_unit: 0.005,
        move_action_energy_cost: 0.0,
        action_temperature: 0.5,
        intent_parallel_threads: 8,
        food_regrowth_interval: 10,
        food_regrowth_jitter: 2,
        food_fertility_threshold: 0.6,
        food_fertility_jitter_strength: 1.0,
        terrain_noise_scale: 0.02,
        terrain_threshold: 1.0,
        spike_density: 0.0,
        global_mutation_rate_modifier: 1.0,
        meta_mutation_enabled: true,
        runtime_plasticity_enabled: true,
        force_random_actions: false,
        seed_genome_config: SeedGenomeConfig {
            num_neurons: 20,
            num_synapses: 80,
            spatial_prior_sigma: 3.5,
            vision_distance: 10,
            max_health: 1_000_000_000.0,
            age_of_maturity: 0,
            gestation_ticks: 2,
            max_organism_age: u32::MAX,
            plasticity_start_age: 0,
            hebb_eta_gain: 0.0,
            juvenile_eta_scale: 0.5,
            eligibility_retention: 0.9,
            max_weight_delta_per_tick: 0.05,
            synapse_prune_threshold: 0.01,
            mutation_rate_age_of_maturity: 0.05,
            mutation_rate_gestation_ticks: 0.05,
            mutation_rate_max_organism_age: 0.05,
            mutation_rate_vision_distance: 0.04,
            mutation_rate_max_health: 0.05,
            mutation_rate_inter_bias: 0.2,
            mutation_rate_inter_update_rate: 0.12,
            mutation_rate_eligibility_retention: 0.05,
            mutation_rate_synapse_prune_threshold: 0.05,
            mutation_rate_neuron_location: 0.02,
            mutation_rate_synapse_weight_perturbation: 0.1,
            mutation_rate_add_synapse: 0.05,
            mutation_rate_remove_synapse: 0.05,
            mutation_rate_remove_neuron: 0.02,
            mutation_rate_add_neuron_split_edge: 0.05,
        },
    }
}

fn main() {
    let config = stable_perf_config();
    let mut sim = Simulation::new(config, 42).expect("init");
    sim.advance_n(200); // warmup

    eprintln!("Organisms after warmup: {}", sim.metrics().organisms);

    // Measure total
    let started = Instant::now();
    sim.advance_n(1000);
    let elapsed = started.elapsed();
    eprintln!(
        "1000 turns: {:.1}ms ({:.1} us/turn)",
        elapsed.as_secs_f64() * 1000.0,
        elapsed.as_micros() as f64 / 1000.0
    );
    eprintln!("Organisms after run: {}", sim.metrics().organisms);
}
