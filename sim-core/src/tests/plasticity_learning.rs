use super::support::{configure_sim, test_genome};
use super::*;
use crate::brain::{action_index, express_genome, BrainScratch, ACTION_ID_BASE};
use crate::grid::world_capacity;
use crate::plasticity::{apply_runtime_weight_updates, compute_pending_coactivations};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

fn repo_default_world_config() -> WorldConfig {
    sim_types::world_config_from_toml_str(include_str!("../../../sim-config/config.toml"))
        .expect("repo default config should parse")
}

/// Integration test: a single organism with one weak food→consume synapse faces
/// a food source that regrows every tick.  We run two identical simulations —
/// one with runtime plasticity enabled, one without — and verify that the
/// plastic organism's synapse weight strengthens through the dopamine-gated
/// Hebbian update, while the static organism's weight stays unchanged.
#[test]
#[ignore = "tuning-sensitive smoke test; use sim-validation for parameter assessment"]
fn lifetime_plasticity_strengthens_food_consume_synapse() {
    let max_age: u32 = 500;
    let initial_weight: f32 = 0.2;
    let initial_energy: f32 = 5_000.0;
    let seed = 42_u64;

    // Sensory neuron 0 = LookRay { offset: 0, target: Food } (food directly ahead).
    let food_ahead_sensory_id = 0_u32;
    let consume_action_id = NeuronId(ACTION_ID_BASE + action_index(ActionType::Consume) as u32);

    // Genome: zero inter neurons, one sensory→action synapse.
    let mut genome = test_genome();
    genome.num_neurons = 0;
    genome.num_synapses = 1;
    genome.inter_biases.clear();
    genome.inter_log_time_constants.clear();
    genome.inter_locations.clear();
    genome.hebb_eta_gain = 0.05;
    genome.eligibility_retention = 0.9;
    genome.synapse_prune_threshold = 0.0;
    genome.age_of_maturity = 0;
    genome.starting_energy = 500.0;
    genome.vision_distance = 5;
    genome.edges = vec![SynapseEdge {
        pre_neuron_id: NeuronId(food_ahead_sensory_id),
        post_neuron_id: consume_action_id,
        weight: initial_weight,
        eligibility: 0.0,
        pending_coactivation: 0.0,
    }];

    // Shared config: tiny world, fast food regrowth, no injections.
    let base_cfg = {
        let mut cfg = super::support::stable_test_config();
        cfg.world_width = 6;
        cfg.num_organisms = 1;
        cfg.food_energy = 100.0;
        cfg.food_regrowth_interval = 1;
        cfg.food_regrowth_jitter = 0;
        cfg.action_temperature = 0.08;
        cfg.max_organism_age = max_age;
        cfg.seed_genome_config.starting_energy = 10_000.0;
        cfg
    };

    // Place one organism at (2,2) facing East, food at (3,2).
    let organism_q = 2;
    let organism_r = 2;
    let food_q = 3;
    let food_r = 2;

    let setup = |plasticity_enabled: bool| -> Simulation {
        let mut cfg = base_cfg.clone();
        cfg.runtime_plasticity_enabled = plasticity_enabled;

        let mut sim = Simulation::new(cfg, seed).expect("sim should init");

        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let brain = express_genome(&genome, &mut rng);
        let organism = OrganismState {
            id: sim_types::OrganismId(0),
            species_id: sim_types::SpeciesId(0),
            q: organism_q,
            r: organism_r,
            generation: 0,
            age_turns: 0,
            facing: FacingDirection::East,
            energy: initial_energy,
            energy_prev: initial_energy,
            dopamine: 0.0,
            consumptions_count: 0,
            reproductions_count: 0,
            last_action_taken: ActionType::Idle,
            brain,
            genome: genome.clone(),
        };

        configure_sim(&mut sim, vec![organism]);

        // Manually place food ahead of the organism.
        let width = sim.config.world_width as usize;
        let food_idx = food_r as usize * width + food_q as usize;
        sim.foods.push(sim_types::FoodState {
            id: sim_types::FoodId(0),
            q: food_q,
            r: food_r,
            energy: sim.config.food_energy,
        });
        sim.occupancy[food_idx] = Some(Occupant::Food(sim_types::FoodId(0)));
        sim.next_food_id = 1;

        // Initialise food ecology so the food cell is fertile (enables regrowth).
        let capacity = world_capacity(sim.config.world_width);
        sim.food_fertility = vec![false; capacity];
        sim.food_fertility[food_idx] = true;
        sim.food_regrowth_due_turn = vec![u64::MAX; capacity];

        sim
    };

    let mut sim_plastic = setup(true);
    let mut sim_static = setup(false);

    // Run for the organism's full lifetime.
    sim_plastic.advance_n(max_age as u32);
    sim_static.advance_n(max_age as u32);

    // Both organisms must survive the full run (they die at the START of the
    // tick after age_turns reaches max_organism_age, so after advance_n(max_age)
    // they are still alive with age_turns == max_age).
    assert_eq!(
        sim_plastic.organisms().len(),
        1,
        "plastic organism should survive its full lifetime ({max_age} ticks)"
    );
    assert_eq!(
        sim_static.organisms().len(),
        1,
        "static organism should survive its full lifetime ({max_age} ticks)"
    );

    // Extract the food→consume synapse weight from each organism's brain.
    let synapse_weight = |sim: &Simulation| -> f32 {
        sim.organisms()[0].brain.sensory[food_ahead_sensory_id as usize]
            .synapses
            .iter()
            .find(|s| s.post_neuron_id == consume_action_id)
            .expect("food→consume synapse must exist")
            .weight
    };

    let plastic_weight = synapse_weight(&sim_plastic);
    let static_weight = synapse_weight(&sim_static);
    // Without plasticity the weight must be unchanged.
    assert_eq!(
        static_weight, initial_weight,
        "without plasticity, synapse weight should remain at initial value"
    );

    assert!(
        plastic_weight > initial_weight * 1.5,
        "with plasticity, food→consume synapse should strengthen substantially over a full \
         lifetime under favorable conditions (initial={initial_weight}, final={plastic_weight}, expected >{:.2})",
        initial_weight * 1.5
    );

    // Sanity: both organisms should have consumed food.
    assert!(
        sim_plastic.organisms()[0].consumptions_count > 0,
        "plastic organism should have consumed food"
    );
    assert!(
        sim_static.organisms()[0].consumptions_count > 0,
        "static organism should have consumed food"
    );
}

#[test]
#[ignore = "tuning-sensitive smoke test; use sim-validation for parameter assessment"]
fn repo_default_plasticity_params_still_produce_learning_signal() {
    let default_cfg = repo_default_world_config();
    let default_seed = &default_cfg.seed_genome_config;
    let max_age = default_cfg.max_organism_age;
    let initial_weight = 0.3_f32;
    let initial_energy = 5_000.0_f32;
    let seed = 43_u64;

    let food_ahead_sensory_id = 0_u32;
    let consume_action_id = NeuronId(ACTION_ID_BASE + action_index(ActionType::Consume) as u32);

    let mut genome = test_genome();
    genome.num_neurons = 0;
    genome.num_synapses = 1;
    genome.inter_biases.clear();
    genome.inter_log_time_constants.clear();
    genome.inter_locations.clear();
    genome.hebb_eta_gain = default_seed.hebb_eta_gain;
    genome.eligibility_retention = default_seed.eligibility_retention;
    genome.synapse_prune_threshold = default_seed.synapse_prune_threshold;
    genome.age_of_maturity = default_seed.age_of_maturity;
    genome.starting_energy = default_seed.starting_energy;
    genome.vision_distance = default_seed.vision_distance;
    genome.edges = vec![SynapseEdge {
        pre_neuron_id: NeuronId(food_ahead_sensory_id),
        post_neuron_id: consume_action_id,
        weight: initial_weight,
        eligibility: 0.0,
        pending_coactivation: 0.0,
    }];

    let base_cfg = {
        let mut cfg = default_cfg.clone();
        cfg.world_width = 6;
        cfg.num_organisms = 1;
        cfg.food_regrowth_interval = 1;
        cfg.food_regrowth_jitter = 0;
        cfg.periodic_injection_interval_turns = 0;
        cfg.periodic_injection_count = 0;
        cfg.terrain_threshold = 1.0;
        cfg.max_organism_age = max_age;
        cfg.seed_genome_config.starting_energy = 10_000.0;
        cfg
    };

    let organism_q = 2;
    let organism_r = 2;
    let food_q = 3;
    let food_r = 2;

    let setup = |plasticity_enabled: bool| -> Simulation {
        let mut cfg = base_cfg.clone();
        cfg.runtime_plasticity_enabled = plasticity_enabled;

        let mut sim = Simulation::new(cfg, seed).expect("sim should init");

        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let brain = express_genome(&genome, &mut rng);
        let organism = OrganismState {
            id: sim_types::OrganismId(0),
            species_id: sim_types::SpeciesId(0),
            q: organism_q,
            r: organism_r,
            generation: 0,
            age_turns: 0,
            facing: FacingDirection::East,
            energy: initial_energy,
            energy_prev: initial_energy,
            dopamine: 0.0,
            consumptions_count: 0,
            reproductions_count: 0,
            last_action_taken: ActionType::Idle,
            brain,
            genome: genome.clone(),
        };

        configure_sim(&mut sim, vec![organism]);

        let width = sim.config.world_width as usize;
        let food_idx = food_r as usize * width + food_q as usize;
        sim.foods.push(sim_types::FoodState {
            id: sim_types::FoodId(0),
            q: food_q,
            r: food_r,
            energy: sim.config.food_energy,
        });
        sim.occupancy[food_idx] = Some(Occupant::Food(sim_types::FoodId(0)));
        sim.next_food_id = 1;

        let capacity = world_capacity(sim.config.world_width);
        sim.food_fertility = vec![false; capacity];
        sim.food_fertility[food_idx] = true;
        sim.food_regrowth_due_turn = vec![u64::MAX; capacity];

        sim
    };

    let mut sim_plastic = setup(true);
    let mut sim_static = setup(false);

    sim_plastic.advance_n(max_age as u32);
    sim_static.advance_n(max_age as u32);

    let synapse_weight = |sim: &Simulation| -> f32 {
        sim.organisms()[0].brain.sensory[food_ahead_sensory_id as usize]
            .synapses
            .iter()
            .find(|s| s.post_neuron_id == consume_action_id)
            .expect("food->consume synapse must exist")
            .weight
    };

    let plastic_weight = synapse_weight(&sim_plastic);
    let static_weight = synapse_weight(&sim_static);
    assert_eq!(static_weight, initial_weight);
    assert!(
        plastic_weight > static_weight,
        "repo default plasticity params should still produce a positive learning signal \
         (plastic={plastic_weight}, static={static_weight})"
    );
    assert!(
        plastic_weight > initial_weight * 1.1,
        "repo default plasticity params should produce a modest but non-trivial learning signal \
         over one lifetime (initial={initial_weight}, final={plastic_weight}, expected >{:.2})",
        initial_weight * 1.1
    );
    assert!(
        sim_plastic.organisms()[0].consumptions_count > 0,
        "plastic organism should have consumed food"
    );
    assert!(
        sim_static.organisms()[0].consumptions_count > 0,
        "static organism should have consumed food"
    );
}

#[test]
#[ignore = "tuning-sensitive smoke test; use sim-validation for parameter assessment"]
fn repo_default_plasticity_learns_to_prefer_rewarded_consume_over_forward() {
    let default_cfg = repo_default_world_config();
    let default_seed = &default_cfg.seed_genome_config;
    let max_age = default_cfg.max_organism_age;
    let initial_energy = 5_000.0_f32;
    let seed = 44_u64;

    let food_ahead_sensory_id = 0_u32;
    let forward_action_id = NeuronId(ACTION_ID_BASE + action_index(ActionType::Forward) as u32);
    let consume_action_id = NeuronId(ACTION_ID_BASE + action_index(ActionType::Consume) as u32);

    let mut genome = test_genome();
    genome.num_neurons = 0;
    genome.num_synapses = 2;
    genome.inter_biases.clear();
    genome.inter_log_time_constants.clear();
    genome.inter_locations.clear();
    genome.hebb_eta_gain = default_seed.hebb_eta_gain;
    genome.eligibility_retention = default_seed.eligibility_retention;
    genome.synapse_prune_threshold = default_seed.synapse_prune_threshold;
    genome.age_of_maturity = default_seed.age_of_maturity;
    genome.starting_energy = default_seed.starting_energy;
    genome.vision_distance = default_seed.vision_distance;
    genome.edges = vec![
        SynapseEdge {
            pre_neuron_id: NeuronId(food_ahead_sensory_id),
            post_neuron_id: forward_action_id,
            weight: 0.25,
            eligibility: 0.0,
            pending_coactivation: 0.0,
        },
        SynapseEdge {
            pre_neuron_id: NeuronId(food_ahead_sensory_id),
            post_neuron_id: consume_action_id,
            weight: 0.3,
            eligibility: 0.0,
            pending_coactivation: 0.0,
        },
    ];

    let base_cfg = {
        let mut cfg = default_cfg.clone();
        cfg.world_width = 6;
        cfg.num_organisms = 1;
        cfg.food_regrowth_interval = 1;
        cfg.food_regrowth_jitter = 0;
        cfg.periodic_injection_interval_turns = 0;
        cfg.periodic_injection_count = 0;
        cfg.terrain_threshold = 1.0;
        cfg.max_organism_age = max_age;
        cfg.seed_genome_config.starting_energy = 10_000.0;
        cfg
    };

    let organism_q = 2;
    let organism_r = 2;
    let food_q = 3;
    let food_r = 2;

    let setup = |plasticity_enabled: bool| -> Simulation {
        let mut cfg = base_cfg.clone();
        cfg.runtime_plasticity_enabled = plasticity_enabled;

        let mut sim = Simulation::new(cfg, seed).expect("sim should init");

        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let brain = express_genome(&genome, &mut rng);
        let organism = OrganismState {
            id: sim_types::OrganismId(0),
            species_id: sim_types::SpeciesId(0),
            q: organism_q,
            r: organism_r,
            generation: 0,
            age_turns: 0,
            facing: FacingDirection::East,
            energy: initial_energy,
            energy_prev: initial_energy,
            dopamine: 0.0,
            consumptions_count: 0,
            reproductions_count: 0,
            last_action_taken: ActionType::Idle,
            brain,
            genome: genome.clone(),
        };

        configure_sim(&mut sim, vec![organism]);

        let width = sim.config.world_width as usize;
        let food_idx = food_r as usize * width + food_q as usize;
        sim.foods.push(sim_types::FoodState {
            id: sim_types::FoodId(0),
            q: food_q,
            r: food_r,
            energy: sim.config.food_energy,
        });
        sim.occupancy[food_idx] = Some(Occupant::Food(sim_types::FoodId(0)));
        sim.next_food_id = 1;

        let capacity = world_capacity(sim.config.world_width);
        sim.food_fertility = vec![false; capacity];
        sim.food_fertility[food_idx] = true;
        sim.food_regrowth_due_turn = vec![u64::MAX; capacity];

        sim
    };

    let mut sim_plastic = setup(true);
    let mut sim_static = setup(false);

    sim_plastic.advance_n(max_age as u32);
    sim_static.advance_n(max_age as u32);

    let edge_weight = |sim: &Simulation, post_neuron_id: NeuronId| -> f32 {
        sim.organisms()[0].brain.sensory[food_ahead_sensory_id as usize]
            .synapses
            .iter()
            .find(|s| s.post_neuron_id == post_neuron_id)
            .map_or(0.0, |synapse| synapse.weight)
    };

    let plastic_forward_weight = edge_weight(&sim_plastic, forward_action_id);
    let plastic_consume_weight = edge_weight(&sim_plastic, consume_action_id);
    let static_forward_weight = edge_weight(&sim_static, forward_action_id);
    let static_consume_weight = edge_weight(&sim_static, consume_action_id);
    let plastic_consumptions = sim_plastic.organisms()[0].consumptions_count;
    let static_consumptions = sim_static.organisms()[0].consumptions_count;
    assert_eq!(static_forward_weight, 0.25);
    assert_eq!(static_consume_weight, 0.3);
    assert!(
        plastic_consume_weight > plastic_forward_weight,
        "plasticity should make the rewarded consume pathway outweigh competing forward \
         (consume={plastic_consume_weight}, forward={plastic_forward_weight})"
    );
    assert!(
        plastic_consume_weight > static_consume_weight * 1.05,
        "plasticity should strengthen the rewarded consume pathway beyond the static control \
         (plastic={plastic_consume_weight}, static={static_consume_weight})"
    );
    assert!(
        plastic_forward_weight < static_forward_weight,
        "the weaker competing forward pathway should decay relative to the static control \
         under repo-default plasticity dynamics (plastic={plastic_forward_weight}, static={static_forward_weight})"
    );
    assert!(
        plastic_consumptions > static_consumptions + 10,
        "plasticity should increase successful consumptions over one lifetime in the \
         forward-vs-consume competition setup (plastic={plastic_consumptions}, static={static_consumptions})"
    );
}

#[test]
#[ignore = "known limitation of centered-logit action credit; validation currently favors this rule"]
fn sampled_action_credit_breaks_symmetric_forward_consume_tie() {
    let max_age = 500_u32;
    let initial_energy = 5_000.0_f32;
    let seed = 45_u64;

    let food_ahead_sensory_id = 0_u32;
    let forward_action_id = NeuronId(ACTION_ID_BASE + action_index(ActionType::Forward) as u32);
    let consume_action_id = NeuronId(ACTION_ID_BASE + action_index(ActionType::Consume) as u32);

    let mut genome = test_genome();
    genome.num_neurons = 0;
    genome.num_synapses = 2;
    genome.inter_biases.clear();
    genome.inter_log_time_constants.clear();
    genome.inter_locations.clear();
    genome.hebb_eta_gain = 0.05;
    genome.eligibility_retention = 0.9;
    genome.synapse_prune_threshold = 0.0;
    genome.age_of_maturity = 0;
    genome.starting_energy = 500.0;
    genome.vision_distance = 5;
    genome.edges = vec![
        SynapseEdge {
            pre_neuron_id: NeuronId(food_ahead_sensory_id),
            post_neuron_id: forward_action_id,
            weight: 0.3,
            eligibility: 0.0,
            pending_coactivation: 0.0,
        },
        SynapseEdge {
            pre_neuron_id: NeuronId(food_ahead_sensory_id),
            post_neuron_id: consume_action_id,
            weight: 0.3,
            eligibility: 0.0,
            pending_coactivation: 0.0,
        },
    ];

    let base_cfg = {
        let mut cfg = super::support::stable_test_config();
        cfg.world_width = 6;
        cfg.num_organisms = 1;
        cfg.food_energy = 100.0;
        cfg.food_regrowth_interval = 1;
        cfg.food_regrowth_jitter = 0;
        cfg.action_temperature = 0.08;
        cfg.max_organism_age = max_age;
        cfg.seed_genome_config.starting_energy = 10_000.0;
        cfg
    };

    let organism_q = 2;
    let organism_r = 2;
    let food_q = 3;
    let food_r = 2;

    let setup = |plasticity_enabled: bool| -> Simulation {
        let mut cfg = base_cfg.clone();
        cfg.runtime_plasticity_enabled = plasticity_enabled;

        let mut sim = Simulation::new(cfg, seed).expect("sim should init");

        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let brain = express_genome(&genome, &mut rng);
        let organism = OrganismState {
            id: sim_types::OrganismId(0),
            species_id: sim_types::SpeciesId(0),
            q: organism_q,
            r: organism_r,
            generation: 0,
            age_turns: 0,
            facing: FacingDirection::East,
            energy: initial_energy,
            energy_prev: initial_energy,
            dopamine: 0.0,
            consumptions_count: 0,
            reproductions_count: 0,
            last_action_taken: ActionType::Idle,
            brain,
            genome: genome.clone(),
        };

        configure_sim(&mut sim, vec![organism]);

        let width = sim.config.world_width as usize;
        let food_idx = food_r as usize * width + food_q as usize;
        sim.foods.push(sim_types::FoodState {
            id: sim_types::FoodId(0),
            q: food_q,
            r: food_r,
            energy: sim.config.food_energy,
        });
        sim.occupancy[food_idx] = Some(Occupant::Food(sim_types::FoodId(0)));
        sim.next_food_id = 1;

        let capacity = world_capacity(sim.config.world_width);
        sim.food_fertility = vec![false; capacity];
        sim.food_fertility[food_idx] = true;
        sim.food_regrowth_due_turn = vec![u64::MAX; capacity];

        sim
    };

    let mut sim_plastic = setup(true);
    let mut sim_static = setup(false);

    sim_plastic.advance_n(max_age as u32);
    sim_static.advance_n(max_age as u32);

    let edge_weight = |sim: &Simulation, post_neuron_id: NeuronId| -> f32 {
        sim.organisms()[0].brain.sensory[food_ahead_sensory_id as usize]
            .synapses
            .iter()
            .find(|s| s.post_neuron_id == post_neuron_id)
            .map_or(0.0, |synapse| synapse.weight)
    };

    let plastic_forward_weight = edge_weight(&sim_plastic, forward_action_id);
    let plastic_consume_weight = edge_weight(&sim_plastic, consume_action_id);
    let static_forward_weight = edge_weight(&sim_static, forward_action_id);
    let static_consume_weight = edge_weight(&sim_static, consume_action_id);
    let plastic_consumptions = sim_plastic.organisms()[0].consumptions_count;
    let static_consumptions = sim_static.organisms()[0].consumptions_count;

    assert_eq!(static_forward_weight, 0.3);
    assert_eq!(static_consume_weight, 0.3);
    assert!(
        plastic_consume_weight > plastic_forward_weight,
        "sampled-action credit should break the symmetric forward-vs-consume tie in favor \
         of the rewarded consume action (consume={plastic_consume_weight}, forward={plastic_forward_weight})"
    );
    assert!(
        plastic_consumptions > static_consumptions + 5,
        "sampled-action credit should convert the symmetric tie into more successful \
         consumptions over one lifetime (plastic={plastic_consumptions}, static={static_consumptions})"
    );
}

fn delayed_credit_assignment_organism(
    initial_weight: f32,
    hebb_eta_gain: f32,
    eligibility_retention: f32,
) -> OrganismState {
    let food_ahead_sensory_id = 0_u32;
    let consume_action_id = NeuronId(ACTION_ID_BASE + action_index(ActionType::Consume) as u32);

    let mut genome = test_genome();
    genome.num_neurons = 0;
    genome.num_synapses = 1;
    genome.inter_biases.clear();
    genome.inter_log_time_constants.clear();
    genome.inter_locations.clear();
    genome.hebb_eta_gain = hebb_eta_gain;
    genome.eligibility_retention = eligibility_retention;
    genome.synapse_prune_threshold = 0.0;
    genome.age_of_maturity = 0;
    genome.starting_energy = 100.0;
    genome.edges = vec![SynapseEdge {
        pre_neuron_id: NeuronId(food_ahead_sensory_id),
        post_neuron_id: consume_action_id,
        weight: initial_weight,
        eligibility: 0.0,
        pending_coactivation: 0.0,
    }];

    let mut rng = ChaCha8Rng::seed_from_u64(7);
    let brain = express_genome(&genome, &mut rng);
    OrganismState {
        id: sim_types::OrganismId(0),
        species_id: sim_types::SpeciesId(0),
        q: 0,
        r: 0,
        generation: 0,
        age_turns: 0,
        facing: FacingDirection::East,
        energy: genome.starting_energy,
        energy_prev: genome.starting_energy,
        dopamine: 0.0,
        consumptions_count: 0,
        reproductions_count: 0,
        last_action_taken: ActionType::Idle,
        brain,
        genome,
    }
}

fn food_consume_edge(organism: &OrganismState) -> &SynapseEdge {
    let consume_action_id = NeuronId(ACTION_ID_BASE + action_index(ActionType::Consume) as u32);
    organism.brain.sensory[0]
        .synapses
        .iter()
        .find(|edge| edge.post_neuron_id == consume_action_id)
        .expect("food->consume synapse should exist")
}

fn set_pending_food_consume_coactivation(
    organism: &mut OrganismState,
    scratch: &mut BrainScratch,
    sensory_activation: f32,
    consume_post_signal: f32,
) {
    organism.brain.sensory[0].neuron.activation = sensory_activation;
    scratch.action_post_signals.fill(0.0);
    scratch.action_post_signals[action_index(ActionType::Consume)] = consume_post_signal;
    compute_pending_coactivations(organism, scratch);
}

fn run_delayed_reward_sequence(
    eligibility_retention: f32,
    blank_ticks_before_reward: usize,
) -> (f32, f32) {
    let mut organism = delayed_credit_assignment_organism(0.2, 0.5, eligibility_retention);
    let mut scratch = BrainScratch::new();

    // Tick 0: coactivation occurs, but reward is absent. This should only write
    // into the eligibility trace, not strengthen the weight yet.
    set_pending_food_consume_coactivation(&mut organism, &mut scratch, 1.0, 1.0);
    apply_runtime_weight_updates(&mut organism, 0.0);

    // Intermediate ticks: no coactivation and no reward, so only the trace
    // should decay according to eligibility_retention.
    for _ in 0..blank_ticks_before_reward {
        set_pending_food_consume_coactivation(&mut organism, &mut scratch, 0.0, 0.0);
        apply_runtime_weight_updates(&mut organism, 0.0);
    }

    let eligibility_before_reward = food_consume_edge(&organism).eligibility;

    // Reward arrives later with no fresh coactivation on this tick. Any weight
    // increase therefore has to come from the retained eligibility trace.
    set_pending_food_consume_coactivation(&mut organism, &mut scratch, 0.0, 0.0);
    organism.energy += 20.0;
    apply_runtime_weight_updates(&mut organism, 0.0);

    (
        food_consume_edge(&organism).weight,
        eligibility_before_reward,
    )
}

#[test]
fn eligibility_traces_assign_delayed_credit_and_decay_with_time() {
    let (short_delay_weight, short_delay_eligibility) = run_delayed_reward_sequence(0.8, 1);
    let (long_delay_weight, long_delay_eligibility) = run_delayed_reward_sequence(0.8, 3);
    let (no_trace_weight, no_trace_eligibility) = run_delayed_reward_sequence(0.0, 3);

    assert!(
        short_delay_eligibility > long_delay_eligibility && long_delay_eligibility > 0.0,
        "eligibility trace should decay over blank ticks but remain non-zero with retention \
         (short={short_delay_eligibility}, long={long_delay_eligibility})"
    );
    assert_eq!(
        no_trace_eligibility, 0.0,
        "without retention, the delayed-credit trace should be gone before reward arrives"
    );

    assert!(
        short_delay_weight > long_delay_weight,
        "a shorter reward delay should assign more credit than a longer delay \
         (short={short_delay_weight}, long={long_delay_weight})"
    );
    assert!(
        long_delay_weight > no_trace_weight,
        "retained eligibility should still strengthen the synapse when reward arrives late \
         (retained={long_delay_weight}, no_trace={no_trace_weight})"
    );
}
