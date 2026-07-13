use anyhow::{anyhow, Result};
use serde::Serialize;
use serde_json::json;
use sim_core::progressive::{
    enforce_retention, exact_fingerprint, verify_extension_effect, AllHistoryRetention,
    ExtensionEffectEvidence, ProtectedResidual, RetentionRequirementHeader, TaskCheckpoint,
    TaskReplay,
};
use sim_core::Simulation;
use sim_types::{
    action_gene_node_id, connection_innovation_id, inter_neuron_id, seed_hidden_gene_node_id,
    sensory_gene_node_id, ActionType, GeneNodeId, HiddenNodeGene, OrganismGenome, SynapseGene,
};

const PROBE_HIDDEN_NODES: u32 = 1_501;

#[derive(Debug, Clone, Serialize)]
struct ProbeTask {
    name: &'static str,
    horizon: u32,
    criterion: ProbeCriterion,
}

#[derive(Debug, Clone, Copy, Serialize)]
enum ProbeCriterion {
    Survive,
    TurnRightAtLeast(u32),
}

#[derive(Debug, Clone, Serialize)]
struct BehaviorPoint {
    turn: u64,
    organisms: Vec<OrganismFact>,
}

#[derive(Debug, Clone, Serialize)]
struct OrganismFact {
    id: u64,
    q: i32,
    r: i32,
    action: ActionType,
    energy_bits: u32,
    consumptions: u64,
}

#[derive(Debug)]
struct TaskRun {
    behavior_fingerprint: String,
    passed: bool,
}

fn main() -> Result<()> {
    let world = probe_world()?;
    let base = base_genome();
    let stage0 = ProtectedResidual::seal(&base)?;

    // A fresh stage is an exact clone: there is no zero-weight edge for the
    // sanitizer to turn into +0.001 and no dormant runtime structure.
    let empty_extension = stage0.seed_extension();
    let empty_extension_exact = base == empty_extension;
    let mut base_world = Simulation::new_with_champion_pool(world.clone(), 77, vec![base.clone()])?;
    let mut empty_world =
        Simulation::new_with_champion_pool(world.clone(), 77, vec![empty_extension])?;
    let initial_worlds_byte_identical = world_bytes(&base_world)? == world_bytes(&empty_world)?;
    base_world.advance_n(64);
    empty_world.advance_n(64);
    let advanced_worlds_byte_identical = world_bytes(&base_world)? == world_bytes(&empty_world)?;

    // Stage 1: corrupt sealed genes, inject an illegal write into a protected
    // hidden node, and add a legal residual. Projection must restore/remove
    // the former while retaining the latter.
    let hidden1 = seed_hidden_gene_node_id(1);
    let mut stage1_genome = stage0.seed_extension();
    stage1_genome.topology.vision_distance += 3;
    stage1_genome.brain.action_biases[0] += 0.75;
    stage1_genome.brain.hidden_nodes[0].bias += 0.5;
    stage1_genome.brain.edges[0].weight = -1.5;
    stage1_genome.brain.hidden_nodes.push(HiddenNodeGene {
        id: hidden1,
        bias: 0.2,
        log_time_constant: -1.203_972_8,
    });
    stage1_genome.brain.edges.extend([
        edge(sensory_gene_node_id(1), seed_hidden_gene_node_id(0), 0.9),
        edge(sensory_gene_node_id(1), hidden1, 0.4),
        edge(hidden1, action_gene_node_id(0), 0.2),
    ]);
    stage0.project(&mut stage1_genome);
    stage0.verify(&stage1_genome)?;
    let illegal_protected_input_removed = !stage1_genome.brain.edges.iter().any(|edge| {
        edge.pre_node_id == sensory_gene_node_id(1)
            && edge.post_node_id == seed_hidden_gene_node_id(0)
    });
    let stage1_residual_retained =
        stage1_genome
            .brain
            .hidden_nodes
            .iter()
            .any(|node| node.id == hidden1)
            && stage1_genome.brain.edges.iter().any(|edge| {
                edge.pre_node_id == hidden1 && edge.post_node_id == action_gene_node_id(0)
            });

    // Stage 2 starts from the accepted stage-1 genome. An adversarial mutation
    // of stage 1 is restored, while a new stage-2 action residual survives.
    let stage1 = ProtectedResidual::seal(&stage1_genome)?;
    let hidden2 = seed_hidden_gene_node_id(2);
    let mut stage2_genome = stage1.seed_extension();
    let stage1_residual_innovation = connection_innovation_id(hidden1, action_gene_node_id(0));
    stage2_genome
        .brain
        .edges
        .iter_mut()
        .find(|edge| edge.innovation == stage1_residual_innovation)
        .ok_or_else(|| anyhow!("stage-1 residual edge missing"))?
        .weight = -1.5;
    stage2_genome.brain.hidden_nodes.push(HiddenNodeGene {
        id: hidden2,
        bias: 1.0,
        log_time_constant: -1.203_972_8,
    });
    stage2_genome
        .brain
        .edges
        .push(edge(hidden2, action_gene_node_id(1), 1.5));
    stage1.project(&mut stage2_genome);
    stage1.verify(&stage2_genome)?;
    let stage1_weight_restored = stage2_genome
        .brain
        .edges
        .iter()
        .find(|edge| edge.innovation == stage1_residual_innovation)
        .map(|edge| edge.weight)
        == stage1_genome
            .brain
            .edges
            .iter()
            .find(|edge| edge.innovation == stage1_residual_innovation)
            .map(|edge| edge.weight);

    let base_fingerprint = exact_fingerprint(&base)?;
    let stage1_fingerprint = exact_fingerprint(&stage1_genome)?;
    let stage2_fingerprint = exact_fingerprint(&stage2_genome)?;
    let knockout = stage1.knockout_extension();
    let knockout_fingerprint = exact_fingerprint(&knockout)?;

    // Full-history archive contract: reproduce each accepted controller's
    // exact stored trace, then replay the candidate on every historical task.
    let task0 = ProbeTask {
        name: "survive-short",
        horizon: 16,
        criterion: ProbeCriterion::Survive,
    };
    let task1 = ProbeTask {
        name: "survive-long",
        horizon: 32,
        criterion: ProbeCriterion::Survive,
    };
    let seeds = vec![11, 29];
    let checkpoint0 = make_checkpoint(0, &task0, &base, &seeds, 2, &world)?;
    let checkpoint1 = make_checkpoint(1, &task1, &stage1_genome, &seeds, 2, &world)?;
    let checkpoints = vec![checkpoint0, checkpoint1];
    let ancestor_replays = vec![
        replay_task(0, &task0, &base, &seeds, &world)?,
        replay_task(1, &task1, &stage1_genome, &seeds, &world)?,
    ];
    let candidate_replays = vec![
        replay_task(0, &task0, &stage2_genome, &seeds, &world)?,
        replay_task(1, &task1, &stage2_genome, &seeds, &world)?,
    ];
    let retention = AllHistoryRetention {
        candidate_controller_fingerprint: stage2_fingerprint.clone(),
        ancestor_replays,
        candidate_replays,
    };
    enforce_retention(&checkpoints, &retention)?;

    // Causal residual audit on a held-out task: the knockout is exactly the
    // prior checkpoint, and enabling stage 2 changes an inspectable trace.
    let effect_task = ProbeTask {
        name: "residual-effect",
        horizon: 32,
        criterion: ProbeCriterion::TurnRightAtLeast(24),
    };
    let enabled_runs = run_panel(&effect_task, &stage2_genome, &seeds, &world)?;
    let knockout_runs = run_panel(&effect_task, &knockout, &seeds, &world)?;
    let effect = ExtensionEffectEvidence {
        task_id: 2,
        task_fingerprint: exact_fingerprint(&effect_task)?,
        trial_seeds: seeds.clone(),
        enabled_controller_fingerprint: stage2_fingerprint.clone(),
        knockout_controller_fingerprint: knockout_fingerprint.clone(),
        enabled_passes: enabled_runs.iter().filter(|run| run.passed).count() as u32,
        knockout_passes: knockout_runs.iter().filter(|run| run.passed).count() as u32,
        minimum_enabled_passes: 2,
        maximum_knockout_passes: 0,
        enabled_behavior_fingerprints: enabled_runs
            .iter()
            .map(|run| run.behavior_fingerprint.clone())
            .collect(),
        knockout_behavior_fingerprints: knockout_runs
            .iter()
            .map(|run| run.behavior_fingerprint.clone())
            .collect(),
    };
    verify_extension_effect(&stage1_fingerprint, &effect)?;

    // Materialization probe crosses the historical boundary and deliberately
    // mixes an inter target whose numeric ID is greater than action IDs on one
    // sensory source. The runtime category sort must still partition it first.
    let wide = wide_genome();
    let mut wide_world = Simulation::new_with_champion_pool(world.clone(), 91, vec![wide.clone()])?;
    let wide_organism = &wide_world.organisms()[0];
    let stored_hidden_nodes = wide_organism.genome.brain.hidden_nodes.len();
    let expressed_hidden_nodes = wide_organism.brain.inter.len();
    let encoded_edges = wide_organism.genome.brain.edges.len();
    let expressed_edges = wide_organism.brain.synapse_count as usize;
    let sensory0_targets = wide_organism.brain.sensory[0]
        .synapses
        .iter()
        .map(|edge| edge.post_neuron_id.0)
        .collect::<Vec<_>>();
    let sensory0_action_split = wide_organism.brain.sensory[0].action_synapse_start;
    let high_inter_runtime_id = inter_neuron_id(PROBE_HIDDEN_NODES - 1).0;
    let action_runtime_id = ActionType::TurnLeft.neuron_id().unwrap().0;
    let wide_initial_bytes = world_bytes(&wide_world)?;
    let mut wide_world_again = Simulation::new_with_champion_pool(world, 91, vec![wide])?;
    wide_world.advance_n(16);
    wide_world_again.advance_n(16);
    let wide_deterministic = world_bytes(&wide_world)? == world_bytes(&wide_world_again)?;

    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "schema": "progressive-capacity-probe-v1",
            "zero_stage": {
                "genotype_exact": empty_extension_exact,
                "initial_world_bytes_exact": initial_worlds_byte_identical,
                "world_bytes_after_64_ticks_exact": advanced_worlds_byte_identical,
                "materialization": "no added node or edge; exact sealed clone"
            },
            "protected_residual": {
                "illegal_input_to_sealed_hidden_removed": illegal_protected_input_removed,
                "stage1_residual_retained": stage1_residual_retained,
                "stage1_parameter_restored_during_stage2": stage1_weight_restored,
                "base_controller_fingerprint": base_fingerprint,
                "stage1_controller_fingerprint": stage1_fingerprint,
                "stage2_controller_fingerprint": stage2_fingerprint,
                "stage2_knockout_fingerprint": knockout_fingerprint,
                "knockout_is_exact_stage1": knockout == stage1_genome
            },
            "all_history_retention": {
                "checkpoint_count": checkpoints.len(),
                "candidate_replay_count": retention.candidate_replays.len(),
                "accepted": true,
                "contract": "exact ancestor controller/task/seed/behavior replay plus candidate threshold on every checkpoint"
            },
            "extension_effect": effect,
            "runtime_capacity": {
                "historical_hidden_node_ceiling": 1000,
                "probe_hidden_nodes": PROBE_HIDDEN_NODES,
                "stored_genotype_hidden_nodes": stored_hidden_nodes,
                "expressed_phenotype_hidden_nodes": expressed_hidden_nodes,
                "encoded_edges": encoded_edges,
                "expressed_edges": expressed_edges,
                "high_inter_runtime_id": high_inter_runtime_id,
                "stable_action_runtime_id": action_runtime_id,
                "sensory0_runtime_target_order": sensory0_targets,
                "sensory0_action_split": sensory0_action_split,
                "initial_serialized_bytes": wide_initial_bytes.len(),
                "deterministic_after_16_ticks": wide_deterministic
            },
            "limits": [
                "capacity is finite under u32 IDs and machine memory; this removes the accidental 1000-node action-ID collision, not physical bounds",
                "residuals can change old actions, so all-history retention replay remains mandatory",
                "runtime plasticity can couple residual actions back into lifetime weight trajectories; exact prefix-isolation claims require plasticity off or a separate plasticity-aware audit"
            ]
        }))?
    );
    Ok(())
}

fn probe_world() -> Result<sim_types::WorldConfig> {
    let mut world = sim_config::load_default_world_config()?;
    world.world_width = 25;
    world.num_organisms = 1;
    world.intent_parallel_threads = 1;
    world.runtime_plasticity_enabled = false;
    Ok(world)
}

fn base_genome() -> OrganismGenome {
    let mut genome = OrganismGenome::test_fixture();
    let hidden = seed_hidden_gene_node_id(0);
    genome.brain.hidden_nodes[0].bias = 0.1;
    genome.brain.edges = vec![
        edge(sensory_gene_node_id(0), hidden, 0.5),
        edge(hidden, action_gene_node_id(0), 0.4),
    ];
    genome
}

fn wide_genome() -> OrganismGenome {
    let mut genome = OrganismGenome::test_fixture();
    genome.brain.hidden_nodes = (0..PROBE_HIDDEN_NODES)
        .map(|index| HiddenNodeGene {
            id: seed_hidden_gene_node_id(index),
            bias: if index + 1 == PROBE_HIDDEN_NODES {
                0.5
            } else {
                0.0
            },
            log_time_constant: -1.203_972_8,
        })
        .collect();
    let high = seed_hidden_gene_node_id(PROBE_HIDDEN_NODES - 1);
    genome.brain.edges = vec![
        edge(sensory_gene_node_id(0), high, 0.7),
        edge(sensory_gene_node_id(0), action_gene_node_id(0), 0.2),
        edge(high, action_gene_node_id(1), 0.8),
    ];
    genome
}

fn edge(pre_node_id: GeneNodeId, post_node_id: GeneNodeId, weight: f32) -> SynapseGene {
    SynapseGene {
        innovation: connection_innovation_id(pre_node_id, post_node_id),
        pre_node_id,
        post_node_id,
        weight,
        enabled: true,
    }
}

fn world_bytes(sim: &Simulation) -> Result<Vec<u8>> {
    let mut bytes = Vec::new();
    sim.save(&mut bytes)?;
    Ok(bytes)
}

fn make_checkpoint(
    task_id: u64,
    task: &ProbeTask,
    genome: &OrganismGenome,
    seeds: &[u64],
    minimum_passes: u32,
    world: &sim_types::WorldConfig,
) -> Result<TaskCheckpoint> {
    let runs = run_panel(task, genome, seeds, world)?;
    Ok(TaskCheckpoint {
        requirement: RetentionRequirementHeader {
            task_id,
            minimum_passes,
        },
        task_fingerprint: exact_fingerprint(task)?,
        accepted_controller_fingerprint: exact_fingerprint(genome)?,
        trial_seeds: seeds.to_vec(),
        accepted_passes: runs.iter().filter(|run| run.passed).count() as u32,
        accepted_behavior_fingerprints: runs
            .into_iter()
            .map(|run| run.behavior_fingerprint)
            .collect(),
    })
}

fn replay_task(
    task_id: u64,
    task: &ProbeTask,
    genome: &OrganismGenome,
    seeds: &[u64],
    world: &sim_types::WorldConfig,
) -> Result<TaskReplay> {
    let runs = run_panel(task, genome, seeds, world)?;
    Ok(TaskReplay {
        task_id,
        task_fingerprint: exact_fingerprint(task)?,
        controller_fingerprint: exact_fingerprint(genome)?,
        trial_seeds: seeds.to_vec(),
        passes: runs.iter().filter(|run| run.passed).count() as u32,
        behavior_fingerprints: runs
            .into_iter()
            .map(|run| run.behavior_fingerprint)
            .collect(),
    })
}

fn run_panel(
    task: &ProbeTask,
    genome: &OrganismGenome,
    seeds: &[u64],
    world: &sim_types::WorldConfig,
) -> Result<Vec<TaskRun>> {
    seeds
        .iter()
        .map(|&seed| run_task(task, genome, seed, world))
        .collect()
}

fn run_task(
    task: &ProbeTask,
    genome: &OrganismGenome,
    seed: u64,
    world: &sim_types::WorldConfig,
) -> Result<TaskRun> {
    let mut sim = Simulation::new_with_champion_pool(world.clone(), seed, vec![genome.clone()])?;
    let mut trace = Vec::with_capacity(task.horizon as usize);
    let mut right_turns = 0_u32;
    for _ in 0..task.horizon {
        sim.advance_n(1);
        right_turns += sim
            .organisms()
            .iter()
            .filter(|organism| organism.last_action_taken == ActionType::TurnRight)
            .count() as u32;
        trace.push(BehaviorPoint {
            turn: sim.turn(),
            organisms: sim
                .organisms()
                .iter()
                .map(|organism| OrganismFact {
                    id: organism.id.0,
                    q: organism.q,
                    r: organism.r,
                    action: organism.last_action_taken,
                    energy_bits: organism.energy.to_bits(),
                    consumptions: organism.consumptions_count,
                })
                .collect(),
        });
    }
    Ok(TaskRun {
        behavior_fingerprint: exact_fingerprint(&trace)?,
        passed: match task.criterion {
            ProbeCriterion::Survive => !sim.organisms().is_empty(),
            ProbeCriterion::TurnRightAtLeast(minimum) => right_turns >= minimum,
        },
    })
}
