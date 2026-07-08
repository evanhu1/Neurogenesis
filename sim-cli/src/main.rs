//! sim-cli — stateless, world-as-file research CLI for the NeuroGenesis engine
//! (new substrate). Each invocation reads a `HexSim` world from `--in`, runs one
//! command, and (for mutating commands) writes the advanced world to `--out`
//! (default `--in`). Output is JSON.
//!
//! Commands:
//!   new [--seed N] [--width W] [--founders F] [--energy E] --out w.bin
//!   run-to <turn> --in w.bin [--out w.bin]      # stops early on extinction
//!   step [N] --in w.bin [--out w.bin]
//!   state --in w.bin
//!   inspect <id> --in w.bin
//!   brain <id> --in w.bin
//!   genome <id> --in w.bin
//!   decide <id> --in w.bin
//!   find <field> <op> <value> --in w.bin        # op: gt lt ge le eq
//!   lineage --in w.bin

use anyhow::{anyhow, bail, Context, Result};
use serde_json::{json, Value};
use sim_hexworld::{HexConfig, HexSim};
use sim_substrate::brain::{EXPLICIT_IDLE_LOGIT_BIAS, MIN_ACTION_TEMPERATURE};
use std::collections::BTreeMap;

fn main() {
    if let Err(e) = run() {
        eprintln!("error: {e:#}");
        std::process::exit(1);
    }
}

struct Args {
    command: String,
    positionals: Vec<String>,
    flags: BTreeMap<String, String>,
}

fn parse_args() -> Result<Args> {
    let raw: Vec<String> = std::env::args().skip(1).collect();
    if raw.is_empty() {
        bail!("usage: sim-cli <command> [args] [--in w.bin] [--out w.bin]");
    }
    let command = raw[0].clone();
    let mut positionals = Vec::new();
    let mut flags = BTreeMap::new();
    let mut i = 1;
    while i < raw.len() {
        let tok = &raw[i];
        if let Some(key) = tok.strip_prefix("--") {
            let val = raw.get(i + 1).cloned().unwrap_or_default();
            flags.insert(key.to_string(), val);
            i += 2;
        } else {
            positionals.push(tok.clone());
            i += 1;
        }
    }
    Ok(Args {
        command,
        positionals,
        flags,
    })
}

fn load(args: &Args) -> Result<HexSim> {
    let path = args
        .flags
        .get("in")
        .ok_or_else(|| anyhow!("this command needs --in <world.bin>"))?;
    let bytes = std::fs::read(path).with_context(|| format!("reading {path}"))?;
    bincode::deserialize(&bytes).with_context(|| format!("decoding world {path}"))
}

fn save(args: &Args, sim: &HexSim) -> Result<()> {
    let out = args
        .flags
        .get("out")
        .or_else(|| args.flags.get("in"))
        .ok_or_else(|| anyhow!("mutating command needs --out or --in"))?;
    let bytes = bincode::serialize(sim).context("encoding world")?;
    std::fs::write(out, bytes).with_context(|| format!("writing {out}"))?;
    Ok(())
}

fn print(value: Value) {
    println!("{}", serde_json::to_string_pretty(&value).unwrap());
}

fn flag_parse<T: std::str::FromStr>(args: &Args, key: &str, default: T) -> Result<T> {
    match args.flags.get(key) {
        Some(v) => v.parse().map_err(|_| anyhow!("bad value for --{key}: {v}")),
        None => Ok(default),
    }
}

fn body_id_arg(args: &Args) -> Result<u64> {
    args.positionals
        .first()
        .ok_or_else(|| anyhow!("this command needs an organism id"))?
        .parse()
        .context("organism id must be an integer")
}

fn run() -> Result<()> {
    let args = parse_args()?;
    match args.command.as_str() {
        "new" => cmd_new(&args),
        "run-to" => cmd_run_to(&args),
        "step" => cmd_step(&args),
        "state" => cmd_state(&args),
        "inspect" => cmd_inspect(&args),
        "brain" => cmd_brain(&args),
        "genome" => cmd_genome(&args),
        "decide" => cmd_decide(&args),
        "find" => cmd_find(&args),
        "lineage" => cmd_lineage(&args),
        other => bail!("unknown command '{other}'"),
    }
}

fn cmd_new(args: &Args) -> Result<()> {
    let config = HexConfig {
        world_width: flag_parse(args, "width", HexConfig::default().world_width)?,
        num_founders: flag_parse(args, "founders", HexConfig::default().num_founders)?,
        founder_energy: flag_parse(args, "energy", HexConfig::default().founder_energy)?,
        ..HexConfig::default()
    };
    let seed = flag_parse(args, "seed", 0u64)?;
    let sim = HexSim::new(config, seed);
    save(args, &sim)?;
    print(json!({
        "created": true, "seed": seed, "width": config.world_width,
        "founders": config.num_founders, "alive": sim.alive_count(),
    }));
    Ok(())
}

fn cmd_run_to(args: &Args) -> Result<()> {
    let target: u64 = args
        .positionals
        .first()
        .ok_or_else(|| anyhow!("run-to needs a target turn"))?
        .parse()
        .context("target turn must be an integer")?;
    let mut sim = load(args)?;
    let mut extinct = false;
    while sim.turn() < target {
        if !sim.tick() {
            extinct = true;
            break;
        }
    }
    save(args, &sim)?;
    print(json!({
        "turn": sim.turn(), "alive": sim.alive_count(),
        "extinct": extinct, "extinct_at": sim.extinct_at,
    }));
    Ok(())
}

fn cmd_step(args: &Args) -> Result<()> {
    let n: u64 = args.positionals.first().map(|s| s.parse()).transpose()?.unwrap_or(1);
    let mut sim = load(args)?;
    let mut extinct = false;
    for _ in 0..n {
        if !sim.tick() {
            extinct = true;
            break;
        }
    }
    save(args, &sim)?;
    print(json!({ "turn": sim.turn(), "alive": sim.alive_count(), "extinct": extinct }));
    Ok(())
}

fn cmd_state(args: &Args) -> Result<()> {
    let sim = load(args)?;
    print(serde_json::to_value(sim.population_stats())?);
    Ok(())
}

fn cmd_inspect(args: &Args) -> Result<()> {
    let sim = load(args)?;
    let id = body_id_arg(args)?;
    let (_, body) = sim
        .body_by_id(id)
        .ok_or_else(|| anyhow!("no living organism with id {id}"))?;
    let h = &body.genome.header;
    print(json!({
        "id": body.id,
        "energy": body.energy,
        "health": body.health,
        "age_turns": body.age_turns,
        "generation": body.generation,
        "is_gestating": body.is_gestating,
        "brain": { "neurons": body.brain.neurons.len(), "edges": body.brain.edges.len(),
                   "inputs": body.brain.input_count, "hidden": body.brain.hidden_count,
                   "outputs": body.brain.output_count },
        "cppn": { "nodes": body.genome.cppn.nodes.len(), "conns": body.genome.cppn.conns.len() },
        "morphology": body.morphology,
        "lifecycle": { "age_of_maturity": h.lifecycle.age_of_maturity,
                       "gestation_ticks": h.lifecycle.gestation_ticks,
                       "max_organism_age": h.lifecycle.max_organism_age },
        "plasticity": { "hebb_eta_gain": h.plasticity.hebb_eta_gain,
                        "juvenile_eta_scale": h.plasticity.juvenile_eta_scale,
                        "eligibility_retention": h.plasticity.eligibility_retention },
    }));
    Ok(())
}

fn cmd_brain(args: &Args) -> Result<()> {
    let sim = load(args)?;
    let id = body_id_arg(args)?;
    let (_, body) = sim
        .body_by_id(id)
        .ok_or_else(|| anyhow!("no living organism with id {id}"))?;
    // BrainNet derives Serialize — hand back the full phenotype network.
    print(serde_json::to_value(&body.brain)?);
    Ok(())
}

fn cmd_genome(args: &Args) -> Result<()> {
    let sim = load(args)?;
    let id = body_id_arg(args)?;
    let (_, body) = sim
        .body_by_id(id)
        .ok_or_else(|| anyhow!("no living organism with id {id}"))?;
    print(serde_json::to_value(&body.genome)?);
    Ok(())
}

fn cmd_decide(args: &Args) -> Result<()> {
    let sim = load(args)?;
    let id = body_id_arg(args)?;
    let (handle, body) = sim
        .body_by_id(id)
        .ok_or_else(|| anyhow!("no living organism with id {id}"))?;
    let obs = sim.observe_body(handle);
    let mut brain = body.brain.clone();
    let logits: Vec<f32> = brain.step(&obs).to_vec();
    let catalog = sim.catalog();
    let temperature = 1.0f32;
    let t = temperature.max(MIN_ACTION_TEMPERATURE);
    let max_logit = logits.iter().copied().fold(EXPLICIT_IDLE_LOGIT_BIAS, f32::max);
    let mut sum = ((EXPLICIT_IDLE_LOGIT_BIAS - max_logit) / t).exp();
    for &l in &logits {
        sum += ((l - max_logit) / t).exp();
    }
    let mut actions = Vec::new();
    for (slot, &logit) in logits.iter().enumerate() {
        let actuator = body
            .action_layout
            .actuator_indices
            .get(slot)
            .map(|&ai| catalog.actuators[ai].key.clone())
            .unwrap_or_else(|| format!("slot{slot}"));
        let prob = ((logit - max_logit) / t).exp() / sum;
        actions.push(json!({ "action": actuator, "logit": logit, "prob": prob }));
    }
    let idle_prob = ((EXPLICIT_IDLE_LOGIT_BIAS - max_logit) / t).exp() / sum;
    print(json!({ "id": id, "actions": actions, "idle_prob": idle_prob, "observation": obs }));
    Ok(())
}

fn cmd_find(args: &Args) -> Result<()> {
    let sim = load(args)?;
    if args.positionals.len() < 3 {
        bail!("find needs: <field> <op> <value>  (op: gt lt ge le eq)");
    }
    let field = &args.positionals[0];
    let op = &args.positionals[1];
    let value: f64 = args.positionals[2].parse().context("value must be a number")?;
    let mut matches = Vec::new();
    for body in sim.living_bodies() {
        let field_val = match field.as_str() {
            "energy" => body.energy as f64,
            "health" => body.health as f64,
            "age" => body.age_turns as f64,
            "generation" => body.generation as f64,
            "neurons" => body.brain.neurons.len() as f64,
            "edges" => body.brain.edges.len() as f64,
            other => bail!("unknown field '{other}'"),
        };
        let hit = match op.as_str() {
            "gt" => field_val > value,
            "lt" => field_val < value,
            "ge" => field_val >= value,
            "le" => field_val <= value,
            "eq" => (field_val - value).abs() < f64::EPSILON,
            other => bail!("unknown op '{other}' (use gt lt ge le eq)"),
        };
        if hit {
            matches.push(json!({ "id": body.id, field.clone(): field_val }));
        }
    }
    print(json!({ "field": field, "op": op, "value": value, "count": matches.len(), "matches": matches }));
    Ok(())
}

fn cmd_lineage(args: &Args) -> Result<()> {
    let sim = load(args)?;
    let mut hist: BTreeMap<u64, u64> = BTreeMap::new();
    for body in sim.living_bodies() {
        *hist.entry(body.generation).or_insert(0) += 1;
    }
    let generations: Vec<Value> = hist
        .iter()
        .map(|(gen, count)| json!({ "generation": gen, "count": count }))
        .collect();
    print(json!({ "alive": sim.alive_count(), "generations": generations }));
    Ok(())
}
