//! views — the shared data plane for the NeuroGenesis research tooling.
//!
//! Owns world/metric-sidecar file IO, recording-aware advancement, world
//! construction, and every read command as a function that renders to a writer
//! (text or JSON, chosen per call). Both `cli` (agent-facing shell) and
//! `sim-server` (the human-facing web backend) are thin frontends over this
//! crate, so the numbers the agent and the researcher see are identical by
//! construction.
//!
//! Read commands take a [`ReadCtx`] (a borrowed view of a loaded world + its
//! optional metric recorder) and write to any [`std::io::Write`]. They are pure
//! reads: no world mutation, no `--out`.

pub mod dashboards;
pub mod inspect_ext;
pub mod output;

// Flat re-exports so callers use `views::eco` / `views::find` alongside
// the reads defined directly in this module (`state`, `pillars`, …).
pub use dashboards::{eco, genome, lineage, timeseries};
pub use inspect_ext::{brain, decide, find};

use anyhow::{anyhow, Result};
use config::{load_world_config_from_path, world_config_from_toml_parts, WorldConfig};
use metrics::{
    compute_pillar_scores, derive_interval_metrics, ingest_tick, register_existing,
    register_founders, BehaviorIntervalRow, IntervalMetrics, Ledger, OrganismLifetimeRow,
    PillarScores, TickSummaryRow,
};
use output::{opt, opt_json, Format, Stats};
use serde_json::json;
use std::fs::File;
use std::io::{BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use types::{EntityId, OrganismGenome};
use world_sim::Simulation;

/// In-memory metric recorder. Holds the shared per-organism ledger plus raw
/// tick, action-time interval, and lifetime fact streams. Behavioral metrics
/// use `behavior_intervals`; lifetime rows remain cohort/lifecycle facts.
#[derive(serde::Serialize, serde::Deserialize)]
pub struct Recorder {
    pub ledger: Ledger,
    pub tick_summary: Vec<TickSummaryRow>,
    pub behavior_intervals: Vec<BehaviorIntervalRow>,
    pub lifetimes: Vec<OrganismLifetimeRow>,
    /// Richer per-recorded-tick ecology samples for `eco`/`timeseries`
    /// trajectories (the metric `tick_summary` only carries descendant pop).
    pub samples: Vec<EcoSample>,
    /// Boundary cadence used to close action-time behavior rows.
    pub report_every: u64,
    /// Last world turn whose action records were ingested. This can differ
    /// from the loaded world's turn when it was advanced under `--no-metrics`.
    pub recorded_through_turn: u64,
    /// Turn at which recording began. `> 0` means already-alive organisms were
    /// back-registered with partial lifetime counters (windows are approximate).
    pub started_turn: u64,
}

/// One per recorded tick: a cheap, whole-world ecology snapshot from the
/// `TickDelta` metrics (no extra organism scans), powering `eco`/`timeseries`.
#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub struct EcoSample {
    pub population: u32,
    pub food: u32,
    pub deaths: u32,
    pub starvations: u32,
    pub age_deaths: u32,
    pub predations: u32,
    pub consumptions: u32,
}

/// A borrowed, read-only view of a loaded world plus its optional metric
/// recorder — the argument every read command takes. `format`/`scaled` carry
/// the caller's presentation defaults (per-command `--json`/`--text` still
/// override `format`).
pub struct ReadCtx<'a> {
    pub sim: &'a Simulation,
    pub recorder: Option<&'a Recorder>,
    pub report_every: u64,
    pub format: Format,
    pub scaled: bool,
}

impl ReadCtx<'_> {
    fn recorded_behavior_rows(&self) -> Option<Vec<BehaviorIntervalRow>> {
        let rec = self.recorder?;
        let mut rows = rec.behavior_intervals.clone();
        let open = rec
            .ledger
            .behavior_interval_snapshot(rec.recorded_through_turn);
        if open.start_tick < open.end_tick {
            rows.push(open);
        }
        Some(rows)
    }

    /// Compute live pillars over the recorded span using the exact `metrics`
    /// pipeline. Returns the scores, the interval count, and whether the window
    /// is partial (recording started after turn 0). `None` when not recording.
    pub fn live_pillars(&self) -> Option<(PillarScores, usize, bool)> {
        let rec = self.recorder?;
        let rows = self.recorded_behavior_rows()?;
        let intervals = derive_interval_metrics(&rows);
        let scores = compute_pillar_scores(&intervals);
        Some((
            scores,
            intervals.len(),
            rec.started_turn > 0 || rec.recorded_through_turn != self.sim.turn(),
        ))
    }

    /// The full per-interval metric series behind the pillar scores — the
    /// granular data `pillars` reports alongside the windowed means.
    pub fn live_intervals(&self) -> Option<Vec<IntervalMetrics>> {
        let rows = self.recorded_behavior_rows()?;
        Some(derive_interval_metrics(&rows))
    }
}

/// Strip `--json` / `--text` from `args`, returning the effective format (the
/// per-command flag overriding `default`) and the rest.
pub fn take_format<'a>(default: Format, args: &[&'a str]) -> (Format, Vec<&'a str>) {
    let mut fmt = default;
    let mut rest = Vec::with_capacity(args.len());
    for &a in args {
        match a {
            "--json" => fmt = Format::Json,
            "--text" => fmt = Format::Text,
            _ => rest.push(a),
        }
    }
    (fmt, rest)
}

// ---------------------------------------------------------------------------
// World / metric-sidecar file IO
// ---------------------------------------------------------------------------

/// Write to `<path>.tmp` then atomically rename into place, so a crash or write
/// error mid-save can't truncate an existing file — worlds are often advanced in
/// place (`--out` defaults to `--in`), so the input is the only copy. Creates
/// the parent directory and flushes before the rename.
pub fn atomic_write(
    path: &str,
    write: impl FnOnce(&mut BufWriter<File>) -> Result<()>,
) -> Result<()> {
    if let Some(parent) = Path::new(path).parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent).ok();
        }
    }
    let tmp = format!("{path}.tmp");
    {
        let mut w =
            BufWriter::new(File::create(&tmp).map_err(|e| anyhow!("cannot write `{tmp}`: {e}"))?);
        write(&mut w)?;
        w.flush().map_err(|e| anyhow!("flushing `{tmp}`: {e}"))?;
    }
    std::fs::rename(&tmp, path).map_err(|e| anyhow!("renaming `{tmp}` -> `{path}`: {e}"))
}

pub fn load_world(path: &str) -> Result<Simulation> {
    let file = File::open(path).map_err(|e| anyhow!("cannot open world `{path}`: {e}"))?;
    Simulation::load(BufReader::new(file)).map_err(|e| anyhow!("loading world `{path}`: {e}"))
}

pub fn save_world(sim: &Simulation, path: &str) -> Result<()> {
    atomic_write(path, |w| {
        sim.save(w)
            .map_err(|e| anyhow!("saving world `{path}`: {e}"))
    })
}

/// Load a metric sidecar: `(report_every, recorder)`, serialized as a CBOR
/// 2-tuple by [`save_sidecar`].
pub fn load_sidecar(path: &str) -> Result<(u64, Recorder)> {
    let file = File::open(path).map_err(|e| anyhow!("cannot open metrics `{path}`: {e}"))?;
    let (report_every, recorder): (u64, Recorder) = ciborium::from_reader(BufReader::new(file))
        .map_err(|e| anyhow!("loading metrics `{path}`: {e}"))?;
    if report_every != recorder.report_every {
        anyhow::bail!(
            "metrics `{path}` has inconsistent report cadence: envelope={report_every}, recorder={}",
            recorder.report_every
        );
    }
    Ok((report_every, recorder))
}

pub fn save_sidecar(report_every: u64, recorder: &Recorder, path: &str) -> Result<()> {
    if report_every != recorder.report_every {
        anyhow::bail!(
            "refusing to save metrics `{path}` with inconsistent report cadence: envelope={report_every}, recorder={}",
            recorder.report_every
        );
    }
    atomic_write(path, |w| {
        // Serialize a borrowed `(report_every, recorder)` tuple — no clone of the
        // (large) recorder, and no hand-mirrored struct to keep in sync.
        ciborium::into_writer(&(report_every, recorder), w)
            .map_err(|e| anyhow!("saving metrics `{path}`: {e}"))
    })
}

/// Resolve the metric sidecar path: `None` when `no_metrics`; the explicit path
/// when given; otherwise the `<world>.metrics` sibling.
pub fn resolve_metrics_path(
    world_path: &str,
    explicit: Option<&str>,
    no_metrics: bool,
) -> Option<String> {
    if no_metrics {
        return None;
    }
    if let Some(p) = explicit {
        return Some(p.to_string());
    }
    Some(sibling_metrics_path(world_path))
}

/// The `<world>.metrics` sibling path for a world file.
pub fn sibling_metrics_path(world_path: &str) -> String {
    PathBuf::from(world_path)
        .with_extension("metrics")
        .to_string_lossy()
        .into_owned()
}

// ---------------------------------------------------------------------------
// Config construction
// ---------------------------------------------------------------------------

/// Build a `WorldConfig` from a config file path + inline `--set` overrides.
/// No overrides → the plain loader.
pub fn world_config_with_overrides(
    config_path: &str,
    sets: &[(String, String)],
) -> Result<WorldConfig> {
    if sets.is_empty() {
        return load_world_config_from_path(Path::new(config_path));
    }
    let world_raw = std::fs::read_to_string(config_path)
        .map_err(|e| anyhow!("reading config `{config_path}`: {e}"))?;
    let seed_genome_path = Path::new(config_path).with_file_name("seed_genome.toml");
    let seed_genome_raw = std::fs::read_to_string(&seed_genome_path)
        .map_err(|e| anyhow!("reading {}: {e}", seed_genome_path.display()))?;
    world_config_from_raw_overrides(&world_raw, &seed_genome_raw, sets)
}

/// The patch+reparse half of [`world_config_with_overrides`], taking raw TOML so
/// a caller (`sweep`) can read the config files once and reuse them across many
/// override-sets.
pub fn world_config_from_raw_overrides(
    world_raw: &str,
    seed_genome_raw: &str,
    sets: &[(String, String)],
) -> Result<WorldConfig> {
    let patched = apply_config_overrides(world_raw, sets)?;
    world_config_from_toml_parts(&patched, seed_genome_raw)
        .map_err(|e| anyhow!("config after --set failed schema validation: {e}"))
}

/// Patch a world-config TOML document with `key=value` overrides before parsing.
/// Keys are matched against any `[section]` table first (the eval config keys are
/// section-unique, e.g. `food_energy` under `[food]`), then the top level.
/// Values are coerced to int/float/bool, falling back to string.
fn apply_config_overrides(world_raw: &str, sets: &[(String, String)]) -> Result<String> {
    if sets.is_empty() {
        return Ok(world_raw.to_string());
    }
    let mut doc: toml::Table = world_raw
        .parse()
        .map_err(|e| anyhow!("config TOML parse failed: {e}"))?;
    for (key, val) in sets {
        let value = coerce_toml_value(val);
        if !set_in_tables(&mut doc, key, &value) {
            doc.insert(key.clone(), value);
        }
    }
    toml::to_string(&doc).map_err(|e| anyhow!("re-serializing config failed: {e}"))
}

/// Set `key` in the first `[section]` sub-table that already contains it.
/// Returns true if a home was found.
fn set_in_tables(doc: &mut toml::Table, key: &str, value: &toml::Value) -> bool {
    if doc.contains_key(key) {
        doc.insert(key.to_string(), value.clone());
        return true;
    }
    for (_, v) in doc.iter_mut() {
        if let toml::Value::Table(section) = v {
            if section.contains_key(key) {
                section.insert(key.to_string(), value.clone());
                return true;
            }
        }
    }
    false
}

fn coerce_toml_value(raw: &str) -> toml::Value {
    if let Ok(i) = raw.parse::<i64>() {
        return toml::Value::Integer(i);
    }
    if let Ok(f) = raw.parse::<f64>() {
        return toml::Value::Float(f);
    }
    if let Ok(b) = raw.parse::<bool>() {
        return toml::Value::Boolean(b);
    }
    toml::Value::String(raw.to_string())
}

// ---------------------------------------------------------------------------
// World construction + recording-aware advancement
// ---------------------------------------------------------------------------

/// Inputs to [`build_world`] — a config path plus the overrides `new` accepts.
pub struct NewWorldParams {
    pub config_path: String,
    pub seed: u64,
    pub report_every: u64,
    pub threads: Option<u32>,
    pub scale: Option<(u32, u32)>,
    pub sets: Vec<(String, String)>,
    /// Optional exact founder genomes (e.g. a NEAT champion snapshot).
    pub champion_pool: Vec<OrganismGenome>,
}

/// A freshly constructed world plus the derived flags a caller reports/persists.
pub struct BuiltWorld {
    pub sim: Simulation,
    pub report_every: u64,
    pub scaled: bool,
}

/// Construct a world from a config file + inline `--set` overrides + optional
/// `--scale`/`--threads`. Presentation (the `new` summary line) stays with the
/// caller; this only builds.
pub fn build_world(params: &NewWorldParams) -> Result<BuiltWorld> {
    let mut config = world_config_with_overrides(&params.config_path, &params.sets)?;
    if let Some(t) = params.threads {
        config.intent_parallel_threads = t;
    }
    if let Some((w, p)) = params.scale {
        config.world_width = w;
        config.num_organisms = p;
    }
    let mut sim =
        Simulation::new_with_champion_pool(config, params.seed, params.champion_pool.clone())
            .map_err(|e| anyhow!("{e}"))?;
    sim.set_experiment_scaled(params.scale.is_some());
    Ok(BuiltWorld {
        sim,
        report_every: params.report_every,
        scaled: params.scale.is_some(),
    })
}

/// Begin recording on a world: build a fresh ledger and back-register the live
/// population (founders exactly at turn 0, otherwise a partial window). Used when
/// `new` mints a sidecar and when a mutating command finds no existing sidecar.
pub fn start_recording(sim: &Simulation, report_every: u64) -> Recorder {
    assert!(report_every > 0, "report_every must be greater than zero");
    let mut ledger = Ledger::new();
    let started_turn = sim.turn();
    ledger.begin_behavior_recording(started_turn);
    if started_turn == 0 {
        register_founders(&mut ledger, sim.organisms());
    } else {
        register_existing(&mut ledger, sim.organisms());
    }
    Recorder {
        ledger,
        tick_summary: Vec::new(),
        behavior_intervals: Vec::new(),
        lifetimes: Vec::new(),
        samples: Vec::new(),
        report_every,
        recorded_through_turn: started_turn,
        started_turn,
    }
}

/// Advance `n` ticks. When a recorder is passed, each tick's delta is fed to the
/// ledger via the shared `ingest_tick`; otherwise this is the fast,
/// allocation-free scrub path (delta dropped).
pub fn advance(sim: &mut Simulation, mut recorder: Option<&mut Recorder>, n: u64) {
    for _ in 0..n {
        tick_recording(sim, recorder.as_deref_mut());
    }
}

/// Advance the simulation exactly one tick, feeding the delta to the recorder
/// ledger when one is present, and return the delta. This is the per-tick unit
/// behind [`advance`]; a resident driver (e.g. the server's live stream) uses
/// the returned delta to animate while still recording metrics identically.
pub fn tick_recording(sim: &mut Simulation, recorder: Option<&mut Recorder>) -> types::TickDelta {
    let delta = sim.tick();
    if let Some(rec) = recorder {
        let turn = sim.turn();
        let deaths = ingest_tick(&mut rec.ledger, turn, &delta, sim.action_records());
        rec.recorded_through_turn = turn;
        rec.lifetimes.extend(deaths);
        rec.tick_summary.push(TickSummaryRow {
            tick: turn,
            population: rec.ledger.population(),
        });
        if turn.is_multiple_of(rec.report_every) {
            rec.behavior_intervals
                .push(rec.ledger.take_behavior_interval(turn));
        }
        let org_deaths = delta
            .removed_positions
            .iter()
            .filter(|r| matches!(r.entity_id, EntityId::Organism(_)))
            .count() as u32;
        rec.samples.push(EcoSample {
            population: delta.metrics.organisms,
            food: sim.foods().len() as u32,
            deaths: org_deaths,
            starvations: delta.metrics.starvations_last_turn as u32,
            age_deaths: delta.metrics.age_deaths_last_turn as u32,
            predations: delta.metrics.predations_last_turn as u32,
            consumptions: delta.metrics.consumptions_last_turn as u32,
        });
    }
    delta
}

// ---------------------------------------------------------------------------
// Reads (formerly `impl App` methods in cli/src/main.rs)
// ---------------------------------------------------------------------------

pub fn turn(ctx: &ReadCtx, args: &[&str], out: &mut impl Write) -> Result<()> {
    let (fmt, _) = take_format(ctx.format, args);
    let t = ctx.sim.turn();
    if fmt.is_json() {
        writeln!(out, "{}", json!({ "turn": t })).map_err(Into::into)
    } else {
        writeln!(out, "turn = {t}").map_err(Into::into)
    }
}

pub fn pillars(ctx: &ReadCtx, args: &[&str], out: &mut impl Write) -> Result<()> {
    let (fmt, _) = take_format(ctx.format, args);
    let Some((p, n_intervals, partial)) = ctx.live_pillars() else {
        anyhow::bail!("recording is off; advance a world with a metric sidecar before `pillars`");
    };
    let intervals = ctx.live_intervals().unwrap_or_default();
    if fmt.is_json() {
        let mut v = pillars_value(&p, n_intervals, partial);
        v["scaled"] = json!(ctx.scaled);
        // All the granular data behind the scores: the full per-interval series
        // (each interval's sub-signals), so the windowed pillar means can be read
        // against their underlying trajectory.
        v["granular"] = json!({
            "report_every": ctx.report_every,
            "window_start_tick": p.window_start_tick,
            "window_end_tick": p.window_end_tick,
            "intervals": serde_json::to_value(&intervals).unwrap_or(serde_json::Value::Null),
        });
        return writeln!(out, "{v}").map_err(Into::into);
    }
    let tag = match (partial, ctx.scaled) {
        (true, true) => "  [PARTIAL window; scaled: non-canonical]",
        (true, false) => "  [PARTIAL window]",
        (false, true) => "  [scaled: non-canonical]",
        (false, false) => "",
    };
    writeln!(
        out,
        "pillars over ({}, {}]  ({n_intervals} interval(s), report_every={}){tag}",
        p.window_start_tick, p.window_end_tick, ctx.report_every
    )?;
    writeln!(
        out,
        "  foraging      plant_consumption_rate {}",
        opt(p.mean_plant_consumption_rate, 4)
    )?;
    writeln!(
        out,
        "  predation     prey_consumption_rate  {}",
        opt(p.mean_prey_consumption_rate, 4)
    )?;
    writeln!(
        out,
        "  intelligence  action_effectiveness {}  mi_sa {}",
        opt(p.mean_action_effectiveness, 4),
        opt(p.mean_mi_sa, 4),
    )?;
    writeln!(
        out,
        "  learning      learning_slope {}",
        opt(p.mean_learning_slope, 6),
    )?;
    // Granular per-interval series behind the scores (the window marked *).
    writeln!(out, "  granular intervals (tick: eff plant prey mi slope):")?;
    for m in &intervals {
        let in_window = m.tick > p.window_start_tick && m.tick <= p.window_end_tick;
        writeln!(
            out,
            "  {}{:>7}: {} {} {} {} {}",
            if in_window { "*" } else { " " },
            m.tick,
            opt(m.action_effectiveness, 3),
            opt(m.plant_consumption_rate, 4),
            opt(m.prey_consumption_rate, 4),
            opt(m.mi_sa, 3),
            opt(m.learning_slope, 6),
        )?;
    }
    Ok(())
}

pub fn state(ctx: &ReadCtx, args: &[&str], out: &mut impl Write) -> Result<()> {
    let (fmt, _) = take_format(ctx.format, args);
    let pillars = ctx.live_pillars();
    let scaled = ctx.scaled;
    let sim = ctx.sim;
    let orgs = sim.organisms();
    let pop = orgs.len();
    let descendants = orgs.iter().filter(|o| o.generation > 0).count();
    let m = sim.metrics().clone();
    let turn = sim.turn();

    let energy = Stats::of(&orgs.iter().map(|o| o.energy as f64).collect::<Vec<_>>());
    let age = Stats::of(&orgs.iter().map(|o| o.age_turns as f64).collect::<Vec<_>>());
    let gen = Stats::of(&orgs.iter().map(|o| o.generation as f64).collect::<Vec<_>>());

    let (plants, food_energy) = food_summary(sim);
    let food_tiles = sim.food_tile_count();
    let habitable_cells = sim.habitable_cell_count();
    let realized_food_tile_fraction = if habitable_cells == 0 {
        0.0
    } else {
        food_tiles as f64 / habitable_cells as f64
    };

    if fmt.is_json() {
        let mut v = json!({
            "turn": turn,
            "population": pop,
            "descendants": descendants,
            "founders": pop - descendants,
            "energy": energy.map(|s| s.json()),
            "age_turns": age.map(|s| s.json()),
            "generation": gen.map(|s| s.json()),
            "food": { "plants": plants, "total_energy": food_energy },
            "food_tiles": {
                "selected": food_tiles,
                "habitable_cells": habitable_cells,
                "realized_fraction": realized_food_tile_fraction,
                "configured_fraction": sim.config().food_tile_fraction,
            },
            "brain_dynamics": {
                "leaky_neurons_enabled": sim.config().leaky_neurons_enabled,
                "runtime_plasticity_enabled": sim.config().runtime_plasticity_enabled,
                "predation_enabled": sim.config().predation_enabled,
                "active_sensory_inputs": types::SensoryReceptor::active(sim.config().predation_enabled).count(),
                "active_actions_excluding_idle": types::ActionType::active(sim.config().predation_enabled).count(),
            },
            "last_turn": {
                "consumptions": m.consumptions_last_turn,
                "plant_consumptions": m.plant_consumptions_last_turn,
                "predations": m.predations_last_turn,
                "starvations": m.starvations_last_turn,
                "age_deaths": m.age_deaths_last_turn,
                "energy_ledger": m.energy_ledger_last_turn,
            },
            "totals": {
                "consumptions": m.total_consumptions,
                "plant_consumptions": m.total_plant_consumptions,
            },
            "scaled": scaled,
        });
        if let Some((p, n, partial)) = pillars {
            v["pillars"] = pillars_value(&p, n, partial);
        }
        return writeln!(out, "{v}").map_err(Into::into);
    }

    let fmt_stats = |s: Option<Stats>| s.map(|s| s.text()).unwrap_or_else(|| "(none)".into());
    writeln!(
        out,
        "turn = {turn}{}",
        if scaled {
            "  [scaled: non-canonical]"
        } else {
            ""
        }
    )?;
    writeln!(
        out,
        "population = {pop}  (descendants gen>0 = {descendants}, founders gen0 = {})",
        pop - descendants
    )?;
    writeln!(out, "  energy:     {}", fmt_stats(energy))?;
    writeln!(out, "  age_turns:  {}", fmt_stats(age))?;
    writeln!(out, "  generation: {}", fmt_stats(gen))?;
    writeln!(out, "food: plants={plants} total_energy={food_energy:.0}")?;
    writeln!(
        out,
        "last-turn: consumptions={} (plant={}) predations={} starvations={} age_deaths={}",
        m.consumptions_last_turn,
        m.plant_consumptions_last_turn,
        m.predations_last_turn,
        m.starvations_last_turn,
        m.age_deaths_last_turn,
    )?;
    let ledger = m.energy_ledger_last_turn;
    writeln!(
        out,
        "  energy-ledger: org={:.0}->{:.0} food={:.0}->{:.0} plant_spawn={:.0} tick_drain={:.0} attack_debit={:.0} attack_credit={:.0} food_residual={:.3e} attack_residual={:.3e} total_residual={:.3e} tol={:.1e}",
        ledger.organism_energy_before,
        ledger.organism_energy_after,
        ledger.food_energy_before,
        ledger.food_energy_after,
        ledger.plant_spawn_energy,
        ledger.tick_drain_energy,
        ledger.attack_transfer_debit,
        ledger.attack_transfer_credit,
        ledger.food_transfer_residual,
        ledger.attack_transfer_residual,
        ledger.total_residual,
        ledger.residual_tolerance,
    )?;
    writeln!(
        out,
        "cumulative: consumptions={} (plant={})",
        m.total_consumptions, m.total_plant_consumptions,
    )?;
    if let Some((p, _, partial)) = pillars {
        writeln!(
            out,
            "metrics: plant_rate {} | prey_rate {} | action_eff {} | mi_sa {} | learn_slope {}{}",
            opt(p.mean_plant_consumption_rate, 4),
            opt(p.mean_prey_consumption_rate, 4),
            opt(p.mean_action_effectiveness, 4),
            opt(p.mean_mi_sa, 4),
            opt(p.mean_learning_slope, 6),
            if partial { "  [PARTIAL]" } else { "" },
        )?;
    }
    Ok(())
}

pub fn food(ctx: &ReadCtx, args: &[&str], out: &mut impl Write) -> Result<()> {
    let (fmt, _) = take_format(ctx.format, args);
    let sim = ctx.sim;
    let (plants, energy) = food_summary(sim);
    let total = plants;
    let cells = (sim.config().world_width as u64).pow(2);
    let coverage = total as f64 / cells as f64 * 100.0;
    if fmt.is_json() {
        return writeln!(
            out,
            "{}",
            json!({
                "plants": plants,
                "total": total,
                "total_energy": energy,
                "coverage_pct": coverage,
                "cells": cells,
            })
        )
        .map_err(Into::into);
    }
    writeln!(
        out,
        "food: plants={plants} total={total} total_energy={energy:.0} \
         coverage={coverage:.3}% of {cells} cells"
    )
    .map_err(Into::into)
}

pub fn inspect(ctx: &ReadCtx, args: &[&str], out: &mut impl Write) -> Result<()> {
    let (fmt, rest) = take_format(ctx.format, args);
    let id: u64 = rest
        .first()
        .ok_or_else(|| anyhow!("inspect needs an organism id"))?
        .parse()?;
    if let Some(arg) = rest.get(1) {
        anyhow::bail!("unknown inspect arg `{arg}`");
    }
    let sim = ctx.sim;
    let idx = sim
        .organisms()
        .iter()
        .position(|o| o.id.0 == id)
        .ok_or_else(|| anyhow!("no live organism with id {id}"))?;
    let rec = sim.action_records().get(idx).cloned().flatten();
    let o = &sim.organisms()[idx];
    if fmt.is_json() {
        let action_logits: Vec<_> = o
            .brain
            .action
            .iter()
            .map(|a| json!({ "action": a.action_type, "logit": a.logit }))
            .collect();
        let value = json!({
            "id": o.id.0,
            "position": { "q": o.q, "r": o.r },
            "facing": o.facing,
            "energy": o.energy,
            "energy_flow_last_tick": o.energy_flow_last_tick,
            "age": o.age_turns,
            "generation": o.generation,
            "species": o.species_id.0,
            "last_action": o.last_action_taken,
            "consumptions": {
                "total": o.consumptions_count,
                "plant": o.plant_consumptions_count,
                "prey": o.prey_consumptions_count,
            },
            "genome": {
                "hidden_nodes": o.genome.hidden_node_count(),
                "synapses": o.brain.synapse_count,
                "vision_range": sim.config().vision_range,
                "hebb_eta": o.genome.plasticity.hebb_eta_gain,
                "juvenile_eta_scale": o.genome.plasticity.juvenile_eta_scale,
                "eligibility_retention": o.genome.plasticity.eligibility_retention,
                "max_weight_delta_per_tick": o.genome.plasticity.max_weight_delta_per_tick,
                "synapse_prune_threshold": o.genome.plasticity.synapse_prune_threshold,
                "plasticity_maturity_ticks": o.genome.lifecycle.plasticity_maturity_ticks,
            },
            "action_logits": action_logits,
            "last_decision": rec,
        });
        return writeln!(out, "{value}").map_err(Into::into);
    }
    writeln!(out, "organism {id}:")?;
    writeln!(
        out,
        "  pos=({}, {}) facing={:?} energy={} energy_flow={} age={} gen={} species={}",
        o.q,
        o.r,
        o.facing,
        o.energy,
        o.energy_flow_last_tick,
        o.age_turns,
        o.generation,
        o.species_id.0
    )?;
    writeln!(
        out,
        "  last_action={:?} consumptions={} (plant={}, prey={})",
        o.last_action_taken,
        o.consumptions_count,
        o.plant_consumptions_count,
        o.prey_consumptions_count,
    )?;
    let g = &o.genome;
    writeln!(
        out,
        "  genome: num_neurons={} synapses={} vision_range={} | hebb_eta={:.4} juv_scale={:.3} elig_ret={:.3} max_dw={:.4} prune={:.4} | plasticity_maturity={}",
        g.hidden_node_count(),
        o.brain.synapse_count,
        sim.config().vision_range,
        g.plasticity.hebb_eta_gain,
        g.plasticity.juvenile_eta_scale,
        g.plasticity.eligibility_retention,
        g.plasticity.max_weight_delta_per_tick,
        g.plasticity.synapse_prune_threshold,
        g.lifecycle.plasticity_maturity_ticks,
    )?;
    write!(out, "  action logits:")?;
    for a in &o.brain.action {
        write!(out, " {:?}={:.3}", a.action_type, a.logit)?;
    }
    writeln!(out)?;
    if let Some(rec) = rec {
        writeln!(
            out,
            "  this-tick: selected={:?} failed={} food_visible(rays -1/0/+1)={:?} utilization={:.3}",
            rec.selected_action, rec.action_failed, rec.food_visible, rec.utilization
        )?;
    }
    Ok(())
}

pub fn top(ctx: &ReadCtx, args: &[&str], out: &mut impl Write) -> Result<()> {
    let (fmt, rest) = take_format(ctx.format, args);
    let field = *rest.first().ok_or_else(|| anyhow!("top needs a field"))?;
    if !matches!(
        field,
        "energy" | "energy_flow" | "age" | "generation" | "consumptions"
    ) {
        anyhow::bail!("unknown field `{field}` (energy|energy_flow|age|generation|consumptions)");
    }
    let n: usize = rest.get(1).map(|s| s.parse()).transpose()?.unwrap_or(10);
    if let Some(arg) = rest.get(2) {
        anyhow::bail!("unknown top arg `{arg}`");
    }
    let sim = ctx.sim;
    let mut idx: Vec<usize> = (0..sim.organisms().len()).collect();
    let key = |o: &types::OrganismState| -> f64 {
        match field {
            "energy" => o.energy as f64,
            "energy_flow" => o.energy_flow_last_tick as f64,
            "age" => o.age_turns as f64,
            "generation" => o.generation as f64,
            "consumptions" => o.consumptions_count as f64,
            _ => f64::NAN,
        }
    };
    let orgs = sim.organisms();
    idx.sort_by(|&a, &b| {
        key(&orgs[b])
            .total_cmp(&key(&orgs[a]))
            .then_with(|| orgs[a].id.cmp(&orgs[b].id))
    });
    if fmt.is_json() {
        let rows: Vec<_> = idx
            .iter()
            .take(n)
            .map(|&i| {
                let o = &orgs[i];
                json!({
                    "id": o.id.0,
                    "value": key(o),
                    "energy": o.energy,
                    "age": o.age_turns,
                    "generation": o.generation,
                    "consumptions": o.consumptions_count,
                    "last_action": o.last_action_taken,
                })
            })
            .collect();
        return writeln!(
            out,
            "{}",
            json!({ "field": field, "requested": n, "rows": rows })
        )
        .map_err(Into::into);
    }
    writeln!(out, "top {n} by {field}:")?;
    for &i in idx.iter().take(n) {
        let o = &orgs[i];
        writeln!(
            out,
            "  id={:<6} {field}={:<12.3} energy={:.1} age={} gen={} consum={} last={:?}",
            o.id.0,
            key(o),
            o.energy,
            o.age_turns,
            o.generation,
            o.consumptions_count,
            o.last_action_taken
        )?;
    }
    Ok(())
}

pub fn hist(ctx: &ReadCtx, args: &[&str], out: &mut impl Write) -> Result<()> {
    let (fmt, rest) = take_format(ctx.format, args);
    let field = *rest.first().ok_or_else(|| anyhow!("hist needs a field"))?;
    if let Some(arg) = rest.get(1) {
        anyhow::bail!("unknown hist arg `{arg}`");
    }
    let sim = ctx.sim;
    let vals: Vec<f64> = sim
        .organisms()
        .iter()
        .map(|o| match field {
            "energy" => o.energy as f64,
            "energy_flow" => o.energy_flow_last_tick as f64,
            "age" => o.age_turns as f64,
            "generation" => o.generation as f64,
            _ => f64::NAN,
        })
        .collect();
    if vals.iter().any(|v| v.is_nan()) {
        anyhow::bail!("unknown field `{field}` (energy|energy_flow|age|generation)");
    }
    if vals.is_empty() {
        if fmt.is_json() {
            return writeln!(out, "{}", json!({ "field": field, "n": 0, "bins": [] }))
                .map_err(Into::into);
        }
        writeln!(out, "(no organisms)")?;
        return Ok(());
    }
    let min = vals.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    if min == max {
        if fmt.is_json() {
            return writeln!(
                out,
                "{}",
                json!({
                    "field": field,
                    "n": vals.len(),
                    "min": min,
                    "max": max,
                    "bins": [{ "lo": min, "hi": max, "count": vals.len(), "inclusive": true }],
                })
            )
            .map_err(Into::into);
        }
        writeln!(out, "{field} histogram ({} organisms):", vals.len())?;
        writeln!(out, "  [{min:>9.1}] {:>8} {}", vals.len(), "#".repeat(40))?;
        return Ok(());
    }
    let bins = 10usize;
    let width = ((max - min) / bins as f64).max(f64::MIN_POSITIVE);
    let mut counts = vec![0u64; bins];
    for v in &vals {
        let b = (((v - min) / width) as usize).min(bins - 1);
        counts[b] += 1;
    }
    let peak = counts.iter().copied().max().unwrap_or(1).max(1);
    if fmt.is_json() {
        let rendered: Vec<_> = counts
            .iter()
            .enumerate()
            .map(|(i, count)| {
                let lo = min + i as f64 * width;
                let hi = lo + width;
                json!({ "lo": lo, "hi": hi, "count": count })
            })
            .collect();
        return writeln!(
            out,
            "{}",
            json!({ "field": field, "n": vals.len(), "min": min, "max": max, "bins": rendered })
        )
        .map_err(Into::into);
    }
    writeln!(out, "{field} histogram ({} organisms):", vals.len())?;
    for (i, c) in counts.iter().enumerate() {
        let lo = min + i as f64 * width;
        let hi = lo + width;
        let bar = "#".repeat((c * 40 / peak) as usize);
        writeln!(out, "  [{lo:>9.1}, {hi:>9.1}) {c:>8} {bar}")?;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Shared read helpers
// ---------------------------------------------------------------------------

/// Shared JSON encoding of the windowed metric readout (used by `pillars` and
/// `state`). Raw values only — no [0,1] pillar scoring.
pub(crate) fn pillars_value(
    p: &PillarScores,
    n_intervals: usize,
    partial: bool,
) -> serde_json::Value {
    json!({
        "window_start_tick": p.window_start_tick,
        "window_end_tick": p.window_end_tick,
        "intervals": n_intervals,
        "partial": partial,
        "plant_consumption_rate": opt_json(p.mean_plant_consumption_rate),
        "prey_consumption_rate": opt_json(p.mean_prey_consumption_rate),
        "action_effectiveness": opt_json(p.mean_action_effectiveness),
        "mi_sa": opt_json(p.mean_mi_sa),
        "learning_slope": opt_json(p.mean_learning_slope),
    })
}

/// Plant count and total standing plant energy across the world.
pub fn food_summary(sim: &Simulation) -> (u64, f64) {
    let plants = sim.foods().len() as u64;
    let energy = sim.foods().iter().map(|food| food.energy as f64).sum();
    (plants, energy)
}
