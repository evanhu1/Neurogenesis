//! sim-cli — headless, stateful research client for the NeuroGenesis engine.
//!
//! Reads commands from stdin (one per line), holding a single `Simulation` in
//! memory so the world can be scrubbed forward and inspected repeatedly without
//! rebuilding. Live metrics (pillars, ecology, lineage, genome drift) are
//! computed from the shared `sim-metrics` crate, so they match the offline eval
//! byte-for-byte. Output defaults to compact text; `--json` (or the `format`
//! command) emits machine-parseable JSON.
//!
//! See docs/sim-cli-design.md and SPEC.md.

mod dashboards;
mod inspect_ext;
mod output;
mod sweep;

use anyhow::{anyhow, bail, Result};
use output::{opt, opt_json, Format, Stats};
use serde_json::json;
use sim_config::{load_world_config_from_path, world_config_from_toml_parts};
use sim_core::Simulation;
use sim_metrics::{
    compute_pillar_scores, derive_interval_metrics, ingest_tick, register_existing,
    register_founders, Ledger, OrganismLifetimeRow, PillarScores, TickSummaryRow,
};
use sim_types::EntityId;
use std::fs::File;
use std::io::{self, BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};

const DEFAULT_CONFIG: &str = "sim-evaluation/config.toml";
/// Default reporting-interval width (matches the eval; override at `new`).
const REPORT_EVERY: u64 = 10_000;
/// Default directory for run-mode result files (sweep, etc.). Override with the
/// global `--out-dir` flag. Under `artifacts/` so results survive a session.
const DEFAULT_OUT_DIR: &str = "artifacts/runs";

/// Build a timestamped result-file path under `out_dir`, creating the directory.
/// Run modes (sweep, …) write their result artifact here so output is durable
/// and discoverable rather than scrolled past in stdout.
fn run_output_path(out_dir: &str, prefix: &str) -> Result<PathBuf> {
    std::fs::create_dir_all(out_dir)
        .map_err(|e| anyhow!("cannot create out-dir `{out_dir}`: {e}"))?;
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0);
    Ok(PathBuf::from(out_dir).join(format!("{prefix}-{stamp}.json")))
}

/// Commands that advance / create a world and therefore persist it (`--out`,
/// defaulting to `--in`). Everything else is a pure read.
fn is_mutating(cmd: &str) -> bool {
    matches!(cmd, "new" | "step" | "run-to" | "watch")
}

/// Pure-read commands: read the loaded world, write only stdout, never `--out`.
/// These are the commands allowed inside `query` batch mode.
fn is_read_only(cmd: &str) -> bool {
    matches!(
        cmd,
        "turn"
            | "pillars"
            | "state"
            | "eco"
            | "lineage"
            | "genome"
            | "timeseries"
            | "food"
            | "inspect"
            | "top"
            | "hist"
            | "find"
            | "brain"
            | "decide"
    )
}

fn print_help(out: &mut impl Write) -> Result<()> {
    writeln!(
        out,
        "sim-cli — stateless world-as-file research CLI (agent-facing). JSON output by default; `--text` to override.\n\
         \n\
         WORLD I/O\n\
         \x20 --in <world.bin>     read a world (required by every command except `new`)\n\
         \x20 --out <world.bin>    write the advanced world (mutating cmds; defaults to --in = advance in place)\n\
         \x20 --metrics <path>     metric sidecar (defaults to the `<world>.metrics` sibling; --no-metrics to disable)\n\
         \n\
         MUTATING (persist the world)\n\
         \x20 new [--config P] [--seed N] [--set k=v]... [--scale W,POP] [--threads K] [--report-every R] --out w.bin\n\
         \x20 step [N] --in w.bin [--out w.bin]            advance N ticks (default 1)\n\
         \x20 run-to T --in w.bin [--out w.bin]            advance until turn == T\n\
         \x20 watch T [--every E] --in w.bin [--out w.bin] advance to T, emitting a metrics row every E ticks\n\
         \x20 bench [N] --in w.bin                         time N ticks (world discarded)\n\
         \n\
         READS (stdout only)\n\
         \x20 turn | state | pillars | eco | lineage | genome [--gene G] | food --in w.bin\n\
         \x20 timeseries [--cols LIST] [--last K] --in w.bin\n\
         \x20 inspect ID | top FIELD [N] | hist FIELD | find EXPR | brain ID [--view V] | decide ID --in w.bin\n\
         \x20 query --in w.bin                             read-only commands from stdin (one per line), one load\n\
         \n\
         pillars/eco-trajectory/timeseries need a metric sidecar (minted by `new`, follows the world).\n\
         Snapshot/fork a world with `cp`; full reference in docs/sim-cli.md."
    )
    .map_err(Into::into)
}

/// On-disk metric sidecar: the recorder accumulators plus the `report_every`
/// the interval layer needs. Lives beside the world as `<world>.metrics`.
#[derive(serde::Serialize, serde::Deserialize)]
struct Sidecar {
    report_every: u64,
    recorder: Recorder,
}

fn main() {
    if let Err(e) = run() {
        // Structured, single-line error for the agent driving us over the shell.
        eprintln!("{}", json!({ "error": e.to_string() }));
        std::process::exit(1);
    }
}

/// One-shot entry point. Parses `argv` into `<command> [global flags] [command
/// args]`, loads the world (and auto-following metric sidecar), runs the single
/// command, then persists the world + sidecar for mutating commands.
fn run() -> Result<()> {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let cmd: String = argv.first().cloned().unwrap_or_default();
    if cmd.is_empty() || cmd == "help" || cmd == "--help" || cmd == "-h" {
        let mut out = io::stdout().lock();
        print_help(&mut out)?;
        return Ok(());
    }
    let cmd = cmd.as_str();

    // Pull global flags (orthogonal to every command) out of the arg stream.
    let mut in_path: Option<String> = None;
    let mut out_path: Option<String> = None;
    let mut metrics_flag: Option<String> = None;
    let mut out_dir: String = DEFAULT_OUT_DIR.to_string();
    let mut no_metrics = false;
    let mut rest: Vec<String> = Vec::new();
    let mut it = argv.into_iter().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--in" => in_path = Some(it.next().ok_or_else(|| anyhow!("--in needs a path"))?),
            "--out" => out_path = Some(it.next().ok_or_else(|| anyhow!("--out needs a path"))?),
            "--metrics" => {
                metrics_flag = Some(it.next().ok_or_else(|| anyhow!("--metrics needs a path"))?)
            }
            "--out-dir" => {
                out_dir = it.next().ok_or_else(|| anyhow!("--out-dir needs a path"))?
            }
            "--no-metrics" => no_metrics = true,
            _ => rest.push(a),
        }
    }
    let cmd_args: Vec<&str> = rest.iter().map(String::as_str).collect();

    // Run modes that own their world(s) and emit a result file, not --in/--out.
    if cmd == "sweep" {
        let mut out = io::stdout().lock();
        return sweep::run_sweep(&cmd_args, &out_dir, &mut out);
    }

    let mut app = App {
        sim: None,
        recorder: None,
        report_every: REPORT_EVERY,
        format: Format::Json, // JSON by default for the agent CLI; `--text` overrides.
        scaled: false,
    };
    let mut out = io::stdout().lock();
    let mutating = is_mutating(cmd);

    if cmd == "new" {
        // Constructor: build the world from config + overrides; mint a sidecar
        // unless suppressed. Persisted below.
        app.build_world(&cmd_args, &mut out)?;
        if !no_metrics {
            app.start_recording()?;
        }
    } else {
        let world_path = in_path
            .clone()
            .ok_or_else(|| anyhow!("`{cmd}` needs --in <world.bin>"))?;
        app.sim = Some(load_world(&world_path)?);

        // The metric sidecar follows the world: explicit --metrics, else the
        // `<world>.metrics` sibling if present. Mutating commands without an
        // existing sidecar mint a fresh one (back-registering live organisms)
        // so recording is on by default; --no-metrics opts out.
        let metrics_path = resolve_metrics_path(&world_path, metrics_flag.as_deref(), no_metrics);
        if let Some(mp) = metrics_path.as_ref() {
            if Path::new(mp).exists() {
                let sidecar = load_sidecar(mp)?;
                app.report_every = sidecar.report_every;
                app.recorder = Some(sidecar.recorder);
            } else if mutating {
                app.start_recording()?;
            }
        }

        app.run_oneshot(cmd, &cmd_args, &mut out)?;
        out.flush()?;

        if mutating {
            let dest = out_path.clone().unwrap_or(world_path.clone());
            save_world(app.sim.as_ref().expect("world loaded"), &dest)?;
            if let (Some(rec), Some(mp)) = (app.recorder.as_ref(), metrics_path.as_ref()) {
                // Sidecar follows the written world (so `--out fork.bin` forks
                // its metrics to `fork.metrics` too).
                let mp = if out_path.is_some() {
                    resolve_metrics_path(&dest, metrics_flag.as_deref(), no_metrics)
                        .unwrap_or_else(|| mp.clone())
                } else {
                    mp.clone()
                };
                save_sidecar(&app, rec, &mp)?;
            }
        }
        return Ok(());
    }

    // `new` persistence (world + minted sidecar).
    out.flush()?;
    let dest = out_path
        .clone()
        .ok_or_else(|| anyhow!("`new` needs --out <world.bin>"))?;
    save_world(app.sim.as_ref().expect("world built"), &dest)?;
    if let Some(rec) = app.recorder.as_ref() {
        let mp = resolve_metrics_path(&dest, metrics_flag.as_deref(), no_metrics)
            .expect("recorder present implies metrics enabled");
        save_sidecar(&app, rec, &mp)?;
    }
    Ok(())
}

/// Resolve the metric sidecar path: `None` when `--no-metrics`; the explicit
/// `--metrics` path when given; otherwise the `<world>.metrics` sibling.
fn resolve_metrics_path(world_path: &str, explicit: Option<&str>, no_metrics: bool) -> Option<String> {
    if no_metrics {
        return None;
    }
    if let Some(p) = explicit {
        return Some(p.to_string());
    }
    Some(
        PathBuf::from(world_path)
            .with_extension("metrics")
            .to_string_lossy()
            .into_owned(),
    )
}

fn load_world(path: &str) -> Result<Simulation> {
    let file =
        File::open(path).map_err(|e| anyhow!("cannot open world `{path}`: {e}"))?;
    Simulation::load(BufReader::new(file)).map_err(|e| anyhow!("loading world `{path}`: {e}"))
}

fn save_world(sim: &Simulation, path: &str) -> Result<()> {
    if let Some(parent) = Path::new(path).parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent).ok();
        }
    }
    let file = File::create(path).map_err(|e| anyhow!("cannot write world `{path}`: {e}"))?;
    sim.save(BufWriter::new(file))
        .map_err(|e| anyhow!("saving world `{path}`: {e}"))
}

fn load_sidecar(path: &str) -> Result<Sidecar> {
    let file =
        File::open(path).map_err(|e| anyhow!("cannot open metrics `{path}`: {e}"))?;
    ciborium::from_reader(BufReader::new(file))
        .map_err(|e| anyhow!("loading metrics `{path}`: {e}"))
}

fn save_sidecar(app: &App, recorder: &Recorder, path: &str) -> Result<()> {
    if let Some(parent) = Path::new(path).parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent).ok();
        }
    }
    let sidecar = SidecarRef {
        report_every: app.report_every,
        recorder,
    };
    let file = File::create(path).map_err(|e| anyhow!("cannot write metrics `{path}`: {e}"))?;
    ciborium::into_writer(&sidecar, BufWriter::new(file))
        .map_err(|e| anyhow!("saving metrics `{path}`: {e}"))
}

/// Borrowed mirror of [`Sidecar`] so we can serialize without cloning the
/// recorder. Field order/names must match `Sidecar`.
#[derive(serde::Serialize)]
struct SidecarRef<'a> {
    report_every: u64,
    recorder: &'a Recorder,
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

struct App {
    sim: Option<Simulation>,
    /// Live metric accumulator; `Some` only while `record on`. Built on the
    /// shared `sim-metrics` ledger so live pillars match the eval byte-for-byte.
    recorder: Option<Recorder>,
    /// Reporting-interval width used for pillar windowing (eval default 10k;
    /// override at `load --report-every`).
    report_every: u64,
    /// Global default output format; per-command `--json`/`--text` override it.
    format: Format,
    /// True when the loaded world used `--scale` (or otherwise diverges from the
    /// canonical eval config), so metrics are not eval-comparable.
    scaled: bool,
}

/// In-memory metric recorder. Holds the shared per-organism ledger plus the two
/// raw fact streams the analysis layer consumes (`derive_interval_metrics`):
/// the per-tick descendant-population line and the lifetime rows of organisms
/// that died while recording. Survivors are intentionally absent — the interval
/// layer skips `death_tick == None` rows, exactly as the eval does.
#[derive(serde::Serialize, serde::Deserialize)]
struct Recorder {
    ledger: Ledger,
    tick_summary: Vec<TickSummaryRow>,
    lifetimes: Vec<OrganismLifetimeRow>,
    /// Richer per-recorded-tick ecology samples for `eco`/`timeseries`
    /// trajectories (the metric `tick_summary` only carries descendant pop).
    samples: Vec<EcoSample>,
    /// Turn at which recording began. `> 0` means already-alive organisms were
    /// back-registered with partial lifetime counters (windows are approximate).
    started_turn: u64,
}

/// One per recorded tick: a cheap, whole-world ecology snapshot from the
/// `TickDelta` metrics (no extra organism scans), powering `eco`/`timeseries`.
#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub(crate) struct EcoSample {
    pub population: u32,
    pub descendants: u32,
    pub food: u32,
    pub births: u32,
    pub deaths: u32,
    pub starvations: u32,
    pub age_deaths: u32,
    pub predations: u32,
    pub consumptions: u32,
    pub reproductions: u32,
}

impl App {
    fn sim(&mut self) -> Result<&mut Simulation> {
        self.sim
            .as_mut()
            .ok_or_else(|| anyhow!("no simulation loaded; run `load` first"))
    }

    /// Advance `n` ticks. When recording is active each tick's delta is fed to
    /// the ledger via the shared `ingest_tick`; otherwise this is the fast,
    /// allocation-free scrub path (delta dropped).
    fn advance(&mut self, n: u64) -> Result<()> {
        let sim = self
            .sim
            .as_mut()
            .ok_or_else(|| anyhow!("no simulation loaded; run `load` first"))?;
        match self.recorder.as_mut() {
            None => {
                for _ in 0..n {
                    sim.tick();
                }
            }
            Some(rec) => {
                for _ in 0..n {
                    let delta = sim.tick();
                    let turn = sim.turn();
                    let deaths = ingest_tick(&mut rec.ledger, turn, &delta, sim.action_records());
                    rec.lifetimes.extend(deaths);
                    let descendants = rec.ledger.descendant_population();
                    rec.tick_summary.push(TickSummaryRow {
                        tick: turn,
                        descendant_population: descendants,
                    });
                    let org_deaths = delta
                        .removed_positions
                        .iter()
                        .filter(|r| matches!(r.entity_id, EntityId::Organism(_)))
                        .count() as u32;
                    rec.samples.push(EcoSample {
                        population: delta.metrics.organisms,
                        descendants,
                        food: sim.foods().len() as u32,
                        births: delta.spawned.len() as u32,
                        deaths: org_deaths,
                        starvations: delta.metrics.starvations_last_turn as u32,
                        age_deaths: delta.metrics.age_deaths_last_turn as u32,
                        predations: delta.metrics.predations_last_turn as u32,
                        consumptions: delta.metrics.consumptions_last_turn as u32,
                        reproductions: delta.metrics.reproductions_last_turn as u32,
                    });
                }
            }
        }
        Ok(())
    }

    /// Compute live pillars over the recorded span using the exact `sim-metrics`
    /// pipeline. Returns the scores, the interval count, and whether the window
    /// is partial (recording started after turn 0). `None` when not recording.
    fn live_pillars(&self) -> Option<(PillarScores, usize, bool)> {
        let rec = self.recorder.as_ref()?;
        let total_ticks = self.sim.as_ref()?.turn();
        let intervals = derive_interval_metrics(
            &rec.tick_summary,
            &rec.lifetimes,
            self.report_every,
            total_ticks,
        );
        let scores = compute_pillar_scores(&intervals);
        Some((scores, intervals.len(), rec.started_turn > 0))
    }

    /// The full per-interval metric series behind the pillar scores — the
    /// granular data `pillars` reports alongside the windowed means.
    fn live_intervals(&self) -> Option<Vec<sim_metrics::IntervalMetrics>> {
        let rec = self.recorder.as_ref()?;
        let total_ticks = self.sim.as_ref()?.turn();
        Some(derive_interval_metrics(
            &rec.tick_summary,
            &rec.lifetimes,
            self.report_every,
            total_ticks,
        ))
    }

    /// Run one command against the already-loaded world. `new` is handled by
    /// the orchestrator (it constructs rather than reads), so it is not here.
    fn run_oneshot(&mut self, cmd: &str, args: &[&str], out: &mut impl Write) -> Result<()> {
        if is_read_only(cmd) {
            return self.run_read(cmd, args, out);
        }
        match cmd {
            "step" => self.step(args, out),
            "run-to" => self.run_to(args, out),
            "bench" => self.bench(args, out),
            "watch" => self.watch(args, out),
            "query" => self.query(out),
            other => bail!("unknown command `{other}` (try `help`)"),
        }
    }

    /// Dispatch a pure-read command (no world mutation, no `--out`).
    fn run_read(&mut self, cmd: &str, args: &[&str], out: &mut impl Write) -> Result<()> {
        match cmd {
            "turn" => {
                let (fmt, _) = self.take_format(args);
                let t = self.sim()?.turn();
                if fmt.is_json() {
                    writeln!(out, "{}", json!({ "turn": t })).map_err(Into::into)
                } else {
                    writeln!(out, "turn = {t}").map_err(Into::into)
                }
            }
            "pillars" => self.pillars(args, out),
            "state" => self.state(args, out),
            "eco" => self.eco(args, out),
            "lineage" => self.lineage(args, out),
            "genome" => self.genome(args, out),
            "timeseries" => self.timeseries(args, out),
            "food" => self.food(args, out),
            "inspect" => self.inspect(args, out),
            "top" => self.top(args, out),
            "hist" => self.hist(args, out),
            "find" => self.find(args, out),
            "brain" => self.brain(args, out),
            "decide" => self.decide(args, out),
            other => bail!("unknown read command `{other}` (try `help`)"),
        }
    }

    /// Batch-read mode: load the world once, then run many pure-read commands
    /// from stdin (one per line), so a burst of probes pays a single
    /// deserialize. Each line emits its own result; a failing line emits an
    /// `{"error": …}` line and the batch continues. Mutating commands are
    /// rejected (use a real `step`/`run-to` invocation for those).
    fn query(&mut self, out: &mut impl Write) -> Result<()> {
        let stdin = io::stdin();
        for line in stdin.lock().lines() {
            let line = line?;
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let mut parts = line.split_whitespace();
            let sub = parts.next().unwrap_or("");
            let sub_args: Vec<&str> = parts.collect();
            let result = if is_read_only(sub) {
                self.run_read(sub, &sub_args, out)
            } else {
                Err(anyhow!("`{sub}` is not a read command (not allowed in query)"))
            };
            if let Err(e) = result {
                writeln!(out, "{}", json!({ "error": e.to_string(), "command": line }))?;
            }
            out.flush()?;
        }
        Ok(())
    }

    /// Strip `--json` / `--text` from `args`, returning the effective format
    /// (per-command flag overriding the session default) and the rest.
    fn take_format<'a>(&self, args: &[&'a str]) -> (Format, Vec<&'a str>) {
        let mut fmt = self.format;
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

    // (REPL-era `record`/`format`/`help` removed: recording is now the metric
    // sidecar's presence, format is the global `--json`/`--text` flag, and help
    // is the free `print_help`.)

    /// `new`: construct a world from a config file + inline `--set k=v`
    /// overrides + optional `--scale`/`--threads`. Sets `self.sim` and reports;
    /// the orchestrator persists it to `--out`.
    fn build_world(&mut self, args: &[&str], out: &mut impl Write) -> Result<()> {
        let mut config_path = DEFAULT_CONFIG.to_string();
        let mut seed: u64 = 0;
        let mut report_every: u64 = REPORT_EVERY;
        let mut threads: Option<u32> = None;
        let mut scale: Option<(u32, u32)> = None;
        let mut sets: Vec<(String, String)> = Vec::new();
        let (fmt, args) = self.take_format(args);
        let mut i = 0;
        while i < args.len() {
            match args[i] {
                "--config" => {
                    config_path = args
                        .get(i + 1)
                        .ok_or_else(|| anyhow!("--config needs a path"))?
                        .to_string();
                    i += 2;
                }
                "--seed" => {
                    seed = args
                        .get(i + 1)
                        .ok_or_else(|| anyhow!("--seed needs a value"))?
                        .parse()?;
                    i += 2;
                }
                "--report-every" => {
                    report_every = args
                        .get(i + 1)
                        .ok_or_else(|| anyhow!("--report-every needs a value"))?
                        .parse()?;
                    if report_every == 0 {
                        bail!("--report-every must be >= 1");
                    }
                    i += 2;
                }
                "--threads" => {
                    let t: u32 = args
                        .get(i + 1)
                        .ok_or_else(|| anyhow!("--threads needs a value"))?
                        .parse()?;
                    if t == 0 {
                        bail!("--threads must be >= 1");
                    }
                    threads = Some(t);
                    i += 2;
                }
                "--scale" => {
                    let spec = args
                        .get(i + 1)
                        .ok_or_else(|| anyhow!("--scale needs WIDTH,POP"))?;
                    let (w, p) = spec
                        .split_once(',')
                        .ok_or_else(|| anyhow!("--scale wants WIDTH,POP (e.g. 60,300)"))?;
                    let w: u32 = w.trim().parse().map_err(|_| anyhow!("bad --scale width"))?;
                    let p: u32 = p.trim().parse().map_err(|_| anyhow!("bad --scale pop"))?;
                    if w == 0 || p == 0 {
                        bail!("--scale width and pop must be >= 1");
                    }
                    scale = Some((w, p));
                    i += 2;
                }
                "--set" => {
                    let kv = args
                        .get(i + 1)
                        .ok_or_else(|| anyhow!("--set needs key=value"))?;
                    let (k, v) = kv
                        .split_once('=')
                        .ok_or_else(|| anyhow!("--set wants key=value (e.g. food_energy=12)"))?;
                    sets.push((k.trim().to_string(), v.trim().to_string()));
                    i += 2;
                }
                other => bail!("unknown new arg `{other}`"),
            }
        }

        let mut config = if sets.is_empty() {
            load_world_config_from_path(Path::new(&config_path))?
        } else {
            let world_raw = std::fs::read_to_string(&config_path)
                .map_err(|e| anyhow!("reading config `{config_path}`: {e}"))?;
            let seed_genome_path = Path::new(&config_path).with_file_name("seed_genome.toml");
            let seed_genome_raw = std::fs::read_to_string(&seed_genome_path)
                .map_err(|e| anyhow!("reading {}: {e}", seed_genome_path.display()))?;
            let patched = apply_config_overrides(&world_raw, &sets)?;
            world_config_from_toml_parts(&patched, &seed_genome_raw)
                .map_err(|e| anyhow!("config after --set failed schema validation: {e}"))?
        };
        if let Some(t) = threads {
            config.intent_parallel_threads = t;
        }
        if let Some((w, p)) = scale {
            config.world_width = w;
            config.num_organisms = p;
        }
        let sim = Simulation::new(config, seed).map_err(|e| anyhow!("{e}"))?;
        let scaled = scale.is_some();
        if fmt.is_json() {
            writeln!(
                out,
                "{}",
                json!({
                    "config": config_path,
                    "seed": seed,
                    "report_every": report_every,
                    "threads": sim.config().intent_parallel_threads,
                    "world_width": sim.config().world_width,
                    "num_organisms": sim.config().num_organisms,
                    "food_energy": sim.config().food_energy,
                    "overrides": sets.iter().map(|(k, v)| format!("{k}={v}")).collect::<Vec<_>>(),
                    "turn": 0,
                    "population": sim.organisms().len(),
                    "scaled": scaled,
                })
            )?;
        } else {
            let scaled_tag = if scaled { "  [scaled: non-canonical]" } else { "" };
            writeln!(
                out,
                "created config={config_path} seed={seed} report_every={report_every} threads={}: world_width={} num_organisms={} food_energy={} turn=0 population={}{scaled_tag}",
                sim.config().intent_parallel_threads,
                sim.config().world_width,
                sim.config().num_organisms,
                sim.config().food_energy,
                sim.organisms().len(),
            )?;
        }
        self.sim = Some(sim);
        self.recorder = None;
        self.report_every = report_every;
        self.scaled = scaled;
        Ok(())
    }

    /// Time `n` ticks through the current advance path (respects the recorder
    /// state) and report throughput. Advances the simulation by `n`.
    fn bench(&mut self, args: &[&str], out: &mut impl Write) -> Result<()> {
        let (fmt, rest) = self.take_format(args);
        let n: u64 = rest.first().map(|s| s.parse()).transpose()?.unwrap_or(100_000);
        if n == 0 {
            bail!("bench needs N >= 1");
        }
        let recording = self.recorder.is_some();
        let threads = self.sim()?.config().intent_parallel_threads;
        let start = std::time::Instant::now();
        self.advance(n)?;
        let elapsed = start.elapsed().as_secs_f64();
        let tps = if elapsed > 0.0 { n as f64 / elapsed } else { 0.0 };
        let ns_per_tick = if n > 0 {
            elapsed * 1e9 / n as f64
        } else {
            0.0
        };
        let turn = self.sim()?.turn();
        if fmt.is_json() {
            return writeln!(
                out,
                "{}",
                json!({
                    "ticks": n,
                    "seconds": elapsed,
                    "ticks_per_sec": tps,
                    "ns_per_tick": ns_per_tick,
                    "recording": recording,
                    "threads": threads,
                    "turn": turn,
                })
            )
            .map_err(Into::into);
        }
        writeln!(
            out,
            "bench: {n} ticks in {elapsed:.3}s = {tps:.0} ticks/s ({ns_per_tick:.0} ns/tick) \
             [recording={recording} threads={threads}] turn={turn}{}",
            if cfg!(debug_assertions) {
                "  (debug build — rebuild with --release for real throughput)"
            } else {
                ""
            }
        )
        .map_err(Into::into)
    }

    fn step(&mut self, args: &[&str], out: &mut impl Write) -> Result<()> {
        let (fmt, rest) = self.take_format(args);
        let n: u64 = rest.first().map(|s| s.parse()).transpose()?.unwrap_or(1);
        self.advance(n)?;
        self.emit_turn(fmt, out)
    }

    fn run_to(&mut self, args: &[&str], out: &mut impl Write) -> Result<()> {
        let (fmt, rest) = self.take_format(args);
        let target: u64 = rest
            .first()
            .ok_or_else(|| anyhow!("run-to needs a target turn"))?
            .parse()?;
        let current = self.sim()?.turn();
        if target > current {
            self.advance(target - current)?;
        }
        self.emit_turn(fmt, out)
    }

    fn emit_turn(&mut self, fmt: Format, out: &mut impl Write) -> Result<()> {
        let t = self.sim()?.turn();
        if fmt.is_json() {
            writeln!(out, "{}", json!({ "turn": t })).map_err(Into::into)
        } else {
            writeln!(out, "turn = {t}").map_err(Into::into)
        }
    }

    /// Begin recording on the current world: build a fresh ledger and
    /// back-register the live population (founders exactly at turn 0, otherwise
    /// a partial window). Used when `new` mints a sidecar and when a mutating
    /// command finds no existing sidecar to extend.
    fn start_recording(&mut self) -> Result<()> {
        let sim = self
            .sim
            .as_ref()
            .ok_or_else(|| anyhow!("no world loaded to record"))?;
        let mut ledger = Ledger::new();
        let started_turn = sim.turn();
        if started_turn == 0 {
            register_founders(&mut ledger, sim.organisms());
        } else {
            register_existing(&mut ledger, sim.organisms());
        }
        self.recorder = Some(Recorder {
            ledger,
            tick_summary: Vec::new(),
            lifetimes: Vec::new(),
            samples: Vec::new(),
            started_turn,
        });
        Ok(())
    }

    fn pillars(&mut self, args: &[&str], out: &mut impl Write) -> Result<()> {
        let (fmt, _) = self.take_format(args);
        let Some((p, n_intervals, partial)) = self.live_pillars() else {
            bail!("recording is off; advance a world with a metric sidecar before `pillars`");
        };
        let intervals = self.live_intervals().unwrap_or_default();
        if fmt.is_json() {
            let mut v = pillars_value(&p, n_intervals, partial);
            v["scaled"] = json!(self.scaled);
            // All the granular data behind the scores: the full per-interval
            // series (each interval's sub-signals), so the windowed pillar means
            // can be read against their underlying trajectory.
            v["granular"] = json!({
                "report_every": self.report_every,
                "window_start_tick": p.window_start_tick,
                "window_end_tick": p.window_end_tick,
                "intervals": serde_json::to_value(&intervals).unwrap_or(serde_json::Value::Null),
            });
            return writeln!(out, "{v}").map_err(Into::into);
        }
        let tag = match (partial, self.scaled) {
            (true, true) => "  [PARTIAL window; scaled: non-canonical]",
            (true, false) => "  [PARTIAL window]",
            (false, true) => "  [scaled: non-canonical]",
            (false, false) => "",
        };
        writeln!(
            out,
            "pillars over ({}, {}]  ({n_intervals} interval(s), report_every={}){tag}",
            p.window_start_tick, p.window_end_tick, self.report_every
        )?;
        writeln!(
            out,
            "  foraging      {:.3}   plant_consumption_rate {}",
            p.foraging_pillar,
            opt(p.mean_plant_consumption_rate, 4)
        )?;
        writeln!(
            out,
            "  predation     {:.3}   prey_consumption_rate  {}",
            p.predation_pillar,
            opt(p.mean_prey_consumption_rate, 4)
        )?;
        writeln!(
            out,
            "  intelligence  {:.3}   action_effectiveness {}  mi_sa {}",
            p.intelligence_pillar,
            opt(p.mean_action_effectiveness, 4),
            opt(p.mean_mi_sa, 4),
        )?;
        writeln!(
            out,
            "  learning      {:.3}   learning_slope {}",
            p.learning_pillar,
            opt(p.mean_learning_slope, 6),
        )?;
        // Granular per-interval series behind the scores (the window marked *).
        writeln!(
            out,
            "  granular intervals (tick: eff plant prey mi slope):"
        )?;
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

    fn state(&mut self, args: &[&str], out: &mut impl Write) -> Result<()> {
        let (fmt, _) = self.take_format(args);
        let pillars = self.live_pillars();
        let scaled = self.scaled;
        let sim = self.sim()?;
        let orgs = sim.organisms();
        let pop = orgs.len();
        let descendants = orgs.iter().filter(|o| o.generation > 0).count();
        let m = sim.metrics().clone();
        let turn = sim.turn();

        let energy = Stats::of(&orgs.iter().map(|o| o.energy as f64).collect::<Vec<_>>());
        let health = Stats::of(&orgs.iter().map(|o| o.health as f64).collect::<Vec<_>>());
        let age = Stats::of(&orgs.iter().map(|o| o.age_turns as f64).collect::<Vec<_>>());
        let gen = Stats::of(&orgs.iter().map(|o| o.generation as f64).collect::<Vec<_>>());

        let (plants, corpses, food_energy) = food_summary(sim);

        if fmt.is_json() {
            let mut v = json!({
                "turn": turn,
                "population": pop,
                "descendants": descendants,
                "founders": pop - descendants,
                "energy": energy.map(|s| s.json()),
                "health": health.map(|s| s.json()),
                "age_turns": age.map(|s| s.json()),
                "generation": gen.map(|s| s.json()),
                "food": { "plants": plants, "corpses": corpses, "total_energy": food_energy },
                "last_turn": {
                    "consumptions": m.consumptions_last_turn,
                    "predations": m.predations_last_turn,
                    "reproductions": m.reproductions_last_turn,
                    "starvations": m.starvations_last_turn,
                    "age_deaths": m.age_deaths_last_turn,
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
            if scaled { "  [scaled: non-canonical]" } else { "" }
        )?;
        writeln!(
            out,
            "population = {pop}  (descendants gen>0 = {descendants}, founders/injections = {})",
            pop - descendants
        )?;
        writeln!(out, "  energy:     {}", fmt_stats(energy))?;
        writeln!(out, "  health:     {}", fmt_stats(health))?;
        writeln!(out, "  age_turns:  {}", fmt_stats(age))?;
        writeln!(out, "  generation: {}", fmt_stats(gen))?;
        writeln!(
            out,
            "food: plants={plants} corpses={corpses} total_energy={food_energy:.0}"
        )?;
        writeln!(
            out,
            "last-turn: consumptions={} predations={} reproductions={} starvations={} age_deaths={}",
            m.consumptions_last_turn,
            m.predations_last_turn,
            m.reproductions_last_turn,
            m.starvations_last_turn,
            m.age_deaths_last_turn,
        )?;
        if let Some((p, _, partial)) = pillars {
            writeln!(
                out,
                "pillars: forage {:.3} | pred {:.3} | intel {:.3} | learn {:.3}{}",
                p.foraging_pillar,
                p.predation_pillar,
                p.intelligence_pillar,
                p.learning_pillar,
                if partial { "  [PARTIAL]" } else { "" },
            )?;
        }
        Ok(())
    }

    fn food(&mut self, args: &[&str], out: &mut impl Write) -> Result<()> {
        let (fmt, _) = self.take_format(args);
        let sim = self.sim()?;
        let (plants, corpses, energy) = food_summary(sim);
        let total = plants + corpses;
        let cells = (sim.config().world_width as u64).pow(2);
        let coverage = total as f64 / cells as f64 * 100.0;
        if fmt.is_json() {
            return writeln!(
                out,
                "{}",
                json!({
                    "plants": plants,
                    "corpses": corpses,
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
            "food: plants={plants} corpses={corpses} total={total} total_energy={energy:.0} \
             coverage={coverage:.3}% of {cells} cells"
        )
        .map_err(Into::into)
    }

    fn inspect(&mut self, args: &[&str], out: &mut impl Write) -> Result<()> {
        let id: u64 = args
            .first()
            .ok_or_else(|| anyhow!("inspect needs an organism id"))?
            .parse()?;
        let sim = self.sim()?;
        let idx = sim
            .organisms()
            .iter()
            .position(|o| o.id.0 == id)
            .ok_or_else(|| anyhow!("no live organism with id {id}"))?;
        let rec = sim.action_records().get(idx).cloned().flatten();
        let o = &sim.organisms()[idx];
        writeln!(out, "organism {id}:")?;
        writeln!(
            out,
            "  pos=({}, {}) facing={:?} energy={:.2} health={:.2}/{:.2} age={} gen={} species={} gestating={}",
            o.q, o.r, o.facing, o.energy, o.health, o.max_health, o.age_turns, o.generation, o.species_id.0, o.is_gestating
        )?;
        writeln!(
            out,
            "  last_action={:?} consumptions={} (plant={}, prey={}) reproductions={}",
            o.last_action_taken,
            o.consumptions_count,
            o.plant_consumptions_count,
            o.prey_consumptions_count,
            o.reproductions_count
        )?;
        let g = &o.genome;
        writeln!(
            out,
            "  genome: num_neurons={} synapses={} vision_distance={} | hebb_eta={:.4} juv_scale={:.3} elig_ret={:.3} max_dw={:.4} prune={:.4} | maturity={} gestation={} max_age={}",
            g.topology.num_neurons,
            o.brain.synapse_count,
            g.topology.vision_distance,
            g.plasticity.hebb_eta_gain,
            g.plasticity.juvenile_eta_scale,
            g.plasticity.eligibility_retention,
            g.plasticity.max_weight_delta_per_tick,
            g.plasticity.synapse_prune_threshold,
            g.lifecycle.age_of_maturity,
            g.lifecycle.gestation_ticks,
            g.lifecycle.max_organism_age,
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

    fn top(&mut self, args: &[&str], out: &mut impl Write) -> Result<()> {
        let field = *args.first().ok_or_else(|| anyhow!("top needs a field"))?;
        if !matches!(
            field,
            "energy" | "health" | "age" | "generation" | "consumptions" | "reproductions"
        ) {
            bail!(
                "unknown field `{field}` (energy|health|age|generation|consumptions|reproductions)"
            );
        }
        let n: usize = args.get(1).map(|s| s.parse()).transpose()?.unwrap_or(10);
        let sim = self.sim()?;
        let mut idx: Vec<usize> = (0..sim.organisms().len()).collect();
        let key = |o: &sim_types::OrganismState| -> f64 {
            match field {
                "energy" => o.energy as f64,
                "health" => o.health as f64,
                "age" => o.age_turns as f64,
                "generation" => o.generation as f64,
                "consumptions" => o.consumptions_count as f64,
                "reproductions" => o.reproductions_count as f64,
                _ => f64::NAN,
            }
        };
        let orgs = sim.organisms();
        idx.sort_by(|&a, &b| key(&orgs[b]).total_cmp(&key(&orgs[a])));
        writeln!(out, "top {n} by {field}:")?;
        for &i in idx.iter().take(n) {
            let o = &orgs[i];
            writeln!(
                out,
                "  id={:<6} {field}={:<12.3} energy={:.1} age={} gen={} consum={} repro={} last={:?}",
                o.id.0,
                key(o),
                o.energy,
                o.age_turns,
                o.generation,
                o.consumptions_count,
                o.reproductions_count,
                o.last_action_taken
            )?;
        }
        Ok(())
    }

    fn hist(&mut self, args: &[&str], out: &mut impl Write) -> Result<()> {
        let field = *args.first().ok_or_else(|| anyhow!("hist needs a field"))?;
        let sim = self.sim()?;
        let vals: Vec<f64> = sim
            .organisms()
            .iter()
            .map(|o| match field {
                "energy" => o.energy as f64,
                "health" => o.health as f64,
                "age" => o.age_turns as f64,
                "generation" => o.generation as f64,
                _ => f64::NAN,
            })
            .collect();
        if vals.iter().any(|v| v.is_nan()) {
            bail!("unknown field `{field}` (energy|health|age|generation)");
        }
        if vals.is_empty() {
            writeln!(out, "(no organisms)")?;
            return Ok(());
        }
        let min = vals.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let bins = 10usize;
        let width = ((max - min) / bins as f64).max(f64::MIN_POSITIVE);
        let mut counts = vec![0u64; bins];
        for v in &vals {
            let b = (((v - min) / width) as usize).min(bins - 1);
            counts[b] += 1;
        }
        let peak = counts.iter().copied().max().unwrap_or(1).max(1);
        writeln!(out, "{field} histogram ({} organisms):", vals.len())?;
        for (i, c) in counts.iter().enumerate() {
            let lo = min + i as f64 * width;
            let hi = lo + width;
            let bar = "#".repeat((c * 40 / peak) as usize);
            writeln!(out, "  [{lo:>9.1}, {hi:>9.1}) {c:>8} {bar}")?;
        }
        Ok(())
    }
}

/// Shared JSON encoding of a pillar readout (used by `pillars` and `state`).
fn pillars_value(p: &PillarScores, n_intervals: usize, partial: bool) -> serde_json::Value {
    json!({
        "window_start_tick": p.window_start_tick,
        "window_end_tick": p.window_end_tick,
        "intervals": n_intervals,
        "partial": partial,
        "foraging": p.foraging_pillar,
        "predation": p.predation_pillar,
        "intelligence": p.intelligence_pillar,
        "learning": p.learning_pillar,
        "plant_consumption_rate": opt_json(p.mean_plant_consumption_rate),
        "prey_consumption_rate": opt_json(p.mean_prey_consumption_rate),
        "action_effectiveness": opt_json(p.mean_action_effectiveness),
        "mi_sa": opt_json(p.mean_mi_sa),
        "learning_slope": opt_json(p.mean_learning_slope),
    })
}

fn food_summary(sim: &Simulation) -> (u64, u64, f64) {
    let mut plants = 0u64;
    let mut corpses = 0u64;
    let mut energy = 0f64;
    for f in sim.foods() {
        match f.kind {
            sim_types::FoodKind::Plant => plants += 1,
            sim_types::FoodKind::Corpse => corpses += 1,
        }
        energy += f.energy as f64;
    }
    (plants, corpses, energy)
}

