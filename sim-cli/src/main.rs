//! sim-cli — stateless, world-as-file research CLI for the NeuroGenesis engine.
//!
//! Each invocation reads a world from `--in`, runs one command, and (for
//! mutating commands) writes the advanced world to `--out` (default `--in`).
//! Metrics live in a `<world>.metrics` sidecar that follows the world. All
//! world IO, advancement, and read rendering live in the shared `sim-views`
//! crate (which the web server also uses), so live metrics match the offline
//! eval and the agent/human views are identical. Output is JSON by default;
//! `--text` overrides per call.
//!
//! See docs/sim-cli.md (usage) and docs/sim-cli-stateless-spec.md + SPEC.md.

mod dashboards;
mod neat;
mod sweep;
mod tui;

use anyhow::{anyhow, bail, Result};
use serde_json::json;
use sim_core::Simulation;
use sim_views::output::Format;
use sim_views::{
    advance, build_world, load_sidecar, load_world, resolve_metrics_path, save_sidecar, save_world,
    sibling_metrics_path, start_recording, take_format, NewWorldParams, ReadCtx, Recorder,
};
use std::io::{self, BufRead, Write};
use std::path::{Path, PathBuf};

pub(crate) const DEFAULT_CONFIG: &str = "sim-evaluation/config.toml";
/// Default reporting-interval width (matches the eval; override at `new`).
pub(crate) const REPORT_EVERY: u64 = 10_000;
/// Default directory for run-mode result files (sweep, etc.). Override with the
/// global `--out-dir` flag. Under `artifacts/` so results survive a session.
const DEFAULT_OUT_DIR: &str = "artifacts/runs";

/// Build a timestamped result-file path under `out_dir`, creating the directory.
/// Run modes (sweep, …) write their result artifact here so output is durable
/// and discoverable rather than scrolled past in stdout.
pub(crate) fn run_output_path(out_dir: &str, prefix: &str) -> Result<PathBuf> {
    std::fs::create_dir_all(out_dir)
        .map_err(|e| anyhow!("cannot create out-dir `{out_dir}`: {e}"))?;
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0);
    // PID suffix so two sweeps started in the same millisecond don't collide.
    Ok(PathBuf::from(out_dir).join(format!("{prefix}-{stamp}-{}.json", std::process::id())))
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
         \x20 new [--config P] [--seed N] [--seed-genome-snapshot P] [--set k=v]... [--scale W,POP] [--threads K] [--report-every R] --out w.bin\n\
         \x20 step [N] --in w.bin [--out w.bin]            advance N ticks (default 1)\n\
         \x20 run-to T --in w.bin [--out w.bin]            advance until turn == T\n\
         \x20 watch T [--every E] --in w.bin [--out w.bin] advance to T, emitting a metrics row every E ticks\n\
         \x20 bench [N] --in w.bin                         time N ticks (world discarded)\n\
         \x20 sweep --grid k=v,v... --seeds N,N --to T [--out-dir D]  parallel grid×seed runs → result file\n\
         \x20 neat [--population N] [--generations N] [--episode-horizons T[,T...]] [--world-seeds N,N] [--audit-seeds N,N] [--holdout-seeds N,N] [--audit-levels N,N] [--audit-every N] [--scale W,POP] [--param k=v]  canonical generational NEAT → result json + champion world.bin\n\
         \x20 neat analyze RESULT.json [RESULT2.json ...]    derive trend, saturation, lesion, and innovation diagnostics\n\
         \n\
         READS (stdout only)\n\
         \x20 turn | state | pillars | eco | lineage | genome [--gene G] | food --in w.bin\n\
         \x20 timeseries [--cols LIST] [--last K] --in w.bin\n\
         \x20 inspect ID | top FIELD [N] | hist FIELD | find EXPR | brain ID [--view V] | decide ID --in w.bin\n\
         \x20 query --in w.bin                             read-only commands from stdin (one per line), one load\n\
         \n\
         INTERACTIVE (human-facing, not for agents)\n\
         \x20 tui --in w.bin | tui --new [--seed N] [--set k=v]...  split-pane REPL over a resident world; explicit `save`\n\
         \n\
         pillars/eco-trajectory/timeseries need a metric sidecar (minted by `new`, follows the world).\n\
         Snapshot/fork a world with `cp`; full reference in docs/sim-cli.md."
    )
    .map_err(Into::into)
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
            "--out-dir" => out_dir = it.next().ok_or_else(|| anyhow!("--out-dir needs a path"))?,
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
    if cmd == "neat" {
        let mut out = io::stdout().lock();
        return neat::run_neat_cli(&cmd_args, &out_dir, &mut out);
    }
    // Human-facing interactive mode: a resident world driven from a split-pane
    // TUI, as opposed to the agent-facing one-shot commands below. `--in` is a
    // global flag (already pulled above), so hand it through explicitly.
    if cmd == "tui" {
        return tui::run_tui_cli(&cmd_args, in_path.as_deref());
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
        build_world_cli(&mut app, &cmd_args, &mut out)?;
        if !no_metrics {
            app.start_recording()?;
        }
    } else {
        let world_path = in_path
            .clone()
            .ok_or_else(|| anyhow!("`{cmd}` needs --in <world.bin>"))?;
        let sim = load_world(&world_path)?;
        app.scaled = sim.experiment_scaled();
        app.sim = Some(sim);

        // The metric sidecar follows the world: explicit --metrics, else the
        // `<world>.metrics` sibling if present. Mutating commands without an
        // existing sidecar mint a fresh one (back-registering live organisms)
        // so recording is on by default; --no-metrics opts out.
        let metrics_path = resolve_metrics_path(&world_path, metrics_flag.as_deref(), no_metrics);
        if let Some(mp) = metrics_path.as_ref() {
            if Path::new(mp).exists() {
                let (report_every, recorder) = load_sidecar(mp)?;
                app.report_every = report_every;
                let world_turn = app.sim.as_ref().map(|s| s.turn()).unwrap_or(0);
                let sidecar_last = recorder.recorded_through_turn;
                if sidecar_last > world_turn {
                    bail!(
                        "metric sidecar is ahead of its world: covers ticks <= {sidecar_last} but world is at turn {world_turn}"
                    );
                }
                if sidecar_last < world_turn {
                    if mutating {
                        eprintln!(
                            "{}",
                            json!({ "warning": format!(
                                "metric sidecar is stale: covers ticks <= {sidecar_last} but world is at turn {world_turn}; \
                                 restarting recording at the current turn before mutation"
                            ) })
                        );
                        app.start_recording()?;
                    } else {
                        eprintln!(
                            "{}",
                            json!({ "warning": format!(
                                "metric sidecar is stale: covers ticks <= {sidecar_last} but world is at turn {world_turn}; \
                                 history-based reads use the recorded span only"
                            ) })
                        );
                        app.recorder = Some(recorder);
                    }
                } else {
                    app.recorder = Some(recorder);
                }
            } else if mutating {
                app.start_recording()?;
            }
        }

        app.run_oneshot(cmd, &cmd_args, &mut out)?;
        out.flush()?;

        if mutating {
            let dest = out_path.clone().unwrap_or_else(|| world_path.clone());
            save_world(app.sim.as_ref().expect("world loaded"), &dest)?;
            if let Some(rec) = app.recorder.as_ref() {
                // The sidecar follows the WRITTEN world. On a fork (--out != --in)
                // it lands beside the fork, so the source world's sidecar is never
                // overwritten with a higher turn count (which would desync it).
                let sidecar_dest = if dest != world_path {
                    sibling_metrics_path(&dest)
                } else {
                    metrics_path
                        .clone()
                        .expect("recorder present implies a metrics path")
                };
                save_sidecar(app.report_every, rec, &sidecar_dest)?;
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
        save_sidecar(app.report_every, rec, &mp)?;
    }
    Ok(())
}

/// CLI-side state: the loaded world, its optional metric recorder, and the
/// presentation defaults. All heavy lifting (IO, advancement, reads) is
/// delegated to `sim-views`; this struct just holds the pieces a `ReadCtx`
/// borrows and the mutating-command loop advances.
pub(crate) struct App {
    pub(crate) sim: Option<Simulation>,
    pub(crate) recorder: Option<Recorder>,
    pub(crate) report_every: u64,
    pub(crate) format: Format,
    pub(crate) scaled: bool,
}

impl App {
    fn sim(&mut self) -> Result<&mut Simulation> {
        self.sim
            .as_mut()
            .ok_or_else(|| anyhow!("no world loaded (use --in <world.bin>)"))
    }

    /// Borrow the loaded world + recorder as a read context for `sim-views`.
    pub(crate) fn read_ctx(&self) -> Result<ReadCtx<'_>> {
        Ok(ReadCtx {
            sim: self
                .sim
                .as_ref()
                .ok_or_else(|| anyhow!("no world loaded (use --in <world.bin>)"))?,
            recorder: self.recorder.as_ref(),
            report_every: self.report_every,
            format: self.format,
            scaled: self.scaled,
        })
    }

    /// Begin recording on the current world (fresh ledger, live population
    /// back-registered). Used when `new` mints a sidecar and when a mutating
    /// command finds no existing sidecar to extend.
    pub(crate) fn start_recording(&mut self) -> Result<()> {
        let sim = self
            .sim
            .as_ref()
            .ok_or_else(|| anyhow!("no world loaded to record"))?;
        self.recorder = Some(start_recording(sim, self.report_every));
        Ok(())
    }

    /// Advance `n` ticks through the shared recording-aware path.
    pub(crate) fn advance(&mut self, n: u64) -> Result<()> {
        let sim = self
            .sim
            .as_mut()
            .ok_or_else(|| anyhow!("no world loaded (use --in <world.bin>)"))?;
        advance(sim, self.recorder.as_mut(), n);
        Ok(())
    }

    /// Run one command against the already-loaded world. `new` is handled by the
    /// orchestrator (it constructs rather than reads), so it is not here.
    pub(crate) fn run_oneshot(
        &mut self,
        cmd: &str,
        args: &[&str],
        out: &mut impl Write,
    ) -> Result<()> {
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

    /// Dispatch a pure-read command to `sim-views` against a borrowed context.
    pub(crate) fn run_read(
        &mut self,
        cmd: &str,
        args: &[&str],
        out: &mut impl Write,
    ) -> Result<()> {
        let ctx = self.read_ctx()?;
        match cmd {
            "turn" => sim_views::turn(&ctx, args, out),
            "pillars" => sim_views::pillars(&ctx, args, out),
            "state" => sim_views::state(&ctx, args, out),
            "eco" => sim_views::eco(&ctx, args, out),
            "lineage" => sim_views::lineage(&ctx, args, out),
            "genome" => sim_views::genome(&ctx, args, out),
            "timeseries" => sim_views::timeseries(&ctx, args, out),
            "food" => sim_views::food(&ctx, args, out),
            "inspect" => sim_views::inspect(&ctx, args, out),
            "top" => sim_views::top(&ctx, args, out),
            "hist" => sim_views::hist(&ctx, args, out),
            "find" => sim_views::find(&ctx, args, out),
            "brain" => sim_views::brain(&ctx, args, out),
            "decide" => sim_views::decide(&ctx, args, out),
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
        let mut failures = 0usize;
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
                Err(anyhow!(
                    "`{sub}` is not a read command (not allowed in query)"
                ))
            };
            if let Err(e) = result {
                failures += 1;
                writeln!(
                    out,
                    "{}",
                    json!({ "error": e.to_string(), "command": line })
                )?;
            }
            out.flush()?;
        }
        if failures > 0 {
            bail!("query completed with {failures} failed command(s)");
        }
        Ok(())
    }

    /// Time `n` ticks through the current advance path (respects the recorder
    /// state) and report throughput. Advances the simulation by `n`.
    fn bench(&mut self, args: &[&str], out: &mut impl Write) -> Result<()> {
        let (fmt, rest) = take_format(self.format, args);
        let n: u64 = rest
            .first()
            .map(|s| s.parse())
            .transpose()?
            .unwrap_or(100_000);
        if n == 0 {
            bail!("bench needs N >= 1");
        }
        let recording = self.recorder.is_some();
        let threads = self.sim()?.config().intent_parallel_threads;
        let start = std::time::Instant::now();
        self.advance(n)?;
        let elapsed = start.elapsed().as_secs_f64();
        let tps = if elapsed > 0.0 {
            n as f64 / elapsed
        } else {
            0.0
        };
        let ns_per_tick = if n > 0 { elapsed * 1e9 / n as f64 } else { 0.0 };
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
        let (fmt, rest) = take_format(self.format, args);
        let n: u64 = rest.first().map(|s| s.parse()).transpose()?.unwrap_or(1);
        self.advance(n)?;
        self.emit_turn(fmt, out)
    }

    fn run_to(&mut self, args: &[&str], out: &mut impl Write) -> Result<()> {
        let (fmt, rest) = take_format(self.format, args);
        let target: u64 = rest
            .first()
            .ok_or_else(|| anyhow!("run-to needs a target turn"))?
            .parse()?;
        let current = self.sim()?.turn();
        if target < current {
            bail!("run-to target {target} is behind current turn {current}");
        }
        if target > current {
            self.advance(target - current)?;
        }
        self.emit_turn(fmt, out)
    }

    pub(crate) fn emit_turn(&mut self, fmt: Format, out: &mut impl Write) -> Result<()> {
        let t = self.sim()?.turn();
        if fmt.is_json() {
            writeln!(out, "{}", json!({ "turn": t })).map_err(Into::into)
        } else {
            writeln!(out, "turn = {t}").map_err(Into::into)
        }
    }
}

/// `new`: parse the constructor args, build the world via `sim-views`, print the
/// summary line, and install the world into `app`. Persistence is the caller's.
pub(crate) fn build_world_cli(app: &mut App, args: &[&str], out: &mut impl Write) -> Result<()> {
    let mut config_path = DEFAULT_CONFIG.to_string();
    let mut seed: u64 = 0;
    let mut report_every: u64 = REPORT_EVERY;
    let mut threads: Option<u32> = None;
    let mut scale: Option<(u32, u32)> = None;
    let mut seed_genome_snapshot: Option<String> = None;
    let mut sets: Vec<(String, String)> = Vec::new();
    let (fmt, args) = take_format(app.format, args);
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
            "--seed-genome-snapshot" => {
                seed_genome_snapshot = Some(
                    args.get(i + 1)
                        .ok_or_else(|| anyhow!("--seed-genome-snapshot needs a path"))?
                        .to_string(),
                );
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

    let champion_pool = if let Some(path) = seed_genome_snapshot.as_deref() {
        let file = std::fs::File::open(path)
            .map_err(|e| anyhow!("cannot open seed genome snapshot `{path}`: {e}"))?;
        let genome = bincode::deserialize_from(file)
            .map_err(|e| anyhow!("cannot decode seed genome snapshot `{path}`: {e}"))?;
        vec![genome]
    } else {
        Vec::new()
    };
    let built = build_world(&NewWorldParams {
        config_path: config_path.clone(),
        seed,
        report_every,
        threads,
        scale,
        sets: sets.clone(),
        champion_pool,
    })?;
    let sim = built.sim;
    let scaled = built.scaled;
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
        let scaled_tag = if scaled {
            "  [scaled: non-canonical]"
        } else {
            ""
        };
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
    app.sim = Some(sim);
    app.recorder = None;
    app.report_every = report_every;
    app.scaled = scaled;
    Ok(())
}
