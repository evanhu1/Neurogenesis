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

use anyhow::{anyhow, bail, Result};
use output::{opt, opt_json, Format, Stats};
use serde_json::json;
use sim_config::load_world_config_from_path;
use sim_core::Simulation;
use sim_metrics::{
    compute_pillar_scores, derive_interval_metrics, ingest_tick, register_existing,
    register_founders, Ledger, OrganismLifetimeRow, PillarScores, TickSummaryRow,
};
use sim_types::EntityId;
use std::io::{self, BufRead, Write};
use std::path::Path;

const DEFAULT_CONFIG: &str = "sim-evaluation/config.toml";
/// Default reporting-interval width (matches the eval; override at `load`).
const REPORT_EVERY: u64 = 10_000;

fn main() -> Result<()> {
    let mut app = App {
        sim: None,
        recorder: None,
        report_every: REPORT_EVERY,
        format: Format::default(),
        scaled: false,
    };
    let stdin = io::stdin();
    let mut out = io::stdout();
    writeln!(
        out,
        "sim-cli ready. `help` for commands, `quit` to exit.\n\
         Commands are read from stdin, one per line."
    )?;
    out.flush()?;

    for line in stdin.lock().lines() {
        let line = line?;
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if line == "quit" || line == "exit" {
            break;
        }
        if let Err(e) = app.dispatch(line, &mut out) {
            writeln!(out, "error: {e}")?;
        }
        out.flush()?;
    }
    Ok(())
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
#[derive(Debug, Clone, Copy)]
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

    fn dispatch(&mut self, line: &str, out: &mut impl Write) -> Result<()> {
        let mut parts = line.split_whitespace();
        let cmd = parts.next().unwrap_or("");
        let args: Vec<&str> = parts.collect();
        match cmd {
            "help" => self.help(out),
            "load" => self.load(&args, out),
            "step" => self.step(&args, out),
            "run-to" => self.run_to(&args, out),
            "bench" => self.bench(&args, out),
            "turn" => {
                let t = self.sim()?.turn();
                writeln!(out, "turn = {t}").map_err(Into::into)
            }
            "record" => self.record(&args, out),
            "format" => self.set_format(&args, out),
            "pillars" => self.pillars(&args, out),
            "state" => self.state(&args, out),
            "eco" => self.eco(&args, out),
            "lineage" => self.lineage(&args, out),
            "genome" => self.genome(&args, out),
            "timeseries" => self.timeseries(&args, out),
            "watch" => self.watch(&args, out),
            "food" => self.food(&args, out),
            "inspect" => self.inspect(&args, out),
            "top" => self.top(&args, out),
            "hist" => self.hist(&args, out),
            "find" => self.find(&args, out),
            "brain" => self.brain(&args, out),
            "decide" => self.decide(&args, out),
            other => bail!("unknown command `{other}` (try `help`)"),
        }
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

    fn set_format(&mut self, args: &[&str], out: &mut impl Write) -> Result<()> {
        match args.first().copied() {
            Some("json") => self.format = Format::Json,
            Some("text") => self.format = Format::Text,
            None => {}
            Some(other) => bail!("unknown format `{other}` (json|text)"),
        }
        writeln!(
            out,
            "format = {}",
            if self.format.is_json() { "json" } else { "text" }
        )
        .map_err(Into::into)
    }

    fn help(&self, out: &mut impl Write) -> Result<()> {
        writeln!(
            out,
            "commands (most accept --json | --text):\n\
             \x20 load [--config PATH] [--seed N] [--report-every R] [--threads K] [--scale W,POP]  build Simulation::new (defaults: config {DEFAULT_CONFIG}, seed 0, report-every 10000)\n\
             \x20 step [N]                          advance N ticks (default 1)\n\
             \x20 run-to T                          advance until turn == T\n\
             \x20 bench [N]                         time N ticks (default 100000); report ticks/sec\n\
             \x20 watch T [--every E]               advance to T, emitting a metrics row every E ticks\n\
             \x20 turn                              print current turn\n\
             \x20 record on|off|status             toggle live metric recording (required for pillars/eco/lineage history)\n\
             \x20 format json|text                  set the default output format\n\
             \x20 pillars                           foraging/predation/intelligence/learning pillars + sub-signals (needs recording)\n\
             \x20 state                             population / energy / food / last-turn metrics (+ pillars if recording)\n\
             \x20 eco                               ecological dynamics: population/food trajectory, deaths-by-cause, rates\n\
             \x20 lineage                           generation distribution + founder-lineage (species) composition\n\
             \x20 genome [--gene G] [--drift]       population genome-gene distribution (what evolution is selecting for)\n\
             \x20 timeseries [--cols LIST] [--last K]  recorded metric columns as sparklines\n\
             \x20 food                              food (plant/corpse) summary\n\
             \x20 inspect ID                        full dump of one organism\n\
             \x20 top FIELD [N]                     top-N organisms by a field\n\
             \x20 hist FIELD                        histogram of a scalar field\n\
             \x20 find EXPR [--fields LIST] [--limit N]  filter organisms by a predicate (and/or evaluated left-to-right, no precedence)\n\
             \x20 brain ID [--view summary|synapses|activations|dot]  neural inspection\n\
             \x20 decide ID                         explain one organism's current-tick decision\n\
             \x20 quit"
        )
        .map_err(Into::into)
    }

    fn load(&mut self, args: &[&str], out: &mut impl Write) -> Result<()> {
        let mut config_path = DEFAULT_CONFIG.to_string();
        let mut seed: u64 = 0;
        let mut report_every: u64 = REPORT_EVERY;
        let mut threads: Option<u32> = None;
        let mut scale: Option<(u32, u32)> = None;
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
                other => bail!("unknown load arg `{other}`"),
            }
        }
        let mut config = load_world_config_from_path(Path::new(&config_path))?;
        if let Some(t) = threads {
            config.intent_parallel_threads = t;
        }
        if let Some((w, p)) = scale {
            config.world_width = w;
            config.num_organisms = p;
        }
        let sim = Simulation::new(config, seed).map_err(|e| anyhow!("{e}"))?;
        let scaled_tag = if scale.is_some() {
            "  [scaled: non-canonical]"
        } else {
            ""
        };
        writeln!(
            out,
            "loaded config={config_path} seed={seed} report_every={report_every} threads={}: world_width={} num_organisms={} food_energy={} turn=0 population={}{scaled_tag}",
            sim.config().intent_parallel_threads,
            sim.config().world_width,
            sim.config().num_organisms,
            sim.config().food_energy,
            sim.organisms().len(),
        )?;
        self.sim = Some(sim);
        self.recorder = None;
        self.report_every = report_every;
        self.scaled = scale.is_some();
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
        let n: u64 = args.first().map(|s| s.parse()).transpose()?.unwrap_or(1);
        self.advance(n)?;
        writeln!(out, "turn = {}", self.sim()?.turn()).map_err(Into::into)
    }

    fn run_to(&mut self, args: &[&str], out: &mut impl Write) -> Result<()> {
        let target: u64 = args
            .first()
            .ok_or_else(|| anyhow!("run-to needs a target turn"))?
            .parse()?;
        let current = self.sim()?.turn();
        if target > current {
            self.advance(target - current)?;
        }
        writeln!(out, "turn = {}", self.sim()?.turn()).map_err(Into::into)
    }

    fn record(&mut self, args: &[&str], out: &mut impl Write) -> Result<()> {
        match args.first().copied() {
            Some("on") => {
                let sim = self
                    .sim
                    .as_ref()
                    .ok_or_else(|| anyhow!("no simulation loaded; run `load` first"))?;
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
                let mode = if started_turn == 0 {
                    "exact from turn 0"
                } else {
                    "partial (back-registered live population)"
                };
                writeln!(
                    out,
                    "recording ON at turn {started_turn} ({mode}); report_every={}",
                    self.report_every
                )
                .map_err(Into::into)
            }
            Some("off") => {
                let was = self.recorder.is_some();
                self.recorder = None;
                writeln!(
                    out,
                    "recording OFF{}",
                    if was { "" } else { " (was already off)" }
                )
                .map_err(Into::into)
            }
            Some("status") | None => match self.recorder.as_ref() {
                None => writeln!(out, "recording: off").map_err(Into::into),
                Some(rec) => {
                    let turn = self.sim.as_ref().map(|s| s.turn()).unwrap_or(0);
                    writeln!(
                        out,
                        "recording: on  started_turn={} now={} ticks_recorded={} deaths_captured={} {}",
                        rec.started_turn,
                        turn,
                        rec.tick_summary.len(),
                        rec.lifetimes.len(),
                        if rec.started_turn > 0 { "[PARTIAL]" } else { "[exact]" },
                    )
                    .map_err(Into::into)
                }
            },
            Some(other) => bail!("unknown record arg `{other}` (on|off|status)"),
        }
    }

    fn pillars(&mut self, args: &[&str], out: &mut impl Write) -> Result<()> {
        let (fmt, _) = self.take_format(args);
        let Some((p, n_intervals, partial)) = self.live_pillars() else {
            bail!("recording is off; `record on` then advance before `pillars`");
        };
        if fmt.is_json() {
            let mut v = pillars_value(&p, n_intervals, partial);
            v["scaled"] = json!(self.scaled);
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
        )
        .map_err(Into::into)
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

