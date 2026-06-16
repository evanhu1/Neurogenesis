//! sim-cli — headless, stateful command client for the NeuroGenesis engine.
//!
//! Reads commands from stdin (one per line), holding a single `Simulation` in
//! memory so the world can be scrubbed forward and inspected repeatedly without
//! rebuilding. Designed to reproduce a `sim-evaluation` seed run exactly
//! (`Simulation::new(load_world_config_from_path("sim-evaluation/config.toml"),
//! seed)`, no champion pool) and to recompute the foraging signal locally from
//! the `instrumentation` `ActionRecord`s.
//!
//! See docs/sim-cli-design.md.

use anyhow::{anyhow, bail, Result};
use sim_config::load_world_config_from_path;
use sim_core::Simulation;
use sim_metrics::{
    compute_pillar_scores, derive_interval_metrics, ingest_tick, register_existing,
    register_founders, Ledger, OrganismLifetimeRow, PillarScores, TickSummaryRow,
};
use sim_types::{ActionType, EntityId};
use std::collections::HashMap;
use std::io::{self, BufRead, Write};
use std::path::Path;

const DEFAULT_CONFIG: &str = "sim-evaluation/config.toml";
/// Matches the eval's reporting interval (`sim-evaluation/src/cli.rs`).
const REPORT_EVERY: u64 = 10_000;
/// Foraging baseline: `1 / ACTION_COUNT`, `ACTION_COUNT = ActionType::ALL.len()
/// + 1` (the implicit Idle), i.e. 7. Verified in `dataset/schema.rs`.
const P_BASELINE: f64 = 1.0 / 7.0;
/// Upper anchor of the foraging pillar normalization (`analysis/pillars.rs`).
const PILLAR_TARGET: f64 = 0.55;

fn main() -> Result<()> {
    let mut app = App {
        sim: None,
        recorder: None,
        report_every: REPORT_EVERY,
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
    /// Turn at which recording began. `> 0` means already-alive organisms were
    /// back-registered with partial lifetime counters (windows are approximate).
    started_turn: u64,
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
                    rec.tick_summary.push(TickSummaryRow {
                        tick: turn,
                        descendant_population: rec.ledger.descendant_population(),
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
            "turn" => {
                let t = self.sim()?.turn();
                writeln!(out, "turn = {t}").map_err(Into::into)
            }
            "record" => self.record(&args, out),
            "pillars" => self.pillars(out),
            "state" => self.state(out),
            "forage" => self.forage(&args, out),
            "food" => self.food(out),
            "inspect" => self.inspect(&args, out),
            "top" => self.top(&args, out),
            "hist" => self.hist(&args, out),
            other => bail!("unknown command `{other}` (try `help`)"),
        }
    }

    fn help(&self, out: &mut impl Write) -> Result<()> {
        writeln!(
            out,
            "commands:\n\
             \x20 load [--config PATH] [--seed N] [--report-every R]  build Simulation::new (defaults: config {DEFAULT_CONFIG}, seed 0, report-every 10000)\n\
             \x20 step [N]                          advance N ticks (default 1)\n\
             \x20 run-to T                          advance until turn == T\n\
             \x20 turn                              print current turn\n\
             \x20 record on|off|status             toggle live metric recording (required for pillars)\n\
             \x20 pillars                           foraging/predation/intelligence/learning pillars + sub-signals (needs recording)\n\
             \x20 state                             population / energy / food / last-turn metrics summary (+ pillars if recording)\n\
             \x20 forage [FROM] TO                  trace FROM..TO; live foraging probe + eval-style window pillar\n\
             \x20 food                              food (plant/corpse) summary\n\
             \x20 inspect ID                        full dump of one organism\n\
             \x20 top FIELD [N]                     top-N by energy|health|age|generation|consumptions|reproductions\n\
             \x20 hist FIELD                        histogram of energy|health|age|generation\n\
             \x20 quit"
        )
        .map_err(Into::into)
    }

    fn load(&mut self, args: &[&str], out: &mut impl Write) -> Result<()> {
        let mut config_path = DEFAULT_CONFIG.to_string();
        let mut seed: u64 = 0;
        let mut report_every: u64 = REPORT_EVERY;
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
                other => bail!("unknown load arg `{other}`"),
            }
        }
        let config = load_world_config_from_path(Path::new(&config_path))?;
        let sim = Simulation::new(config, seed).map_err(|e| anyhow!("{e}"))?;
        writeln!(
            out,
            "loaded config={config_path} seed={seed} report_every={report_every}: world_width={} num_organisms={} food_energy={} turn=0 population={}",
            sim.config().world_width,
            sim.config().num_organisms,
            sim.config().food_energy,
            sim.organisms().len(),
        )?;
        self.sim = Some(sim);
        self.recorder = None;
        self.report_every = report_every;
        Ok(())
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

    fn pillars(&mut self, out: &mut impl Write) -> Result<()> {
        let Some((p, n_intervals, partial)) = self.live_pillars() else {
            bail!("recording is off; `record on` then advance before `pillars`");
        };
        let tag = if partial { "  [PARTIAL window]" } else { "" };
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

    fn state(&mut self, out: &mut impl Write) -> Result<()> {
        let sim = self.sim()?;
        let orgs = sim.organisms();
        let pop = orgs.len();
        let descendants = orgs.iter().filter(|o| o.generation > 0).count();
        let m = sim.metrics().clone();

        let energy: Vec<f64> = orgs.iter().map(|o| o.energy as f64).collect();
        let health: Vec<f64> = orgs.iter().map(|o| o.health as f64).collect();
        let age: Vec<f64> = orgs.iter().map(|o| o.age_turns as f64).collect();
        let gen: Vec<f64> = orgs.iter().map(|o| o.generation as f64).collect();

        let (plants, corpses, food_energy) = food_summary(sim);

        writeln!(out, "turn = {}", sim.turn())?;
        writeln!(
            out,
            "population = {pop}  (descendants gen>0 = {descendants}, founders/injections = {})",
            pop - descendants
        )?;
        writeln!(out, "  energy:     {}", stats_line(&energy))?;
        writeln!(out, "  health:     {}", stats_line(&health))?;
        writeln!(out, "  age_turns:  {}", stats_line(&age))?;
        writeln!(out, "  generation: {}", stats_line(&gen))?;
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
        if let Some((p, _, partial)) = self.live_pillars() {
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

    fn food(&mut self, out: &mut impl Write) -> Result<()> {
        let sim = self.sim()?;
        let (plants, corpses, energy) = food_summary(sim);
        let total = plants + corpses;
        let cells = (sim.config().world_width as u64).pow(2);
        writeln!(
            out,
            "food: plants={plants} corpses={corpses} total={total} total_energy={energy:.0} \
             coverage={:.3}% of {cells} cells",
            total as f64 / cells as f64 * 100.0
        )
        .map_err(Into::into)
    }

    /// Trace [FROM, TO] tick-by-tick, accumulating the foraging signal two ways:
    /// a **live behavioral probe** over descendant organism-ticks, and an
    /// **eval-style window pillar** from death-tick-bucketed lifetime counters.
    fn forage(&mut self, args: &[&str], out: &mut impl Write) -> Result<()> {
        let (from, to) = match args {
            [to] => (None, to.parse::<u64>()?),
            [from, to] => (Some(from.parse::<u64>()?), to.parse::<u64>()?),
            _ => bail!("usage: forage [FROM] TO"),
        };
        let sim = self.sim()?;
        if let Some(from) = from {
            while sim.turn() < from {
                sim.tick();
            }
        }
        let start = sim.turn();
        if to <= start {
            bail!("TO ({to}) must be greater than current turn ({start})");
        }

        // Live probe accumulators (descendant organism-ticks over the window).
        let mut desc_ticks: u64 = 0;
        let mut food_ahead_ticks: u64 = 0;
        let mut fwd_when_food_ahead: u64 = 0;
        let mut any_food_visible_ticks: u64 = 0;
        // Action histogram restricted to ticks where food is ahead.
        let mut action_hist_when_ahead = [0u64; 7];
        let mut desc_pop_sum: u64 = 0;
        let mut food_count_sum: u64 = 0;
        let mut samples: u64 = 0;

        // Eval-style: per-organism lifetime counters (since FROM), finalized on
        // death into death-tick buckets. Approximate for organisms born before
        // FROM (counters start at FROM); labeled as such in the output.
        let mut lifetime: HashMap<u64, Life> = HashMap::new();
        let mut buckets: HashMap<u64, (u64, u64)> = HashMap::new(); // bucket -> (food_ahead, fwd)

        while sim.turn() < to {
            let delta = sim.tick();
            let turn = sim.turn();

            // Finalize descendants that died this tick.
            for removed in &delta.removed_positions {
                if let EntityId::Organism(id) = removed.entity_id {
                    if let Some(life) = lifetime.remove(&id.0) {
                        if life.is_descendant {
                            let b = bucket_of(turn);
                            let e = buckets.entry(b).or_insert((0, 0));
                            e.0 += life.food_ahead;
                            e.1 += life.fwd;
                        }
                    }
                }
            }

            // Accumulate this tick from survivors' action records.
            let records = sim.action_records();
            let orgs = sim.organisms();
            let mut desc_pop = 0u64;
            for (org, rec) in orgs.iter().zip(records.iter()) {
                let is_desc = org.generation > 0;
                if is_desc {
                    desc_pop += 1;
                }
                let Some(rec) = rec else { continue };
                let food_ahead = rec.food_visible_at_offset(0);
                let fwd = matches!(rec.selected_action, ActionType::Forward);

                // Lifetime ledger (all organisms; filtered to descendants at finalize).
                let life = lifetime.entry(org.id.0).or_insert(Life {
                    is_descendant: is_desc,
                    food_ahead: 0,
                    fwd: 0,
                });
                if food_ahead {
                    life.food_ahead += 1;
                    if fwd {
                        life.fwd += 1;
                    }
                }

                // Live probe (descendants only).
                if is_desc {
                    desc_ticks += 1;
                    if rec.food_visible.iter().any(|v| *v) {
                        any_food_visible_ticks += 1;
                    }
                    if food_ahead {
                        food_ahead_ticks += 1;
                        action_hist_when_ahead[action_index(rec.selected_action)] += 1;
                        if fwd {
                            fwd_when_food_ahead += 1;
                        }
                    }
                }
            }
            desc_pop_sum += desc_pop;
            food_count_sum += sim.foods().len() as u64;
            samples += 1;
        }

        // Flush still-alive descendants at TO (mirrors the eval's end-of-run flush).
        let end_bucket = bucket_of(to);
        for life in lifetime.values() {
            if life.is_descendant {
                let e = buckets.entry(end_bucket).or_insert((0, 0));
                e.0 += life.food_ahead;
                e.1 += life.fwd;
            }
        }

        // ---- Report: live behavioral probe ----
        writeln!(
            out,
            "== foraging trace [{start}, {to}] ({samples} ticks) =="
        )?;
        writeln!(
            out,
            "-- live behavioral probe (descendant organism-ticks) --"
        )?;
        let mean_desc_pop = ratio(desc_pop_sum, samples);
        let mean_food = ratio(food_count_sum, samples);
        writeln!(
            out,
            "  mean descendant population = {mean_desc_pop:.0}   mean food items = {mean_food:.0}"
        )?;
        writeln!(out, "  descendant ticks = {desc_ticks}")?;
        writeln!(
            out,
            "  any-food-visible ticks = {any_food_visible_ticks} ({:.2}% of descendant ticks)",
            pct(any_food_visible_ticks, desc_ticks)
        )?;
        writeln!(
            out,
            "  food-AHEAD ticks       = {food_ahead_ticks} ({:.2}% of descendant ticks)",
            pct(food_ahead_ticks, desc_ticks)
        )?;
        if food_ahead_ticks == 0 {
            writeln!(
                out,
                "  p_fwd_food = N/A (no food-ahead ticks → eval denominator 0 → pillar 0)"
            )?;
        } else {
            let p = fwd_when_food_ahead as f64 / food_ahead_ticks as f64;
            writeln!(
                out,
                "  p_fwd_food = {p:.4}  (fwd {fwd_when_food_ahead} / food-ahead {food_ahead_ticks}; baseline {P_BASELINE:.4})"
            )?;
            writeln!(out, "  action taken WHEN FOOD AHEAD:")?;
            for (i, name) in ACTION_NAMES.iter().enumerate() {
                let c = action_hist_when_ahead[i];
                if c > 0 {
                    writeln!(
                        out,
                        "    {name:<9} {c:>10} ({:.2}%)",
                        pct(c, food_ahead_ticks)
                    )?;
                }
            }
        }

        // ---- Report: eval-style window pillar ----
        writeln!(
            out,
            "-- eval-style window pillar (death-tick-bucketed lifetimes; counters since FROM, approximate) --"
        )?;
        let mut bkeys: Vec<u64> = buckets.keys().copied().collect();
        bkeys.sort_unstable();
        let mut ratios: Vec<f64> = Vec::new();
        for b in &bkeys {
            let (fa, fw) = buckets[b];
            let lo = (b - 1) * REPORT_EVERY + 1;
            let hi = b * REPORT_EVERY;
            if fa == 0 {
                writeln!(
                    out,
                    "  interval ({lo},{hi}]: no food-ahead deaths → skipped"
                )?;
            } else {
                let r = fw as f64 / fa as f64;
                ratios.push(r);
                writeln!(
                    out,
                    "  interval ({lo},{hi}]: p_fwd_food = {r:.4}  (fwd {fw} / food-ahead {fa})"
                )?;
            }
        }
        if ratios.is_empty() {
            writeln!(out, "  mean p_fwd_food = N/A → foraging pillar = 0.000")?;
        } else {
            let mean = ratios.iter().sum::<f64>() / ratios.len() as f64;
            let pillar = (((mean - P_BASELINE) / (PILLAR_TARGET - P_BASELINE)).clamp(0.0, 1.0)
                * 1000.0)
                .round()
                / 1000.0;
            writeln!(
                out,
                "  mean p_fwd_food = {mean:.4} over {} interval(s) → foraging pillar = {pillar:.3}",
                ratios.len()
            )?;
        }
        Ok(())
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

struct Life {
    is_descendant: bool,
    food_ahead: u64,
    fwd: u64,
}

const ACTION_NAMES: [&str; 7] = [
    "Idle",
    "TurnLeft",
    "TurnRight",
    "Forward",
    "Eat",
    "Attack",
    "Reproduce",
];

fn action_index(a: ActionType) -> usize {
    match a {
        ActionType::Idle => 0,
        ActionType::TurnLeft => 1,
        ActionType::TurnRight => 2,
        ActionType::Forward => 3,
        ActionType::Eat => 4,
        ActionType::Attack => 5,
        ActionType::Reproduce => 6,
    }
}

/// 1-based interval index containing `turn` (interval i covers ((i-1)*R, i*R]).
fn bucket_of(turn: u64) -> u64 {
    turn.saturating_sub(1) / REPORT_EVERY + 1
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

fn stats_line(vals: &[f64]) -> String {
    if vals.is_empty() {
        return "(none)".to_string();
    }
    let mut v = vals.to_vec();
    v.sort_by(f64::total_cmp);
    let min = v[0];
    let max = v[v.len() - 1];
    let mean = v.iter().sum::<f64>() / v.len() as f64;
    let median = v[v.len() / 2];
    format!("min={min:.2} median={median:.2} mean={mean:.2} max={max:.2}")
}

fn ratio(num: u64, den: u64) -> f64 {
    if den == 0 {
        0.0
    } else {
        num as f64 / den as f64
    }
}

/// Format an optional metric, printing `NA` when the signal is absent.
fn opt(value: Option<f64>, decimals: usize) -> String {
    value
        .map(|v| format!("{v:.decimals$}"))
        .unwrap_or_else(|| "NA".to_string())
}

fn pct(num: u64, den: u64) -> f64 {
    if den == 0 {
        0.0
    } else {
        num as f64 / den as f64 * 100.0
    }
}
