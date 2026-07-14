//! The `watch` mutating loop: advance to a target turn, emitting a compact
//! metrics row every `--every` ticks. Advancement and metric derivation come
//! from `views`; this file owns only the loop and its row formatting (the
//! read dashboards themselves — eco/lineage/genome/timeseries — live in
//! `views`).

use crate::App;
use anyhow::{anyhow, Result};
use serde_json::json;
use std::io::Write;
use views::output::{opt, Format};
use views::{food_summary, take_format};

impl App {
    pub(crate) fn watch(&mut self, args: &[&str], out: &mut impl Write) -> Result<()> {
        let (fmt, rest) = take_format(self.format, args);
        let target: u64 = rest
            .first()
            .ok_or_else(|| anyhow!("watch needs a target turn"))?
            .parse()?;
        let mut every = self.report_every;
        let mut i = 1;
        while i < rest.len() {
            match rest[i] {
                "--every" => {
                    every = rest
                        .get(i + 1)
                        .copied()
                        .ok_or_else(|| anyhow!("--every needs a value"))?
                        .parse()?;
                    i += 2;
                }
                other => return Err(anyhow!("unknown watch arg `{other}`")),
            }
        }
        if every == 0 {
            return Err(anyhow!("--every must be >= 1"));
        }
        let current = self
            .sim
            .as_ref()
            .map(|s| s.turn())
            .ok_or_else(|| anyhow!("no simulation loaded; run `load` first"))?;
        if target <= current {
            return Err(anyhow!(
                "watch target {target} must be > current turn {current}"
            ));
        }

        // Advance in `every`-sized chunks, emitting a row after each, plus a
        // final partial chunk landing exactly on `target`.
        while self.sim.as_ref().map(|s| s.turn()).unwrap_or(target) < target {
            let now = self.sim.as_ref().map(|s| s.turn()).unwrap_or(target);
            let step = every.min(target - now);
            self.advance(step)?;
            self.emit_watch_row(fmt, out)?;
        }
        Ok(())
    }

    /// Emit one compact metrics row for `watch` (text or one JSON object/line).
    fn emit_watch_row(&self, fmt: Format, out: &mut impl Write) -> Result<()> {
        let ctx = self.read_ctx()?;
        let pillars = ctx.live_pillars();
        let sim = ctx.sim;
        let turn = sim.turn();
        let pop = sim.organisms().len() as u64;
        let (plants, food_energy) = food_summary(sim);

        if fmt.is_json() {
            let mut v = json!({
                "turn": turn,
                "population": pop,
                "food": { "plants": plants, "total_energy": food_energy },
            });
            if let Some((p, _, partial)) = pillars {
                v["metrics"] = json!({
                    "plant_consumption_rate": p.mean_plant_consumption_rate,
                    "prey_consumption_rate": p.mean_prey_consumption_rate,
                    "action_effectiveness": p.mean_action_effectiveness,
                    "mi_sa": p.mean_mi_sa,
                    "learning_slope": p.mean_learning_slope,
                    "partial": partial,
                });
            }
            return writeln!(out, "{v}").map_err(Into::into);
        }

        let food = plants;
        if let Some((p, _, partial)) = pillars {
            writeln!(
                out,
                "t={turn:<8} pop={pop:<6} food={food:<6} \
                 plant={} prey={} eff={} mi={} slope={}{}",
                opt(p.mean_plant_consumption_rate, 4),
                opt(p.mean_prey_consumption_rate, 4),
                opt(p.mean_action_effectiveness, 4),
                opt(p.mean_mi_sa, 4),
                opt(p.mean_learning_slope, 6),
                if partial { " [PARTIAL]" } else { "" },
            )
            .map_err(Into::into)
        } else {
            writeln!(
                out,
                "t={turn:<8} pop={pop:<6} food={food:<6} energy={food_energy:.0}"
            )
            .map_err(Into::into)
        }
    }
}
