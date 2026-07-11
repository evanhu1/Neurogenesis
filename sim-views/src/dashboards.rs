//! Aggregate dashboard reads: ecological dynamics, lineage/species composition,
//! genome-gene drift, and recorded timeseries. Free functions over a
//! [`crate::ReadCtx`], rendering to a writer (text or JSON).

use crate::output::{opt, sparkline, Stats};
use crate::{take_format, EcoSample, ReadCtx};
use anyhow::{anyhow, Result};
use serde_json::{json, Value};
use sim_metrics::derive_interval_metrics;
use sim_types::OrganismState;
use std::io::Write;

/// Cap on sparkline width: long series are bucket-averaged down to this many
/// chars so output stays token-cheap.
const SPARK_CAP: usize = 40;

/// Downsample `vals` to at most `SPARK_CAP` points by averaging contiguous
/// buckets, then render a sparkline. Short series pass through untouched.
fn spark(vals: &[f64]) -> String {
    sparkline(&downsample(vals))
}

/// Average-pool `vals` into at most `SPARK_CAP` buckets. Returns a clone when
/// already short enough.
fn downsample(vals: &[f64]) -> Vec<f64> {
    if vals.len() <= SPARK_CAP {
        return vals.to_vec();
    }
    let mut out = Vec::with_capacity(SPARK_CAP);
    for b in 0..SPARK_CAP {
        let lo = b * vals.len() / SPARK_CAP;
        let hi = ((b + 1) * vals.len() / SPARK_CAP)
            .max(lo + 1)
            .min(vals.len());
        let slice = &vals[lo..hi];
        let mean = slice.iter().sum::<f64>() / slice.len() as f64;
        out.push(mean);
    }
    out
}

fn mean(vals: &[f64]) -> f64 {
    if vals.is_empty() {
        0.0
    } else {
        vals.iter().sum::<f64>() / vals.len() as f64
    }
}

pub fn eco(ctx: &ReadCtx, args: &[&str], out: &mut impl Write) -> Result<()> {
    let (fmt, _) = take_format(ctx.format, args);
    let sim = ctx.sim;
    let turn = sim.turn();
    let pop = sim.organisms().len() as u64;
    let (plants, corpses, food_energy) = crate::food_summary(sim);
    let food_tiles = sim.food_tile_count();
    let habitable_cells = sim.habitable_cell_count();
    let realized_food_tile_fraction = if habitable_cells == 0 {
        0.0
    } else {
        food_tiles as f64 / habitable_cells as f64
    };

    // Point-in-time block (always present).
    let recorder = ctx.recorder;
    let have_traj = recorder.map(|r| !r.samples.is_empty()).unwrap_or(false);

    if fmt.is_json() {
        let mut v = json!({
            "turn": turn,
            "population": pop,
            "food": { "plants": plants, "corpses": corpses, "total_energy": food_energy },
            "food_tiles": {
                "selected": food_tiles,
                "habitable_cells": habitable_cells,
                "realized_fraction": realized_food_tile_fraction,
                "configured_fraction": sim.config().food_tile_fraction,
            },
        });
        if let Some(rec) = recorder.filter(|_| have_traj) {
            let s = &rec.samples;
            let pops: Vec<f64> = s.iter().map(|x| x.population as f64).collect();
            let foods: Vec<f64> = s.iter().map(|x| x.food as f64).collect();
            let deaths = s.iter().map(|x| x.deaths as u64).sum::<u64>();
            let starv = s.iter().map(|x| x.starvations as u64).sum::<u64>();
            let aged = s.iter().map(|x| x.age_deaths as u64).sum::<u64>();
            let preyed = s.iter().map(|x| x.predations as u64).sum::<u64>();
            let n = s.len() as f64;
            let cons: Vec<f64> = s.iter().map(|x| x.consumptions as f64).collect();
            let preds: Vec<f64> = s.iter().map(|x| x.predations as f64).collect();
            // Start of the last ~20% of samples (at least one sample).
            let tail = (s.len() - s.len() / 5).min(s.len() - 1);
            let cc: Vec<f64> = s[tail..].iter().map(|x| x.population as f64).collect();
            v["trajectory"] = json!({
                "ticks": s.len(),
                "population_series": downsample(&pops),
                "food_series": downsample(&foods),
                "deaths_per_tick": deaths as f64 / n,
                "deaths_by_cause": {
                    "total": deaths,
                    "starvation": starv,
                    "age": aged,
                    "predation": preyed,
                    // total - itemized deaths, if any future cause is not separately counted
                    "other": deaths.saturating_sub(starv + aged + preyed),
                },
                "consumptions_per_tick": mean(&cons),
                "predations_per_tick": mean(&preds),
                "carrying_capacity_est": mean(&cc),
            });
        } else {
            v["trajectory"] = Value::Null;
            v["note"] = json!("trajectories need `record on` then advance");
        }
        return writeln!(out, "{v}").map_err(Into::into);
    }

    writeln!(
        out,
        "eco @ turn {turn}: population={pop} \
         food: plants={plants} corpses={corpses} energy={food_energy:.0}"
    )?;
    if let Some(rec) = recorder.filter(|_| have_traj) {
        let s = &rec.samples;
        let pops: Vec<f64> = s.iter().map(|x| x.population as f64).collect();
        let foods: Vec<f64> = s.iter().map(|x| x.food as f64).collect();
        let deaths = s.iter().map(|x| x.deaths as u64).sum::<u64>();
        let starv = s.iter().map(|x| x.starvations as u64).sum::<u64>();
        let aged = s.iter().map(|x| x.age_deaths as u64).sum::<u64>();
        let preyed = s.iter().map(|x| x.predations as u64).sum::<u64>();
        let n = s.len() as f64;
        let cons: Vec<f64> = s.iter().map(|x| x.consumptions as f64).collect();
        let preds: Vec<f64> = s.iter().map(|x| x.predations as f64).collect();
        let tail = (s.len() - s.len() / 5).min(s.len() - 1);
        let cc = mean(
            &s[tail..]
                .iter()
                .map(|x| x.population as f64)
                .collect::<Vec<_>>(),
        );
        let plast = pops.last().copied().unwrap_or(0.0);
        let flast = foods.last().copied().unwrap_or(0.0);
        writeln!(out, "trajectory over {} recorded ticks:", s.len())?;
        writeln!(out, "  population {} (now {plast:.0})", spark(&pops))?;
        writeln!(out, "  food       {} (now {flast:.0})", spark(&foods))?;
        writeln!(
            out,
            "  rates/tick: deaths={:.4} consumptions={:.4} predations={:.4}",
            deaths as f64 / n,
            mean(&cons),
            mean(&preds),
        )?;
        writeln!(
            out,
            "  deaths-by-cause (window sum, total={deaths}): starvation={starv} age={aged} predation={preyed} other={}",
            deaths.saturating_sub(starv + aged + preyed)
        )?;
        writeln!(out, "  carrying_capacity_est (last ~20%) = {cc:.1}")?;
    } else {
        writeln!(
            out,
            "  (no trajectory: `record on` then advance to populate)"
        )?;
    }
    Ok(())
}

pub fn lineage(ctx: &ReadCtx, args: &[&str], out: &mut impl Write) -> Result<()> {
    let (fmt, _) = take_format(ctx.format, args);
    let sim = ctx.sim;
    let orgs = sim.organisms();
    let pop = orgs.len();
    let gens: Vec<f64> = orgs.iter().map(|o| o.generation as f64).collect();
    let gen_stats = Stats::of(&gens);

    // Founder-lineage (species) composition: count per species_id.
    let mut species: Vec<(u64, u64)> = Vec::new();
    for o in orgs {
        let sid = o.species_id.0;
        match species.binary_search_by_key(&sid, |&(s, _)| s) {
            Ok(i) => species[i].1 += 1,
            Err(i) => species.insert(i, (sid, 1)),
        }
    }
    let distinct = species.len();
    let mut by_count = species.clone();
    by_count.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    let top: Vec<(u64, u64)> = by_count.iter().take(8).copied().collect();

    // Generation-distribution histogram (per integer generation up to max).
    let max_gen = gens.iter().cloned().fold(0.0, f64::max) as usize;
    let mut gen_hist = vec![0u64; max_gen + 1];
    for o in orgs {
        gen_hist[o.generation as usize] += 1;
    }
    let gen_hist_f: Vec<f64> = gen_hist.iter().map(|&c| c as f64).collect();

    if fmt.is_json() {
        let v = json!({
            "population": pop,
            "generation": {
                "stats": gen_stats.map(|s| s.json()),
                "histogram": gen_hist,
            },
            "lineages": {
                "distinct": distinct,
                "top": top.iter().map(|&(sid, c)| json!({
                    "species_id": sid,
                    "count": c,
                    "pct": if pop > 0 { c as f64 / pop as f64 * 100.0 } else { 0.0 },
                })).collect::<Vec<_>>(),
            },
            "generation_time": Value::Null,
            "note": "generation_time (parent-age-at-reproduction) not tracked by the recorder",
        });
        return writeln!(out, "{v}").map_err(Into::into);
    }

    writeln!(out, "lineage @ population {pop}:")?;
    match gen_stats {
        Some(s) => writeln!(
            out,
            "  generation: {}  dist {}",
            s.text(),
            spark(&gen_hist_f)
        )?,
        None => writeln!(out, "  generation: (no organisms)")?,
    }
    writeln!(out, "  founder-lineages: {distinct} distinct (diversity)")?;
    for (sid, c) in &top {
        let pct = if pop > 0 {
            *c as f64 / pop as f64 * 100.0
        } else {
            0.0
        };
        writeln!(out, "    species {sid:<6} count={c:<6} {pct:5.1}%")?;
    }
    if ctx.recorder.is_none() {
        writeln!(
            out,
            "  (generation-time needs `record on`; parent-age not tracked)"
        )?;
    } else {
        writeln!(
            out,
            "  (generation-time: parent-age-at-reproduction not tracked by recorder)"
        )?;
    }
    Ok(())
}

pub fn genome(ctx: &ReadCtx, args: &[&str], out: &mut impl Write) -> Result<()> {
    let (fmt, rest) = take_format(ctx.format, args);
    let mut only_gene: Option<&str> = None;
    let mut drift = false;
    let mut i = 0;
    while i < rest.len() {
        match rest[i] {
            "--gene" => {
                only_gene = Some(
                    rest.get(i + 1)
                        .copied()
                        .ok_or_else(|| anyhow!("--gene needs a name"))?,
                );
                i += 2;
            }
            "--drift" => {
                drift = true;
                i += 1;
            }
            other => return Err(anyhow!("unknown genome arg `{other}`")),
        }
    }

    let sim = ctx.sim;
    let orgs = sim.organisms();

    // (name, group, extractor). Structural counts are derived from the
    // canonical direct graph; synapses use the expressed runtime count.
    type Extractor = fn(&OrganismState) -> f64;
    let genes: &[(&str, &str, Extractor)] = &[
        ("num_neurons", "topology", |o| {
            o.genome.hidden_node_count() as f64
        }),
        ("num_synapses", "topology", |o| o.brain.synapse_count as f64),
        ("vision_distance", "topology", |o| {
            o.genome.topology.vision_distance as f64
        }),
        ("age_of_maturity", "lifecycle", |o| {
            o.genome.lifecycle.age_of_maturity as f64
        }),
        ("gestation_ticks", "lifecycle", |o| {
            o.genome.lifecycle.gestation_ticks as f64
        }),
        ("max_organism_age", "lifecycle", |o| {
            o.genome.lifecycle.max_organism_age as f64
        }),
        ("hebb_eta_gain", "plasticity", |o| {
            o.genome.plasticity.hebb_eta_gain as f64
        }),
        ("juvenile_eta_scale", "plasticity", |o| {
            o.genome.plasticity.juvenile_eta_scale as f64
        }),
        ("eligibility_retention", "plasticity", |o| {
            o.genome.plasticity.eligibility_retention as f64
        }),
        ("max_weight_delta_per_tick", "plasticity", |o| {
            o.genome.plasticity.max_weight_delta_per_tick as f64
        }),
        ("synapse_prune_threshold", "plasticity", |o| {
            o.genome.plasticity.synapse_prune_threshold as f64
        }),
    ];

    let collect = |f: Extractor| -> Vec<f64> { orgs.iter().map(f).collect() };

    // Single-gene mode.
    if let Some(name) = only_gene {
        let found = genes.iter().find(|(g, _, _)| *g == name);
        let vals = match found {
            Some((_, _, f)) => collect(*f),
            None => return Err(anyhow!("unknown gene `{name}`")),
        };
        let stats = Stats::of(&vals);
        if fmt.is_json() {
            let v = json!({ "gene": name, "stats": stats.map(|s| s.json()) });
            return writeln!(out, "{v}").map_err(Into::into);
        }
        match stats {
            Some(s) => writeln!(out, "{name}: {}", s.text())?,
            None => writeln!(out, "{name}: (no organisms)")?,
        }
        return Ok(());
    }

    // Drift baseline: compare current mean to start-of-window mean from the
    // recorder. The recorder does not retain per-gene history, so two population
    // samples aren't available — emit a clear note instead.
    let drift_note = if drift {
        Some("drift needs two genome samples; the recorder does not retain per-gene history")
    } else {
        None
    };

    // Plasticity genes only drive behavior when within-life learning is on;
    // flag when the world has it disabled so the values aren't misread.
    let plasticity_enabled = ctx.sim.config().runtime_plasticity_enabled;

    if fmt.is_json() {
        let mut gene_obj = serde_json::Map::new();
        for (name, group, f) in genes {
            let s = Stats::of(&collect(*f));
            gene_obj.insert(
                (*name).to_string(),
                json!({ "group": group, "stats": s.map(|s| s.json()) }),
            );
        }
        let mut v = json!({
            "population": orgs.len(),
            "runtime_plasticity_enabled": plasticity_enabled,
            "genes": gene_obj,
        });
        if let Some(n) = drift_note {
            v["drift_note"] = json!(n);
        }
        return writeln!(out, "{v}").map_err(Into::into);
    }

    writeln!(out, "genome-gene distribution @ population {}:", orgs.len())?;
    let mut last_group = "";
    for (name, group, f) in genes {
        if *group != last_group {
            let tag = if *group == "plasticity" && !plasticity_enabled {
                "  (disabled — runtime_plasticity_enabled=false)"
            } else {
                ""
            };
            writeln!(out, "  [{group}]{tag}")?;
            last_group = group;
        }
        match Stats::of(&collect(*f)) {
            Some(s) => writeln!(out, "    {name:<26} {}", s.text())?,
            None => writeln!(out, "    {name:<26} (none)")?,
        }
    }
    if let Some(n) = drift_note {
        writeln!(out, "  {n}")?;
    }
    Ok(())
}

pub fn timeseries(ctx: &ReadCtx, args: &[&str], out: &mut impl Write) -> Result<()> {
    let (fmt, rest) = take_format(ctx.format, args);
    let mut cols: Vec<String> = Vec::new();
    let mut last: Option<usize> = None;
    let mut i = 0;
    while i < rest.len() {
        match rest[i] {
            "--cols" => {
                let list = rest
                    .get(i + 1)
                    .copied()
                    .ok_or_else(|| anyhow!("--cols needs a comma-separated list"))?;
                cols = list.split(',').map(|s| s.trim().to_string()).collect();
                i += 2;
            }
            "--last" => {
                let k: usize = rest
                    .get(i + 1)
                    .copied()
                    .ok_or_else(|| anyhow!("--last needs a value"))?
                    .parse()?;
                if k == 0 {
                    return Err(anyhow!("--last must be >= 1"));
                }
                last = Some(k);
                i += 2;
            }
            other => return Err(anyhow!("unknown timeseries arg `{other}`")),
        }
    }
    if cols.is_empty() {
        cols = ["population", "food", "plant_consumption_rate", "mi_sa"]
            .iter()
            .map(|s| s.to_string())
            .collect();
    }

    let report_every = ctx.report_every;
    let rec = ctx
        .recorder
        .ok_or_else(|| anyhow!("timeseries requires recording; `record on` then advance"))?;
    if rec.samples.is_empty() {
        return Err(anyhow!("no recorded samples yet; advance the sim first"));
    }
    let current_turn = ctx.sim.turn();

    // Sample-keyed columns (per recorded tick).
    let s = &rec.samples;
    let sample_col = |name: &str| -> Option<Vec<f64>> {
        let f: fn(&EcoSample) -> f64 = match name {
            "population" => |x| x.population as f64,
            "food" => |x| x.food as f64,
            "deaths" => |x| x.deaths as f64,
            "consumptions" => |x| x.consumptions as f64,
            "predations" => |x| x.predations as f64,
            _ => return None,
        };
        Some(s.iter().map(f).collect())
    };

    // Interval-keyed columns (per reporting interval, from the metrics layer).
    let intervals = derive_interval_metrics(
        &rec.tick_summary,
        &rec.lifetimes,
        report_every,
        current_turn,
    );
    let interval_col = |name: &str| -> Option<Vec<f64>> {
        let f: fn(&sim_metrics::IntervalMetrics) -> f64 = match name {
            "action_effectiveness" => |m| m.action_effectiveness.unwrap_or(f64::NAN),
            "plant_consumption_rate" => |m| m.plant_consumption_rate.unwrap_or(f64::NAN),
            "prey_consumption_rate" => |m| m.prey_consumption_rate.unwrap_or(f64::NAN),
            "mi_sa" => |m| m.mi_sa.unwrap_or(f64::NAN),
            "learning_slope" => |m| m.learning_slope.unwrap_or(f64::NAN),
            "interval_population" => |m| m.pop as f64,
            _ => return None,
        };
        Some(intervals.iter().map(f).collect())
    };

    // Resolve each requested column to a numeric series, trimming to --last.
    let mut resolved: Vec<(String, Vec<f64>)> = Vec::new();
    for name in &cols {
        let series = sample_col(name).or_else(|| interval_col(name));
        let mut series = series.ok_or_else(|| {
            anyhow!(
                "unknown column `{name}`; valid: population food \
                 deaths consumptions predations action_effectiveness \
                 plant_consumption_rate prey_consumption_rate mi_sa learning_slope \
                 interval_population"
            )
        })?;
        if let Some(k) = last {
            if series.len() > k {
                series = series[series.len() - k..].to_vec();
            }
        }
        resolved.push((name.clone(), series));
    }

    if fmt.is_json() {
        let mut obj = serde_json::Map::new();
        for (name, series) in &resolved {
            // NaN isn't valid JSON; encode missing interval values as null.
            let arr: Vec<Value> = series
                .iter()
                .map(|&v| if v.is_nan() { Value::Null } else { json!(v) })
                .collect();
            obj.insert(name.clone(), Value::Array(arr));
        }
        return writeln!(out, "{}", Value::Object(obj)).map_err(Into::into);
    }

    for (name, series) in &resolved {
        // For sparkline + last value, treat NaN as gaps by substituting the
        // series mean so scaling stays sane; report last real value.
        let finite: Vec<f64> = series.iter().copied().filter(|v| v.is_finite()).collect();
        let fill = mean(&finite);
        let plot: Vec<f64> = series
            .iter()
            .map(|&v| if v.is_finite() { v } else { fill })
            .collect();
        let last_v = series.iter().rev().find(|v| v.is_finite()).copied();
        writeln!(
            out,
            "  {name:<26} {} last={} (n={})",
            spark(&plot),
            opt(last_v, 4),
            series.len(),
        )?;
    }
    Ok(())
}
