use crate::{
    ledger::N_ACTIONS,
    metrics::{action_baseline_entropy, action_baseline_probability, IntervalMetrics},
};
use anyhow::Result;
use serde::Serialize;
use std::fmt::Write as _;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

pub struct Reporter {
    csv: BufWriter<File>,
}

impl Reporter {
    pub fn new(out_dir: &Path) -> Result<Self> {
        let csv_path = out_dir.join("timeseries.csv");
        let mut csv = BufWriter::new(File::create(csv_path)?);
        writeln!(
            csv,
            "tick,pop,births,deaths,food,max_generation,life_mean,predation_rate,foraging_rate,attack_attempt_rate,attack_success_rate,ate_pct,cons_mean,brain_size,brain_size_stddev,brain_size_p10,brain_size_p50,brain_size_p90,lineage_diversity,p_fwd_food,mi_sa,mi_sa_juvenile,mi_sa_adult,h_action,idle_fraction,reproduction_efficiency,damage_avoidance,reward_reversal_shift,util"
        )?;
        Ok(Self { csv })
    }

    pub fn emit(&mut self, metrics: &IntervalMetrics) -> Result<()> {
        writeln!(
            self.csv,
            "{tick},{pop},{births},{deaths},{food},{max_generation},{life_mean},{predation_rate},{foraging_rate},{attack_attempt_rate},{attack_success_rate},{ate_pct},{cons_mean},{brain_size},{brain_size_stddev},{brain_size_p10},{brain_size_p50},{brain_size_p90},{lineage_diversity},{p_fwd_food},{mi_sa},{mi_sa_juvenile},{mi_sa_adult},{h_action},{idle_fraction},{reproduction_efficiency},{damage_avoidance},{reward_reversal_shift},{util}",
            tick = metrics.tick,
            pop = metrics.pop,
            births = metrics.births,
            deaths = metrics.deaths,
            food = metrics.food,
            max_generation = csv_opt_u64(metrics.max_generation),
            life_mean = csv_opt(metrics.life_mean),
            predation_rate = csv_opt(metrics.predation_rate),
            foraging_rate = csv_opt(metrics.foraging_rate),
            attack_attempt_rate = csv_opt(metrics.attack_attempt_rate),
            attack_success_rate = csv_opt(metrics.attack_success_rate),
            ate_pct = csv_opt(metrics.ate_pct),
            cons_mean = csv_opt(metrics.cons_mean),
            brain_size = csv_opt(metrics.brain_size),
            brain_size_stddev = csv_opt(metrics.brain_size_stddev),
            brain_size_p10 = csv_opt(metrics.brain_size_p10),
            brain_size_p50 = csv_opt(metrics.brain_size_p50),
            brain_size_p90 = csv_opt(metrics.brain_size_p90),
            lineage_diversity = csv_opt(metrics.lineage_diversity),
            p_fwd_food = csv_opt(metrics.p_fwd_food),
            mi_sa = csv_opt(metrics.mi_sa),
            mi_sa_juvenile = csv_opt(metrics.mi_sa_juvenile),
            mi_sa_adult = csv_opt(metrics.mi_sa_adult),
            h_action = csv_opt(metrics.h_action),
            idle_fraction = csv_opt(metrics.idle_fraction),
            reproduction_efficiency = csv_opt(metrics.reproduction_efficiency),
            damage_avoidance = csv_opt(metrics.damage_avoidance),
            reward_reversal_shift = csv_opt(metrics.reward_reversal_shift),
            util = csv_opt(metrics.util),
        )?;

        Ok(())
    }

    pub fn flush(&mut self) -> Result<()> {
        self.csv.flush()?;
        Ok(())
    }
}

pub struct HtmlReportMeta {
    pub title: Option<String>,
    pub seed_label: String,
    pub seed_count: usize,
    pub ticks: u64,
    pub report_every: u64,
    pub min_lifetime: u64,
    pub baseline: bool,
    pub total_time_seconds: f64,
    pub generated_at_utc: String,
    pub aggregate_score: f64,
    pub aggregate_score_median: f64,
    pub aggregate_score_stddev: f64,
    pub aggregate_score_min: f64,
    pub aggregate_score_max: f64,
    pub aggregate_window_start_tick: u64,
    pub aggregate_window_end_tick: u64,
    pub aggregate_p_component: f64,
    pub aggregate_mi_component: f64,
    pub aggregate_entropy_component: f64,
    pub aggregate_predation_component: f64,
    pub aggregate_mean_p_fwd_food: Option<f64>,
    pub aggregate_mean_mi_sa: Option<f64>,
    pub aggregate_mean_mi_sa_juvenile: Option<f64>,
    pub aggregate_mean_mi_sa_adult: Option<f64>,
    pub aggregate_mean_h_action: Option<f64>,
    pub aggregate_mean_predation_rate: Option<f64>,
    pub aggregate_mean_foraging_rate: Option<f64>,
    pub aggregate_mean_attack_attempt_rate: Option<f64>,
    pub aggregate_mean_attack_success_rate: Option<f64>,
    pub aggregate_mean_idle_fraction: Option<f64>,
    pub aggregate_mean_reproduction_efficiency: Option<f64>,
    pub aggregate_mean_lineage_diversity: Option<f64>,
    pub aggregate_mean_damage_avoidance: Option<f64>,
    pub aggregate_mean_reward_reversal_shift: Option<f64>,
    pub aggregate_reward_reversal_adaptation_ticks: Option<u64>,
    pub timeseries_label: String,
    pub per_seed_rows: Vec<PerSeedReportRow>,
}

#[derive(Debug, Clone, Serialize)]
pub struct PerSeedReportRow {
    pub seed: u64,
    pub score: f64,
    pub total_time_seconds: f64,
    pub state_hash: String,
    pub report_href: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct ComparisonMetricRow {
    pub label: String,
    pub control_mean: Option<f64>,
    pub treatment_mean: Option<f64>,
    pub mean_diff: Option<f64>,
    pub ci_low: Option<f64>,
    pub ci_high: Option<f64>,
}

#[derive(Debug, Clone, Serialize)]
pub struct PerSeedComparisonRow {
    pub seed: u64,
    pub control_score: f64,
    pub treatment_score: f64,
    pub diff_score: f64,
    pub control_report_href: String,
    pub treatment_report_href: String,
}

pub struct ComparisonHtmlReportMeta {
    pub title: Option<String>,
    pub seed_label: String,
    pub ticks: u64,
    pub control_label: String,
    pub treatment_label: String,
    pub total_time_seconds: f64,
    pub metric_rows: Vec<ComparisonMetricRow>,
    pub per_seed_rows: Vec<PerSeedComparisonRow>,
    pub control_report_href: String,
    pub treatment_report_href: String,
}

pub fn write_html_report(
    out_dir: &Path,
    meta: &HtmlReportMeta,
    rows: &[IntervalMetrics],
) -> Result<()> {
    let report_path = out_dir.join("report.html");
    let mut html = String::new();
    html.push_str(
        "<!doctype html><html><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">",
    );
    html.push_str("<title>Sim Validation Report</title>");
    html.push_str(
        "<style>\
        :root{--bg:#f5f7fb;--panel:#ffffff;--ink:#0f172a;--muted:#64748b;--line:#dbe2ea;--accent:#0f766e;--base:#b45309;}\
        *{box-sizing:border-box}body{margin:0;font-family:ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,Arial,sans-serif;background:var(--bg);color:var(--ink)}\
        .wrap{max-width:1200px;margin:24px auto;padding:0 16px}\
        .panel{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:16px;margin-bottom:16px}\
        h1,h2{margin:0 0 10px 0}h1{font-size:24px}h2{font-size:18px}\
        .meta{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:10px}\
        .k{color:var(--muted);font-size:12px;text-transform:uppercase;letter-spacing:.04em}.v{font-weight:600}\
        table{width:100%;border-collapse:collapse;font-size:13px}th,td{padding:8px;border-bottom:1px solid var(--line);text-align:right}th:first-child,td:first-child{text-align:left}\
        .chart{margin:8px 0 20px 0}svg{width:100%;height:auto;border:1px solid var(--line);border-radius:8px;background:#fff}\
        .note{color:var(--muted);font-size:12px}\
        .score-big{font-size:42px;font-weight:700;line-height:1;margin:0}\
        .score-sub{margin:8px 0 0 0;color:var(--muted);font-size:12px}\
        .score-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:10px;margin-top:12px}\
        .guide h3{margin:14px 0 8px 0;font-size:16px}.guide p{margin:0 0 10px 0;line-height:1.45}.guide ul{margin:0 0 12px 20px;line-height:1.45}\
        .guide li{margin:4px 0}.guide code{background:#f1f5f9;padding:1px 5px;border-radius:4px;border:1px solid #e2e8f0}\
        </style></head><body><div class=\"wrap\">",
    );

    html.push_str("<div class=\"panel\"><h1>Simulation Validation Report</h1><div class=\"meta\">");
    if let Some(title) = &meta.title {
        kv(&mut html, "Title", title);
    }
    kv(
        &mut html,
        if meta.seed_count == 1 {
            "Seed"
        } else {
            "Seeds"
        },
        &meta.seed_label,
    );
    kv(&mut html, "Ticks", &meta.ticks.to_string());
    kv(&mut html, "Report Every", &meta.report_every.to_string());
    kv(&mut html, "Min Lifetime", &meta.min_lifetime.to_string());
    kv(
        &mut html,
        "Baseline",
        if meta.baseline { "true" } else { "false" },
    );
    kv(
        &mut html,
        "Total Time",
        &format!("{:.3}s", meta.total_time_seconds),
    );
    kv(&mut html, "Generated At (UTC)", &meta.generated_at_utc);
    html.push_str("</div></div>");

    html.push_str("<div class=\"panel\"><h2>Aggregate Score</h2>");
    if meta.seed_count == 1 {
        let _ = write!(
            html,
            "<p class=\"score-big\">{:.2}</p>\
             <p class=\"score-sub\">window: ticks {}-{} | higher is better for quick run-to-run comparison</p>",
            meta.aggregate_score, meta.aggregate_window_start_tick, meta.aggregate_window_end_tick
        );
    } else {
        let _ = write!(
            html,
            "<p class=\"score-big\">{:.2}</p>\
             <p class=\"score-sub\">mean across {} seeds | median {:.2} | stddev {:.2} | min {:.2} | max {:.2} | window: ticks {}-{}</p>",
            meta.aggregate_score,
            meta.seed_count,
            meta.aggregate_score_median,
            meta.aggregate_score_stddev,
            meta.aggregate_score_min,
            meta.aggregate_score_max,
            meta.aggregate_window_start_tick,
            meta.aggregate_window_end_tick
        );
    }
    html.push_str("<div class=\"score-grid\">");
    if meta.seed_count > 1 {
        kv(
            &mut html,
            "Median Score",
            &format!("{:.2}", meta.aggregate_score_median),
        );
        kv(
            &mut html,
            "Score Stddev",
            &format!("{:.2}", meta.aggregate_score_stddev),
        );
        kv(
            &mut html,
            "Score Min",
            &format!("{:.2}", meta.aggregate_score_min),
        );
        kv(
            &mut html,
            "Score Max",
            &format!("{:.2}", meta.aggregate_score_max),
        );
    }
    kv(
        &mut html,
        "P(Fwd|food) component",
        &format!("{:.3}", meta.aggregate_p_component),
    );
    kv(
        &mut html,
        "MI(S;A) component",
        &format!("{:.3}", meta.aggregate_mi_component),
    );
    kv(
        &mut html,
        "Entropy component",
        &format!("{:.3}", meta.aggregate_entropy_component),
    );
    kv(
        &mut html,
        "Predation component",
        &format!("{:.3}", meta.aggregate_predation_component),
    );
    kv(
        &mut html,
        "Window mean P(Fwd|food)",
        &fmt_opt(meta.aggregate_mean_p_fwd_food, 4),
    );
    kv(
        &mut html,
        "Window mean MI(S;A)",
        &fmt_opt(meta.aggregate_mean_mi_sa, 4),
    );
    kv(
        &mut html,
        "Window mean juvenile MI(S;A)",
        &fmt_opt(meta.aggregate_mean_mi_sa_juvenile, 4),
    );
    kv(
        &mut html,
        "Window mean adult MI(S;A)",
        &fmt_opt(meta.aggregate_mean_mi_sa_adult, 4),
    );
    kv(
        &mut html,
        "Window mean H(action)",
        &fmt_opt(meta.aggregate_mean_h_action, 4),
    );
    kv(
        &mut html,
        "Window mean predation rate",
        &fmt_opt(meta.aggregate_mean_predation_rate, 6),
    );
    kv(
        &mut html,
        "Window mean foraging rate",
        &fmt_opt(meta.aggregate_mean_foraging_rate, 6),
    );
    kv(
        &mut html,
        "Window mean attack attempt rate",
        &fmt_opt(meta.aggregate_mean_attack_attempt_rate, 6),
    );
    kv(
        &mut html,
        "Window mean attack success rate",
        &fmt_opt(meta.aggregate_mean_attack_success_rate, 4),
    );
    kv(
        &mut html,
        "Window mean idle fraction",
        &fmt_opt(meta.aggregate_mean_idle_fraction, 4),
    );
    kv(
        &mut html,
        "Window mean reproduction efficiency",
        &fmt_opt(meta.aggregate_mean_reproduction_efficiency, 4),
    );
    kv(
        &mut html,
        "Window mean lineage diversity",
        &fmt_opt(meta.aggregate_mean_lineage_diversity, 4),
    );
    kv(
        &mut html,
        "Window mean damage avoidance",
        &fmt_opt(meta.aggregate_mean_damage_avoidance, 4),
    );
    kv(
        &mut html,
        "Window mean reversal shift",
        &fmt_opt(meta.aggregate_mean_reward_reversal_shift, 4),
    );
    kv(
        &mut html,
        "Reversal adaptation ticks",
        &fmt_opt_u64(meta.aggregate_reward_reversal_adaptation_ticks),
    );
    html.push_str("</div></div>");

    if !meta.per_seed_rows.is_empty() {
        html.push_str("<div class=\"panel\"><h2>Per-Seed Results</h2><table><thead><tr>");
        for header in ["seed", "score", "time_s", "state_hash", "report"] {
            let _ = write!(html, "<th>{header}</th>");
        }
        html.push_str("</tr></thead><tbody>");
        for row in &meta.per_seed_rows {
            let _ = write!(
                html,
                "<tr><td>{}</td><td>{:.2}</td><td>{:.3}</td><td>{}</td><td><a href=\"{}\">open</a></td></tr>",
                row.seed, row.score, row.total_time_seconds, row.state_hash, row.report_href
            );
        }
        html.push_str("</tbody></table></div>");
    }

    let _ = write!(
        html,
        "<div class=\"panel\"><h2>Timeseries</h2><p class=\"note\">{}</p><table><thead><tr>",
        meta.timeseries_label
    );
    for header in [
        "tick",
        "pop",
        "births",
        "deaths",
        "food",
        "max_generation",
        "life_mean",
        "predation_rate",
        "foraging_rate",
        "attack_attempt_rate",
        "attack_success_rate",
        "ate_pct",
        "cons_mean",
        "brain_size",
        "brain_size_stddev",
        "lineage_diversity",
        "p_fwd_food",
        "mi_sa",
        "mi_sa_juvenile",
        "mi_sa_adult",
        "h_action",
        "idle_fraction",
        "reproduction_efficiency",
        "damage_avoidance",
        "reward_reversal_shift",
        "util",
    ] {
        let _ = write!(html, "<th>{header}</th>");
    }
    html.push_str("</tr></thead><tbody>");
    for row in rows {
        html.push_str("<tr>");
        for cell in [
            row.tick.to_string(),
            row.pop.to_string(),
            row.births.to_string(),
            row.deaths.to_string(),
            row.food.to_string(),
            fmt_opt_u64(row.max_generation),
            fmt_opt(row.life_mean, 2),
            fmt_opt(row.predation_rate, 6),
            fmt_opt(row.foraging_rate, 6),
            fmt_opt(row.attack_attempt_rate, 6),
            fmt_opt(row.attack_success_rate, 4),
            fmt_opt(row.ate_pct, 2),
            fmt_opt(row.cons_mean, 2),
            fmt_opt(row.brain_size, 2),
            fmt_opt(row.brain_size_stddev, 2),
            fmt_opt(row.lineage_diversity, 4),
            fmt_opt(row.p_fwd_food, 4),
            fmt_opt(row.mi_sa, 4),
            fmt_opt(row.mi_sa_juvenile, 4),
            fmt_opt(row.mi_sa_adult, 4),
            fmt_opt(row.h_action, 4),
            fmt_opt(row.idle_fraction, 4),
            fmt_opt(row.reproduction_efficiency, 4),
            fmt_opt(row.damage_avoidance, 4),
            fmt_opt(row.reward_reversal_shift, 4),
            fmt_opt(row.util, 4),
        ] {
            let _ = write!(html, "<td>{cell}</td>");
        }
        html.push_str("</tr>");
    }
    html.push_str("</tbody></table></div>");

    let charts = [
        (
            "Population",
            metric_series(rows, |r| Some(r.pop as f64)),
            None,
            "#0f766e",
        ),
        (
            "Births",
            metric_series(rows, |r| Some(r.births as f64)),
            None,
            "#2563eb",
        ),
        (
            "Deaths",
            metric_series(rows, |r| Some(r.deaths as f64)),
            None,
            "#dc2626",
        ),
        (
            "Food",
            metric_series(rows, |r| Some(r.food as f64)),
            None,
            "#65a30d",
        ),
        (
            "Max Generation",
            metric_series(rows, |r| r.max_generation.map(|value| value as f64)),
            None,
            "#7e22ce",
        ),
        (
            "Life Length Mean",
            metric_series(rows, |r| r.life_mean),
            None,
            "#7c3aed",
        ),
        (
            "Predation Rate",
            metric_series(rows, |r| r.predation_rate),
            Some(0.0),
            "#be123c",
        ),
        (
            "Foraging Rate",
            metric_series(rows, |r| r.foraging_rate),
            Some(0.0),
            "#16a34a",
        ),
        (
            "Attack Attempt Rate",
            metric_series(rows, |r| r.attack_attempt_rate),
            Some(0.0),
            "#ef4444",
        ),
        (
            "Attack Success Rate",
            metric_series(rows, |r| r.attack_success_rate),
            Some(0.0),
            "#b91c1c",
        ),
        ("Ate %", metric_series(rows, |r| r.ate_pct), None, "#c2410c"),
        (
            "Consumptions Mean",
            metric_series(rows, |r| r.cons_mean),
            None,
            "#0e7490",
        ),
        (
            "Brain Size",
            metric_series(rows, |r| r.brain_size),
            None,
            "#0891b2",
        ),
        (
            "P(Forward | Food Ahead)",
            metric_series(rows, |r| r.p_fwd_food),
            Some(action_baseline_probability()),
            "#0f766e",
        ),
        (
            "MI(S;A)",
            metric_series(rows, |r| r.mi_sa),
            Some(0.0),
            "#9333ea",
        ),
        (
            "MI(S;A) Juvenile",
            metric_series(rows, |r| r.mi_sa_juvenile),
            Some(0.0),
            "#7c3aed",
        ),
        (
            "MI(S;A) Adult",
            metric_series(rows, |r| r.mi_sa_adult),
            Some(0.0),
            "#581c87",
        ),
        (
            "H(Action)",
            metric_series(rows, |r| r.h_action),
            Some(action_baseline_entropy()),
            "#b45309",
        ),
        (
            "Idle Fraction",
            metric_series(rows, |r| r.idle_fraction),
            Some(action_baseline_probability()),
            "#f97316",
        ),
        (
            "Reproduction Efficiency",
            metric_series(rows, |r| r.reproduction_efficiency),
            Some(0.0),
            "#0ea5e9",
        ),
        (
            "Lineage Diversity",
            metric_series(rows, |r| r.lineage_diversity),
            Some(0.0),
            "#0f766e",
        ),
        (
            "Damage Avoidance",
            metric_series(rows, |r| r.damage_avoidance),
            Some(1.0),
            "#059669",
        ),
        (
            "Reward Reversal Shift",
            metric_series(rows, |r| r.reward_reversal_shift),
            Some(0.0),
            "#1d4ed8",
        ),
        (
            "Inter Utilization",
            metric_series(rows, |r| r.util),
            None,
            "#334155",
        ),
    ];

    html.push_str("<div class=\"panel\"><h2>Metric Charts</h2>");
    for (title, series, baseline, color) in charts {
        html.push_str("<div class=\"chart\">");
        let _ = write!(html, "<h3>{title}</h3>");
        html.push_str(&line_chart_svg(&series, baseline, color));
        if let Some(base) = baseline {
            let _ = write!(
                html,
                "<div class=\"note\">dashed line = baseline ({base:.4})</div>"
            );
        }
        html.push_str("</div>");
    }
    html.push_str("</div>");
    append_interpretation_guidance(&mut html);

    html.push_str("</div></body></html>");
    std::fs::write(report_path, html)?;
    Ok(())
}

pub fn write_comparison_html_report(
    out_dir: &Path,
    meta: &ComparisonHtmlReportMeta,
) -> Result<()> {
    let report_path = out_dir.join("comparison_report.html");
    let mut html = String::new();
    html.push_str(
        "<!doctype html><html><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">",
    );
    html.push_str("<title>Sim Validation Comparison</title>");
    html.push_str(
        "<style>\
        :root{--bg:#f5f7fb;--panel:#ffffff;--ink:#0f172a;--muted:#64748b;--line:#dbe2ea;}\
        *{box-sizing:border-box}body{margin:0;font-family:ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,Arial,sans-serif;background:var(--bg);color:var(--ink)}\
        .wrap{max-width:1200px;margin:24px auto;padding:0 16px}\
        .panel{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:16px;margin-bottom:16px}\
        h1,h2{margin:0 0 10px 0}h1{font-size:24px}h2{font-size:18px}\
        .meta{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:10px}\
        .k{color:var(--muted);font-size:12px;text-transform:uppercase;letter-spacing:.04em}.v{font-weight:600}\
        table{width:100%;border-collapse:collapse;font-size:13px}th,td{padding:8px;border-bottom:1px solid var(--line);text-align:right}th:first-child,td:first-child{text-align:left}\
        a{color:#1d4ed8;text-decoration:none}a:hover{text-decoration:underline}\
        </style></head><body><div class=\"wrap\">",
    );
    html.push_str("<div class=\"panel\"><h1>Simulation Validation Comparison</h1><div class=\"meta\">");
    if let Some(title) = &meta.title {
        kv(&mut html, "Title", title);
    }
    kv(&mut html, "Seeds", &meta.seed_label);
    kv(&mut html, "Ticks", &meta.ticks.to_string());
    kv(&mut html, "Control", &meta.control_label);
    kv(&mut html, "Treatment", &meta.treatment_label);
    kv(
        &mut html,
        "Total Time",
        &format!("{:.3}s", meta.total_time_seconds),
    );
    html.push_str("</div></div>");

    html.push_str("<div class=\"panel\"><h2>Run Reports</h2><table><tbody>");
    let _ = write!(
        html,
        "<tr><td>Control</td><td><a href=\"{}\">open report</a></td></tr>",
        meta.control_report_href
    );
    let _ = write!(
        html,
        "<tr><td>Treatment</td><td><a href=\"{}\">open report</a></td></tr>",
        meta.treatment_report_href
    );
    html.push_str("</tbody></table></div>");

    html.push_str("<div class=\"panel\"><h2>Metric Diffs</h2><table><thead><tr>");
    for header in [
        "metric",
        "control_mean",
        "treatment_mean",
        "mean_diff",
        "ci_low",
        "ci_high",
    ] {
        let _ = write!(html, "<th>{header}</th>");
    }
    html.push_str("</tr></thead><tbody>");
    for row in &meta.metric_rows {
        let _ = write!(
            html,
            "<tr><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>",
            row.label,
            fmt_opt(row.control_mean, 4),
            fmt_opt(row.treatment_mean, 4),
            fmt_opt(row.mean_diff, 4),
            fmt_opt(row.ci_low, 4),
            fmt_opt(row.ci_high, 4),
        );
    }
    html.push_str("</tbody></table></div>");

    html.push_str("<div class=\"panel\"><h2>Per-Seed Score Diffs</h2><table><thead><tr>");
    for header in [
        "seed",
        "control_score",
        "treatment_score",
        "diff_score",
        "control_report",
        "treatment_report",
    ] {
        let _ = write!(html, "<th>{header}</th>");
    }
    html.push_str("</tr></thead><tbody>");
    for row in &meta.per_seed_rows {
        let _ = write!(
            html,
            "<tr><td>{}</td><td>{:.2}</td><td>{:.2}</td><td>{:.2}</td><td><a href=\"{}\">open</a></td><td><a href=\"{}\">open</a></td></tr>",
            row.seed,
            row.control_score,
            row.treatment_score,
            row.diff_score,
            row.control_report_href,
            row.treatment_report_href,
        );
    }
    html.push_str("</tbody></table></div>");

    html.push_str("</div></body></html>");
    std::fs::write(report_path, html)?;
    Ok(())
}

fn fmt_opt(value: Option<f64>, decimals: usize) -> String {
    match value {
        Some(v) => format!("{v:.decimals$}"),
        None => "NA".to_owned(),
    }
}

fn fmt_opt_u64(value: Option<u64>) -> String {
    value
        .map(|v| v.to_string())
        .unwrap_or_else(|| "NA".to_owned())
}

fn csv_opt(value: Option<f64>) -> String {
    value.map(|v| v.to_string()).unwrap_or_default()
}

fn csv_opt_u64(value: Option<u64>) -> String {
    value.map(|v| v.to_string()).unwrap_or_default()
}

fn kv(html: &mut String, key: &str, value: &str) {
    let _ = write!(
        html,
        "<div><div class=\"k\">{key}</div><div class=\"v\">{value}</div></div>"
    );
}

fn metric_series<F>(rows: &[IntervalMetrics], mut accessor: F) -> Vec<(u64, f64)>
where
    F: FnMut(&IntervalMetrics) -> Option<f64>,
{
    rows.iter()
        .filter_map(|row| accessor(row).map(|value| (row.tick, value)))
        .filter(|(_, value)| value.is_finite())
        .collect()
}

fn line_chart_svg(series: &[(u64, f64)], baseline: Option<f64>, color: &str) -> String {
    const WIDTH: f64 = 1024.0;
    const HEIGHT: f64 = 220.0;
    const LEFT: f64 = 56.0;
    const RIGHT: f64 = 16.0;
    const TOP: f64 = 14.0;
    const BOTTOM: f64 = 32.0;
    let plot_w = WIDTH - LEFT - RIGHT;
    let plot_h = HEIGHT - TOP - BOTTOM;

    if series.is_empty() {
        return "<svg viewBox=\"0 0 1024 220\"><text x=\"16\" y=\"36\" fill=\"#64748b\">NA</text></svg>"
            .to_owned();
    }

    let min_tick = series.first().map(|(tick, _)| *tick).unwrap_or(0);
    let max_tick = series.last().map(|(tick, _)| *tick).unwrap_or(min_tick);
    let x_span = (max_tick.saturating_sub(min_tick)).max(1) as f64;

    let mut y_min = series
        .iter()
        .map(|(_, value)| *value)
        .fold(f64::INFINITY, f64::min);
    let mut y_max = series
        .iter()
        .map(|(_, value)| *value)
        .fold(f64::NEG_INFINITY, f64::max);
    if let Some(base) = baseline {
        y_min = y_min.min(base);
        y_max = y_max.max(base);
    }
    if (y_max - y_min).abs() < f64::EPSILON {
        y_min -= 1.0;
        y_max += 1.0;
    } else {
        let padding = (y_max - y_min) * 0.08;
        y_min -= padding;
        y_max += padding;
    }
    let y_span = y_max - y_min;

    let map_x = |tick: u64| LEFT + (tick.saturating_sub(min_tick) as f64 / x_span) * plot_w;
    let map_y = |value: f64| TOP + (1.0 - (value - y_min) / y_span) * plot_h;

    let mut d = String::new();
    for (idx, (tick, value)) in series.iter().enumerate() {
        let x = map_x(*tick);
        let y = map_y(*value);
        if idx == 0 {
            let _ = write!(d, "M{:.2},{:.2}", x, y);
        } else {
            let _ = write!(d, " L{:.2},{:.2}", x, y);
        }
    }

    let mut svg = String::new();
    let _ = write!(
        svg,
        "<svg viewBox=\"0 0 {WIDTH} {HEIGHT}\" xmlns=\"http://www.w3.org/2000/svg\">"
    );
    let _ = write!(
        svg,
        "<rect x=\"0\" y=\"0\" width=\"{WIDTH}\" height=\"{HEIGHT}\" fill=\"#fff\"/>"
    );

    for i in 0..=4 {
        let frac = i as f64 / 4.0;
        let y = TOP + frac * plot_h;
        let value = y_max - frac * y_span;
        let _ = write!(
            svg,
            "<line x1=\"{LEFT}\" y1=\"{y:.2}\" x2=\"{x2:.2}\" y2=\"{y:.2}\" stroke=\"#e2e8f0\" stroke-width=\"1\"/>\
             <text x=\"8\" y=\"{ly:.2}\" fill=\"#64748b\" font-size=\"11\">{value:.3}</text>",
            x2 = LEFT + plot_w,
            ly = y + 4.0
        );
    }

    if let Some(base) = baseline {
        let by = map_y(base);
        let _ = write!(
            svg,
            "<line x1=\"{LEFT}\" y1=\"{by:.2}\" x2=\"{x2:.2}\" y2=\"{by:.2}\" stroke=\"#b45309\" stroke-width=\"1.5\" stroke-dasharray=\"6,5\"/>",
            x2 = LEFT + plot_w
        );
    }

    let _ = write!(
        svg,
        "<path d=\"{d}\" fill=\"none\" stroke=\"{color}\" stroke-width=\"2.5\"/>"
    );
    let _ = write!(
        svg,
        "<text x=\"{LEFT}\" y=\"{y}\" fill=\"#64748b\" font-size=\"11\">tick {min_tick}</text>\
         <text x=\"{x}\" y=\"{y}\" fill=\"#64748b\" font-size=\"11\" text-anchor=\"end\">tick {max_tick}</text>",
        y = HEIGHT - 10.0,
        x = LEFT + plot_w
    );
    svg.push_str("</svg>");
    svg
}

fn append_interpretation_guidance(html: &mut String) {
    let baseline_probability = action_baseline_probability();
    let baseline_entropy = action_baseline_entropy();

    html.push_str("<div class=\"panel guide\"><h2>Interpreting The Metrics</h2>");

    html.push_str("<h3>P(Fwd|food) -- \"Can they see?\"</h3>");
    html.push_str("<p>This is the single most important number. It answers: when food is directly ahead, does the organism walk toward it more often than chance?</p>");
    html.push_str("<ul>");
    let _ = write!(
        html,
        "<li><code>{baseline_probability:.2}</code> ({N_ACTIONS} actions): random. Brains are not influencing behavior in a useful way. Evolution has not discovered stimulus-response coupling.</li>",
    );
    html.push_str("<li><code>0.30-0.40</code>: weak signal. Something is working but unreliably. Could be a small subpopulation of competent foragers diluted by many random walkers.</li>");
    html.push_str("<li><code>0.50+</code>: strong directed foraging. Evolution has found brains that reliably turn sensory input into adaptive action.</li>");
    html.push_str("<li>Below baseline: actively food-avoidant. Possible if the action encoding or sensory wiring has an inversion bug.</li>");
    html.push_str("</ul>");
    html.push_str("<p>If this number is flat at baseline after thousands of generations, the evolutionary loop is broken. Check energy economics first (is eating actually rewarded enough to matter?), then mutation rates (can evolution explore fast enough?).</p>");

    html.push_str("<h3>H(action) -- \"Are they decisive?\"</h3>");
    html.push_str("<p>Shannon entropy of the action distribution. In the aggregate score, the preferred regime is low but nonzero entropy: a small, purposeful repertoire is better than random wandering, but better than single-action collapse too.</p>");
    html.push_str("<ul>");
    html.push_str("<li><code>~= 0</code>: over-collapsed. Usually a rigid fixed-action policy rather than intelligent sequencing.</li>");
    html.push_str("<li><code>~0.8-1.1</code> bits: preferred band. Typically means mostly forward movement, periodic consumes, and occasional turns when the state demands them.</li>");
    let _ = write!(
        html,
        "<li><code>~= log2(N_ACTIONS)</code> (<code>{baseline_entropy:.2}</code> for {N_ACTIONS} actions): uniform random. No preferences. Brain output is noise.</li>",
    );
    html.push_str("<li><code>1.5+</code> bits: too diffuse. There may be some structure, but too much behavior is still being left to randomness.</li>");
    html.push_str("</ul>");
    html.push_str("<p>Read H together with P(Fwd|food) and MI(S;A). The ideal is low entropy plus strong sensory coupling: a policy that commits hard within each situation while still using different actions across situations. Near-zero entropy with weak coupling is a dumb reflex. High entropy is just noise.</p>");

    html.push_str("<h3>MI(S;A) -- \"Do they react?\"</h3>");
    html.push_str("<p>Mutual information between what the organism sees and what it does. The general version of P(Fwd|food) - captures all sensory-action coupling, not just the food-ahead case.</p>");
    html.push_str("<ul>");
    html.push_str("<li><code>0.00</code>: actions are statistically independent of sensory input. The brain is ignoring its eyes.</li>");
    html.push_str("<li><code>0.01-0.05</code>: weak coupling. Noisy but present. Typical of early evolution where a few organisms have partial stimulus-response wiring.</li>");
    html.push_str("<li><code>0.10+</code>: meaningful sensory-motor coupling. Organisms are making different decisions in different sensory contexts.</li>");
    html.push_str("</ul>");
    html.push_str("<p>MI is biased upward with small sample sizes (short-lived organisms produce noisy histograms that look like they have structure). The Miller-Madow correction helps but does not eliminate this. If MI looks suspiciously high in early intervals when organisms are dying young, it is probably bias. Trust MI trends over absolute values. Trust it more when life_mean is high (more samples per organism).</p>");
    html.push_str("<p>The relationship between MI and P(Fwd|food): MI can be positive even when P(Fwd|food) is at baseline, if organisms are reacting to non-food stimuli (avoiding walls, turning away from other organisms). This is still interesting - it means brains are sensory-responsive, just not for the behavior you expected.</p>");

    html.push_str("<h3>Inter-neuron utilization -- \"Is the brain earning its keep?\"</h3>");
    html.push_str("<p>Fraction of inter neurons with sustained nonzero activation.</p>");
    html.push_str("<ul>");
    html.push_str(
        "<li>Low util + small brain: fine. A minimal brain where every neuron matters.</li>",
    );
    html.push_str("<li>Low util + large brain: metabolic waste. Evolution added neurons but they are dead weight. Usually means neuron-addition mutations are happening but selection is not strong enough to prune useless neurons (or metabolism cost is too low to penalize them).</li>");
    html.push_str("<li>High util + rising brain size: the best signal. Evolution is growing brains AND using the new capacity. Complexity is paying for itself.</li>");
    html.push_str("<li>Falling util over time: brains are growing faster than they are being utilized. Topology mutations outpacing functional integration.</li>");
    html.push_str("</ul>");

    html.push_str("<h3>Reading the table as a whole</h3>");
    html.push_str("<p>The diagnostic story emerges from metric combinations:</p>");
    html.push_str("<table><thead><tr><th>Pattern</th><th>Diagnosis</th></tr></thead><tbody>");
    html.push_str("<tr><td>Pop stable, all Tier 3 flat at baseline</td><td>Evolution running but not finding anything. Fix energy economics or mutation rates.</td></tr>");
    html.push_str("<tr><td>Pop crashing, high death rate</td><td>Organisms cannot survive. Food too scarce, metabolism too high, or starting energy too low.</td></tr>");
    html.push_str("<tr><td>Pop stable, life_mean rising, Tier 3 flat</td><td>Selection is working but optimizing non-cognitive traits (reproduction timing, energy hoarding). Brains are not the path to fitness.</td></tr>");
    html.push_str("<tr><td>P(Fwd|food) rising, MI rising, H intermediate</td><td>The good outcome. Evolution is discovering sensory-motor coupling.</td></tr>");
    html.push_str("<tr><td>P(Fwd|food) rising, brain shrinking</td><td>Evolution found a minimal circuit for foraging and is trimming waste. Efficient, possibly a local optimum.</td></tr>");
    html.push_str("<tr><td>H ~= 0, pop stable</td><td>Idle-degenerate niche. Organisms survive by not moving (if idle is cheap enough). Fix by making idle cost energy too.</td></tr>");
    html.push_str("<tr><td>MI rising but P(Fwd|food) flat</td><td>Organisms are reacting to stimuli but not food specifically. Check if they are responding to walls or other organisms instead. Might need to adjust sensory salience or food density.</td></tr>");
    html.push_str("</tbody></table>");

    html.push_str("</div>");
}
