use crate::{
    analysis::intervals::action_baseline_probability,
    types::{IntervalMetrics, PillarScores},
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
            "tick,pop,births,deaths,food,max_generation,attack_attempt_rate,attack_success_rate,failed_action_rate,ate_pct,cons_mean,brain_size,brain_size_stddev,brain_size_p10,brain_size_p50,brain_size_p90,p_fwd_food,mi_sa,idle_fraction,util"
        )?;
        Ok(Self { csv })
    }

    pub fn emit(&mut self, metrics: &IntervalMetrics) -> Result<()> {
        writeln!(
            self.csv,
            "{tick},{pop},{births},{deaths},{food},{max_generation},{attack_attempt_rate},{attack_success_rate},{failed_action_rate},{ate_pct},{cons_mean},{brain_size},{brain_size_stddev},{brain_size_p10},{brain_size_p50},{brain_size_p90},{p_fwd_food},{mi_sa},{idle_fraction},{util}",
            tick = metrics.tick,
            pop = metrics.pop,
            births = metrics.births,
            deaths = metrics.deaths,
            food = metrics.food,
            max_generation = csv_opt_u64(metrics.max_generation),
            attack_attempt_rate = csv_opt(metrics.attack_attempt_rate),
            attack_success_rate = csv_opt(metrics.attack_success_rate),
            failed_action_rate = csv_opt(metrics.failed_action_rate),
            ate_pct = csv_opt(metrics.ate_pct),
            cons_mean = csv_opt(metrics.cons_mean),
            brain_size = csv_opt(metrics.brain_size),
            brain_size_stddev = csv_opt(metrics.brain_size_stddev),
            brain_size_p10 = csv_opt(metrics.brain_size_p10),
            brain_size_p50 = csv_opt(metrics.brain_size_p50),
            brain_size_p90 = csv_opt(metrics.brain_size_p90),
            p_fwd_food = csv_opt(metrics.p_fwd_food),
            mi_sa = csv_opt(metrics.mi_sa),
            idle_fraction = csv_opt(metrics.idle_fraction),
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
    pub ticks: u64,
    pub report_every: u64,
    pub control: bool,
    pub total_time_seconds: f64,
    pub generated_at_utc: String,
    pub pillar_window_start_tick: u64,
    pub pillar_window_end_tick: u64,
    pub foraging_pillar: f64,
    pub intelligence_pillar: f64,
    pub competition_pillar: f64,
    pub foraging_p_fwd_food_component: f64,
    pub intelligence_mi_component: f64,
    pub intelligence_action_effectiveness_component: f64,
    pub intelligence_anti_idle_component: f64,
    pub intelligence_util_component: f64,
    pub competition_attack_success_component: f64,
    pub competition_attack_attempt_component: f64,
    pub timeseries_label: String,
    pub per_seed_rows: Vec<PerSeedReportRow>,
}

pub struct HtmlReportContext {
    pub title: Option<String>,
    pub ticks: u64,
    pub report_every: u64,
    pub control: bool,
    pub total_time_seconds: f64,
    pub generated_at_utc: String,
    pub timeseries_label: String,
    pub per_seed_rows: Vec<PerSeedReportRow>,
}

impl HtmlReportMeta {
    pub fn from_pillars(pillars: &PillarScores, ctx: HtmlReportContext) -> Self {
        Self {
            title: ctx.title,
            ticks: ctx.ticks,
            report_every: ctx.report_every,
            control: ctx.control,
            total_time_seconds: ctx.total_time_seconds,
            generated_at_utc: ctx.generated_at_utc,
            pillar_window_start_tick: pillars.window_start_tick,
            pillar_window_end_tick: pillars.window_end_tick,
            foraging_pillar: pillars.foraging_pillar,
            intelligence_pillar: pillars.intelligence_pillar,
            competition_pillar: pillars.competition_pillar,
            foraging_p_fwd_food_component: pillars.foraging_p_fwd_food_component,
            intelligence_mi_component: pillars.intelligence_mi_component,
            intelligence_action_effectiveness_component: pillars
                .intelligence_action_effectiveness_component,
            intelligence_anti_idle_component: pillars.intelligence_anti_idle_component,
            intelligence_util_component: pillars.intelligence_util_component,
            competition_attack_success_component: pillars.competition_attack_success_component,
            competition_attack_attempt_component: pillars.competition_attack_attempt_component,
            timeseries_label: ctx.timeseries_label,
            per_seed_rows: ctx.per_seed_rows,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct PerSeedReportRow {
    pub seed: u64,
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
    pub control_report_href: String,
    pub treatment_report_href: String,
}

pub struct ComparisonHtmlReportMeta {
    pub title: Option<String>,
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
    html.push_str("<title>Sim Evaluation Report</title>");
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
        .table-scroll{max-width:100%;overflow-x:auto;overflow-y:hidden;-webkit-overflow-scrolling:touch}\
        .table-scroll table{min-width:1280px}\
        .th-tip{position:relative;display:inline-block;cursor:help;border-bottom:1px dotted var(--muted);padding-bottom:1px}\
        .th-tip .tip{position:fixed;left:0;top:0;transform:translate(-50%,-100%);width:min(320px,70vw);padding:10px 12px;border-radius:10px;background:#0f172a;color:#f8fafc;font-size:12px;font-weight:500;line-height:1.45;text-align:left;box-shadow:0 12px 30px rgba(15,23,42,.22);opacity:0;visibility:hidden;pointer-events:none;z-index:1000}\
        .th-tip .tip::after{content:\"\";position:absolute;left:50%;top:100%;transform:translateX(-50%);border:6px solid transparent;border-top-color:#0f172a}\
        .th-tip:hover .tip,.th-tip:focus-visible .tip{opacity:1;visibility:visible}\
        .chart{margin:8px 0 20px 0}svg{width:100%;height:auto;border:1px solid var(--line);border-radius:8px;background:#fff}\
        .note{color:var(--muted);font-size:12px}\
        .score-big{font-size:42px;font-weight:700;line-height:1;margin:0}\
        .score-sub{margin:8px 0 0 0;color:var(--muted);font-size:12px}\
        .pillar-list{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:12px;margin-top:16px}\
        .pillar-card{border:1px solid var(--line);border-radius:14px;padding:16px;background:linear-gradient(180deg,#fff 0%,#f8fafc 100%)}\
        .pillar-head{display:flex;align-items:flex-end;justify-content:space-between;gap:12px;margin-bottom:12px}\
        .pillar-title{font-size:18px;font-weight:700;line-height:1.1}\
        .pillar-score{font-size:28px;font-weight:700;line-height:1}\
        .pillar-subscores{display:grid;gap:8px}\
        .pillar-subscore{display:flex;justify-content:space-between;gap:12px;padding-top:8px;border-top:1px solid var(--line);font-size:13px}\
        .pillar-subscore-name{color:var(--muted)}\
        .pillar-subscore-value{font-weight:600}\
        </style></head><body><div class=\"wrap\">",
    );

    html.push_str("<div class=\"panel\"><h1>Simulation Evaluation Report</h1><div class=\"meta\">");
    if let Some(title) = &meta.title {
        kv(&mut html, "Title", title);
    }
    kv(&mut html, "Ticks", &meta.ticks.to_string());
    kv(&mut html, "Report Every", &meta.report_every.to_string());
    kv(
        &mut html,
        "Random-action control",
        if meta.control { "true" } else { "false" },
    );
    kv(
        &mut html,
        "Total Time",
        &format!("{:.3}s", meta.total_time_seconds),
    );
    kv(&mut html, "Generated At (UTC)", &meta.generated_at_utc);
    html.push_str("</div></div>");

    html.push_str("<div class=\"panel\"><h2>Pillar Scores</h2>");
    let _ = write!(
        html,
        "<p class=\"score-sub\">window: ticks {}-{} | no aggregate score — each pillar stands on its own, because different niches excel at different pillars.</p>",
        meta.pillar_window_start_tick, meta.pillar_window_end_tick
    );
    html.push_str("<div class=\"pillar-list\">");
    pillar_card(
        &mut html,
        "Foraging",
        &format!("{:.3}", meta.foraging_pillar),
        &[("P(Fwd|food)", meta.foraging_p_fwd_food_component)],
    );
    pillar_card(
        &mut html,
        "Intelligence",
        &format!("{:.3}", meta.intelligence_pillar),
        &[
            (
                "Action effectiveness",
                meta.intelligence_action_effectiveness_component,
            ),
            ("MI(S;A)", meta.intelligence_mi_component),
            ("Anti-idle", meta.intelligence_anti_idle_component),
            ("Util", meta.intelligence_util_component),
        ],
    );
    pillar_card(
        &mut html,
        "Competition",
        &format!("{:.3}", meta.competition_pillar),
        &[
            ("Attack success", meta.competition_attack_success_component),
            ("Attack attempts", meta.competition_attack_attempt_component),
        ],
    );
    html.push_str("</div></div>");

    if !meta.per_seed_rows.is_empty() {
        html.push_str("<div class=\"panel\"><h2>Per-Seed Results</h2><table><thead><tr>");
        for header in ["seed", "time_s", "state_hash", "report"] {
            let _ = write!(html, "<th>{header}</th>");
        }
        html.push_str("</tr></thead><tbody>");
        for row in &meta.per_seed_rows {
            let _ = write!(
                html,
                "<tr><td>{}</td><td>{:.3}</td><td>{}</td><td><a href=\"{}\">open</a></td></tr>",
                row.seed, row.total_time_seconds, row.state_hash, row.report_href
            );
        }
        html.push_str("</tbody></table></div>");
    }

    let _ = write!(
        html,
        "<div class=\"panel\"><h2>Timeseries</h2><p class=\"note\">{}</p><div class=\"table-scroll\"><table><thead><tr>",
        meta.timeseries_label
    );
    let baseline_probability = action_baseline_probability();
    let timeseries_headers = [
        (
            "tick",
            "End tick of this reporting interval. Each row summarizes the ticks after the previous boundary and up to this tick.".to_owned(),
        ),
        (
            "pop",
            "Population at the final tick of the interval, not an interval average.".to_owned(),
        ),
        (
            "births",
            "Total births during the interval, summed across ticks.".to_owned(),
        ),
        (
            "deaths",
            "Total deaths during the interval, summed across ticks.".to_owned(),
        ),
        (
            "max_generation",
            "Highest generation present at the end of the interval, taken from the latest tick summary in the window.".to_owned(),
        ),
        (
            "attack_attempt_rate",
            "Attack selections per unit population exposure: total Attack actions in the interval divided by the sum of population counts over ticks.".to_owned(),
        ),
        (
            "attack_success_rate",
            "Share of Attack selections that did not fail: successful attacks divided by total attack attempts in the interval.".to_owned(),
        ),
        (
            "failed_action_rate",
            "Share of contingent actions that failed. Counts Forward, Eat, Attack, and Reproduce failures; excludes Idle and turning.".to_owned(),
        ),
        (
            "ate_pct",
            "Percent of sampled organisms in this interval's lifetime cohort that consumed food at least once. Uses organisms that died in the interval.".to_owned(),
        ),
        (
            "cons_mean",
            "Mean lifetime food consumptions for the interval's deceased-organism cohort.".to_owned(),
        ),
        (
            "brain_size",
            "Mean brain size at the latest flush snapshot at or before the interval end. Brain size is neurons plus synapses across living organisms in that snapshot.".to_owned(),
        ),
        (
            "p_fwd_food",
            format!(
                "Among recorded action ticks where food was visible straight ahead, the fraction that chose Forward. Random-choice baseline is {baseline_probability:.4}."
            ),
        ),
        (
            "mi_sa",
            "Miller-Madow-corrected mutual information between coarse food-visibility context and selected action, pooled from deceased-organism lifetime histograms.".to_owned(),
        ),
        (
            "idle_fraction",
            "Share of all selected actions in the interval that were Idle.".to_owned(),
        ),
        (
            "util",
            "Mean inter-neuron utilization across the interval's deceased-organism cohort. This is the per-organism fraction of inter neurons with sustained nonzero activation.".to_owned(),
        ),
    ];
    for (header, tooltip) in timeseries_headers {
        tooltip_th(&mut html, header, &tooltip);
    }
    html.push_str("</tr></thead><tbody>");
    for row in rows {
        html.push_str("<tr>");
        for cell in [
            row.tick.to_string(),
            row.pop.to_string(),
            row.births.to_string(),
            row.deaths.to_string(),
            fmt_opt_u64(row.max_generation),
            fmt_opt(row.attack_attempt_rate, 6),
            fmt_opt(row.attack_success_rate, 4),
            fmt_opt(row.failed_action_rate, 4),
            fmt_opt(row.ate_pct, 2),
            fmt_opt(row.cons_mean, 2),
            fmt_opt(row.brain_size, 2),
            fmt_opt(row.p_fwd_food, 4),
            fmt_opt(row.mi_sa, 4),
            fmt_opt(row.idle_fraction, 4),
            fmt_opt(row.util, 4),
        ] {
            let _ = write!(html, "<td>{cell}</td>");
        }
        html.push_str("</tr>");
    }
    html.push_str("</tbody></table></div></div>");

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
        (
            "Failed Action Rate",
            metric_series(rows, |r| r.failed_action_rate),
            Some(0.0),
            "#991b1b",
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
            "Idle Fraction",
            metric_series(rows, |r| r.idle_fraction),
            Some(action_baseline_probability()),
            "#f97316",
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

    html.push_str(
        "<script>(function(){\
        function place(el){\
          var tip=el.querySelector('.tip');if(!tip)return;\
          var r=el.getBoundingClientRect();\
          tip.style.left=(r.left+r.width/2)+'px';\
          tip.style.top=(r.top-10)+'px';\
        }\
        document.querySelectorAll('.th-tip').forEach(function(el){\
          el.addEventListener('mouseenter',function(){place(el);});\
          el.addEventListener('focus',function(){place(el);});\
        });\
        })();</script>",
    );
    html.push_str("</div></body></html>");
    std::fs::write(report_path, html)?;
    Ok(())
}

pub fn write_comparison_html_report(out_dir: &Path, meta: &ComparisonHtmlReportMeta) -> Result<()> {
    let report_path = out_dir.join("comparison_report.html");
    let mut html = String::new();
    html.push_str(
        "<!doctype html><html><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">",
    );
    html.push_str("<title>Sim Evaluation Comparison</title>");
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
    html.push_str(
        "<div class=\"panel\"><h1>Simulation Evaluation Comparison</h1><div class=\"meta\">",
    );
    if let Some(title) = &meta.title {
        kv(&mut html, "Title", title);
    }
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

    html.push_str("<div class=\"panel\"><h2>Per-Seed Reports</h2><table><thead><tr>");
    for header in ["seed", "control_report", "treatment_report"] {
        let _ = write!(html, "<th>{header}</th>");
    }
    html.push_str("</tr></thead><tbody>");
    for row in &meta.per_seed_rows {
        let _ = write!(
            html,
            "<tr><td>{}</td><td><a href=\"{}\">open</a></td><td><a href=\"{}\">open</a></td></tr>",
            row.seed, row.control_report_href, row.treatment_report_href,
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

fn pillar_card(html: &mut String, title: &str, score: &str, subscores: &[(&str, f64)]) {
    let _ = write!(
        html,
        "<div class=\"pillar-card\"><div class=\"pillar-head\"><div class=\"pillar-title\">{title}</div><div class=\"pillar-score\">{score}</div></div><div class=\"pillar-subscores\">"
    );
    for (name, value) in subscores {
        let _ = write!(
            html,
            "<div class=\"pillar-subscore\"><span class=\"pillar-subscore-name\">{name}</span><span class=\"pillar-subscore-value\">{value:.3}</span></div>"
        );
    }
    html.push_str("</div></div>");
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

fn tooltip_th(html: &mut String, label: &str, tooltip: &str) {
    let _ = write!(
        html,
        "<th><span class=\"th-tip\" tabindex=\"0\">{}<span class=\"tip\">{}</span></span></th>",
        escape_html(label),
        escape_html(tooltip),
    );
}

fn escape_html(text: &str) -> String {
    let mut escaped = String::with_capacity(text.len());
    for ch in text.chars() {
        match ch {
            '&' => escaped.push_str("&amp;"),
            '<' => escaped.push_str("&lt;"),
            '>' => escaped.push_str("&gt;"),
            '"' => escaped.push_str("&quot;"),
            '\'' => escaped.push_str("&#39;"),
            _ => escaped.push(ch),
        }
    }
    escaped
}
