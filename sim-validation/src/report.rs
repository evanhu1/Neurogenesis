use crate::metrics::{action_baseline_entropy, action_baseline_probability, IntervalMetrics};
use anyhow::Result;
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
            "tick,pop,births,deaths,food,max_generation,life_mean,life_max,ate_pct,cons_mean,brain_size,p_fwd_food,mi_sa,h_action,util"
        )?;
        Ok(Self { csv })
    }

    pub fn emit(&mut self, metrics: &IntervalMetrics) -> Result<()> {
        writeln!(
            self.csv,
            "{tick},{pop},{births},{deaths},{food},{max_generation},{life_mean},{life_max},{ate_pct},{cons_mean},{brain_size},{p_fwd_food},{mi_sa},{h_action},{util}",
            tick = metrics.tick,
            pop = metrics.pop,
            births = metrics.births,
            deaths = metrics.deaths,
            food = metrics.food,
            max_generation = csv_opt_u64(metrics.max_generation),
            life_mean = csv_opt(metrics.life_mean),
            life_max = csv_opt_u64(metrics.life_max),
            ate_pct = csv_opt(metrics.ate_pct),
            cons_mean = csv_opt(metrics.cons_mean),
            brain_size = csv_opt(metrics.brain_size),
            p_fwd_food = csv_opt(metrics.p_fwd_food),
            mi_sa = csv_opt(metrics.mi_sa),
            h_action = csv_opt(metrics.h_action),
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
    pub seed: u64,
    pub ticks: u64,
    pub report_every: u64,
    pub min_lifetime: u64,
    pub baseline: bool,
    pub total_time_seconds: f64,
    pub aggregate_score: f64,
    pub aggregate_window_start_tick: u64,
    pub aggregate_window_end_tick: u64,
    pub aggregate_p_component: f64,
    pub aggregate_mi_component: f64,
    pub aggregate_entropy_component: f64,
    pub aggregate_mean_p_fwd_food: Option<f64>,
    pub aggregate_mean_mi_sa: Option<f64>,
    pub aggregate_mean_h_action: Option<f64>,
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
    kv(&mut html, "Seed", &meta.seed.to_string());
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
    html.push_str("</div></div>");

    html.push_str("<div class=\"panel\"><h2>Aggregate Score</h2>");
    let _ = write!(
        html,
        "<p class=\"score-big\">{:.2}</p>\
         <p class=\"score-sub\">window: ticks {}-{} | higher is better for quick run-to-run comparison</p>",
        meta.aggregate_score, meta.aggregate_window_start_tick, meta.aggregate_window_end_tick
    );
    html.push_str("<div class=\"score-grid\">");
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
        "Window mean H(action)",
        &fmt_opt(meta.aggregate_mean_h_action, 4),
    );
    html.push_str("</div></div>");

    html.push_str("<div class=\"panel\"><h2>Timeseries</h2><table><thead><tr>");
    for header in [
        "tick",
        "pop",
        "births",
        "deaths",
        "food",
        "max_generation",
        "life_mean",
        "life_max",
        "ate_pct",
        "cons_mean",
        "brain_size",
        "p_fwd_food",
        "mi_sa",
        "h_action",
        "util",
    ] {
        let _ = write!(html, "<th>{header}</th>");
    }
    html.push_str("</tr></thead><tbody>");
    for row in rows {
        let _ = write!(
            html,
            "<tr><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td>\
             <td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>",
            row.tick,
            row.pop,
            row.births,
            row.deaths,
            row.food,
            fmt_opt_u64(row.max_generation),
            fmt_opt(row.life_mean, 2),
            fmt_opt_u64(row.life_max),
            fmt_opt(row.ate_pct, 2),
            fmt_opt(row.cons_mean, 2),
            fmt_opt(row.brain_size, 2),
            fmt_opt(row.p_fwd_food, 4),
            fmt_opt(row.mi_sa, 4),
            fmt_opt(row.h_action, 4),
            fmt_opt(row.util, 4)
        );
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
            "H(Action)",
            metric_series(rows, |r| r.h_action),
            Some(action_baseline_entropy()),
            "#b45309",
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
    html.push_str("<div class=\"panel guide\"><h2>Interpreting The Metrics</h2>");

    html.push_str("<h3>P(Fwd|food) -- \"Can they see?\"</h3>");
    html.push_str("<p>This is the single most important number. It answers: when food is directly ahead, does the organism walk toward it more often than chance?</p>");
    html.push_str("<ul>");
    html.push_str("<li><code>0.25</code> (4 actions) or <code>0.14</code> (7 actions): random. Brains are not influencing behavior in a useful way. Evolution has not discovered stimulus-response coupling.</li>");
    html.push_str("<li><code>0.30-0.40</code>: weak signal. Something is working but unreliably. Could be a small subpopulation of competent foragers diluted by many random walkers.</li>");
    html.push_str("<li><code>0.50+</code>: strong directed foraging. Evolution has found brains that reliably turn sensory input into adaptive action.</li>");
    html.push_str("<li>Below baseline: actively food-avoidant. Possible if the action encoding or sensory wiring has an inversion bug.</li>");
    html.push_str("</ul>");
    html.push_str("<p>If this number is flat at baseline after thousands of generations, the evolutionary loop is broken. Check energy economics first (is eating actually rewarded enough to matter?), then mutation rates (can evolution explore fast enough?).</p>");

    html.push_str("<h3>H(action) -- \"Are they interesting?\"</h3>");
    html.push_str("<p>Shannon entropy of the action distribution. Measures behavioral diversity, not quality.</p>");
    html.push_str("<ul>");
    html.push_str("<li><code>~= 0</code>: organism does one thing every tick (usually Idle). Degenerate. Common when metabolism is so punishing that any movement is a net loss.</li>");
    html.push_str("<li><code>~= log2(N_ACTIONS)</code> (<code>2.0</code> for 4 actions): uniform random. No preferences. Brain output is noise.</li>");
    html.push_str("<li><code>0.8-1.5</code> bits (for 4 actions): organism has preferences but uses multiple actions. This is where directed foragers live - they mostly go Forward but turn when needed.</li>");
    html.push_str("</ul>");
    html.push_str("<p>Read H together with P(Fwd|food). If H is intermediate AND P(Fwd|food) is above baseline, you have genuine adaptive behavior. If H is intermediate but P(Fwd|food) is at baseline, the organism has arbitrary preferences that are not connected to sensory input - a fixed motor program, not intelligence.</p>");

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
