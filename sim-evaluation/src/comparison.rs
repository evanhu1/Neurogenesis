use crate::{
    cli::FeatureOverrides,
    orchestration::run_evaluation_across_seeds,
    output::{mean_option, write_summary_json},
    report::{
        write_comparison_html_report, ComparisonHtmlReportMeta, ComparisonMetricRow,
        PerSeedComparisonRow,
    },
    types::{ComparisonSummary, EvaluationSummary, HarnessRunOptions},
};
use anyhow::Result;
use sim_types::WorldConfig;
use std::fs;
use std::time::Instant;

pub(crate) fn apply_feature_overrides(
    mut config: WorldConfig,
    overrides: &FeatureOverrides,
) -> WorldConfig {
    if overrides.disable_plasticity {
        config.runtime_plasticity_enabled = false;
    }
    config
}

pub(crate) fn run_comparison_evaluation(
    control_config: WorldConfig,
    treatment_config: WorldConfig,
    options: &HarnessRunOptions,
    overrides: &FeatureOverrides,
) -> Result<ComparisonSummary> {
    let run_started = Instant::now();
    fs::create_dir_all(&options.out_dir)?;

    let control = run_evaluation_across_seeds(
        control_config,
        &HarnessRunOptions {
            out_dir: options.out_dir.join("control"),
            title: options
                .title
                .as_ref()
                .map(|title| format!("{title} [control]")),
            ..options.clone()
        },
    )?;
    let treatment = run_evaluation_across_seeds(
        treatment_config,
        &HarnessRunOptions {
            out_dir: options.out_dir.join("treatment"),
            title: options
                .title
                .as_ref()
                .map(|title| format!("{title} [treatment]")),
            ..options.clone()
        },
    )?;

    let control_label = if options.control {
        "random-action control".to_owned()
    } else {
        "control".to_owned()
    };
    let treatment_label = overrides.label();
    let metric_rows = comparison_metric_rows(&control, &treatment);
    let per_seed_rows = control
        .seed_summaries
        .iter()
        .zip(&treatment.seed_summaries)
        .map(|(control_seed, treatment_seed)| PerSeedComparisonRow {
            seed: control_seed.seed,
            control_report_href: format!("control/seed_{}/report.html", control_seed.seed),
            treatment_report_href: format!("treatment/seed_{}/report.html", treatment_seed.seed),
        })
        .collect::<Vec<_>>();
    let total_time_seconds = run_started.elapsed().as_secs_f64();
    let comparison = ComparisonSummary {
        title: options.title.clone(),
        seeds: options.seeds.clone(),
        ticks: options.ticks,
        control_label: control_label.clone(),
        treatment_label: treatment_label.clone(),
        total_time_seconds,
        control: control.clone(),
        treatment: treatment.clone(),
        metric_rows: metric_rows.clone(),
    };
    write_summary_json(&options.out_dir, &comparison)?;
    write_comparison_html_report(
        &options.out_dir,
        &ComparisonHtmlReportMeta {
            title: options.title.clone(),
            ticks: options.ticks,
            control_label,
            treatment_label,
            total_time_seconds,
            metric_rows,
            per_seed_rows,
            control_report_href: "control/report.html".to_owned(),
            treatment_report_href: "treatment/report.html".to_owned(),
        },
    )?;
    Ok(comparison)
}

fn comparison_metric_rows(
    control: &EvaluationSummary,
    treatment: &EvaluationSummary,
) -> Vec<ComparisonMetricRow> {
    let paired = |label: &str, f: fn(&crate::types::PillarScores) -> Option<f64>| {
        paired_metric_row(
            label,
            control
                .seed_summaries
                .iter()
                .map(|s| f(&s.pillars))
                .collect(),
            treatment
                .seed_summaries
                .iter()
                .map(|s| f(&s.pillars))
                .collect(),
        )
    };
    vec![
        paired("foraging_pillar", |p| Some(p.foraging_pillar)),
        paired("intelligence_pillar", |p| Some(p.intelligence_pillar)),
        paired("competition_pillar", |p| Some(p.competition_pillar)),
        paired("idle_fraction", |p| p.mean_idle_fraction),
        paired("p_fwd_food", |p| p.mean_p_fwd_food),
        paired("mi_sa", |p| p.mean_mi_sa),
        paired("foraging_rate", |p| p.mean_foraging_rate),
        paired("attack_attempt_rate", |p| p.mean_attack_attempt_rate),
        paired("attack_success_rate", |p| p.mean_attack_success_rate),
        paired("failed_action_rate", |p| p.mean_failed_action_rate),
        paired("util", |p| p.mean_util),
    ]
}

fn paired_metric_row(
    label: &str,
    control_values: Vec<Option<f64>>,
    treatment_values: Vec<Option<f64>>,
) -> ComparisonMetricRow {
    let control_mean = mean_option(control_values.iter().copied());
    let treatment_mean = mean_option(treatment_values.iter().copied());
    let diffs = control_values
        .iter()
        .zip(&treatment_values)
        .filter_map(|(control, treatment)| match (*control, *treatment) {
            (Some(control), Some(treatment)) => Some(treatment - control),
            _ => None,
        })
        .collect::<Vec<_>>();
    let (mean_diff, ci_low, ci_high) = diff_confidence_interval(&diffs);
    ComparisonMetricRow {
        label: label.to_owned(),
        control_mean,
        treatment_mean,
        mean_diff,
        ci_low,
        ci_high,
    }
}

fn diff_confidence_interval(diffs: &[f64]) -> (Option<f64>, Option<f64>, Option<f64>) {
    if diffs.is_empty() {
        return (None, None, None);
    }
    let mean = diffs.iter().sum::<f64>() / diffs.len() as f64;
    if diffs.len() == 1 {
        return (Some(mean), Some(mean), Some(mean));
    }
    let variance = diffs
        .iter()
        .map(|diff| {
            let delta = *diff - mean;
            delta * delta
        })
        .sum::<f64>()
        / (diffs.len() as f64 - 1.0);
    let se = variance.sqrt() / (diffs.len() as f64).sqrt();
    let margin = 1.96 * se;
    (Some(mean), Some(mean - margin), Some(mean + margin))
}
