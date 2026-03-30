use crate::{
    cli::FeatureOverrides,
    orchestration::run_validation_across_seeds,
    output::{format_seed_list, mean_option, write_summary_json},
    report::{
        write_comparison_html_report, ComparisonHtmlReportMeta, ComparisonMetricRow,
        PerSeedComparisonRow,
    },
    types::{ComparisonSummary, HarnessRunOptions, ValidationSummary},
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
    if let Some(value) = overrides.executed_action_credit {
        config.executed_action_credit = value;
    }
    if let Some(value) = overrides.explicit_idle_softmax {
        config.explicit_idle_softmax = value;
    }
    if let Some(value) = overrides.juvenile_plasticity {
        config.juvenile_plasticity_enabled = value;
    }
    if let Some(value) = overrides.split_attack {
        config.split_attack_actions = value;
    }
    config
}

pub(crate) fn run_comparison_validation(
    control_config: WorldConfig,
    treatment_config: WorldConfig,
    options: &HarnessRunOptions,
    overrides: &FeatureOverrides,
) -> Result<ComparisonSummary> {
    let run_started = Instant::now();
    fs::create_dir_all(&options.out_dir)?;

    let control = run_validation_across_seeds(
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
    let treatment = run_validation_across_seeds(
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

    let control_label = if options.baseline {
        "control (baseline)".to_owned()
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
            control_score: control_seed.aggregate_score.score,
            treatment_score: treatment_seed.aggregate_score.score,
            diff_score: treatment_seed.aggregate_score.score - control_seed.aggregate_score.score,
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
            seed_label: format_seed_list(&options.seeds),
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
    control: &ValidationSummary,
    treatment: &ValidationSummary,
) -> Vec<ComparisonMetricRow> {
    vec![
        paired_metric_row(
            "aggregate_score",
            control
                .seed_summaries
                .iter()
                .map(|seed| Some(seed.aggregate_score.score))
                .collect(),
            treatment
                .seed_summaries
                .iter()
                .map(|seed| Some(seed.aggregate_score.score))
                .collect(),
        ),
        paired_metric_row(
            "idle_fraction",
            control
                .seed_summaries
                .iter()
                .map(|seed| seed.aggregate_score.mean_idle_fraction)
                .collect(),
            treatment
                .seed_summaries
                .iter()
                .map(|seed| seed.aggregate_score.mean_idle_fraction)
                .collect(),
        ),
        paired_metric_row(
            "action_entropy",
            control
                .seed_summaries
                .iter()
                .map(|seed| seed.aggregate_score.mean_h_action)
                .collect(),
            treatment
                .seed_summaries
                .iter()
                .map(|seed| seed.aggregate_score.mean_h_action)
                .collect(),
        ),
        paired_metric_row(
            "p_fwd_food",
            control
                .seed_summaries
                .iter()
                .map(|seed| seed.aggregate_score.mean_p_fwd_food)
                .collect(),
            treatment
                .seed_summaries
                .iter()
                .map(|seed| seed.aggregate_score.mean_p_fwd_food)
                .collect(),
        ),
        paired_metric_row(
            "mi_sa",
            control
                .seed_summaries
                .iter()
                .map(|seed| seed.aggregate_score.mean_mi_sa)
                .collect(),
            treatment
                .seed_summaries
                .iter()
                .map(|seed| seed.aggregate_score.mean_mi_sa)
                .collect(),
        ),
        paired_metric_row(
            "mi_sa_juvenile",
            control
                .seed_summaries
                .iter()
                .map(|seed| seed.aggregate_score.mean_mi_sa_juvenile)
                .collect(),
            treatment
                .seed_summaries
                .iter()
                .map(|seed| seed.aggregate_score.mean_mi_sa_juvenile)
                .collect(),
        ),
        paired_metric_row(
            "mi_sa_adult",
            control
                .seed_summaries
                .iter()
                .map(|seed| seed.aggregate_score.mean_mi_sa_adult)
                .collect(),
            treatment
                .seed_summaries
                .iter()
                .map(|seed| seed.aggregate_score.mean_mi_sa_adult)
                .collect(),
        ),
        paired_metric_row(
            "reproduction_efficiency",
            control
                .seed_summaries
                .iter()
                .map(|seed| seed.aggregate_score.mean_reproduction_efficiency)
                .collect(),
            treatment
                .seed_summaries
                .iter()
                .map(|seed| seed.aggregate_score.mean_reproduction_efficiency)
                .collect(),
        ),
        paired_metric_row(
            "foraging_rate",
            control
                .seed_summaries
                .iter()
                .map(|seed| seed.aggregate_score.mean_foraging_rate)
                .collect(),
            treatment
                .seed_summaries
                .iter()
                .map(|seed| seed.aggregate_score.mean_foraging_rate)
                .collect(),
        ),
        paired_metric_row(
            "attack_attempt_rate",
            control
                .seed_summaries
                .iter()
                .map(|seed| seed.aggregate_score.mean_attack_attempt_rate)
                .collect(),
            treatment
                .seed_summaries
                .iter()
                .map(|seed| seed.aggregate_score.mean_attack_attempt_rate)
                .collect(),
        ),
        paired_metric_row(
            "attack_success_rate",
            control
                .seed_summaries
                .iter()
                .map(|seed| seed.aggregate_score.mean_attack_success_rate)
                .collect(),
            treatment
                .seed_summaries
                .iter()
                .map(|seed| seed.aggregate_score.mean_attack_success_rate)
                .collect(),
        ),
        paired_metric_row(
            "damage_avoidance",
            control
                .seed_summaries
                .iter()
                .map(|seed| seed.aggregate_score.mean_damage_avoidance)
                .collect(),
            treatment
                .seed_summaries
                .iter()
                .map(|seed| seed.aggregate_score.mean_damage_avoidance)
                .collect(),
        ),
        paired_metric_row(
            "reward_reversal_shift",
            control
                .seed_summaries
                .iter()
                .map(|seed| seed.aggregate_score.mean_reward_reversal_shift)
                .collect(),
            treatment
                .seed_summaries
                .iter()
                .map(|seed| seed.aggregate_score.mean_reward_reversal_shift)
                .collect(),
        ),
        paired_metric_row(
            "reward_reversal_adaptation_ticks",
            control
                .seed_summaries
                .iter()
                .map(|seed| {
                    seed.aggregate_score
                        .reward_reversal_adaptation_ticks
                        .map(|v| v as f64)
                })
                .collect(),
            treatment
                .seed_summaries
                .iter()
                .map(|seed| {
                    seed.aggregate_score
                        .reward_reversal_adaptation_ticks
                        .map(|v| v as f64)
                })
                .collect(),
        ),
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
