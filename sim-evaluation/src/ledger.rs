use crate::types::ReproductionAnalytics;
use sim_types::{ActionRecord, ActionType, OrganismId, OrganismState, ReproductionEvent};
use std::collections::HashMap;

pub const N_ACTIONS: usize = 7;
pub const SENSORY_BIN_COUNT: usize = 5;
const INTER_EMA_ALPHA: f32 = 0.05;
const UTILIZATION_THRESHOLD: f32 = 0.03;
const SURVIVAL_AGE_30: u64 = 30;

#[derive(Debug, Clone)]
pub struct CompletedLifetime {
    pub lifetime: u64,
    pub consumptions: u64,
    pub action_counts: [u32; N_ACTIONS],
    pub joint: [[u32; N_ACTIONS]; SENSORY_BIN_COUNT],
    pub food_ahead_ticks: u32,
    pub fwd_when_food_ahead: u32,
    pub utilization: f32,
}

#[derive(Debug, Clone, Default)]
pub struct IntervalActionStats {
    pub action_counts: [u64; N_ACTIONS],
    pub joint: [[u64; N_ACTIONS]; SENSORY_BIN_COUNT],
    pub juvenile_joint: [[u64; N_ACTIONS]; SENSORY_BIN_COUNT],
    pub adult_joint: [[u64; N_ACTIONS]; SENSORY_BIN_COUNT],
    pub reproduction_attempts: u64,
    pub total_damage_taken: f64,
}

impl IntervalActionStats {
    pub fn total_actions(&self) -> u64 {
        self.action_counts.iter().sum()
    }
}

#[derive(Debug)]
struct OrganismEntry {
    birth_tick: u64,
    tracked_reproduction_birth: bool,
    survived_to_30_recorded: bool,
    survived_to_maturity_recorded: bool,
    last_successful_birth_tick: Option<u64>,
    last_consumptions: u64,
    action_counts: [u32; N_ACTIONS],
    joint: [[u32; N_ACTIONS]; SENSORY_BIN_COUNT],
    food_ahead_ticks: u32,
    fwd_when_food_ahead: u32,
    inter_ema: Vec<f32>,
    ema_initialized: bool,
}

impl OrganismEntry {
    fn new(birth_tick: u64, tracked_reproduction_birth: bool) -> Self {
        Self {
            birth_tick,
            tracked_reproduction_birth,
            survived_to_30_recorded: false,
            survived_to_maturity_recorded: false,
            last_successful_birth_tick: None,
            last_consumptions: 0,
            action_counts: [0; N_ACTIONS],
            joint: [[0; N_ACTIONS]; SENSORY_BIN_COUNT],
            food_ahead_ticks: 0,
            fwd_when_food_ahead: 0,
            inter_ema: Vec::new(),
            ema_initialized: false,
        }
    }
}

#[derive(Debug)]
pub struct Ledger {
    sidecar: HashMap<OrganismId, OrganismEntry>,
    recently_deceased: Vec<CompletedLifetime>,
    interval_action_stats: IntervalActionStats,
    births: u64,
    successful_births: u64,
    blocked_births: u64,
    parent_died_during_reproduction: u64,
    survived_to_30: u64,
    survived_to_maturity: u64,
    pending_tracked_births: HashMap<OrganismId, ()>,
    parent_energy_after_successful_birth_sum: f64,
    parent_energy_after_successful_birth_count: u64,
    age_at_first_successful_reproduction_sum: f64,
    age_at_first_successful_reproduction_count: u64,
    successful_birth_interval_sum: f64,
    successful_birth_interval_count: u64,
    pub neonatal_deaths: u64,
    min_lifetime: u64,
}

impl Ledger {
    pub fn new(min_lifetime: u64) -> Self {
        Self {
            sidecar: HashMap::new(),
            recently_deceased: Vec::new(),
            interval_action_stats: IntervalActionStats::default(),
            births: 0,
            successful_births: 0,
            blocked_births: 0,
            parent_died_during_reproduction: 0,
            survived_to_30: 0,
            survived_to_maturity: 0,
            pending_tracked_births: HashMap::new(),
            parent_energy_after_successful_birth_sum: 0.0,
            parent_energy_after_successful_birth_count: 0,
            age_at_first_successful_reproduction_sum: 0.0,
            age_at_first_successful_reproduction_count: 0,
            successful_birth_interval_sum: 0.0,
            successful_birth_interval_count: 0,
            neonatal_deaths: 0,
            min_lifetime,
        }
    }

    pub fn birth(&mut self, id: OrganismId, tick: u64) {
        let tracked_reproduction_birth = self.pending_tracked_births.remove(&id).is_some();
        if tracked_reproduction_birth {
            self.births = self.births.saturating_add(1);
        }
        self.sidecar
            .insert(id, OrganismEntry::new(tick, tracked_reproduction_birth));
    }

    pub fn update(&mut self, record: ActionRecord) {
        let action_idx = action_index(record.selected_action);
        let sensory_bin = sensory_bin(&record);
        self.interval_action_stats.action_counts[action_idx] =
            self.interval_action_stats.action_counts[action_idx].saturating_add(1);
        self.interval_action_stats.joint[sensory_bin][action_idx] =
            self.interval_action_stats.joint[sensory_bin][action_idx].saturating_add(1);
        if record.age_turns < u64::from(record.age_of_maturity) {
            self.interval_action_stats.juvenile_joint[sensory_bin][action_idx] =
                self.interval_action_stats.juvenile_joint[sensory_bin][action_idx]
                    .saturating_add(1);
        } else {
            self.interval_action_stats.adult_joint[sensory_bin][action_idx] =
                self.interval_action_stats.adult_joint[sensory_bin][action_idx].saturating_add(1);
        }
        if record.selected_action == ActionType::Reproduce {
            self.interval_action_stats.reproduction_attempts = self
                .interval_action_stats
                .reproduction_attempts
                .saturating_add(1);
        }
        self.interval_action_stats.total_damage_taken +=
            f64::from(record.damage_taken_last_turn.max(0.0));

        let Some(entry) = self.sidecar.get_mut(&record.organism_id) else {
            return;
        };

        entry.last_consumptions = record.consumptions_count;
        entry.action_counts[action_idx] = entry.action_counts[action_idx].saturating_add(1);
        entry.joint[sensory_bin][action_idx] =
            entry.joint[sensory_bin][action_idx].saturating_add(1);

        if record.food_ahead {
            entry.food_ahead_ticks = entry.food_ahead_ticks.saturating_add(1);
            if record.selected_action == ActionType::Forward {
                entry.fwd_when_food_ahead = entry.fwd_when_food_ahead.saturating_add(1);
            }
        }

        if !entry.ema_initialized {
            entry.inter_ema = record
                .inter_activations
                .iter()
                .map(|value| value.abs())
                .collect();
            entry.ema_initialized = true;
            return;
        }

        if entry.inter_ema.len() != record.inter_activations.len() {
            entry.inter_ema = record
                .inter_activations
                .iter()
                .map(|value| value.abs())
                .collect();
            return;
        }

        for (ema, activation) in entry.inter_ema.iter_mut().zip(&record.inter_activations) {
            *ema = (1.0 - INTER_EMA_ALPHA) * *ema + INTER_EMA_ALPHA * activation.abs();
        }
    }

    pub fn death(&mut self, id: OrganismId, tick: u64) {
        let Some(entry) = self.sidecar.remove(&id) else {
            return;
        };

        let lifetime = tick.saturating_sub(entry.birth_tick);
        if lifetime < self.min_lifetime {
            self.neonatal_deaths = self.neonatal_deaths.saturating_add(1);
            return;
        }

        let utilization = if entry.inter_ema.is_empty() {
            0.0
        } else {
            let utilized = entry
                .inter_ema
                .iter()
                .filter(|value| **value > UTILIZATION_THRESHOLD)
                .count() as f32;
            utilized / entry.inter_ema.len() as f32
        };

        self.recently_deceased.push(CompletedLifetime {
            lifetime,
            consumptions: entry.last_consumptions,
            action_counts: entry.action_counts,
            joint: entry.joint,
            food_ahead_ticks: entry.food_ahead_ticks,
            fwd_when_food_ahead: entry.fwd_when_food_ahead,
            utilization,
        });
    }

    pub fn recently_deceased(&self) -> &[CompletedLifetime] {
        &self.recently_deceased
    }

    pub fn handle_reproduction_event(&mut self, tick: u64, event: ReproductionEvent) {
        match event.failure_cause {
            None => {
                self.successful_births = self.successful_births.saturating_add(1);
                self.parent_energy_after_successful_birth_sum +=
                    f64::from(event.parent_energy_after_event);
                self.parent_energy_after_successful_birth_count = self
                    .parent_energy_after_successful_birth_count
                    .saturating_add(1);
                if let Some(entry) = self.sidecar.get_mut(&event.parent_id) {
                    if entry.last_successful_birth_tick.is_none() {
                        self.age_at_first_successful_reproduction_sum +=
                            event.parent_age_turns as f64;
                        self.age_at_first_successful_reproduction_count = self
                            .age_at_first_successful_reproduction_count
                            .saturating_add(1);
                    } else if let Some(last_tick) = entry.last_successful_birth_tick {
                        self.successful_birth_interval_sum += tick.saturating_sub(last_tick) as f64;
                        self.successful_birth_interval_count =
                            self.successful_birth_interval_count.saturating_add(1);
                    }
                    entry.last_successful_birth_tick = Some(tick);
                }
                if let Some(child_id) = event.child_id {
                    self.pending_tracked_births.insert(child_id, ());
                }
            }
            Some(sim_types::ReproductionFailureCause::BlockedBirth) => {
                self.blocked_births = self.blocked_births.saturating_add(1);
            }
            Some(sim_types::ReproductionFailureCause::ParentDied) => {
                self.parent_died_during_reproduction =
                    self.parent_died_during_reproduction.saturating_add(1);
            }
        }
    }

    pub fn update_survival_thresholds(&mut self, organisms: &[OrganismState]) {
        for organism in organisms {
            let Some(entry) = self.sidecar.get_mut(&organism.id) else {
                continue;
            };
            if !entry.tracked_reproduction_birth {
                continue;
            }
            if !entry.survived_to_30_recorded && organism.age_turns >= SURVIVAL_AGE_30 {
                self.survived_to_30 = self.survived_to_30.saturating_add(1);
                entry.survived_to_30_recorded = true;
            }
            if !entry.survived_to_maturity_recorded
                && organism.age_turns >= u64::from(organism.genome.age_of_maturity)
            {
                self.survived_to_maturity = self.survived_to_maturity.saturating_add(1);
                entry.survived_to_maturity_recorded = true;
            }
        }
    }

    pub fn reproduction_analytics(&self) -> ReproductionAnalytics {
        ReproductionAnalytics {
            births: self.births,
            successful_births: self.successful_births,
            blocked_births: self.blocked_births,
            parent_died_during_reproduction: self.parent_died_during_reproduction,
            survived_to_30: self.survived_to_30,
            survived_to_maturity: self.survived_to_maturity,
            mean_parent_energy_after_successful_birth: mean_or_none(
                self.parent_energy_after_successful_birth_sum,
                self.parent_energy_after_successful_birth_count,
            ),
            mean_age_at_first_successful_reproduction: mean_or_none(
                self.age_at_first_successful_reproduction_sum,
                self.age_at_first_successful_reproduction_count,
            ),
            mean_successful_birth_interval: mean_or_none(
                self.successful_birth_interval_sum,
                self.successful_birth_interval_count,
            ),
        }
    }

    pub fn interval_action_stats(&self) -> &IntervalActionStats {
        &self.interval_action_stats
    }

    pub fn clear_interval(&mut self) {
        self.recently_deceased.clear();
        self.interval_action_stats = IntervalActionStats::default();
        self.neonatal_deaths = 0;
    }
}

fn mean_or_none(sum: f64, count: u64) -> Option<f64> {
    if count == 0 {
        None
    } else {
        Some(sum / count as f64)
    }
}

fn sensory_bin(record: &ActionRecord) -> usize {
    if record.food_ahead {
        1
    } else if record.food_left {
        2
    } else if record.food_right {
        3
    } else if record.food_behind {
        4
    } else {
        0
    }
}

fn action_index(action: ActionType) -> usize {
    match action {
        ActionType::Idle => 0,
        ActionType::TurnLeft => 1,
        ActionType::TurnRight => 2,
        ActionType::Forward => 3,
        ActionType::Eat => 4,
        ActionType::Attack => 5,
        ActionType::Reproduce => 6,
    }
}
