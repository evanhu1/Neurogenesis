use crate::types::ReproductionAnalytics;
use sim_types::{ActionRecord, ActionType, OrganismId, OrganismState, ReproductionEvent};
use std::collections::HashMap;

pub const N_ACTIONS: usize = 7;
pub const SENSORY_BIN_COUNT: usize = 5;
const SURVIVAL_AGE_30: u64 = 30;

#[derive(Debug, Clone, Default)]
pub struct IntervalLifetimeSummary {
    pub count: u64,
    pub lifetime_sum: u64,
    pub ate_count: u64,
    pub consumptions_sum: u64,
    pub action_counts: [u64; N_ACTIONS],
    pub joint: [[u64; N_ACTIONS]; SENSORY_BIN_COUNT],
    pub food_ahead_ticks_sum: u64,
    pub fwd_when_food_ahead_sum: u64,
    pub utilization_sum: f64,
}

impl IntervalLifetimeSummary {
    fn record(&mut self, entry: &OrganismEntry, lifetime: u64) {
        self.count = self.count.saturating_add(1);
        self.lifetime_sum = self.lifetime_sum.saturating_add(lifetime);
        self.ate_count = self
            .ate_count
            .saturating_add(u64::from(entry.last_consumptions > 0));
        self.consumptions_sum = self
            .consumptions_sum
            .saturating_add(entry.last_consumptions);
        self.food_ahead_ticks_sum = self
            .food_ahead_ticks_sum
            .saturating_add(u64::from(entry.food_ahead_ticks));
        self.fwd_when_food_ahead_sum = self
            .fwd_when_food_ahead_sum
            .saturating_add(u64::from(entry.fwd_when_food_ahead));
        self.utilization_sum += f64::from(entry.utilization.clamp(0.0, 1.0));
        for (idx, count) in entry.action_counts.iter().enumerate() {
            self.action_counts[idx] = self.action_counts[idx].saturating_add(u64::from(*count));
        }
        for (sensory_idx, row) in entry.joint.iter().enumerate() {
            for (action_idx, count) in row.iter().enumerate() {
                self.joint[sensory_idx][action_idx] =
                    self.joint[sensory_idx][action_idx].saturating_add(u64::from(*count));
            }
        }
    }
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
    age_of_maturity: u32,
    tracked_reproduction_birth: bool,
    survived_to_30_recorded: bool,
    survived_to_maturity_recorded: bool,
    last_successful_birth_tick: Option<u64>,
    last_consumptions: u64,
    action_counts: [u32; N_ACTIONS],
    joint: [[u32; N_ACTIONS]; SENSORY_BIN_COUNT],
    food_ahead_ticks: u32,
    fwd_when_food_ahead: u32,
    utilization: f32,
}

impl OrganismEntry {
    fn new(birth_tick: u64, age_of_maturity: u32, tracked_reproduction_birth: bool) -> Self {
        Self {
            birth_tick,
            age_of_maturity,
            tracked_reproduction_birth,
            survived_to_30_recorded: false,
            survived_to_maturity_recorded: false,
            last_successful_birth_tick: None,
            last_consumptions: 0,
            action_counts: [0; N_ACTIONS],
            joint: [[0; N_ACTIONS]; SENSORY_BIN_COUNT],
            food_ahead_ticks: 0,
            fwd_when_food_ahead: 0,
            utilization: 0.0,
        }
    }
}

#[derive(Debug)]
pub struct Ledger {
    sidecar: HashMap<OrganismId, OrganismEntry>,
    recently_deceased: IntervalLifetimeSummary,
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
            recently_deceased: IntervalLifetimeSummary::default(),
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

    pub fn birth(&mut self, id: OrganismId, tick: u64, age_of_maturity: u32) {
        let tracked_reproduction_birth = self.pending_tracked_births.remove(&id).is_some();
        if tracked_reproduction_birth {
            self.births = self.births.saturating_add(1);
        }
        self.sidecar.insert(
            id,
            OrganismEntry::new(tick, age_of_maturity, tracked_reproduction_birth),
        );
    }

    pub fn update(&mut self, record: ActionRecord) {
        let action_idx = action_index(record.selected_action);
        let sensory_bin = sensory_bin(&record);
        self.interval_action_stats.action_counts[action_idx] =
            self.interval_action_stats.action_counts[action_idx].saturating_add(1);
        self.interval_action_stats.joint[sensory_bin][action_idx] =
            self.interval_action_stats.joint[sensory_bin][action_idx].saturating_add(1);
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

        if record.age_turns < u64::from(entry.age_of_maturity) {
            self.interval_action_stats.juvenile_joint[sensory_bin][action_idx] =
                self.interval_action_stats.juvenile_joint[sensory_bin][action_idx]
                    .saturating_add(1);
        } else {
            self.interval_action_stats.adult_joint[sensory_bin][action_idx] =
                self.interval_action_stats.adult_joint[sensory_bin][action_idx].saturating_add(1);
        }
        entry.last_consumptions = record.consumptions_count;
        entry.action_counts[action_idx] = entry.action_counts[action_idx].saturating_add(1);
        entry.joint[sensory_bin][action_idx] =
            entry.joint[sensory_bin][action_idx].saturating_add(1);
        entry.utilization = record.utilization.clamp(0.0, 1.0);

        if record.food_ahead {
            entry.food_ahead_ticks = entry.food_ahead_ticks.saturating_add(1);
            if record.selected_action == ActionType::Forward {
                entry.fwd_when_food_ahead = entry.fwd_when_food_ahead.saturating_add(1);
            }
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

        self.recently_deceased.record(&entry, lifetime);
    }

    pub fn recently_deceased(&self) -> &IntervalLifetimeSummary {
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
                && organism.age_turns >= u64::from(entry.age_of_maturity)
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
        self.recently_deceased = IntervalLifetimeSummary::default();
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
