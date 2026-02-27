use sim_types::{ActionRecord, ActionType, OrganismId};
use std::collections::HashMap;

pub const N_ACTIONS: usize = ActionType::ALL.len();
pub const SENSORY_BIN_COUNT: usize = 5;
const INTER_EMA_ALPHA: f32 = 0.05;
const UTILIZATION_THRESHOLD: f32 = 0.03;

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

#[derive(Debug)]
struct OrganismEntry {
    birth_tick: u64,
    last_consumptions: u64,
    action_counts: [u32; N_ACTIONS],
    joint: [[u32; N_ACTIONS]; SENSORY_BIN_COUNT],
    food_ahead_ticks: u32,
    fwd_when_food_ahead: u32,
    inter_ema: Vec<f32>,
    ema_initialized: bool,
}

impl OrganismEntry {
    fn new(birth_tick: u64) -> Self {
        Self {
            birth_tick,
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
    pub neonatal_deaths: u64,
    min_lifetime: u64,
}

impl Ledger {
    pub fn new(min_lifetime: u64) -> Self {
        Self {
            sidecar: HashMap::new(),
            recently_deceased: Vec::new(),
            neonatal_deaths: 0,
            min_lifetime,
        }
    }

    pub fn birth(&mut self, id: OrganismId, tick: u64) {
        self.sidecar.insert(id, OrganismEntry::new(tick));
    }

    pub fn update(&mut self, record: ActionRecord) {
        let Some(entry) = self.sidecar.get_mut(&record.organism_id) else {
            return;
        };

        let action_idx = action_index(record.selected_action);
        entry.last_consumptions = record.consumptions_count;
        entry.action_counts[action_idx] = entry.action_counts[action_idx].saturating_add(1);

        let sensory_bin = if record.food_ahead {
            1
        } else if record.food_left {
            2
        } else if record.food_right {
            3
        } else if record.food_behind {
            4
        } else {
            0
        };
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

    pub fn clear_interval(&mut self) {
        self.recently_deceased.clear();
        self.neonatal_deaths = 0;
    }
}

fn action_index(action: ActionType) -> usize {
    match action {
        ActionType::Idle => 0,
        ActionType::TurnLeft => 1,
        ActionType::TurnRight => 2,
        ActionType::Forward => 3,
        ActionType::Consume => 4,
        ActionType::Reproduce => 5,
    }
}
