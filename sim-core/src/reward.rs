#[allow(dead_code)]
#[derive(Debug, Clone, Copy)]
pub(crate) enum RewardEvent {
    FoodConsumed { energy: f32 },
    PredationSucceeded { energy: f32 },
    DamageTaken { energy: f32 },
    MoveCost { energy: f32 },
    ReproductionInvested { energy: f32 },
    OffspringSpawned { reward: f32 },
}

#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct RewardLedger {
    pub(crate) food_consumed_energy: f32,
    pub(crate) predation_energy: f32,
    pub(crate) damage_taken_energy: f32,
    pub(crate) move_cost_energy: f32,
    pub(crate) reproduction_investment_energy: f32,
    pub(crate) offspring_spawn_reward: f32,
}

impl RewardLedger {
    pub(crate) fn clear(&mut self) {
        *self = Self::default();
    }

    pub(crate) fn record(&mut self, event: RewardEvent) {
        match event {
            RewardEvent::FoodConsumed { energy } => {
                self.food_consumed_energy += energy.max(0.0);
            }
            RewardEvent::PredationSucceeded { energy } => {
                self.predation_energy += energy.max(0.0);
            }
            RewardEvent::DamageTaken { energy } => {
                self.damage_taken_energy += energy.max(0.0);
            }
            RewardEvent::MoveCost { energy } => {
                self.move_cost_energy += energy.max(0.0);
            }
            RewardEvent::ReproductionInvested { energy } => {
                self.reproduction_investment_energy += energy.max(0.0);
            }
            RewardEvent::OffspringSpawned { reward } => {
                self.offspring_spawn_reward += reward.max(0.0);
            }
        }
    }

    pub(crate) fn reward_signal(self) -> f32 {
        self.food_consumed_energy + self.predation_energy + self.offspring_spawn_reward
            - self.reproduction_investment_energy
            - self.damage_taken_energy
    }
}

#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize, PartialEq, Eq, Default)]
pub(crate) enum PendingActionKind {
    #[default]
    None,
    Reproduce,
}

#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize, PartialEq, Eq, Default)]
pub(crate) struct PendingActionState {
    pub(crate) kind: PendingActionKind,
    pub(crate) turns_remaining: u8,
    pub(crate) reproduction_energy_bits: u32,
}

impl PendingActionState {
    pub(crate) fn reproduction_energy(self) -> f32 {
        f32::from_bits(self.reproduction_energy_bits)
    }
}
