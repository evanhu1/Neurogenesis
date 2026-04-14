use sim_types::OrganismState;

/// Per-tick tonic reward signals. Overwritten each tick by `observe` — not
/// accumulated across events. All channel magnitudes are unit-normalized so the
/// weighted signal naturally sits in roughly [-2, +2] under default weights.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct RewardLedger {
    pub(crate) energy_level: f32,
    pub(crate) energy_delta_gain: f32,
    pub(crate) energy_delta_loss: f32,
    pub(crate) health_level: f32,
    pub(crate) health_delta_gain: f32,
    pub(crate) health_delta_loss: f32,
}

impl RewardLedger {
    /// Overwrites the ledger with tonic signals derived from the organism's
    /// current state and its `energy_prev` / `health_prev` snapshots. Deltas
    /// cover the full tick (from end of previous plasticity step to now).
    pub(crate) fn observe(&mut self, organism: &OrganismState, food_energy: f32) {
        let food_scale = food_energy.max(1e-3);
        let health_scale = organism.max_health.max(1e-3);

        let energy_delta = organism.energy - organism.energy_prev;
        let health_delta = organism.health - organism.health_prev;

        self.energy_level = (organism.energy / (4.0 * food_scale)).clamp(0.0, 1.0) - 0.5;
        self.energy_delta_gain = energy_delta.max(0.0) / food_scale;
        self.energy_delta_loss = (-energy_delta).max(0.0) / food_scale;
        self.health_level = (organism.health / health_scale).clamp(0.0, 1.0) - 0.5;
        self.health_delta_gain = health_delta.max(0.0) / health_scale;
        self.health_delta_loss = (-health_delta).max(0.0) / health_scale;
    }

    /// Weighted linear combination of the tonic channels. A short/malformed
    /// weights slice falls back to `DEFAULT_REWARD_WEIGHTS` element-wise so
    /// legacy organisms and deserialized genomes keep the baseline behavior.
    pub(crate) fn weighted_reward_signal(self, weights: &[f32]) -> f32 {
        let w = |i: usize| {
            weights
                .get(i)
                .copied()
                .unwrap_or(DEFAULT_REWARD_WEIGHTS[i])
        };
        w(0) * self.energy_level
            + w(1) * self.energy_delta_gain
            + w(2) * self.energy_delta_loss
            + w(3) * self.health_level
            + w(4) * self.health_delta_gain
            + w(5) * self.health_delta_loss
    }
}

/// Number of genomic reward-weight coefficients; must match the channel count
/// in `RewardLedger::weighted_reward_signal` and `DEFAULT_REWARD_WEIGHTS`.
pub(crate) const REWARD_WEIGHT_COUNT: usize = 6;

/// Default reward-weight vector, ordered to match the ledger fields:
/// [energy_level, energy_delta_gain, energy_delta_loss,
///  health_level, health_delta_gain, health_delta_loss].
/// Level channels start at 0 / +1 so evolution can discover which absolute
/// states matter; delta gain/loss defaults split into +1 / -1 so signed
/// change in resources drives a symmetric dopamine response.
pub(crate) const DEFAULT_REWARD_WEIGHTS: [f32; REWARD_WEIGHT_COUNT] =
    [0.0, 1.0, -1.0, 1.0, 1.0, -1.0];

/// Bound on an individual reward weight. Evolution can flip signs and scale
/// up to 3x, but cannot produce arbitrarily large dopamine magnitudes.
pub(crate) const REWARD_WEIGHT_MIN: f32 = -3.0;
pub(crate) const REWARD_WEIGHT_MAX: f32 = 3.0;

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
