use sim_types::{ActionType, SensoryReceptor};

pub(crate) const SENSORY_COUNT: u32 = SensoryReceptor::LOOK_NEURON_COUNT + 1;
pub(crate) const ACTION_COUNT: usize = ActionType::ALL.len();
pub(crate) const ACTION_COUNT_U32: u32 = ACTION_COUNT as u32;
pub(crate) const INTER_ID_BASE: u32 = 1000;
pub(crate) const ACTION_ID_BASE: u32 = 2000;
pub(crate) const INTER_UPDATE_RATE_MAX: f32 = 1.0;
pub(crate) const INTER_UPDATE_RATE_MIN: f32 = 0.1;
