use std::time::Duration;

#[derive(Clone, Copy)]
pub enum BrainStage {
    InterSetup,
    InterAccumulation,
    ActionActivationResolve,
    PlasticitySetup,
    PlasticitySensoryTuning,
    PlasticityInterTuning,
    PlasticityPrune,
}

#[inline]
pub fn record_brain_stage(_stage: BrainStage, _elapsed: Duration) {}
