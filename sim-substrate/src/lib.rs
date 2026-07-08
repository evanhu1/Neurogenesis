//! `sim-substrate` — the environment-agnostic evolutionary meta-learner.
//!
//! It owns the genome (a CPPN that indirectly encodes the phenotype), the
//! developmental map `develop()` (ES-HyperNEAT), the `BrainNet` runtime with
//! within-lifetime Hebbian plasticity, the three operators
//! (`mutate`/`crossover`/`reproduce`), the `Environment` trait boundary, the
//! `PopulationDriver` tick loop, and the Quality-Diversity champion archive.
//!
//! It has no knowledge of any world: environments (hex ecology, toy ribbon,
//! ...) implement `Environment` and are driven unchanged. Determinism is a hard
//! invariant — every RNG draw is a pure function of `(seed, turn, id)` and all
//! cross-body decisions live in handle-ordered serial code.

pub mod brain;
pub mod catalog;
pub mod cppn;
pub mod develop;
pub mod driver;
pub mod environment;
pub mod genome;
pub mod operators;
pub mod plasticity;
pub mod qd;
pub mod rng;
pub mod seed;

pub use brain::{sample_action, BrainNet};
pub use catalog::{
    ActionLayout, ActuatorSpec, Coord, MorphologyParam, ObsLayout, SensorSpec, SubstrateCatalog,
};
pub use cppn::{Activation, CppnGenome};
pub use develop::{develop, DevelopConfig, Phenotype};
pub use driver::{DriverConfig, PopulationDriver};
pub use environment::{
    ActionOutput, Body, BodyHandle, BodyView, DerivedBodyParams, EffectSink, Environment,
    Gestation, MateIntent, PopulationRead,
};
pub use genome::{Genome, HeaderGenes, MutationRates};
pub use operators::{crossover, mutate, reproduce, MutateCtx};
pub use qd::{BehaviorDescriptor, QdArchive};
