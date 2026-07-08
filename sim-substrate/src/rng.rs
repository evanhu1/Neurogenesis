//! Deterministic RNG + structural-hashing utilities shared across the substrate
//! and its environments.
//!
//! The whole engine replays from `(seed, turn, id)`, so every RNG draw and every
//! identity derivation must be a pure function of inputs — never a global
//! counter, never wall-clock. `mix_u64` is the SplitMix64 finalizer used by the
//! current engine (`sim-core/src/turn/mod.rs`); we keep it byte-identical so
//! ported determinism tests still hold.

/// The canonical per-stream RNG. ChaCha8 is reproducible and `serde`-friendly,
/// matching the workspace `rand_chacha` dependency.
pub type Rng = rand_chacha::ChaCha8Rng;

/// SplitMix64 finalizing mix — identical to `sim-core`'s `mix_u64`.
#[inline]
pub fn mix_u64(mut value: u64) -> u64 {
    value ^= value >> 30;
    value = value.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value ^= value >> 27;
    value = value.wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^= value >> 31;
    value
}

// Domain separation constants so hashes of the same integers never collide
// across purposes (connection identity vs node identity vs stream seeding).
const DOMAIN_CONN: u64 = 0x0000_C0FF_EE00_0001;
const DOMAIN_NODE: u64 = 0x0000_C0FF_EE00_0002;
const DOMAIN_STREAM: u64 = 0x0000_C0FF_EE00_0003;

/// Structural identity of a connection between two CPPN node ids. Order matters
/// (directed edge). Deterministic and collision-negligible; convergent-identical
/// structure is homologous by design (see plan risk #1).
#[inline]
pub fn hash_conn(from: u64, to: u64) -> u64 {
    mix_u64(DOMAIN_CONN ^ mix_u64(from).rotate_left(17) ^ mix_u64(to).rotate_left(41))
}

/// Structural identity of a node created by splitting a connection with the
/// given innovation id. Stable across independent lineages that split the same
/// homologous connection.
#[inline]
pub fn hash_node(parent_conn_innovation: u64) -> u64 {
    mix_u64(DOMAIN_NODE ^ mix_u64(parent_conn_innovation).rotate_left(29))
}

/// Seed a fresh RNG stream deterministically from a base seed and a purpose tag.
pub fn stream(seed: u64, tag: u64) -> Rng {
    use rand::SeedableRng;
    let mixed = mix_u64(DOMAIN_STREAM ^ mix_u64(seed) ^ mix_u64(tag).rotate_left(23));
    Rng::seed_from_u64(mixed)
}
