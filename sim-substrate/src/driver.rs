//! `PopulationDriver` owns the tick loop: the `Vec<Body>`, the RNG streams,
//! brain evaluation, plasticity, energy/age/death/mating bookkeeping, and phase
//! sequencing. The environment supplies physics through the `Environment`
//! trait. All cross-body decisions and all RNG draws live in deterministic,
//! order-independent code (id-keyed hashing / handle-ordered sink application),
//! so results are identical regardless of thread scheduling.

use crate::brain::sample_action;
use crate::develop::{develop, DevelopConfig};
use crate::environment::{Body, BodyHandle, BodyView, Environment, Gestation, PopulationRead};
use crate::genome::{Genome, MutationRates};
use crate::operators::{reproduce, MutateCtx};
use crate::plasticity::{plasticity_step, PlasticityParams};
use crate::rng::{mix_u64, stream};

const RNG_TURN_MIX: u64 = 0x9E37_79B9_7F4A_7C15;
const RNG_ORGANISM_MIX: u64 = 0xBF58_476D_1CE4_E5B9;

// Phase tags for deriving independent per-turn RNG streams.
const TAG_STEP_WORLD: u64 = 1;
const TAG_RESOLVE: u64 = 2;
const TAG_PLACE: u64 = 3;
const TAG_BIRTH: u64 = 4;

fn deterministic_unit_sample(seed: u64, turn: u64, id: u64) -> f32 {
    let mixed = seed ^ turn.wrapping_mul(RNG_TURN_MIX) ^ id.wrapping_mul(RNG_ORGANISM_MIX);
    let sample = (mix_u64(mixed) >> 40) as u32;
    sample as f32 / ((1_u32 << 24) - 1) as f32
}

/// Read-only view over the body slice handed to the environment.
struct PopSlice<'a> {
    bodies: &'a [Body],
}

impl PopulationRead for PopSlice<'_> {
    fn len(&self) -> usize {
        self.bodies.len()
    }
    fn view(&self, handle: BodyHandle) -> Option<BodyView<'_>> {
        self.bodies.get(handle.0 as usize).map(|b| b.view(handle))
    }
    fn is_alive(&self, handle: BodyHandle) -> bool {
        self.bodies.get(handle.0 as usize).map(|b| b.alive).unwrap_or(false)
    }
}

#[derive(Clone, Copy, serde::Serialize, serde::Deserialize)]
pub struct DriverConfig {
    pub global_mutation_rate_modifier: f32,
    pub meta_mutation_enabled: bool,
    pub action_temperature: f32,
    pub founder_energy: f32,
    pub develop: DevelopConfig,
}

impl Default for DriverConfig {
    fn default() -> Self {
        DriverConfig {
            global_mutation_rate_modifier: 1.0,
            meta_mutation_enabled: true,
            action_temperature: 1.0,
            founder_energy: 100.0,
            develop: DevelopConfig::default(),
        }
    }
}

#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct PopulationDriver {
    pub bodies: Vec<Body>,
    pub turn: u64,
    seed: u64,
    next_id: u64,
    births_this_turn: u64,
    baseline_rates: MutationRates,
    config: DriverConfig,
    // Reused scratch so ticks avoid per-tick allocation churn; never persisted.
    #[serde(skip)]
    sink: crate::environment::EffectSink,
}

/// Per-body action selection carried between phases.
struct ActionRecord {
    handle: BodyHandle,
    logits: Vec<f32>,
    selected: Option<usize>,
    confidence: f32,
}

impl PopulationDriver {
    pub fn new(seed: u64, baseline_rates: MutationRates, config: DriverConfig) -> Self {
        PopulationDriver {
            bodies: Vec::new(),
            turn: 0,
            seed,
            next_id: 1,
            births_this_turn: 0,
            baseline_rates,
            config,
            sink: crate::environment::EffectSink::default(),
        }
    }

    pub fn alive_count(&self) -> usize {
        self.bodies.iter().filter(|b| b.alive).count()
    }

    /// Total organisms ever created (founders + births) — useful for tests.
    pub fn total_ever(&self) -> u64 {
        self.next_id - 1
    }

    fn mutate_ctx(&self) -> MutateCtx<'_> {
        MutateCtx {
            global_mutation_rate_modifier: self.config.global_mutation_rate_modifier,
            meta_mutation_enabled: self.config.meta_mutation_enabled,
            baseline_rates: &self.baseline_rates,
        }
    }

    fn make_body<E: Environment>(
        &mut self,
        env: &E,
        genome: Genome,
        generation: u64,
        starting_energy: f32,
    ) -> Body {
        let pheno = develop(&genome, env.catalog(), &self.config.develop);
        let derived = env.derive_body_params(&pheno.morphology_values);
        let id = self.next_id;
        self.next_id += 1;
        Body {
            id,
            alive: true,
            energy: starting_energy,
            health: derived.max_health,
            age_turns: 0,
            generation,
            is_gestating: false,
            energy_at_last_sensing: starting_energy,
            morphology: pheno.morphology_values,
            derived,
            genome,
            brain: pheno.brain,
            obs_layout: pheno.obs_layout,
            action_layout: pheno.action_layout,
            gestation: None,
        }
    }

    /// Seed `count` founder organisms from `seed_genome`.
    pub fn seed_population<E: Environment>(
        &mut self,
        env: &mut E,
        count: usize,
        seed_genome: &Genome,
    ) {
        self.seed_population_from(env, count, std::slice::from_ref(seed_genome));
    }

    /// Seed `count` founders drawing genomes cyclically from `pool` (a seed
    /// genome, or a champion pool). Deterministic; per-founder placement RNG is
    /// hashed from the base seed and index.
    pub fn seed_population_from<E: Environment>(
        &mut self,
        env: &mut E,
        count: usize,
        pool: &[Genome],
    ) {
        if pool.is_empty() {
            return;
        }
        for i in 0..count {
            let mut rng = stream(self.seed, mix_u64(0xF0 ^ i as u64));
            let Some(site) = env.place_founder(&mut rng) else {
                continue;
            };
            let handle = BodyHandle(self.bodies.len() as u32);
            let genome = pool[i % pool.len()].clone();
            let body = self.make_body(env, genome, 0, self.config.founder_energy);
            self.bodies.push(body);
            env.attach(&self.bodies[handle.0 as usize].view(handle), site);
        }
    }

    fn apply_sink(&mut self) -> Vec<BodyHandle> {
        let mut newly_dead = Vec::new();
        // Snapshot-then-apply in handle order.
        self.sink.energy_deltas.sort_by_key(|(h, _)| h.0);
        self.sink.health_deltas.sort_by_key(|(h, _)| h.0);
        self.sink.deaths.sort_by_key(|h| h.0);
        for (h, d) in &self.sink.energy_deltas {
            if let Some(b) = self.bodies.get_mut(h.0 as usize) {
                if b.alive {
                    b.energy += d;
                }
            }
        }
        for (h, d) in &self.sink.health_deltas {
            if let Some(b) = self.bodies.get_mut(h.0 as usize) {
                if b.alive {
                    b.health = (b.health + d).min(b.derived.max_health);
                }
            }
        }
        for h in &self.sink.deaths {
            if let Some(b) = self.bodies.get_mut(h.0 as usize) {
                if b.alive {
                    b.alive = false;
                    newly_dead.push(*h);
                }
            }
        }
        self.sink.clear();
        newly_dead
    }

    fn kill_and_notify<E: Environment>(&mut self, env: &mut E, dead: Vec<BodyHandle>) {
        if dead.is_empty() {
            return;
        }
        let pop = PopSlice { bodies: &self.bodies };
        env.on_deaths(&dead, &pop, &mut self.sink);
        let _ = self.apply_sink();
    }

    pub fn tick<E: Environment>(&mut self, env: &mut E) {
        let turn = self.turn;
        self.births_this_turn = 0;

        // ---- Phase A: metabolism + lifecycle deaths ------------------------
        let mut dead: Vec<BodyHandle> = Vec::new();
        for i in 0..self.bodies.len() {
            if !self.bodies[i].alive {
                continue;
            }
            let handle = BodyHandle(i as u32);
            let cost = {
                let view = self.bodies[i].view(handle);
                env.metabolic_cost(&view)
            };
            let body = &mut self.bodies[i];
            body.energy -= cost;
            let over_age =
                body.age_turns >= u64::from(body.genome.header.lifecycle.max_organism_age);
            if body.energy <= 0.0 || body.health <= 0.0 || over_age {
                body.alive = false;
                dead.push(handle);
            }
        }
        self.kill_and_notify(env, dead);

        // ---- Phase B: sensing + brain + action selection -------------------
        let mut obs = Vec::new();
        let mut actions: Vec<ActionRecord> = Vec::new();
        for i in 0..self.bodies.len() {
            if !self.bodies[i].alive || self.bodies[i].is_gestating {
                continue;
            }
            let handle = BodyHandle(i as u32);
            obs.clear();
            obs.resize(self.bodies[i].obs_layout.len, 0.0);
            {
                let view = self.bodies[i].view(handle);
                env.observe(&view, &self.bodies[i].obs_layout, &mut obs);
            }
            self.bodies[i].energy_at_last_sensing = self.bodies[i].energy;
            let logits: Vec<f32> = self.bodies[i].brain.step(&obs).to_vec();
            let sample = deterministic_unit_sample(self.seed, turn, self.bodies[i].id);
            let selected = sample_action(&logits, self.config.action_temperature, sample);
            let confidence = selected
                .map(|s| softmax_prob(&logits, self.config.action_temperature, s))
                .unwrap_or(0.0);
            actions.push(ActionRecord {
                handle,
                logits,
                selected,
                confidence,
            });
        }

        // ---- Phase C: mating (deterministic pairing) -----------------------
        self.mating_phase(env, &actions);

        // ---- Phase D: action resolution ------------------------------------
        {
            let pop = PopSlice { bodies: &self.bodies };
            let intents: Vec<E::Intents> = actions
                .iter()
                .filter_map(|rec| {
                    let body = &self.bodies[rec.handle.0 as usize];
                    if !body.alive {
                        return None;
                    }
                    let view = body.view(rec.handle);
                    let action = crate::environment::ActionOutput {
                        logits: &rec.logits,
                        selected: rec.selected,
                        confidence: rec.confidence,
                        layout: &body.action_layout,
                    };
                    Some(env.decode_intents(&view, &action))
                })
                .collect();
            let mut rng = stream(self.seed, mix_u64(TAG_RESOLVE ^ turn.wrapping_mul(RNG_TURN_MIX)));
            env.resolve_actions(&intents, &pop, &mut rng, &mut self.sink);
        }
        let dead = self.apply_sink();
        self.kill_and_notify(env, dead);

        // ---- Phase E: world step -------------------------------------------
        {
            let pop = PopSlice { bodies: &self.bodies };
            let mut rng =
                stream(self.seed, mix_u64(TAG_STEP_WORLD ^ turn.wrapping_mul(RNG_TURN_MIX)));
            env.step_world(&pop, &mut rng, &mut self.sink);
        }
        let dead = self.apply_sink();
        self.kill_and_notify(env, dead);

        // ---- Phase F: gestation + births -----------------------------------
        self.birth_phase(env);

        // ---- Phase G: age + plasticity -------------------------------------
        for i in 0..self.bodies.len() {
            let body = &mut self.bodies[i];
            if !body.alive {
                continue;
            }
            if body.age_turns > 0 && !body.is_gestating {
                let energy_delta = body.energy - body.energy_at_last_sensing;
                let params =
                    PlasticityParams::derive(&body.genome.header, body.age_turns, energy_delta);
                plasticity_step(&mut body.brain, &params);
            }
            body.age_turns += 1;
        }

        self.turn += 1;
    }

    fn mating_phase<E: Environment>(&mut self, env: &mut E, actions: &[ActionRecord]) {
        let mut intents: Vec<(BodyHandle, BodyHandle, f32)> = Vec::new();
        {
            let _pop = PopSlice { bodies: &self.bodies };
            for rec in actions {
                let body = &self.bodies[rec.handle.0 as usize];
                if !body.alive || body.is_gestating {
                    continue;
                }
                let view = body.view(rec.handle);
                let action = crate::environment::ActionOutput {
                    logits: &rec.logits,
                    selected: rec.selected,
                    confidence: rec.confidence,
                    layout: &body.action_layout,
                };
                if let Some(mate) = env.mate_intent(&view, &action) {
                    intents.push((rec.handle, mate.target, mate.confidence));
                }
            }
        }
        if intents.is_empty() {
            return;
        }
        // Deterministic pairing order: (target, confidence desc, initiator id asc).
        intents.sort_by(|a, b| {
            a.1 .0
                .cmp(&b.1 .0)
                .then_with(|| b.2.total_cmp(&a.2))
                .then_with(|| {
                    self.bodies[a.0 .0 as usize].id.cmp(&self.bodies[b.0 .0 as usize].id)
                })
        });
        let mut claimed = vec![false; self.bodies.len()];
        for (initiator, target, _conf) in intents {
            let (ii, ti) = (initiator.0 as usize, target.0 as usize);
            if ii >= claimed.len() || ti >= claimed.len() || claimed[ii] || claimed[ti] {
                continue;
            }
            if !self.eligible_to_mate(ii) || !self.mate_partner_ok(ti) {
                continue;
            }
            claimed[ii] = true;
            claimed[ti] = true;
            // One-sided consent (bootstrap default): initiator pays and gestates.
            let investment = self.bodies[ii].derived.investment_energy;
            let partner_genome = self.bodies[ti].genome.clone();
            let gestation_ticks = self.bodies[ii].genome.header.lifecycle.gestation_ticks;
            let body = &mut self.bodies[ii];
            body.energy -= investment;
            body.is_gestating = true;
            body.gestation = Some(Gestation {
                partner_genome,
                remaining: gestation_ticks,
                investment,
            });
        }
    }

    fn eligible_to_mate(&self, i: usize) -> bool {
        let b = &self.bodies[i];
        b.alive
            && !b.is_gestating
            && b.age_turns >= u64::from(b.genome.header.lifecycle.age_of_maturity)
            && b.energy >= b.derived.investment_energy
    }

    fn mate_partner_ok(&self, i: usize) -> bool {
        let b = &self.bodies[i];
        b.alive && b.age_turns >= u64::from(b.genome.header.lifecycle.age_of_maturity)
    }

    fn birth_phase<E: Environment>(&mut self, env: &mut E) {
        let turn = self.turn;
        let mut ready: Vec<usize> = Vec::new();
        for i in 0..self.bodies.len() {
            let body = &mut self.bodies[i];
            if !body.alive {
                continue;
            }
            let Some(g) = body.gestation.as_mut() else {
                continue;
            };
            if g.remaining > 0 {
                g.remaining -= 1;
                continue;
            }
            ready.push(i);
        }
        ready.sort_by_key(|&i| self.bodies[i].id);
        for i in ready {
            let (parent_genome, partner_genome, generation, investment) = {
                let body = &self.bodies[i];
                let g = body.gestation.as_ref().unwrap();
                (
                    body.genome.clone(),
                    g.partner_genome.clone(),
                    body.generation + 1,
                    g.investment,
                )
            };
            let birth_salt = mix_u64(TAG_BIRTH ^ turn.wrapping_mul(RNG_TURN_MIX))
                ^ mix_u64(self.bodies[i].id.wrapping_mul(RNG_ORGANISM_MIX))
                ^ self.births_this_turn;
            self.births_this_turn += 1;
            let mut rng = stream(self.seed, birth_salt);

            let site = {
                let mut place_rng = stream(self.seed, mix_u64(TAG_PLACE ^ birth_salt));
                let handle = BodyHandle(i as u32);
                let view = self.bodies[handle.0 as usize].view(handle);
                env.place_birth(&view, &mut place_rng)
            };

            // Clear gestation regardless (blocked births are lost, mirroring the
            // current engine's BlockedBirth outcome).
            {
                let body = &mut self.bodies[i];
                body.is_gestating = false;
                body.gestation = None;
            }

            let Some(site) = site else {
                continue;
            };

            let child = reproduce(&parent_genome, &partner_genome, &self.mutate_ctx(), &mut rng);
            let handle = BodyHandle(self.bodies.len() as u32);
            let body = self.make_body(env, child, generation, investment);
            self.bodies.push(body);
            env.attach(&self.bodies[handle.0 as usize].view(handle), site);
        }
    }
}

fn softmax_prob(logits: &[f32], temperature: f32, idx: usize) -> f32 {
    let t = temperature.max(crate::brain::MIN_ACTION_TEMPERATURE);
    let idle_bias = crate::brain::EXPLICIT_IDLE_LOGIT_BIAS;
    let max_logit = logits.iter().copied().fold(idle_bias, f32::max);
    let mut sum = ((idle_bias - max_logit) / t).exp();
    for &l in logits {
        sum += ((l - max_logit) / t).exp();
    }
    if idx >= logits.len() {
        return 0.0;
    }
    ((logits[idx] - max_logit) / t).exp() / sum
}
