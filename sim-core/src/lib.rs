use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use sim_protocol::{
    ActionNeuronState, ActionType, BrainState, InterNeuronState, MetricsSnapshot, NeuronId,
    NeuronState, NeuronType, OccupancyCell, OrganismId, OrganismMove, OrganismState,
    SensoryNeuronState, SensoryReceptorType, SurvivalRule, SynapseEdge, TickDelta, WorldConfig,
    WorldSnapshot,
};
use std::cmp::Ordering;
use std::collections::HashMap;
use thiserror::Error;

const SYNAPSE_STRENGTH_MAX: f32 = 8.0;
const DEFAULT_RESTING: f32 = -70.0;
const DEFAULT_THRESHOLD: f32 = -55.0;
const DEFAULT_DECAY: f32 = 0.7;
const INVERTED_NEURON_RATE: f32 = 0.4;

#[derive(Debug, Error)]
pub enum SimError {
    #[error("invalid world config: {0}")]
    InvalidConfig(String),
}

#[derive(Debug, Clone)]
pub struct Simulation {
    config: WorldConfig,
    epoch: u64,
    tick_in_epoch: u32,
    seed: u64,
    rng: ChaCha8Rng,
    next_organism_id: u64,
    organisms: Vec<OrganismState>,
    occupancy: HashMap<(i32, i32), Vec<OrganismId>>,
    metrics: MetricsSnapshot,
}

impl Simulation {
    pub fn new(config: WorldConfig, seed: u64) -> Result<Self, SimError> {
        validate_config(&config)?;
        let mut sim = Self {
            config,
            epoch: 0,
            tick_in_epoch: 0,
            seed,
            rng: ChaCha8Rng::seed_from_u64(seed),
            next_organism_id: 0,
            organisms: Vec::new(),
            occupancy: HashMap::new(),
            metrics: MetricsSnapshot::default(),
        };
        sim.spawn_initial_population();
        sim.metrics.organisms = sim.organisms.len() as u32;
        Ok(sim)
    }

    pub fn config(&self) -> &WorldConfig {
        &self.config
    }

    pub fn snapshot(&self) -> WorldSnapshot {
        let mut organisms = self.organisms.clone();
        organisms.sort_by_key(|o| o.id);

        let mut occupancy: Vec<OccupancyCell> = self
            .occupancy
            .iter()
            .map(|((x, y), ids)| {
                let mut organism_ids = ids.clone();
                organism_ids.sort();
                OccupancyCell {
                    x: *x,
                    y: *y,
                    organism_ids,
                }
            })
            .collect();
        occupancy.sort_by_key(|c| (c.x, c.y));

        WorldSnapshot {
            epoch: self.epoch,
            tick_in_epoch: self.tick_in_epoch,
            rng_seed: self.seed,
            config: self.config.clone(),
            organisms,
            occupancy,
            metrics: self.metrics.clone(),
        }
    }

    pub fn reset(&mut self, seed: Option<u64>) {
        self.seed = seed.unwrap_or(self.seed);
        self.rng = ChaCha8Rng::seed_from_u64(self.seed);
        self.epoch = 0;
        self.tick_in_epoch = 0;
        self.next_organism_id = 0;
        self.organisms.clear();
        self.occupancy.clear();
        self.metrics = MetricsSnapshot::default();
        self.spawn_initial_population();
        self.metrics.organisms = self.organisms.len() as u32;
    }

    pub fn step_n(&mut self, count: u32) -> Vec<TickDelta> {
        let mut deltas = Vec::with_capacity(count as usize);
        for _ in 0..count {
            deltas.push(self.tick());
        }
        deltas
    }

    pub fn epoch_n(&mut self, count: u32) -> Vec<MetricsSnapshot> {
        let mut out = Vec::with_capacity(count as usize);
        for _ in 0..count {
            self.run_epoch();
            out.push(self.metrics.clone());
        }
        out
    }

    pub fn scatter(&mut self) {
        for idx in 0..self.organisms.len() {
            let id = self.organisms[idx].id;
            loop {
                let x = self.rng.random_range(0..self.config.columns as i32);
                let y = self.rng.random_range(0..self.config.rows as i32);
                if !self.survival_check(x, y) {
                    self.move_organism_to(id, x, y);
                    reset_brain_state(&mut self.organisms[idx].brain);
                    break;
                }
            }
        }
    }

    pub fn process_survivors(&mut self) -> u32 {
        let rule = self.config.survival_rule.clone();
        let columns = self.config.columns;
        let mut survivors = Vec::with_capacity(self.organisms.len());
        for organism in self.organisms.drain(..) {
            let fit = survival_check_for(&rule, columns, organism.x, organism.y);
            if fit || self.rng.random::<f32>() >= self.config.unfit_kill_probability {
                survivors.push(organism);
            }
        }

        let surviving_count = survivors.len() as u32;
        self.organisms = survivors;
        self.rebuild_occupancy();

        let target = self.config.num_organisms as usize;
        let deficit = target.saturating_sub(self.organisms.len());
        if deficit > 0 {
            if !self.organisms.is_empty() {
                let clone_count = ((self.config.offspring_fill_ratio.clamp(0.0, 1.0)
                    * deficit as f32)
                    .floor() as usize)
                    .min(deficit);
                for _ in 0..clone_count {
                    let parent_idx = self.rng.random_range(0..self.organisms.len());
                    let parent = self.organisms[parent_idx].clone();
                    let child = self.spawn_offspring(&parent);
                    self.add_organism(child);
                }
            }

            while self.organisms.len() < target {
                let x = self.rng.random_range(0..self.config.columns as i32);
                let y = self.rng.random_range(0..self.config.rows as i32);
                let id = self.alloc_organism_id();
                let brain = self.generate_brain();
                self.add_organism(OrganismState { id, x, y, brain });
            }
        }

        self.organisms.sort_by_key(|o| o.id);
        self.metrics.organisms = self.organisms.len() as u32;
        self.metrics.survivors_last_epoch = surviving_count;
        surviving_count
    }

    pub fn run_epoch(&mut self) {
        self.scatter();
        for _ in 0..self.config.steps_per_epoch {
            self.tick();
        }
        self.process_survivors();
        self.epoch += 1;
        self.tick_in_epoch = 0;
        self.metrics.epochs = self.epoch;
    }

    pub fn focused_organism(&self, id: OrganismId) -> Option<OrganismState> {
        self.organisms.iter().find(|o| o.id == id).cloned()
    }

    pub fn export_trace_jsonl(&mut self, epochs: u32) -> Vec<String> {
        let mut lines = Vec::new();
        lines.push(
            serde_json::to_string(&self.snapshot())
                .expect("serialize initial snapshot for trace export"),
        );
        for _ in 0..epochs {
            self.run_epoch();
            lines.push(
                serde_json::to_string(&self.snapshot())
                    .expect("serialize epoch snapshot for trace export"),
            );
        }
        lines
    }

    fn tick(&mut self) -> TickDelta {
        let mut moves = Vec::new();
        let mut synapse_ops = 0_u64;
        let mut actions_applied = 0_u64;

        for idx in 0..self.organisms.len() {
            let organism_id = self.organisms[idx].id;
            let start_pos = (self.organisms[idx].x, self.organisms[idx].y);

            let action = {
                let brain = &mut self.organisms[idx].brain;
                reset_action_activity(brain);
                decay_all(brain);
                sum_all(
                    brain,
                    start_pos,
                    self.config.columns as i32,
                    self.config.rows as i32,
                    self.config.vision_depth as i32,
                    &self.occupancy,
                );

                let fire_result = fire_all(brain);
                synapse_ops += fire_result.synapse_ops;
                fire_result.actions
            };

            if let Some((old_pos, new_pos)) = self.apply_actions(organism_id, action) {
                actions_applied += 1;
                moves.push(OrganismMove {
                    id: organism_id,
                    from: old_pos,
                    to: new_pos,
                });
            }
        }

        self.tick_in_epoch = self.tick_in_epoch.saturating_add(1);
        self.metrics.ticks = self.metrics.ticks.saturating_add(1);
        self.metrics.organisms = self.organisms.len() as u32;
        self.metrics.synapse_ops_last_tick = synapse_ops;
        self.metrics.actions_applied_last_tick = actions_applied;

        TickDelta {
            tick_in_epoch: self.tick_in_epoch,
            epoch: self.epoch,
            moves,
            metrics: self.metrics.clone(),
        }
    }

    fn apply_actions(
        &mut self,
        organism_id: OrganismId,
        action_outcomes: [bool; 4],
    ) -> Option<((i32, i32), (i32, i32))> {
        let idx = self.organisms.iter().position(|o| o.id == organism_id)?;
        let mut x = self.organisms[idx].x;
        let mut y = self.organisms[idx].y;
        let from = (x, y);

        for action in ActionType::ALL {
            if !action_outcomes[action_index(action)] {
                continue;
            }
            let (nx, ny) = match action {
                ActionType::MoveUp => (x, y + 1),
                ActionType::MoveDown => (x, y - 1),
                ActionType::MoveLeft => (x - 1, y),
                ActionType::MoveRight => (x + 1, y),
            };
            if self.in_bounds(nx, ny) {
                x = nx;
                y = ny;
            }
        }

        if from == (x, y) {
            return None;
        }

        self.move_organism_to(organism_id, x, y);
        Some((from, (x, y)))
    }

    fn spawn_initial_population(&mut self) {
        for _ in 0..self.config.num_organisms {
            let id = self.alloc_organism_id();
            let x = self.rng.random_range(0..self.config.columns as i32);
            let y = self.rng.random_range(0..self.config.rows as i32);
            let brain = self.generate_brain();
            self.add_organism(OrganismState { id, x, y, brain });
        }
        self.organisms.sort_by_key(|o| o.id);
    }

    fn spawn_offspring(&mut self, parent: &OrganismState) -> OrganismState {
        let id = self.alloc_organism_id();
        let x = parent.x;
        let y = parent.y;
        let mut brain = parent.brain.clone();
        self.mutate_brain(&mut brain);
        OrganismState { id, x, y, brain }
    }

    fn mutate_brain(&mut self, brain: &mut BrainState) {
        if self.rng.random::<f32>() > self.config.mutation_chance {
            return;
        }

        let operations = self.config.mutation_magnitude.max(1.0).round() as usize;
        for _ in 0..operations {
            if self.rng.random::<f32>() < 0.5 {
                self.apply_topology_mutation(brain);
            } else {
                self.apply_synapse_mutation(brain);
            }
        }

        brain.synapse_count = count_synapses(brain) as u32;
    }

    fn apply_topology_mutation(&mut self, brain: &mut BrainState) {
        match self.rng.random_range(0..3) {
            0 => {
                if brain.inter.len() as u32 >= self.config.max_num_neurons {
                    return;
                }
                let next_id = next_inter_neuron_id(brain);
                let mut neuron = make_neuron(
                    NeuronId(next_id),
                    NeuronType::Inter,
                    self.rng.random::<f32>() < INVERTED_NEURON_RATE,
                    self.config.action_potential_length,
                );
                neuron.action_potential_threshold += self.rng.random_range(-1.0..1.0);
                let inter = InterNeuronState {
                    neuron,
                    synapses: Vec::new(),
                };
                brain.inter.push(inter);

                let pre_candidates = output_neuron_ids(brain);
                let post_candidates = post_neuron_ids(brain);
                if !pre_candidates.is_empty() && !post_candidates.is_empty() {
                    let pre = pre_candidates[self.rng.random_range(0..pre_candidates.len())];
                    let post = post_candidates[self.rng.random_range(0..post_candidates.len())];
                    let w = self
                        .rng
                        .random_range(-SYNAPSE_STRENGTH_MAX..SYNAPSE_STRENGTH_MAX);
                    let _ = create_synapse(brain, pre, post, w);
                }
            }
            1 => {
                if brain.inter.is_empty() {
                    return;
                }
                let idx = self.rng.random_range(0..brain.inter.len());
                let removed_id = brain.inter[idx].neuron.neuron_id;
                brain.inter.remove(idx);
                remove_neuron_references(brain, removed_id);
            }
            _ => {
                if brain.inter.is_empty() {
                    return;
                }
                let idx = self.rng.random_range(0..brain.inter.len());
                let n = &mut brain.inter[idx].neuron;
                n.is_inverted = !n.is_inverted;
                n.action_potential_threshold += self.rng.random_range(-2.0..2.0);
                n.potential_decay_rate =
                    (n.potential_decay_rate + self.rng.random_range(-0.1..0.1)).clamp(0.1, 0.99);
            }
        }
    }

    fn apply_synapse_mutation(&mut self, brain: &mut BrainState) {
        match self.rng.random_range(0..3) {
            0 => {
                let pre_candidates = output_neuron_ids(brain);
                let post_candidates = post_neuron_ids(brain);
                if pre_candidates.is_empty() || post_candidates.is_empty() {
                    return;
                }
                let pre = pre_candidates[self.rng.random_range(0..pre_candidates.len())];
                let post = post_candidates[self.rng.random_range(0..post_candidates.len())];
                let weight = self
                    .rng
                    .random_range(-SYNAPSE_STRENGTH_MAX..SYNAPSE_STRENGTH_MAX);
                let _ = create_synapse(brain, pre, post, weight);
            }
            1 => {
                let outputs = output_neuron_ids(brain);
                if outputs.is_empty() {
                    return;
                }
                let pre = outputs[self.rng.random_range(0..outputs.len())];
                remove_random_synapse(brain, pre, &mut self.rng);
            }
            _ => {
                let outputs = output_neuron_ids(brain);
                if outputs.is_empty() {
                    return;
                }
                let pre = outputs[self.rng.random_range(0..outputs.len())];
                perturb_random_synapse(brain, pre, self.config.mutation_magnitude, &mut self.rng);
            }
        }
    }

    fn add_organism(&mut self, organism: OrganismState) {
        self.occupancy
            .entry((organism.x, organism.y))
            .or_default()
            .push(organism.id);
        self.organisms.push(organism);
    }

    fn alloc_organism_id(&mut self) -> OrganismId {
        let id = OrganismId(self.next_organism_id);
        self.next_organism_id += 1;
        id
    }

    fn move_organism_to(&mut self, id: OrganismId, x: i32, y: i32) {
        if let Some(org) = self.organisms.iter_mut().find(|o| o.id == id) {
            let old_pos = (org.x, org.y);
            if let Some(ids) = self.occupancy.get_mut(&old_pos) {
                ids.retain(|oid| *oid != id);
            }
            org.x = x;
            org.y = y;
            self.occupancy.entry((x, y)).or_default().push(id);
        }
    }

    fn rebuild_occupancy(&mut self) {
        self.occupancy.clear();
        for org in &self.organisms {
            self.occupancy
                .entry((org.x, org.y))
                .or_default()
                .push(org.id);
        }
    }

    fn in_bounds(&self, x: i32, y: i32) -> bool {
        x >= 0 && y >= 0 && x < self.config.columns as i32 && y < self.config.rows as i32
    }

    fn survival_check(&self, x: i32, _y: i32) -> bool {
        survival_check_for(&self.config.survival_rule, self.config.columns, x, _y)
    }

    pub fn metrics(&self) -> &MetricsSnapshot {
        &self.metrics
    }
}

fn survival_check_for(rule: &SurvivalRule, columns: u32, x: i32, _y: i32) -> bool {
    match *rule {
        SurvivalRule::CenterBandX {
            min_fraction,
            max_fraction,
        } => {
            let min_x = (columns as f32 * min_fraction) as i32;
            let max_x = (columns as f32 * max_fraction) as i32;
            x > min_x && x < max_x
        }
    }
}

fn validate_config(config: &WorldConfig) -> Result<(), SimError> {
    if config.columns == 0 || config.rows == 0 {
        return Err(SimError::InvalidConfig(
            "columns and rows must be greater than zero".to_owned(),
        ));
    }
    if config.num_organisms == 0 {
        return Err(SimError::InvalidConfig(
            "num_organisms must be greater than zero".to_owned(),
        ));
    }
    if !(0.0..=1.0).contains(&config.mutation_chance) {
        return Err(SimError::InvalidConfig(
            "mutation_chance must be within [0, 1]".to_owned(),
        ));
    }
    if !(0.0..=1.0).contains(&config.unfit_kill_probability) {
        return Err(SimError::InvalidConfig(
            "unfit_kill_probability must be within [0, 1]".to_owned(),
        ));
    }
    if !(0.0..=1.0).contains(&config.offspring_fill_ratio) {
        return Err(SimError::InvalidConfig(
            "offspring_fill_ratio must be within [0, 1]".to_owned(),
        ));
    }
    if config.action_potential_length == 0 {
        return Err(SimError::InvalidConfig(
            "action_potential_length must be >= 1".to_owned(),
        ));
    }
    Ok(())
}

#[derive(Default)]
struct FireResult {
    actions: [bool; 4],
    synapse_ops: u64,
}

fn action_index(action: ActionType) -> usize {
    match action {
        ActionType::MoveUp => 0,
        ActionType::MoveDown => 1,
        ActionType::MoveLeft => 2,
        ActionType::MoveRight => 3,
    }
}

fn make_neuron(
    id: NeuronId,
    neuron_type: NeuronType,
    is_inverted: bool,
    action_potential_length: u32,
) -> NeuronState {
    NeuronState {
        neuron_id: id,
        neuron_type,
        is_inverted,
        action_potential_threshold: DEFAULT_THRESHOLD,
        resting_potential: DEFAULT_RESTING,
        potential: DEFAULT_RESTING,
        incoming_current: 0.0,
        potential_decay_rate: DEFAULT_DECAY,
        action_potential_length,
        action_potential_time: None,
        parent_ids: Vec::new(),
    }
}

fn make_sensory_neuron(
    id: u32,
    receptor_type: SensoryReceptorType,
    apl: u32,
) -> SensoryNeuronState {
    let mut n = make_neuron(NeuronId(id), NeuronType::Sensory, false, apl);
    n.action_potential_threshold = 0.0;
    n.resting_potential = 0.0;
    n.potential = 0.0;
    SensoryNeuronState {
        neuron: n,
        receptor_type,
        synapses: Vec::new(),
    }
}

fn make_action_neuron(id: u32, action_type: ActionType, apl: u32) -> ActionNeuronState {
    ActionNeuronState {
        neuron: make_neuron(NeuronId(id), NeuronType::Action, false, apl),
        action_type,
        is_active: false,
    }
}

fn next_inter_neuron_id(brain: &BrainState) -> u32 {
    brain
        .inter
        .iter()
        .map(|n| n.neuron.neuron_id.0)
        .max()
        .map_or(1000, |id| id + 1)
}

fn reset_action_activity(brain: &mut BrainState) {
    for neuron in &mut brain.action {
        neuron.is_active = false;
    }
}

fn reset_brain_state(brain: &mut BrainState) {
    for n in &mut brain.inter {
        n.neuron.potential = n.neuron.resting_potential;
        n.neuron.incoming_current = 0.0;
        n.neuron.action_potential_time = None;
    }
    for n in &mut brain.action {
        n.neuron.potential = n.neuron.resting_potential;
        n.neuron.incoming_current = 0.0;
        n.neuron.action_potential_time = None;
        n.is_active = false;
    }
    for n in &mut brain.sensory {
        n.neuron.potential = n.neuron.resting_potential;
        n.neuron.incoming_current = 0.0;
        n.neuron.action_potential_time = None;
    }
}

fn decay_all(brain: &mut BrainState) {
    for n in &mut brain.sensory {
        decay_neuron(&mut n.neuron);
    }
    for n in &mut brain.inter {
        decay_neuron(&mut n.neuron);
    }
    for n in &mut brain.action {
        decay_neuron(&mut n.neuron);
    }
}

fn decay_neuron(neuron: &mut NeuronState) {
    neuron.potential = neuron.resting_potential
        + (neuron.potential - neuron.resting_potential) * neuron.potential_decay_rate;
}

fn sum_all(
    brain: &mut BrainState,
    position: (i32, i32),
    columns: i32,
    rows: i32,
    vision_depth: i32,
    occupancy: &HashMap<(i32, i32), Vec<OrganismId>>,
) {
    for sensory in &mut brain.sensory {
        sensory.neuron.potential = receptor_value(
            sensory.receptor_type,
            position,
            columns,
            rows,
            vision_depth,
            occupancy,
        );
    }
    for inter in &mut brain.inter {
        inter.neuron.potential += inter.neuron.incoming_current;
        inter.neuron.incoming_current = 0.0;
    }
    for action in &mut brain.action {
        action.neuron.potential += action.neuron.incoming_current;
        action.neuron.incoming_current = 0.0;
    }
}

fn receptor_value(
    receptor: SensoryReceptorType,
    position: (i32, i32),
    columns: i32,
    rows: i32,
    vision_depth: i32,
    occupancy: &HashMap<(i32, i32), Vec<OrganismId>>,
) -> f32 {
    let (x, y) = position;
    let bound = 8.0_f32;
    match receptor {
        SensoryReceptorType::X => x as f32 / columns.max(1) as f32 * bound,
        SensoryReceptorType::Y => y as f32 / rows.max(1) as f32 * bound,
        SensoryReceptorType::LookLeft => {
            look_density(
                (x - vision_depth, x),
                (y - vision_depth, y + vision_depth + 1),
                occupancy,
                columns,
                rows,
            ) * bound
        }
        SensoryReceptorType::LookRight => {
            look_density(
                (x + 1, x + vision_depth + 1),
                (y - vision_depth, y + vision_depth + 1),
                occupancy,
                columns,
                rows,
            ) * bound
        }
        SensoryReceptorType::LookUp => {
            look_density(
                (x - vision_depth, x + vision_depth + 1),
                (y + 1, y + vision_depth + 1),
                occupancy,
                columns,
                rows,
            ) * bound
        }
        SensoryReceptorType::LookDown => {
            look_density(
                (x - vision_depth, x + vision_depth + 1),
                (y - vision_depth, y),
                occupancy,
                columns,
                rows,
            ) * bound
        }
    }
}

fn look_density(
    x_bounds: (i32, i32),
    y_bounds: (i32, i32),
    occupancy: &HashMap<(i32, i32), Vec<OrganismId>>,
    columns: i32,
    rows: i32,
) -> f32 {
    let mut total = 0_u32;
    let mut hit = 0_u32;

    for y in y_bounds.0..y_bounds.1 {
        for x in x_bounds.0..x_bounds.1 {
            total += 1;
            if x < 0
                || y < 0
                || x >= columns
                || y >= rows
                || occupancy.get(&(x, y)).is_some_and(|ids| !ids.is_empty())
            {
                hit += 1;
            }
        }
    }

    if total == 0 {
        return 0.0;
    }

    hit as f32 / total as f32
}

fn fire_all(brain: &mut BrainState) -> FireResult {
    let mut current_additions: Vec<(NeuronId, f32)> = Vec::new();
    let mut result = FireResult::default();

    for sensory in &mut brain.sensory {
        if progress_action_potential(&mut sensory.neuron) {
            for edge in &sensory.synapses {
                current_additions
                    .push((edge.post_neuron_id, sensory.neuron.potential * edge.weight));
                result.synapse_ops += 1;
            }
            sensory.neuron.potential = sensory.neuron.resting_potential;
        }
    }

    for inter in &mut brain.inter {
        if progress_action_potential(&mut inter.neuron) {
            for edge in &inter.synapses {
                current_additions.push((edge.post_neuron_id, edge.weight));
                result.synapse_ops += 1;
            }
            inter.neuron.potential = inter.neuron.resting_potential;
        }
    }

    for action in &mut brain.action {
        if progress_action_potential(&mut action.neuron) {
            action.is_active = true;
            result.actions[action_index(action.action_type)] = true;
            action.neuron.potential = action.neuron.resting_potential;
        }
    }

    for (id, delta) in current_additions {
        if let Some(neuron) = get_neuron_mut(brain, id) {
            neuron.incoming_current += delta;
        }
    }

    result
}

fn progress_action_potential(neuron: &mut NeuronState) -> bool {
    let threshold_reached =
        neuron.is_inverted ^ (neuron.potential > neuron.action_potential_threshold);
    if threshold_reached && neuron.action_potential_time.is_none() {
        neuron.action_potential_time = Some(0);
    }

    match neuron.action_potential_time {
        Some(t) if t >= neuron.action_potential_length => {
            neuron.action_potential_time = None;
            true
        }
        Some(t) => {
            neuron.action_potential_time = Some(t + 1);
            false
        }
        None => false,
    }
}

fn output_neuron_ids(brain: &BrainState) -> Vec<NeuronId> {
    brain
        .sensory
        .iter()
        .map(|n| n.neuron.neuron_id)
        .chain(brain.inter.iter().map(|n| n.neuron.neuron_id))
        .collect()
}

fn post_neuron_ids(brain: &BrainState) -> Vec<NeuronId> {
    brain
        .inter
        .iter()
        .map(|n| n.neuron.neuron_id)
        .chain(brain.action.iter().map(|n| n.neuron.neuron_id))
        .collect()
}

fn count_synapses(brain: &BrainState) -> usize {
    brain
        .sensory
        .iter()
        .map(|s| s.synapses.len())
        .sum::<usize>()
        + brain.inter.iter().map(|i| i.synapses.len()).sum::<usize>()
}

fn get_neuron_mut(brain: &mut BrainState, id: NeuronId) -> Option<&mut NeuronState> {
    if let Some(s) = brain.sensory.iter_mut().find(|n| n.neuron.neuron_id == id) {
        return Some(&mut s.neuron);
    }
    if let Some(i) = brain.inter.iter_mut().find(|n| n.neuron.neuron_id == id) {
        return Some(&mut i.neuron);
    }
    if let Some(a) = brain.action.iter_mut().find(|n| n.neuron.neuron_id == id) {
        return Some(&mut a.neuron);
    }
    None
}

fn create_synapse(brain: &mut BrainState, pre: NeuronId, post: NeuronId, weight: f32) -> bool {
    if pre == post {
        return false;
    }

    let duplicate = brain
        .sensory
        .iter()
        .find(|n| n.neuron.neuron_id == pre)
        .map(|n| n.synapses.iter().any(|e| e.post_neuron_id == post))
        .or_else(|| {
            brain
                .inter
                .iter()
                .find(|n| n.neuron.neuron_id == pre)
                .map(|n| n.synapses.iter().any(|e| e.post_neuron_id == post))
        })
        .unwrap_or(true);

    if duplicate {
        return false;
    }

    let clamped = weight.clamp(-SYNAPSE_STRENGTH_MAX, SYNAPSE_STRENGTH_MAX);

    if let Some(sensory) = brain.sensory.iter_mut().find(|n| n.neuron.neuron_id == pre) {
        sensory.synapses.push(SynapseEdge {
            post_neuron_id: post,
            weight: clamped,
        });
        sensory
            .synapses
            .sort_by(|a, b| a.post_neuron_id.cmp(&b.post_neuron_id));
    } else if let Some(inter) = brain.inter.iter_mut().find(|n| n.neuron.neuron_id == pre) {
        inter.synapses.push(SynapseEdge {
            post_neuron_id: post,
            weight: clamped,
        });
        inter
            .synapses
            .sort_by(|a, b| a.post_neuron_id.cmp(&b.post_neuron_id));
    } else {
        return false;
    }

    if let Some(post_neuron) = get_neuron_mut(brain, post) {
        if !post_neuron.parent_ids.contains(&pre) {
            post_neuron.parent_ids.push(pre);
            post_neuron.parent_ids.sort();
        }
    }

    true
}

fn remove_neuron_references(brain: &mut BrainState, target: NeuronId) {
    for sensory in &mut brain.sensory {
        sensory.synapses.retain(|e| e.post_neuron_id != target);
        sensory.neuron.parent_ids.retain(|p| *p != target);
    }
    for inter in &mut brain.inter {
        inter.synapses.retain(|e| e.post_neuron_id != target);
        inter.neuron.parent_ids.retain(|p| *p != target);
    }
    for action in &mut brain.action {
        action.neuron.parent_ids.retain(|p| *p != target);
    }
}

fn remove_random_synapse<R: Rng + ?Sized>(brain: &mut BrainState, pre: NeuronId, rng: &mut R) {
    if let Some(sensory) = brain.sensory.iter_mut().find(|n| n.neuron.neuron_id == pre) {
        if sensory.synapses.is_empty() {
            return;
        }
        let idx = rng.random_range(0..sensory.synapses.len());
        let post = sensory.synapses[idx].post_neuron_id;
        sensory.synapses.remove(idx);
        if let Some(post_neuron) = get_neuron_mut(brain, post) {
            post_neuron.parent_ids.retain(|id| *id != pre);
        }
        return;
    }

    if let Some(inter) = brain.inter.iter_mut().find(|n| n.neuron.neuron_id == pre) {
        if inter.synapses.is_empty() {
            return;
        }
        let idx = rng.random_range(0..inter.synapses.len());
        let post = inter.synapses[idx].post_neuron_id;
        inter.synapses.remove(idx);
        if let Some(post_neuron) = get_neuron_mut(brain, post) {
            post_neuron.parent_ids.retain(|id| *id != pre);
        }
    }
}

fn perturb_random_synapse<R: Rng + ?Sized>(
    brain: &mut BrainState,
    pre: NeuronId,
    mutation_magnitude: f32,
    rng: &mut R,
) {
    let magnitude = mutation_magnitude.clamp(0.1, 8.0);
    if let Some(sensory) = brain.sensory.iter_mut().find(|n| n.neuron.neuron_id == pre) {
        if sensory.synapses.is_empty() {
            return;
        }
        let idx = rng.random_range(0..sensory.synapses.len());
        let delta = rng.random_range(-magnitude..magnitude);
        sensory.synapses[idx].weight = (sensory.synapses[idx].weight + delta)
            .clamp(-SYNAPSE_STRENGTH_MAX, SYNAPSE_STRENGTH_MAX);
        return;
    }

    if let Some(inter) = brain.inter.iter_mut().find(|n| n.neuron.neuron_id == pre) {
        if inter.synapses.is_empty() {
            return;
        }
        let idx = rng.random_range(0..inter.synapses.len());
        let delta = rng.random_range(-magnitude..magnitude);
        inter.synapses[idx].weight =
            (inter.synapses[idx].weight + delta).clamp(-SYNAPSE_STRENGTH_MAX, SYNAPSE_STRENGTH_MAX);
    }
}

impl Simulation {
    fn generate_brain(&mut self) -> BrainState {
        let mut sensory = Vec::new();
        for (idx, receptor_type) in SensoryReceptorType::ALL.into_iter().enumerate() {
            sensory.push(make_sensory_neuron(
                idx as u32,
                receptor_type,
                self.config.action_potential_length,
            ));
        }

        let mut inter = Vec::new();
        for i in 0..self.config.num_neurons {
            inter.push(InterNeuronState {
                neuron: make_neuron(
                    NeuronId(1000 + i),
                    NeuronType::Inter,
                    self.rng.random::<f32>() < INVERTED_NEURON_RATE,
                    self.config.action_potential_length,
                ),
                synapses: Vec::new(),
            });
        }

        let mut action = Vec::new();
        for (idx, action_type) in ActionType::ALL.into_iter().enumerate() {
            action.push(make_action_neuron(
                2000 + idx as u32,
                action_type,
                self.config.action_potential_length,
            ));
        }

        let mut brain = BrainState {
            sensory,
            inter,
            action,
            synapse_count: 0,
        };

        for _ in 0..self.config.num_synapses {
            let pre_candidates = output_neuron_ids(&brain);
            let post_candidates = post_neuron_ids(&brain);
            if pre_candidates.is_empty() || post_candidates.is_empty() {
                break;
            }
            let pre = pre_candidates[self.rng.random_range(0..pre_candidates.len())];
            let post = post_candidates[self.rng.random_range(0..post_candidates.len())];
            let weight = self
                .rng
                .random_range(-SYNAPSE_STRENGTH_MAX..SYNAPSE_STRENGTH_MAX);
            let _ = create_synapse(&mut brain, pre, post, weight);
        }

        brain.synapse_count = count_synapses(&brain) as u32;
        brain
    }
}

pub fn compare_snapshots(a: &WorldSnapshot, b: &WorldSnapshot) -> Ordering {
    let sa = serde_json::to_string(a).expect("serialize snapshot A");
    let sb = serde_json::to_string(b).expect("serialize snapshot B");
    sa.cmp(&sb)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_seed() {
        let cfg = WorldConfig::default();
        let mut a = Simulation::new(cfg.clone(), 42).expect("simulation A should initialize");
        let mut b = Simulation::new(cfg, 42).expect("simulation B should initialize");
        a.epoch_n(3);
        b.epoch_n(3);
        assert_eq!(
            compare_snapshots(&a.snapshot(), &b.snapshot()),
            Ordering::Equal
        );
    }

    #[test]
    fn different_seed_changes_state() {
        let cfg = WorldConfig::default();
        let mut a = Simulation::new(cfg.clone(), 42).expect("simulation A should initialize");
        let mut b = Simulation::new(cfg, 43).expect("simulation B should initialize");
        a.step_n(5);
        b.step_n(5);
        assert_ne!(
            compare_snapshots(&a.snapshot(), &b.snapshot()),
            Ordering::Equal
        );
    }

    #[test]
    fn config_validation_rejects_zero_dimensions() {
        let cfg = WorldConfig {
            columns: 0,
            ..WorldConfig::default()
        };
        let err = Simulation::new(cfg, 1).expect_err("expected invalid config error");
        assert!(err.to_string().contains("columns and rows"));
    }

    #[test]
    fn population_count_restored_after_epoch() {
        let cfg = WorldConfig {
            num_organisms: 80,
            ..WorldConfig::default()
        };
        let mut sim = Simulation::new(cfg.clone(), 10).expect("simulation should initialize");
        sim.run_epoch();
        assert_eq!(sim.snapshot().organisms.len(), cfg.num_organisms as usize);
    }

    #[test]
    fn receptor_sampling_directional_non_zero() {
        let cfg = WorldConfig {
            columns: 5,
            rows: 5,
            num_organisms: 1,
            num_neurons: 0,
            num_synapses: 0,
            ..WorldConfig::default()
        };
        let mut sim = Simulation::new(cfg, 7).expect("simulation should initialize");

        // Force known occupancy: organism at (2,2) and synthetic occupancy on the right.
        sim.organisms[0].x = 2;
        sim.organisms[0].y = 2;
        sim.rebuild_occupancy();
        sim.occupancy
            .entry((3, 2))
            .or_default()
            .push(OrganismId(999));

        let v = receptor_value(
            SensoryReceptorType::LookRight,
            (2, 2),
            sim.config.columns as i32,
            sim.config.rows as i32,
            sim.config.vision_depth as i32,
            &sim.occupancy,
        );
        assert!(v > 0.0);
    }
}
