use super::*;

#[cfg(feature = "instrumentation")]
const INTER_EMA_ALPHA: f32 = 0.05;
#[cfg(feature = "instrumentation")]
const UTILIZATION_THRESHOLD: f32 = 0.03;

#[derive(Clone, Copy)]
struct IntentBuildContext<'a> {
    world_width: i32,
    occupancy: &'a [Option<Occupant>],
    action_temperature: f32,
    runtime_plasticity_enabled: bool,
    leaky_neurons_enabled: bool,
    predation_enabled: bool,
    force_random_actions: bool,
    sim_seed: u64,
    tick: u64,
}

#[derive(Clone, Copy)]
struct SelectedActionState {
    selected_action: ActionType,
    selected_symbol: types::Symbol,
    selected_action_mask: u8,
    forward_logit: f32,
    synapse_ops: u64,
}

impl Simulation {
    pub(super) fn build_intents(&mut self, world_width: i32) -> Vec<OrganismIntent> {
        let context = IntentBuildContext {
            world_width,
            occupancy: &self.occupancy,
            action_temperature: self.config.action_temperature,
            runtime_plasticity_enabled: self.config.runtime_plasticity_enabled,
            leaky_neurons_enabled: self.config.leaky_neurons_enabled,
            predation_enabled: self.config.predation_enabled,
            force_random_actions: self.config.force_random_actions,
            sim_seed: self.seed,
            tick: self.turn,
        };
        let thread_pool = self.parallel_pool();
        let mut intents = std::mem::take(&mut self.turn_scratch.intents);
        #[cfg(feature = "profiling")]
        let brain_eval_started = Instant::now();

        #[cfg(feature = "instrumentation")]
        {
            let mut action_records = std::mem::take(&mut self.action_records);
            thread_pool.install(|| {
                self.organisms
                    .par_iter_mut()
                    .with_min_len(INTENT_PARALLEL_MIN_LEN)
                    .enumerate()
                    .map_init(BrainScratch::new, |scratch, (idx, organism)| {
                        let built = build_intent_for_organism(idx, organism, context, scratch);
                        (built.intent, built.action_record)
                    })
                    .unzip_into_vecs(&mut intents, &mut action_records)
            });
            self.action_records = action_records;
        }

        #[cfg(not(feature = "instrumentation"))]
        thread_pool.install(|| {
            self.organisms
                .par_iter_mut()
                .with_min_len(INTENT_PARALLEL_MIN_LEN)
                .enumerate()
                .map_init(BrainScratch::new, |scratch, (idx, organism)| {
                    build_intent_for_organism(idx, organism, context, scratch).intent
                })
                .collect_into_vec(&mut intents)
        });

        #[cfg(feature = "profiling")]
        profiling::record_brain_eval_total(brain_eval_started.elapsed());
        intents
    }
}

fn build_intent_for_organism(
    idx: usize,
    organism: &mut OrganismState,
    context: IntentBuildContext<'_>,
    scratch: &mut BrainScratch,
) -> BuiltIntent {
    let selected = select_action_for_organism(organism, context, scratch);
    organism.last_action_taken = selected.selected_action;
    organism.last_action_symbol = selected.selected_symbol;
    organism.last_action_mask = selected.selected_action_mask;
    let intent = intent_from_selected_action(idx, organism, selected, context);
    BuiltIntent {
        intent,
        #[cfg(feature = "instrumentation")]
        action_record: Some(instrument_action_record(organism, selected)),
    }
}

fn select_action_for_organism(
    organism: &mut OrganismState,
    context: IntentBuildContext<'_>,
    scratch: &mut BrainScratch,
) -> SelectedActionState {
    crate::sensing::begin_sensing_tick(organism);
    let action_sample = deterministic_action_sample(context.sim_seed, context.tick, organism.id);
    crate::sensing::encode_sensory_symbol(organism, context.world_width, context.occupancy);
    let evaluation = evaluate_brain(
        organism,
        BrainEvalContext {
            leaky_neurons_enabled: context.leaky_neurons_enabled,
            action_temperature: context.action_temperature,
            action_sample: (!context.force_random_actions).then_some(action_sample),
        },
        scratch,
    );
    if context.runtime_plasticity_enabled {
        compute_pending_coactivations(organism, scratch);
    }

    let selected_symbol = if context.force_random_actions {
        uniform_random_action_symbol(action_sample, context.predation_enabled)
    } else {
        evaluation.selected_symbol
    };
    let selected_action = if selected_symbol.is_action_enabled(context.predation_enabled) {
        selected_symbol.action_type()
    } else {
        ActionType::Idle
    };
    SelectedActionState {
        selected_action,
        selected_symbol,
        selected_action_mask: selected_action.command_bit(),
        forward_logit: evaluation.action_logits[types::Symbol::D.index()],
        synapse_ops: evaluation.synapse_ops,
    }
}

fn intent_from_selected_action(
    idx: usize,
    organism: &OrganismState,
    selected: SelectedActionState,
    context: IntentBuildContext<'_>,
) -> OrganismIntent {
    let from = (organism.q, organism.r);
    let forward_cell = || hex_neighbor(from, organism.facing, context.world_width);
    let mut intent = OrganismIntent {
        idx,
        id: organism.id,
        from,
        facing_after_actions: organism.facing,
        wants_move: false,
        wants_attack: false,
        move_target: None,
        interaction_target: None,
        snapshot_attack_target: None,
        move_confidence: 0.0,
        command_count: u8::from(selected.selected_action != ActionType::Idle),
        synapse_ops: selected.synapse_ops,
    };

    match selected.selected_action {
        ActionType::Idle => {}
        ActionType::TurnLeft => intent.facing_after_actions = rotate_left(organism.facing),
        ActionType::TurnRight => intent.facing_after_actions = rotate_right(organism.facing),
        ActionType::Forward => {
            intent.wants_move = true;
            intent.move_target = Some(forward_cell());
            intent.move_confidence = selected.forward_logit;
        }
        ActionType::Attack => {
            intent.wants_attack = true;
            let target = forward_cell();
            intent.interaction_target = Some(target);
            let target_idx = target.1 as usize * context.world_width as usize + target.0 as usize;
            intent.snapshot_attack_target = match context.occupancy[target_idx] {
                Some(Occupant::Organism(id)) => Some(id),
                _ => None,
            };
        }
    }
    intent
}

#[cfg(feature = "instrumentation")]
fn instrument_action_record(
    organism: &mut OrganismState,
    selected: SelectedActionState,
) -> ActionRecord {
    ActionRecord {
        organism_id: organism.id,
        selected_action: selected.selected_action,
        selected_action_mask: selected.selected_action_mask,
        failed_action_mask: selected.selected_action_mask
            & (ActionType::Forward.command_bit() | ActionType::Attack.command_bit()),
        action_failed: selected.selected_action.can_fail(),
        age_turns: organism.age_turns,
        utilization: update_instrumentation_utilization(organism),
        successful_attacks_count: organism.successful_attacks_count,
    }
}

fn uniform_random_action_symbol(sample: f32, predation_enabled: bool) -> types::Symbol {
    let count = types::Symbol::ALL
        .into_iter()
        .filter(|symbol| symbol.is_action_enabled(predation_enabled))
        .count();
    let bucket = (sample.clamp(0.0, 1.0 - f32::EPSILON) * count as f32) as usize;
    types::Symbol::ALL
        .into_iter()
        .filter(|symbol| symbol.is_action_enabled(predation_enabled))
        .nth(bucket)
        .unwrap_or(types::Symbol::End)
}

#[cfg(feature = "instrumentation")]
fn update_instrumentation_utilization(organism: &mut OrganismState) -> f32 {
    let activations = organism
        .brain
        .inter
        .iter()
        .map(|inter| inter.neuron.activation.abs());
    let ema = &mut organism.instrumentation.inter_ema;
    let mut inter_count = 0_usize;
    if ema.len() != organism.brain.inter.len() {
        ema.clear();
        ema.extend(activations);
        inter_count = ema.len();
    } else {
        for (ema_value, activation) in ema.iter_mut().zip(activations) {
            *ema_value = (1.0 - INTER_EMA_ALPHA) * *ema_value + INTER_EMA_ALPHA * activation;
            inter_count += 1;
        }
    }
    if inter_count == 0 {
        return 0.0;
    }
    ema.iter()
        .filter(|value| **value > UTILIZATION_THRESHOLD)
        .count() as f32
        / inter_count as f32
}
