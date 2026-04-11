use super::*;

#[cfg(feature = "instrumentation")]
const INTER_EMA_ALPHA: f32 = 0.05;
#[cfg(feature = "instrumentation")]
const UTILIZATION_THRESHOLD: f32 = 0.03;

#[derive(Clone, Copy)]
struct IntentBuildContext<'a> {
    world_width: i32,
    occupancy: &'a [Option<Occupant>],
    spike_map: &'a [bool],
    pending_actions: &'a [PendingActionState],
    action_temperature: f32,
    runtime_plasticity_enabled: bool,
    force_random_actions: bool,
    sim_seed: u64,
    tick: u64,
}

#[derive(Clone, Copy)]
struct SelectedActionState {
    selected_action: ActionType,
    selected_action_logit: f32,
    synapse_ops: u64,
    #[cfg(feature = "instrumentation")]
    food_flags: (bool, bool, bool, bool),
}

#[derive(Clone, Copy)]
struct ActionIntentOutcome {
    facing_after_actions: FacingDirection,
    wants_move: bool,
    wants_eat: bool,
    wants_attack: bool,
    wants_reproduce: bool,
    move_target: Option<(i32, i32)>,
    interaction_target: Option<(i32, i32)>,
}

impl Simulation {
    pub(super) fn build_intents(&mut self, snapshot: &TurnSnapshot) -> Vec<OrganismIntent> {
        let context = IntentBuildContext {
            world_width: self.config.world_width as i32,
            occupancy: &self.occupancy,
            spike_map: &self.spike_map,
            pending_actions: &self.pending_actions,
            action_temperature: self.config.action_temperature,
            runtime_plasticity_enabled: self.config.runtime_plasticity_enabled,
            force_random_actions: self.config.force_random_actions,
            sim_seed: self.seed,
            tick: self.turn,
        };
        let thread_pool = sim_parallel_pool(self.config.intent_parallel_threads);
        #[cfg(feature = "profiling")]
        let brain_eval_started = Instant::now();
        let built_intents: Vec<BuiltIntent> = thread_pool.install(|| {
            self.organisms
                .par_iter_mut()
                .with_min_len(INTENT_PARALLEL_MIN_LEN)
                .enumerate()
                .map_init(BrainScratch::new, |scratch, (idx, organism)| {
                    build_intent_for_organism(
                        idx,
                        organism,
                        snapshot.organism_states[idx],
                        snapshot.organism_ids[idx],
                        context,
                        scratch,
                    )
                })
                .collect()
        });
        #[cfg(feature = "profiling")]
        profiling::record_brain_eval_total(brain_eval_started.elapsed());

        #[cfg(feature = "instrumentation")]
        {
            self.action_records.clear();
            self.action_record_indices.clear();
            self.action_records.reserve(
                built_intents
                    .len()
                    .saturating_sub(self.action_records.capacity()),
            );
        }

        let intents = built_intents
            .into_iter()
            .map(|built| {
                #[cfg(feature = "instrumentation")]
                if let Some(action_record) = built.action_record {
                    self.record_action(action_record);
                }
                built.intent
            })
            .collect();

        intents
    }
}

fn build_intent_for_organism(
    idx: usize,
    organism: &mut OrganismState,
    snapshot_state: SnapshotOrganismState,
    organism_id: OrganismId,
    context: IntentBuildContext<'_>,
    scratch: &mut BrainScratch,
) -> BuiltIntent {
    let pending_action = context.pending_actions[idx];
    if pending_action.turns_remaining > 0 {
        return locked_intent(idx, organism_id, snapshot_state);
    }

    let selected_action_state = select_action_for_organism(organism, organism_id, context, scratch);

    organism.last_action_taken = selected_action_state.selected_action;
    let intent = translate_action_to_intent(
        idx,
        organism_id,
        snapshot_state,
        selected_action_state.selected_action,
        selected_action_state.selected_action_logit,
        selected_action_state.synapse_ops,
        context.world_width,
    );

    BuiltIntent {
        intent,
        #[cfg(feature = "instrumentation")]
        action_record: Some(instrument_action_record(
            organism,
            organism_id,
            selected_action_state.selected_action,
            selected_action_state.food_flags,
        )),
    }
}

fn locked_intent(
    idx: usize,
    organism_id: OrganismId,
    snapshot_state: SnapshotOrganismState,
) -> BuiltIntent {
    BuiltIntent {
        intent: OrganismIntent {
            idx,
            id: organism_id,
            from: (snapshot_state.q, snapshot_state.r),
            facing_after_actions: snapshot_state.facing,
            wants_move: false,
            wants_eat: false,
            wants_attack: false,
            wants_reproduce: false,
            move_target: None,
            interaction_target: None,
            move_confidence: 0.0,
            action_cost_count: 0,
            synapse_ops: 0,
        },
        #[cfg(feature = "instrumentation")]
        action_record: None,
    }
}

fn select_action_for_organism(
    organism: &mut OrganismState,
    organism_id: OrganismId,
    context: IntentBuildContext<'_>,
    scratch: &mut BrainScratch,
) -> SelectedActionState {
    let vision_distance = organism.genome.vision_distance;
    let action_sample = deterministic_action_sample(context.sim_seed, context.tick, organism_id);

    if context.force_random_actions {
        #[cfg(feature = "instrumentation")]
        let food_flags = {
            let ray_scans = scan_rays(
                (organism.q, organism.r),
                organism.facing,
                organism.id,
                context.world_width,
                context.occupancy,
                context.spike_map,
                vision_distance,
            );
            let food_at_offset = |offset: i8| -> bool {
                let Some(ray_idx) = SensoryReceptor::LOOK_RAY_OFFSETS
                    .iter()
                    .position(|candidate| *candidate == offset)
                else {
                    return false;
                };
                ray_scans[ray_idx].food_signal > 0.0
            };
            (
                food_at_offset(0),
                food_at_offset(-1),
                food_at_offset(1),
                food_at_offset(3),
            )
        };

        return SelectedActionState {
            selected_action: uniform_random_action(action_sample),
            selected_action_logit: 0.0,
            synapse_ops: 0,
            #[cfg(feature = "instrumentation")]
            food_flags,
        };
    }

    let evaluation = evaluate_brain(
        organism,
        BrainEvalContext {
            world_width: context.world_width,
            occupancy: context.occupancy,
            spike_map: context.spike_map,
            vision_distance,
            action_temperature: context.action_temperature,
            action_sample,
        },
        scratch,
    );
    if context.runtime_plasticity_enabled {
        compute_pending_coactivations(organism, scratch);
    }

    let selected_action = evaluation.selected_action;
    let selected_action_logit = if selected_action == ActionType::Idle {
        0.0
    } else {
        evaluation.action_logits[action_index(selected_action)]
    };

    SelectedActionState {
        selected_action,
        selected_action_logit,
        synapse_ops: evaluation.synapse_ops,
        #[cfg(feature = "instrumentation")]
        food_flags: (
            evaluation.food_ahead,
            evaluation.food_left,
            evaluation.food_right,
            evaluation.food_behind,
        ),
    }
}

fn translate_action_to_intent(
    idx: usize,
    organism_id: OrganismId,
    snapshot_state: SnapshotOrganismState,
    selected_action: ActionType,
    selected_action_logit: f32,
    synapse_ops: u64,
    world_width: i32,
) -> OrganismIntent {
    let outcome = intent_from_selected_action(selected_action, snapshot_state, world_width);
    let move_confidence = if outcome.wants_move {
        selected_action_logit
    } else {
        0.0
    };

    OrganismIntent {
        idx,
        id: organism_id,
        from: (snapshot_state.q, snapshot_state.r),
        facing_after_actions: outcome.facing_after_actions,
        wants_move: outcome.wants_move,
        wants_eat: outcome.wants_eat,
        wants_attack: outcome.wants_attack,
        wants_reproduce: outcome.wants_reproduce,
        move_target: outcome.move_target,
        interaction_target: outcome.interaction_target,
        move_confidence,
        action_cost_count: u8::from(selected_action != ActionType::Idle),
        synapse_ops,
    }
}

#[cfg(feature = "instrumentation")]
fn instrument_action_record(
    organism: &mut OrganismState,
    organism_id: OrganismId,
    selected_action: ActionType,
    food_flags: (bool, bool, bool, bool),
) -> ActionRecord {
    let (food_ahead, food_left, food_right, food_behind) = food_flags;
    ActionRecord {
        organism_id,
        selected_action,
        action_failed: contingent_action_can_fail(selected_action),
        food_ahead,
        food_left,
        food_right,
        food_behind,
        damage_taken_last_turn: organism.damage_taken_last_turn,
        age_turns: organism.age_turns,
        utilization: update_instrumentation_utilization(organism),
        consumptions_count: organism.consumptions_count,
    }
}

#[cfg(feature = "instrumentation")]
fn contingent_action_can_fail(action: ActionType) -> bool {
    matches!(
        action,
        ActionType::Forward | ActionType::Eat | ActionType::Attack | ActionType::Reproduce
    )
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
    let utilized = ema
        .iter()
        .filter(|value| **value > UTILIZATION_THRESHOLD)
        .count();
    utilized as f32 / inter_count as f32
}

fn intent_from_selected_action(
    selected_action: ActionType,
    snapshot_state: SnapshotOrganismState,
    world_width: i32,
) -> ActionIntentOutcome {
    let from = (snapshot_state.q, snapshot_state.r);
    let current_facing = snapshot_state.facing;

    match selected_action {
        ActionType::Idle => ActionIntentOutcome {
            facing_after_actions: current_facing,
            wants_move: false,
            wants_eat: false,
            wants_attack: false,
            wants_reproduce: false,
            move_target: None,
            interaction_target: None,
        },
        ActionType::TurnLeft => ActionIntentOutcome {
            facing_after_actions: rotate_left(current_facing),
            wants_move: false,
            wants_eat: false,
            wants_attack: false,
            wants_reproduce: false,
            move_target: None,
            interaction_target: None,
        },
        ActionType::TurnRight => ActionIntentOutcome {
            facing_after_actions: rotate_right(current_facing),
            wants_move: false,
            wants_eat: false,
            wants_attack: false,
            wants_reproduce: false,
            move_target: None,
            interaction_target: None,
        },
        ActionType::Forward => ActionIntentOutcome {
            facing_after_actions: current_facing,
            wants_move: true,
            wants_eat: false,
            wants_attack: false,
            wants_reproduce: false,
            move_target: Some(hex_neighbor(from, current_facing, world_width)),
            interaction_target: None,
        },
        ActionType::Eat => ActionIntentOutcome {
            facing_after_actions: current_facing,
            wants_move: false,
            wants_eat: true,
            wants_attack: false,
            wants_reproduce: false,
            move_target: None,
            interaction_target: Some(hex_neighbor(from, current_facing, world_width)),
        },
        ActionType::Attack => ActionIntentOutcome {
            facing_after_actions: current_facing,
            wants_move: false,
            wants_eat: false,
            wants_attack: true,
            wants_reproduce: false,
            move_target: None,
            interaction_target: Some(hex_neighbor(from, current_facing, world_width)),
        },
        ActionType::Reproduce => ActionIntentOutcome {
            facing_after_actions: current_facing,
            wants_move: false,
            wants_eat: false,
            wants_attack: false,
            wants_reproduce: true,
            move_target: None,
            interaction_target: None,
        },
    }
}
