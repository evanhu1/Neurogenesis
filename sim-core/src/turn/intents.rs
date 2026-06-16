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
    spike_visual_map: &'a [VisualProperties],
    visual_map: &'a [VisualProperties],
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
    food_visible: [bool; crate::brain::VISION_RAY_COUNT],
}

impl Simulation {
    pub(super) fn build_intents(&mut self, world_width: i32) -> Vec<OrganismIntent> {
        let context = IntentBuildContext {
            world_width,
            occupancy: &self.occupancy,
            spike_map: &self.spike_map,
            spike_visual_map: &self.spike_visual_map,
            visual_map: &self.visual_map,
            pending_actions: &self.pending_actions,
            action_temperature: self.config.action_temperature,
            runtime_plasticity_enabled: self.config.runtime_plasticity_enabled,
            force_random_actions: self.config.force_random_actions,
            sim_seed: self.seed,
            tick: self.turn,
        };
        let thread_pool = self.parallel_pool();
        // Recycled buffer (see `TurnScratch`); the tick loop hands it back
        // after the commit phase. collect_into_vec/unzip_into_vecs truncate
        // and refill it with identical element order to a fresh collect.
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
    let pending_action = context.pending_actions[idx];
    if pending_action.turns_remaining > 0 {
        return locked_intent(idx, organism);
    }

    let selected_action_state = select_action_for_organism(organism, context, scratch);

    organism.last_action_taken = selected_action_state.selected_action;
    if selected_action_state.selected_action == ActionType::Idle {
        organism.health = (organism.health + organism_health_regeneration(organism))
            .min(organism.max_health.max(1.0));
    }
    let intent = intent_from_selected_action(
        idx,
        organism,
        selected_action_state.selected_action,
        selected_action_state.selected_action_logit,
        selected_action_state.synapse_ops,
        context.world_width,
    );

    BuiltIntent {
        intent,
        #[cfg(feature = "instrumentation")]
        action_record: Some(instrument_action_record(organism, selected_action_state)),
    }
}

fn locked_intent(idx: usize, organism: &OrganismState) -> BuiltIntent {
    BuiltIntent {
        intent: OrganismIntent {
            idx,
            id: organism.id,
            from: (organism.q, organism.r),
            facing_after_actions: organism.facing,
            wants_move: false,
            wants_eat: false,
            wants_attack: false,
            wants_reproduce: false,
            move_target: None,
            interaction_target: None,
            move_confidence: 0.0,
            took_action: false,
            synapse_ops: 0,
        },
        #[cfg(feature = "instrumentation")]
        action_record: None,
    }
}

fn select_action_for_organism(
    organism: &mut OrganismState,
    context: IntentBuildContext<'_>,
    scratch: &mut BrainScratch,
) -> SelectedActionState {
    let vision_distance = organism.genome.topology.vision_distance;
    let action_sample = deterministic_action_sample(context.sim_seed, context.tick, organism.id);

    if context.force_random_actions {
        #[cfg(feature = "instrumentation")]
        let food_visible = {
            let ray_scans = scan_rays(
                (organism.q, organism.r),
                organism.facing,
                organism.id,
                context.world_width,
                context.occupancy,
                context.spike_map,
                context.spike_visual_map,
                context.visual_map,
                vision_distance,
            );
            ray_scans.map(|scan| scan.food_visible)
        };

        return SelectedActionState {
            selected_action: uniform_random_action(action_sample),
            selected_action_logit: 0.0,
            synapse_ops: 0,
            #[cfg(feature = "instrumentation")]
            food_visible,
        };
    }

    let evaluation = evaluate_brain(
        organism,
        BrainEvalContext {
            world_width: context.world_width,
            occupancy: context.occupancy,
            spike_map: context.spike_map,
            spike_visual_map: context.spike_visual_map,
            visual_map: context.visual_map,
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
        food_visible: evaluation.food_visible,
    }
}

fn intent_from_selected_action(
    idx: usize,
    organism: &OrganismState,
    selected_action: ActionType,
    selected_action_logit: f32,
    synapse_ops: u64,
    world_width: i32,
) -> OrganismIntent {
    let from = (organism.q, organism.r);
    let current_facing = organism.facing;
    let forward_cell = || hex_neighbor(from, current_facing, world_width);

    let mut intent = OrganismIntent {
        idx,
        id: organism.id,
        from,
        facing_after_actions: current_facing,
        wants_move: false,
        wants_eat: false,
        wants_attack: false,
        wants_reproduce: false,
        move_target: None,
        interaction_target: None,
        move_confidence: 0.0,
        took_action: selected_action != ActionType::Idle,
        synapse_ops,
    };

    match selected_action {
        ActionType::Idle => {}
        ActionType::TurnLeft => intent.facing_after_actions = rotate_left(current_facing),
        ActionType::TurnRight => intent.facing_after_actions = rotate_right(current_facing),
        ActionType::Forward => {
            intent.wants_move = true;
            intent.move_target = Some(forward_cell());
            intent.move_confidence = selected_action_logit;
        }
        ActionType::Eat => {
            intent.wants_eat = true;
            intent.interaction_target = Some(forward_cell());
        }
        ActionType::Attack => {
            intent.wants_attack = true;
            intent.interaction_target = Some(forward_cell());
        }
        ActionType::Reproduce => intent.wants_reproduce = true,
    }

    intent
}

#[cfg(feature = "instrumentation")]
fn instrument_action_record(
    organism: &mut OrganismState,
    selected_action_state: SelectedActionState,
) -> ActionRecord {
    ActionRecord {
        organism_id: organism.id,
        selected_action: selected_action_state.selected_action,
        action_failed: selected_action_state.selected_action.can_fail(),
        food_visible: selected_action_state.food_visible,
        age_turns: organism.age_turns,
        utilization: update_instrumentation_utilization(organism),
        consumptions_count: organism.consumptions_count,
        plant_consumptions_count: organism.plant_consumptions_count,
        prey_consumptions_count: organism.prey_consumptions_count,
    }
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
