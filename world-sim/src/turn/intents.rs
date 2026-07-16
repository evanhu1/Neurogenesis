use super::*;

#[cfg(feature = "instrumentation")]
const INTER_EMA_ALPHA: f32 = 0.05;
#[cfg(feature = "instrumentation")]
const UTILIZATION_THRESHOLD: f32 = 0.03;

#[derive(Clone, Copy)]
struct IntentBuildContext<'a> {
    world_width: i32,
    occupancy: &'a [Option<Occupant>],
    vision_ray_table: &'a crate::sensing::VisionRayTable,
    action_temperature: f32,
    runtime_plasticity_enabled: bool,
    leaky_neurons_enabled: bool,
    predation_enabled: bool,
    starting_energy: u32,
    energy_flow_scale: u32,
    force_random_actions: bool,
    compositional_actions_enabled: bool,
    sim_seed: u64,
    tick: u64,
}

#[derive(Clone, Copy)]
struct SelectedActionState {
    selected_action: ActionType,
    selected_action_mask: u8,
    forward_logit: f32,
    synapse_ops: u64,
}

impl Simulation {
    pub(super) fn build_intents(&mut self, world_width: i32) -> Vec<OrganismIntent> {
        let context = IntentBuildContext {
            world_width,
            occupancy: &self.occupancy,
            vision_ray_table: &self.vision_ray_table,
            action_temperature: self.config.action_temperature,
            runtime_plasticity_enabled: self.config.runtime_plasticity_enabled,
            leaky_neurons_enabled: self.config.leaky_neurons_enabled,
            predation_enabled: self.config.predation_enabled,
            starting_energy: self.config.starting_energy,
            energy_flow_scale: self.config.attack_energy_transfer,
            force_random_actions: self.config.force_random_actions,
            compositional_actions_enabled: self.config.compositional_actions_enabled,
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
    let selected_action_state = select_action_for_organism(organism, context, scratch);

    organism.last_action_taken = selected_action_state.selected_action;
    organism.last_action_mask = selected_action_state.selected_action_mask;
    let intent = intent_from_selected_action(idx, organism, selected_action_state, context);

    BuiltIntent {
        intent,
        #[cfg(feature = "instrumentation")]
        action_record: Some(instrument_action_record(organism, selected_action_state)),
    }
}

fn select_action_for_organism(
    organism: &mut OrganismState,
    context: IntentBuildContext<'_>,
    scratch: &mut BrainScratch,
) -> SelectedActionState {
    let action_samples = deterministic_action_samples(context.sim_seed, context.tick, organism.id);

    if context.force_random_actions {
        return SelectedActionState {
            selected_action: if context.compositional_actions_enabled {
                primary_action_from_mask(uniform_random_compositional_mask(
                    action_samples,
                    context.predation_enabled,
                ))
            } else {
                uniform_random_action(action_samples[0], context.predation_enabled)
            },
            selected_action_mask: if context.compositional_actions_enabled {
                uniform_random_compositional_mask(action_samples, context.predation_enabled)
            } else {
                uniform_random_action(action_samples[0], context.predation_enabled).command_bit()
            },
            forward_logit: 0.0,
            synapse_ops: 0,
        };
    }

    let ray_scans = crate::sensing::encode_sensory_inputs(
        organism,
        context.vision_ray_table,
        context.occupancy,
        context.starting_energy,
        context.energy_flow_scale,
        context.predation_enabled,
    );
    let _ = ray_scans;
    let evaluation = evaluate_brain(
        organism,
        BrainEvalContext {
            leaky_neurons_enabled: context.leaky_neurons_enabled,
            predation_enabled: context.predation_enabled,
            action_temperature: context.action_temperature,
            action_samples,
            compositional_actions_enabled: context.compositional_actions_enabled,
        },
        scratch,
    );
    if context.runtime_plasticity_enabled {
        compute_pending_coactivations(organism, scratch);
    }

    let selected_action = evaluation.selected_action;
    SelectedActionState {
        selected_action,
        selected_action_mask: evaluation.selected_action_mask,
        forward_logit: evaluation.action_logits[action_index(ActionType::Forward)],
        synapse_ops: evaluation.synapse_ops,
    }
}

fn intent_from_selected_action(
    idx: usize,
    organism: &OrganismState,
    selected: SelectedActionState,
    context: IntentBuildContext<'_>,
) -> OrganismIntent {
    let selected_action_mask = selected.selected_action_mask;
    let world_width = context.world_width;
    let compositional_actions_enabled = context.compositional_actions_enabled;
    let from = (organism.q, organism.r);
    let current_facing = organism.facing;
    let forward_cell = || hex_neighbor(from, current_facing, world_width);

    let mut intent = OrganismIntent {
        idx,
        id: organism.id,
        from,
        facing_after_actions: current_facing,
        wants_move: false,
        wants_attack: false,
        move_target: None,
        interaction_target: None,
        interaction_after_move: compositional_actions_enabled,
        snapshot_attack_target: None,
        move_confidence: 0.0,
        command_count: selected_action_mask.count_ones() as u8,
        synapse_ops: selected.synapse_ops,
    };

    if selected_action_mask & ActionType::TurnLeft.command_bit() != 0 {
        intent.facing_after_actions = rotate_left(current_facing);
    } else if selected_action_mask & ActionType::TurnRight.command_bit() != 0 {
        intent.facing_after_actions = rotate_right(current_facing);
    }
    let command_forward_cell = || hex_neighbor(from, intent.facing_after_actions, world_width);
    if selected_action_mask & ActionType::Forward.command_bit() != 0 {
        intent.wants_move = true;
        intent.move_target = Some(if compositional_actions_enabled {
            command_forward_cell()
        } else {
            forward_cell()
        });
        intent.move_confidence = selected.forward_logit;
    }
    if selected_action_mask & ActionType::Attack.command_bit() != 0 {
        intent.wants_attack = true;
        let target = if compositional_actions_enabled {
            command_forward_cell()
        } else {
            forward_cell()
        };
        intent.interaction_target = Some(target);
        let target_idx = target.1 as usize * world_width as usize + target.0 as usize;
        intent.snapshot_attack_target = match context.occupancy[target_idx] {
            Some(Occupant::Organism(id)) => Some(id),
            _ => None,
        };
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
        selected_action_mask: selected_action_state.selected_action_mask,
        failed_action_mask: selected_action_state.selected_action_mask
            & (ActionType::Forward.command_bit() | ActionType::Attack.command_bit()),
        action_failed: selected_action_state.selected_action_mask
            & (ActionType::Forward.command_bit() | ActionType::Attack.command_bit())
            != 0,
        age_turns: organism.age_turns,
        utilization: update_instrumentation_utilization(organism),
        successful_attacks_count: organism.successful_attacks_count,
    }
}

fn uniform_random_compositional_mask(samples: [f32; 3], predation_enabled: bool) -> u8 {
    let orientation = match (samples[0].clamp(0.0, 1.0 - f32::EPSILON) * 3.0) as usize {
        0 => ActionType::TurnLeft,
        1 => ActionType::TurnRight,
        _ => ActionType::Idle,
    };
    let locomotion = if samples[1] < 0.5 {
        ActionType::Forward
    } else {
        ActionType::Idle
    };
    let mut interactions = [ActionType::Attack]
        .into_iter()
        .filter(|action| action.is_enabled(predation_enabled));
    let interaction_count = interactions.clone().count();
    let bucket =
        (samples[2].clamp(0.0, 1.0 - f32::EPSILON) * (interaction_count + 1) as f32) as usize;
    let interaction = interactions.nth(bucket).unwrap_or(ActionType::Idle);
    orientation.command_bit() | locomotion.command_bit() | interaction.command_bit()
}

fn primary_action_from_mask(mask: u8) -> ActionType {
    ActionType::ALL
        .iter()
        .copied()
        .find(|action| mask & action.command_bit() != 0)
        .unwrap_or(ActionType::Idle)
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
