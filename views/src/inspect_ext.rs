//! Per-organism inspection reads: predicate filtering (`find`), neural
//! inspection (`brain`), and single-tick decision explanation (`decide`). Free
//! functions over a [`crate::ReadCtx`], rendering to a writer (text or JSON).

use crate::output::{Format, Stats};
use crate::{take_format, ReadCtx};
use anyhow::{anyhow, bail, Result};
use serde_json::{json, Value};
use std::io::Write;
use types::{
    ActionType, NeuronId, OrganismState, SensoryReceptor, Symbol, SynapseEdge, SynapseTiming,
};

const MIN_ACTION_TEMPERATURE: f32 = 1.0e-6;

pub fn find(ctx: &ReadCtx, args: &[&str], out: &mut impl Write) -> Result<()> {
    let (fmt, rest) = take_format(ctx.format, args);

    // Split args into the predicate expression and the trailing options.
    let mut expr_tokens: Vec<&str> = Vec::new();
    let mut fields_spec: Option<&str> = None;
    let mut limit: usize = 20;
    let mut i = 0;
    while i < rest.len() {
        match rest[i] {
            "--fields" => {
                fields_spec = Some(
                    rest.get(i + 1)
                        .copied()
                        .ok_or_else(|| anyhow!("--fields needs a comma-separated list"))?,
                );
                i += 2;
            }
            "--limit" => {
                limit = rest
                    .get(i + 1)
                    .ok_or_else(|| anyhow!("--limit needs a value"))?
                    .parse()?;
                i += 2;
            }
            tok => {
                expr_tokens.push(tok);
                i += 1;
            }
        }
    }
    if expr_tokens.is_empty() {
        bail!("find needs a predicate, e.g. `find energy > 100 and age < 50`");
    }
    let predicate = Predicate::parse(&expr_tokens)?;

    let fields: Vec<&str> = match fields_spec {
        Some(s) => s
            .split(',')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .collect(),
        None => DEFAULT_FIND_FIELDS.to_vec(),
    };
    for f in &fields {
        if !is_field(f) {
            bail!("unknown --fields column `{f}`; valid: {FIND_FIELDS}");
        }
    }

    let sim = ctx.sim;
    let orgs = sim.organisms();

    let matched: Vec<&OrganismState> = orgs.iter().filter(|o| predicate.eval(o)).collect();
    let total = matched.len();
    let shown = &matched[..total.min(limit)];

    if fmt.is_json() {
        let rows: Vec<Value> = shown
            .iter()
            .map(|o| {
                let mut obj = serde_json::Map::new();
                for f in &fields {
                    obj.insert((*f).to_string(), field_json(o, f));
                }
                Value::Object(obj)
            })
            .collect();
        let v = json!({ "matched": total, "shown": shown.len(), "rows": rows });
        return writeln!(out, "{v}").map_err(Into::into);
    }

    writeln!(
        out,
        "{total} match(es){}; fields: {}",
        if total > shown.len() {
            format!(" (showing {})", shown.len())
        } else {
            String::new()
        },
        fields.join(",")
    )?;
    for o in shown {
        let cells: Vec<String> = fields
            .iter()
            .map(|f| format!("{f}={}", field_text(o, f)))
            .collect();
        writeln!(out, "  {}", cells.join(" "))?;
    }
    Ok(())
}

pub fn brain(ctx: &ReadCtx, args: &[&str], out: &mut impl Write) -> Result<()> {
    let (fmt, rest) = take_format(ctx.format, args);
    let id: u64 = rest
        .first()
        .ok_or_else(|| anyhow!("brain needs an organism id"))?
        .parse()?;
    let mut view = BrainView::Summary;
    let mut i = 1;
    while i < rest.len() {
        match rest[i] {
            "--view" => {
                view =
                    BrainView::parse(rest.get(i + 1).copied().ok_or_else(|| {
                        anyhow!("--view needs summary|synapses|activations|dot")
                    })?)?;
                i += 2;
            }
            other => bail!("unknown brain arg `{other}`"),
        }
    }

    let sim = ctx.sim;
    // Within-life learning is a world-level toggle; when off, the plasticity
    // genes and eligibility traces are inert and synapse weights stay at their
    // birth values. Surfaced in the plasticity/synapse views below.
    let plasticity_enabled = sim.config().runtime_plasticity_enabled;
    let idx = sim
        .organisms()
        .iter()
        .position(|o| o.id.0 == id)
        .ok_or_else(|| anyhow!("no live organism with id {id}"))?;
    let rec = sim.action_records().get(idx).cloned().flatten();
    let o = &sim.organisms()[idx];

    match view {
        BrainView::Summary => brain_summary(o, rec.as_ref(), plasticity_enabled, fmt, out),
        BrainView::Synapses => brain_synapses(o, plasticity_enabled, fmt, out),
        BrainView::Activations => brain_activations(o, fmt, out),
        BrainView::Dot => brain_dot(o, fmt, out),
    }
}

pub fn decide(ctx: &ReadCtx, args: &[&str], out: &mut impl Write) -> Result<()> {
    let (fmt, rest) = take_format(ctx.format, args);
    let id: u64 = rest
        .first()
        .ok_or_else(|| anyhow!("decide needs an organism id"))?
        .parse()?;
    let temperature = ctx.sim.config().action_temperature;
    let sim = ctx.sim;
    let idx = sim
        .organisms()
        .iter()
        .position(|o| o.id.0 == id)
        .ok_or_else(|| anyhow!("no live organism with id {id}"))?;
    let rec = sim.action_records().get(idx).cloned().flatten();
    let o = &sim.organisms()[idx];

    let Some(rec) = rec else {
        if fmt.is_json() {
            let v = json!({ "id": id, "decided": false,
                "note": "no action record yet; advance one tick (`step`) first" });
            return writeln!(out, "{v}").map_err(Into::into);
        }
        writeln!(
            out,
            "organism {id}: no decision recorded yet; run `step` to advance one tick first"
        )?;
        return Ok(());
    };

    // Reproduce the symbolic action head's categorical softmax.
    let logits: Vec<f32> = o.brain.action.iter().map(|a| a.logit).collect();
    let temp = temperature.max(MIN_ACTION_TEMPERATURE);
    let predation_enabled = sim.config().predation_enabled;
    let mut action_probabilities = vec![0.0_f32; logits.len()];
    let max_logit = Symbol::ALL
        .into_iter()
        .filter(|symbol| symbol.is_action_enabled(predation_enabled))
        .map(|symbol| logits[symbol.index()])
        .fold(f32::NEG_INFINITY, f32::max);
    let mut weight_sum = 0.0;
    for symbol in Symbol::ALL {
        if symbol.is_action_enabled(predation_enabled) {
            action_probabilities[symbol.index()] =
                ((logits[symbol.index()] - max_logit) / temp).exp();
            weight_sum += action_probabilities[symbol.index()];
        }
    }
    for probability in &mut action_probabilities {
        *probability /= weight_sum;
    }

    let inter_act: Vec<f64> = o
        .brain
        .inter
        .iter()
        .map(|n| n.neuron.activation as f64)
        .collect();
    let inter_stats = Stats::of(&inter_act);

    if fmt.is_json() {
        let inputs: Vec<Value> = o
            .brain
            .sensory
            .iter()
            .map(|s| {
                json!({
                    "receptor": receptor_label(&s.receptor),
                    "activation": round3(s.neuron.activation),
                })
            })
            .collect();
        let actions: Vec<Value> = o
            .brain
            .action
            .iter()
            .enumerate()
            .map(|(k, a)| {
                json!({
                    "symbol": a.symbol,
                    "logit": round3(a.logit),
                    "prob": round3(action_probabilities[k]),
                })
            })
            .collect();
        let v = json!({
            "id": id,
            "decided": true,
            "inputs": inputs,
            "inter": inter_stats.map(|s| s.json()),
            "actions": actions,
            "action_mode": "symbolic_categorical",
            "selected": format!("{:?}", rec.selected_action),
            "selected_symbol": o.last_action_symbol,
            "selected_command_mask": rec.selected_action_mask,
            "selected_commands": ActionType::ALL.iter().filter(|action| rec.selected_action_mask & action.command_bit() != 0).map(|action| format!("{action:?}")).collect::<Vec<_>>(),
            "failed_command_mask": rec.failed_action_mask,
            "action_failed": rec.action_failed,
            "note": "logits/probs reflect current (post-tick) brain state; selected commands are from the decision tick just executed",
        });
        return writeln!(out, "{v}").map_err(Into::into);
    }

    writeln!(out, "decision for organism {id} (turn-current):")?;
    writeln!(out, "  sensory inputs (nonzero):")?;
    for s in &o.brain.sensory {
        if s.neuron.activation != 0.0 {
            writeln!(
                out,
                "    {:<22} {:>7.3}",
                receptor_label(&s.receptor),
                s.neuron.activation
            )?;
        }
    }
    match inter_stats {
        Some(st) => writeln!(out, "  inter ({}): {}", o.brain.inter.len(), st.text())?,
        None => writeln!(out, "  inter: (none)")?,
    }
    writeln!(out, "  actions (logit -> prob):")?;
    for (k, a) in o.brain.action.iter().enumerate() {
        writeln!(
            out,
            "    {:<10} logit={:>8.3}  p={:.3}",
            format!("{}", a.symbol),
            a.logit,
            action_probabilities[k]
        )?;
    }
    writeln!(
        out,
        "  primary={:?} command_mask={:#06b} failed_mask={:#06b}",
        rec.selected_action, rec.selected_action_mask, rec.failed_action_mask
    )?;
    writeln!(
        out,
        "  note: logits/probs reflect the brain's CURRENT (post-tick) state; \
         `selected` is from the decision tick just executed, \
         so they may differ after plasticity/state updates."
    )?;
    Ok(())
}

// ---------------------------------------------------------------------------
// `find` predicate grammar + field mapping
// ---------------------------------------------------------------------------

const DEFAULT_FIND_FIELDS: &[&str] = &["id", "energy", "age", "generation", "successful_attacks"];

/// All field names accepted by `find` predicates and `--fields` columns. Shown
/// in error messages so the command is self-documenting.
const FIND_FIELDS: &str = "id energy energy_flow age generation species \
    successful_attacks neurons synapses initial_learning_rate";

/// Whether `name` is a known numeric field. The same table validates both
/// predicate fields and `--fields` columns; `org_field` maps each to a value.
fn is_field(name: &str) -> bool {
    matches!(
        name,
        "id" | "energy"
            | "energy_flow"
            | "age"
            | "generation"
            | "species"
            | "successful_attacks"
            | "neurons"
            | "synapses"
            | "initial_learning_rate"
    )
}

/// Numeric value of `name` for organism `o` (panics-free; callers validate the
/// name first). Used by both the predicate evaluator and field printing.
fn org_field(o: &OrganismState, name: &str) -> f64 {
    match name {
        "id" => o.id.0 as f64,
        "energy" => o.energy as f64,
        "energy_flow" => o.energy_flow_last_tick as f64,
        "age" => o.age_turns as f64,
        "generation" => o.generation as f64,
        "species" => o.species_id.0 as f64,
        "successful_attacks" => o.successful_attacks_count as f64,
        "neurons" => o.genome.hidden_node_count() as f64,
        "synapses" => o.brain.synapse_count as f64,
        "initial_learning_rate" => o.genome.plasticity.initial_learning_rate as f64,
        _ => f64::NAN,
    }
}

fn field_text(o: &OrganismState, name: &str) -> String {
    match name {
        "id" | "age" | "generation" | "species" | "successful_attacks" | "neurons" | "synapses"
        | "energy_flow" => {
            format!("{}", org_field(o, name) as i64)
        }
        "initial_learning_rate" => format!("{:.4}", org_field(o, name)),
        _ => format!("{:.2}", org_field(o, name)),
    }
}

fn field_json(o: &OrganismState, name: &str) -> Value {
    match name {
        "id" | "age" | "generation" | "species" | "successful_attacks" | "neurons" | "synapses"
        | "vision" => json!(org_field(o, name) as u64),
        _ => json!(org_field(o, name)),
    }
}

#[derive(Clone, Copy)]
enum CmpOp {
    Lt,
    Le,
    Gt,
    Ge,
    Eq,
    Ne,
}

impl CmpOp {
    fn parse(s: &str) -> Option<CmpOp> {
        Some(match s {
            "<" => CmpOp::Lt,
            "<=" => CmpOp::Le,
            ">" => CmpOp::Gt,
            ">=" => CmpOp::Ge,
            "==" => CmpOp::Eq,
            "!=" => CmpOp::Ne,
            _ => return None,
        })
    }

    fn apply(self, lhs: f64, rhs: f64) -> bool {
        match self {
            CmpOp::Lt => lhs < rhs,
            CmpOp::Le => lhs <= rhs,
            CmpOp::Gt => lhs > rhs,
            CmpOp::Ge => lhs >= rhs,
            // Relative+absolute tolerance: fields are f32-derived, so an exact
            // `==` against a parsed f64 literal would almost never hold. Scale the
            // tolerance with magnitude so `energy == 100` behaves sensibly.
            CmpOp::Eq => (lhs - rhs).abs() <= 1e-6 * lhs.abs().max(rhs.abs()).max(1.0),
            CmpOp::Ne => (lhs - rhs).abs() > 1e-6 * lhs.abs().max(rhs.abs()).max(1.0),
        }
    }
}

struct Comparison {
    field: String,
    op: CmpOp,
    value: f64,
}

#[derive(Clone, Copy)]
enum Join {
    And,
    Or,
}

/// A flat left-to-right conjunction/disjunction of comparisons (no precedence,
/// no parens): `c0 J0 c1 J1 c2 ...`, evaluated left to right.
struct Predicate {
    first: Comparison,
    rest: Vec<(Join, Comparison)>,
}

impl Predicate {
    fn parse(tokens: &[&str]) -> Result<Predicate> {
        // Comparison tokens may be glued ("energy>100") or split across up to
        // three tokens ("energy > 100"); normalize to a flat token stream then
        // consume `field op value` triples separated by and/or.
        let mut flat: Vec<String> = Vec::new();
        for &t in tokens {
            // Shell quoting deliberately preserves a whole expression as one
            // argv item (`find 'age == 50'`). Normalize its internal whitespace
            // exactly like the unquoted multi-argument spelling.
            for part in t.split_whitespace() {
                split_comparison_token(part, &mut flat);
            }
        }
        let mut iter = flat.iter().map(String::as_str).peekable();

        let first = parse_comparison(&mut iter)?;
        let mut rest = Vec::new();
        while let Some(tok) = iter.next() {
            let join = match tok {
                "and" | "&&" => Join::And,
                "or" | "||" => Join::Or,
                other => bail!("expected `and`/`or`, found `{other}`"),
            };
            rest.push((join, parse_comparison(&mut iter)?));
        }
        Ok(Predicate { first, rest })
    }

    fn eval(&self, o: &OrganismState) -> bool {
        let mut acc = self.first.eval(o);
        for (join, cmp) in &self.rest {
            let v = cmp.eval(o);
            acc = match join {
                Join::And => acc && v,
                Join::Or => acc || v,
            };
        }
        acc
    }
}

impl Comparison {
    fn eval(&self, o: &OrganismState) -> bool {
        self.op.apply(org_field(o, &self.field), self.value)
    }
}

/// Break a glued comparison token like `energy>=100` into `["energy", ">=",
/// "100"]`, while leaving bare words (`energy`, `and`) untouched.
fn split_comparison_token(tok: &str, out: &mut Vec<String>) {
    let bytes = tok.as_bytes();
    if let Some(pos) = bytes
        .iter()
        .position(|&b| matches!(b, b'<' | b'>' | b'=' | b'!'))
    {
        let mut end = pos + 1;
        if end < bytes.len() && bytes[end] == b'=' {
            end += 1;
        }
        if pos > 0 {
            out.push(tok[..pos].to_string());
        }
        out.push(tok[pos..end].to_string());
        if end < tok.len() {
            out.push(tok[end..].to_string());
        }
    } else {
        out.push(tok.to_string());
    }
}

fn parse_comparison<'a, I>(iter: &mut std::iter::Peekable<I>) -> Result<Comparison>
where
    I: Iterator<Item = &'a str>,
{
    let field = iter
        .next()
        .ok_or_else(|| anyhow!("expected a field name"))?
        .to_string();
    if !is_field(&field) {
        bail!("unknown field `{field}` in predicate; valid: {FIND_FIELDS}");
    }
    let op_tok = iter
        .next()
        .ok_or_else(|| anyhow!("expected a comparison operator after `{field}`"))?;
    let op = CmpOp::parse(op_tok)
        .ok_or_else(|| anyhow!("bad operator `{op_tok}` (use <,<=,>,>=,==,!=)"))?;
    let val_tok = iter
        .next()
        .ok_or_else(|| anyhow!("expected a value after `{field} {op_tok}`"))?;
    let value: f64 = val_tok
        .parse()
        .map_err(|_| anyhow!("`{val_tok}` is not a number"))?;
    Ok(Comparison { field, op, value })
}

// ---------------------------------------------------------------------------
// `brain` views
// ---------------------------------------------------------------------------

enum BrainView {
    Summary,
    Synapses,
    Activations,
    Dot,
}

impl BrainView {
    fn parse(s: &str) -> Result<BrainView> {
        Ok(match s {
            "summary" => BrainView::Summary,
            "synapses" => BrainView::Synapses,
            "activations" => BrainView::Activations,
            "dot" => BrainView::Dot,
            other => bail!("unknown view `{other}` (summary|synapses|activations|dot)"),
        })
    }
}

const TOP_SYNAPSES: usize = 12;

/// Human-readable label for any neuron id: sensory receptors by name, action
/// neurons by action type, inter neurons by `i<index>`.
fn neuron_label(id: NeuronId) -> String {
    if let Some(symbol) = Symbol::from_action_neuron_id(id) {
        return format!("act:{symbol:?}");
    }
    if let Some(index) = types::inter_neuron_index(id, u32::MAX) {
        return format!("i{index}");
    }
    if let Some(r) = SensoryReceptor::from_neuron_id(id) {
        return format!("s:{}", receptor_label(&r));
    }
    format!("s#{}", id.0)
}

fn receptor_label(r: &SensoryReceptor) -> String {
    match r {
        SensoryReceptor::Symbol { symbol } => format!("symbol:{symbol:?}"),
    }
}

/// Effective per-tick learning rate, reproducing `learning_rate_scale_at_age` in
/// world-sim/src/brain/plasticity.rs: 1.0 once `age >= age_of_maturity`, else the
/// genome's juvenile eta scale (>= 0).
fn effective_eta(o: &OrganismState) -> f32 {
    let g = &o.genome;
    let scale = if o.age_turns >= u64::from(g.lifecycle.plasticity_maturity_ticks) {
        1.0
    } else {
        g.plasticity.juvenile_eta_scale.max(0.0)
    };
    g.plasticity.initial_learning_rate.max(0.0) * scale
}

/// Collect every expressed current-tick and recurrent edge for synapse views.
fn all_edges(o: &OrganismState) -> Vec<&SynapseEdge> {
    o.brain
        .sensory
        .iter()
        .flat_map(|s| s.synapses.iter())
        .chain(o.brain.inter.iter().flat_map(|n| n.synapses.iter()))
        .chain(o.brain.recurrent_synapses.iter())
        .chain(o.brain.action_feedback_synapses.iter())
        .collect()
}

fn brain_summary(
    o: &OrganismState,
    rec: Option<&types::ActionRecord>,
    plasticity_enabled: bool,
    fmt: Format,
    out: &mut impl Write,
) -> Result<()> {
    let b = &o.brain;
    let mut edges = all_edges(o);
    edges.sort_by(|a, c| c.weight.abs().total_cmp(&a.weight.abs()));
    let top: Vec<&SynapseEdge> = edges.iter().take(TOP_SYNAPSES).copied().collect();

    let alphas: Vec<f64> = b.inter.iter().map(|n| n.alpha as f64).collect();
    let (alpha_min, alpha_max) = alphas
        .iter()
        .copied()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(mn, mx), v| {
            (mn.min(v), mx.max(v))
        });
    let eta = effective_eta(o);
    let g = &o.genome.plasticity;

    if fmt.is_json() {
        let top_json: Vec<Value> = top
            .iter()
            .map(|e| {
                json!({
                    "pre": neuron_label(e.pre_neuron_id),
                    "post": neuron_label(e.post_neuron_id),
                    "timing": e.timing,
                    "w": round3(e.weight),
                    "plasticity_coefficient": round3(e.plasticity_coefficient),
                    "elig": round3(e.eligibility),
                })
            })
            .collect();
        let v = json!({
            "id": o.id.0,
            "neurons": { "sensory": b.sensory.len(), "inter": b.inter.len(), "action": b.action.len() },
            "synapse_count": b.synapse_count,
            "top_synapses": top_json,
            "alpha": if b.inter.is_empty() { Value::Null } else { json!({ "min": alpha_min, "max": alpha_max }) },
            "plasticity": {
                "runtime_plasticity_enabled": plasticity_enabled,
                "initial_learning_rate": g.initial_learning_rate,
                "juvenile_eta_scale": g.juvenile_eta_scale,
                "eligibility_retention": g.eligibility_retention,
                "fast_weight_retention": g.fast_weight_retention,
                "action_temperature_scale": g.action_temperature_scale,
                "max_weight_delta_per_tick": g.max_weight_delta_per_tick,
                "synapse_prune_threshold": g.synapse_prune_threshold,
                "effective_eta": effective_eta_round(eta),
            },
            "utilization": rec.map(|r| round3(r.utilization)),
        });
        return writeln!(out, "{v}").map_err(Into::into);
    }

    writeln!(
        out,
        "brain of organism {}: sensory={} inter={} action={} synapses={}",
        o.id.0,
        b.sensory.len(),
        b.inter.len(),
        b.action.len(),
        b.synapse_count
    )?;
    writeln!(
        out,
        "  top {} synapse(s) by |weight| (of {} expressed):",
        top.len(),
        edges.len()
    )?;
    for e in &top {
        writeln!(
            out,
            "    {:<20} -> {:<14} {:<13} w={:>7.3} plasticity={:>6.3} elig={:>7.3}",
            neuron_label(e.pre_neuron_id),
            neuron_label(e.post_neuron_id),
            timing_label(e.timing),
            e.weight,
            e.plasticity_coefficient,
            e.eligibility
        )?;
    }
    if b.inter.is_empty() {
        writeln!(out, "  alpha: (no inter neurons)")?;
    } else {
        writeln!(out, "  alpha range: [{alpha_min:.3}, {alpha_max:.3}]")?;
    }
    let plasticity_tag = if plasticity_enabled {
        ""
    } else {
        "  [runtime_plasticity_enabled disabled — weights static, no within-life learning]"
    };
    writeln!(
        out,
        "  plasticity: initial_learning_rate={:.4} juv_scale={:.3} elig_ret={:.3} fast_ret={:.3} temp={:.3} max_dw={:.4} prune={:.4} -> effective_eta={:.5}{plasticity_tag}",
        g.initial_learning_rate,
        g.juvenile_eta_scale,
        g.eligibility_retention,
        g.fast_weight_retention,
        g.action_temperature_scale,
        g.max_weight_delta_per_tick,
        g.synapse_prune_threshold,
        eta
    )?;
    match rec {
        Some(r) => writeln!(out, "  utilization (this tick): {:.3}", r.utilization)?,
        None => writeln!(out, "  utilization: (no action record; step first)")?,
    }
    Ok(())
}

fn brain_synapses(
    o: &OrganismState,
    plasticity_enabled: bool,
    fmt: Format,
    out: &mut impl Write,
) -> Result<()> {
    let mut edges = all_edges(o);
    edges.sort_by(|a, b| {
        a.pre_neuron_id
            .0
            .cmp(&b.pre_neuron_id.0)
            .then(a.post_neuron_id.0.cmp(&b.post_neuron_id.0))
            .then(a.timing.cmp(&b.timing))
    });

    if fmt.is_json() {
        let rows: Vec<Value> = edges
            .iter()
            .map(|e| {
                json!({
                    "pre": neuron_label(e.pre_neuron_id),
                    "post": neuron_label(e.post_neuron_id),
                    "timing": e.timing,
                    "weight": round3(e.weight),
                    "eligibility": round3(e.eligibility),
                    "pending": round3(e.pending_coactivation),
                })
            })
            .collect();
        let v = json!({
            "id": o.id.0,
            "runtime_plasticity_enabled": plasticity_enabled,
            "synapses": rows,
        });
        return writeln!(out, "{v}").map_err(Into::into);
    }

    writeln!(
        out,
        "brain of organism {}: {} synapse(s):",
        o.id.0,
        edges.len()
    )?;
    if !plasticity_enabled {
        writeln!(
            out,
            "  [runtime_plasticity_enabled disabled — weights are static within lifetime; elig/pending are inert]"
        )?;
    }
    for e in &edges {
        writeln!(
            out,
            "  {:<20} -> {:<14} {:<13} weight={:>7.3} elig={:>7.3} pending={:>7.3}",
            neuron_label(e.pre_neuron_id),
            neuron_label(e.post_neuron_id),
            timing_label(e.timing),
            e.weight,
            e.eligibility,
            e.pending_coactivation
        )?;
    }
    Ok(())
}

fn brain_activations(o: &OrganismState, fmt: Format, out: &mut impl Write) -> Result<()> {
    let b = &o.brain;
    let inter_vals: Vec<f64> = b.inter.iter().map(|n| n.neuron.activation as f64).collect();
    let inter_stats = Stats::of(&inter_vals);

    if fmt.is_json() {
        let sensory: Vec<Value> = b
            .sensory
            .iter()
            .map(|s| {
                json!({
                    "receptor": receptor_label(&s.receptor),
                    "activation": round3(s.neuron.activation),
                })
            })
            .collect();
        let actions: Vec<Value> = b
            .action
            .iter()
            .map(|a| json!({ "symbol": a.symbol, "logit": round3(a.logit) }))
            .collect();
        let v = json!({
            "id": o.id.0,
            "sensory": sensory,
            "inter": {
                "stats": inter_stats.map(|s| s.json()),
                "values": inter_vals.iter().map(|&x| round3(x as f32)).collect::<Vec<_>>(),
            },
            "actions": actions,
        });
        return writeln!(out, "{v}").map_err(Into::into);
    }

    writeln!(out, "activations of organism {}:", o.id.0)?;
    writeln!(out, "  sensory:")?;
    for s in &b.sensory {
        writeln!(
            out,
            "    {:<22} {:>7.3}",
            receptor_label(&s.receptor),
            s.neuron.activation
        )?;
    }
    match inter_stats {
        Some(st) => {
            writeln!(out, "  inter ({}): {}", b.inter.len(), st.text())?;
            writeln!(out, "    {}", crate::output::sparkline(&inter_vals))?;
        }
        None => writeln!(out, "  inter: (none)")?,
    }
    writeln!(out, "  action logits:")?;
    for a in &b.action {
        writeln!(out, "    {:<10} {:>8.3}", format!("{}", a.symbol), a.logit)?;
    }
    Ok(())
}

fn brain_dot(o: &OrganismState, fmt: Format, out: &mut impl Write) -> Result<()> {
    let mut edges = all_edges(o);
    edges.sort_by(|a, b| {
        a.pre_neuron_id
            .0
            .cmp(&b.pre_neuron_id.0)
            .then(a.post_neuron_id.0.cmp(&b.post_neuron_id.0))
            .then(a.timing.cmp(&b.timing))
    });

    let mut dot = String::from("digraph brain {\n  rankdir=LR;\n");
    for s in &o.brain.sensory {
        dot.push_str(&format!(
            "  \"{}\" [shape=box];\n",
            neuron_label(s.neuron.neuron_id)
        ));
    }
    for n in &o.brain.inter {
        dot.push_str(&format!(
            "  \"{}\" [shape=ellipse];\n",
            neuron_label(n.neuron.neuron_id)
        ));
    }
    for a in &o.brain.action {
        dot.push_str(&format!(
            "  \"{}\" [shape=doublecircle];\n",
            neuron_label(a.neuron_id)
        ));
    }
    for e in &edges {
        dot.push_str(&format!(
            "  \"{}\" -> \"{}\" [label=\"{:.3}\"{}];\n",
            neuron_label(e.pre_neuron_id),
            neuron_label(e.post_neuron_id),
            e.weight,
            if e.timing == SynapseTiming::PreviousTick {
                ", style=dashed, color=purple"
            } else {
                ""
            }
        ));
    }
    dot.push_str("}\n");

    if fmt.is_json() {
        let v = json!({ "id": o.id.0, "dot": dot });
        return writeln!(out, "{v}").map_err(Into::into);
    }
    write!(out, "{dot}").map_err(Into::into)
}

fn timing_label(timing: SynapseTiming) -> &'static str {
    match timing {
        SynapseTiming::CurrentTick => "current",
        SynapseTiming::PreviousTick => "previous",
    }
}

/// Round to 3 decimals for compact, stable output.
fn round3(v: f32) -> f64 {
    ((v as f64) * 1000.0).round() / 1000.0
}

fn effective_eta_round(v: f32) -> f64 {
    ((v as f64) * 100_000.0).round() / 100_000.0
}
