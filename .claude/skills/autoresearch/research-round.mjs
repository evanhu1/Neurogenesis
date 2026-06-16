export const meta = {
  name: 'research-round',
  description:
    'One coordinator iteration: ideate single-surface-area code-change experiments, run each as a worktree-isolated research agent (implement → build → determinism check → sim-cli sweep), then synthesize a coordinator handoff. Returns the handoff; the planner writes the OKF database and gates the merge.',
  phases: [
    { title: 'Ideate', detail: 'coordinator proposes code-change experiments' },
    { title: 'Experiment', detail: 'one worktree-isolated research agent per experiment' },
    { title: 'Synthesize', detail: 'coordinator ranks reports and hands off' },
  ],
}

// ---- args (from the planner) -------------------------------------------------
// { iteration:int, coordinator:string, surface_area:string, goal:string,
//   base_ref:string (sha of autoresearch/best), baseline_metrics:{...5 raw pillars},
//   seeds:[int], screen_seeds:[int], n_experiments:int }
const a = args || {}
const iteration = a.iteration ?? 0
const coordinator = a.coordinator ?? 'unnamed'
const surfaceArea = a.surface_area ?? a.coordinator ?? 'unknown'
const goal = a.goal ?? 'improve a lagging metric within this surface area'
const baseRef = a.base_ref ?? 'autoresearch/best'
const baselineMetrics = a.baseline_metrics ?? null
const seeds = a.seeds ?? [7, 42, 123, 2026]
const screenSeeds = a.screen_seeds ?? [7]
const nExperiments = a.n_experiments ?? 4
const iter4 = String(iteration).padStart(4, '0')

const METRICS = {
  type: 'object',
  properties: {
    plant_consumption_rate: { type: ['number', 'null'] },
    prey_consumption_rate: { type: ['number', 'null'] },
    action_effectiveness: { type: ['number', 'null'] },
    mi_sa: { type: ['number', 'null'] },
    learning_slope: { type: ['number', 'null'] },
  },
}

const IDEATE_SCHEMA = {
  type: 'object',
  required: ['experiments'],
  properties: {
    experiments: {
      type: 'array',
      items: {
        type: 'object',
        required: ['id', 'hypothesis', 'change_summary', 'target_metric'],
        properties: {
          id: { type: 'string', description: 'kebab-case, unique within this coordinator' },
          hypothesis: { type: 'string' },
          change_summary: { type: 'string', description: 'the concrete code change, files involved' },
          target_metric: { type: 'string', description: 'which raw pillar it should move' },
        },
      },
    },
  },
}

const REPORT_SCHEMA = {
  type: 'object',
  required: ['id', 'status', 'recommend'],
  properties: {
    id: { type: 'string' },
    git_ref: { type: ['string', 'null'], description: 'autoresearch/exp-* branch holding the change, or null if not persisted' },
    status: { type: 'string', enum: ['evaluated', 'screened-out', 'build-failed', 'determinism-broken'] },
    determinism: { type: 'string', enum: ['ok', 'broken', 'not-checked'] },
    screen_only: { type: 'boolean', description: 'true if only the cheap screen ran (not cross-seed confirmed)' },
    metrics: METRICS,
    delta: METRICS,
    seeds_used: { type: 'array', items: { type: 'integer' } },
    learnings: { type: 'string' },
    concerns: { type: 'string' },
    recommend: { type: 'string', enum: ['promote', 'screen-further', 'dead-end'] },
  },
}

const HANDOFF_SCHEMA = {
  type: 'object',
  required: ['coordinator', 'reports', 'learnings'],
  properties: {
    coordinator: { type: 'string' },
    surface_area: { type: 'string' },
    iteration: { type: 'integer' },
    reports: { type: 'array', items: REPORT_SCHEMA },
    best: { type: 'array', items: { type: 'string' }, description: 'experiment ids worth promoting, best first' },
    promote_refs: { type: 'array', items: { type: 'string' }, description: 'git_refs the planner should run through the merge gate' },
    learnings: { type: 'string', description: 'what moved which metric, and the mechanism' },
    concerns: { type: 'string' },
    directions: {
      type: 'array',
      items: { type: 'object', properties: { title: { type: 'string' }, rationale: { type: 'string' } } },
      description: 'promising untapped-alpha avenues to record as Directions',
    },
    dead_ends: {
      type: 'array',
      items: { type: 'object', properties: { title: { type: 'string' }, reason: { type: 'string' } } },
    },
  },
}

// ---- Phase 1: Ideate ---------------------------------------------------------
phase('Ideate')
const baselineNote = baselineMetrics
  ? `Baseline (base_ref) cross-seed metrics: ${JSON.stringify(baselineMetrics)}.`
  : 'Baseline metrics were not supplied; the research agents will measure base_ref if needed.'

const ideas = await agent(
  `You are the COORDINATOR for the surface area "${surfaceArea}" in the NeuroGenesis simulation research loop.
PRIMARY GOAL: ${goal}
The current best program is commit ${baseRef}. ${baselineNote}

Read docs/sim-cli.md, docs/research-operating-procedure.md, AGENTS.md, and the relevant engine code for THIS surface area only (do not stray into other lever-families). Then propose ${nExperiments} concrete, INDEPENDENT, single-surface-area CODE-CHANGE experiments that could move the target metric. Each must be a real code edit (not just a config sweep), small enough to implement and evaluate, and must preserve the determinism invariant. Favour diverse, mechanistically-motivated ideas over variations of one.

Return the experiments. Keep ids short and kebab-case.`,
  { schema: IDEATE_SCHEMA, label: `ideate:${coordinator}`, phase: 'Ideate' },
)

const experiments = (ideas?.experiments ?? []).slice(0, nExperiments)
if (experiments.length === 0) {
  log(`coordinator ${coordinator}: no experiments proposed; returning empty handoff`)
  return { coordinator, surface_area: surfaceArea, iteration, reports: [], best: [], promote_refs: [], learnings: 'no experiments proposed', concerns: 'ideation returned nothing', directions: [], dead_ends: [] }
}
log(`coordinator ${coordinator}: ${experiments.length} experiments → fanning out worktree research agents`)

// ---- Phase 2: Experiment (one worktree-isolated agent per experiment) --------
const reports = await parallel(
  experiments.map((exp) => async () => {
    const branch = `autoresearch/exp-${iter4}-${coordinator}-${exp.id}`
    const r = await agent(
      `You are a RESEARCH AGENT in an ISOLATED git worktree. Do everything inside this worktree.

EXPERIMENT "${exp.id}" (surface area: ${surfaceArea})
Hypothesis: ${exp.hypothesis}
Change: ${exp.change_summary}
Target metric: ${exp.target_metric}

Steps — follow exactly, and STOP early at any gate failure:
1. Base on the current champion: \`git checkout --detach ${baseRef}\`.
2. Implement the change. Stay strictly within the "${surfaceArea}" surface area; keep it minimal.
3. Build: \`cargo build -p sim-cli --release\` (and sim-core if you touched it). If it fails to compile → return status="build-failed", git_ref=null. Do not continue.
4. DETERMINISM CHECK (hard invariant):
   \`./target/release/sim-cli new --seed 7 --scale 70,400 --out /tmp/d_${exp.id}.bin\` ; \`cp /tmp/d_${exp.id}.bin /tmp/d2_${exp.id}.bin\` ;
   \`./target/release/sim-cli run-to 4000 --in /tmp/d_${exp.id}.bin --no-metrics\` ; \`./target/release/sim-cli run-to 4000 --in /tmp/d2_${exp.id}.bin --no-metrics\` ; \`cmp /tmp/d_${exp.id}.bin /tmp/d2_${exp.id}.bin\`.
   If the two forks differ → the change broke determinism → return status="determinism-broken", determinism="broken". Do not continue.
5. PERSIST THE CODE (critical — the worktree is ephemeral): \`git checkout -b ${branch}\` ; \`git add -A\` ; \`git commit -m "exp ${exp.id}: ${exp.hypothesis}"\`. The branch survives worktree cleanup; set git_ref="${branch}".
6. EVALUATE with sim-cli. First SCREEN cheaply: \`./target/release/sim-cli sweep --grid food_energy=<baseline value> --seeds ${screenSeeds.join(',')} --to 100000 --out-dir artifacts/runs\` (a 1-cell sweep at the BASELINE config = your changed binary's metrics on the canonical config; pick any single config key=baseline-value as the no-op cell). Read the result file's per-cell metrics. If the target metric does NOT improve vs the supplied baseline (${baselineMetrics ? JSON.stringify(baselineMetrics) : 'measure base_ref yourself with a quick run'}), set screen_only=true, status="screened-out", recommend="dead-end", and return.
   If it DOES improve, CONFIRM cross-seed: rerun the 1-cell sweep with \`--seeds ${seeds.join(',')} --to 500000\`. Use those cross-seed means as metrics.
7. Compute delta = metrics − baseline (per the 5 raw pillars: plant_consumption_rate, prey_consumption_rate, action_effectiveness, mi_sa, learning_slope). Flag in "concerns" any regression in a pillar you were not targeting, and any determinism/eval-time worry.
8. Return the REPORT. recommend="promote" only if the target metric improved cross-seed AND no other pillar regressed beyond noise.

Be honest: a clean null result or a dead-end is valuable. Never fabricate metrics — report exactly what the sweep produced.`,
      { schema: REPORT_SCHEMA, label: `exp:${coordinator}/${exp.id}`, phase: 'Experiment', isolation: 'worktree' },
    )
    // Defensive: ensure id + git_ref are populated even if the agent omitted them.
    if (r && !r.id) r.id = exp.id
    if (r && r.git_ref === undefined) r.git_ref = null
    return r
  }),
)

const valid = reports.filter(Boolean)
log(`coordinator ${coordinator}: ${valid.length}/${experiments.length} reports returned`)

// ---- Phase 3: Synthesize (coordinator handoff to the planner) ----------------
phase('Synthesize')
const handoff = await agent(
  `You are the COORDINATOR for "${surfaceArea}", synthesizing your research agents' reports into a single handoff for the PLANNER.

Goal was: ${goal}
Reports (JSON): ${JSON.stringify(valid)}

Produce the coordinator handoff:
- Rank the experiments; list the ids worth promoting (best first) in "best", and their git_refs in "promote_refs" (only experiments with status="evaluated", recommend="promote", determinism="ok", and a real git_ref).
- "learnings": what actually moved which raw metric and the likely mechanism — synthesize across experiments, don't just concatenate.
- "concerns": confounds, regressions, determinism or eval-time worries, suspected artifacts.
- "directions": promising untapped-alpha avenues this round surfaced (for the planner to record as Directions).
- "dead_ends": avenues these experiments ruled out (with the reason).
Echo coordinator="${coordinator}", surface_area, iteration=${iteration}, and include the full reports array unchanged.

Do NOT write any files — return the handoff only; the planner owns the database.`,
  { schema: HANDOFF_SCHEMA, label: `synthesize:${coordinator}`, phase: 'Synthesize' },
)

return handoff ?? {
  coordinator, surface_area: surfaceArea, iteration, reports: valid,
  best: [], promote_refs: [], learnings: 'synthesis failed', concerns: 'synthesis agent returned nothing',
  directions: [], dead_ends: [],
}
