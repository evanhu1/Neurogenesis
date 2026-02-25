const { renderWorld, renderWorldBaseline } = require('../src/features/world/worldCanvas');

const FACINGS = ['East', 'NorthEast', 'NorthWest', 'West', 'SouthWest', 'SouthEast'];

class NoopContext {
  fillStyle = '';
  strokeStyle = '';
  lineWidth = 1;
  font = '';
  textAlign = 'start';
  textBaseline = 'alphabetic';

  clearRect() {}
  fillText() {}
  save() {}
  restore() {}
  translate() {}
  scale() {}
  beginPath() {}
  moveTo() {}
  lineTo() {}
  closePath() {}
  fill() {}
  stroke() {}
  arc() {}
  fillRect() {}
  bezierCurveTo() {}
  drawImage() {}
  measureText(text: string) {
    return { width: text.length * 7 };
  }
}

class MockOffscreenCanvas {
  width: number;
  height: number;

  constructor(width: number, height: number) {
    this.width = width;
    this.height = height;
  }

  getContext(_kind: string) {
    return new NoopContext();
  }
}

function mulberry32(seed: number) {
  let t = seed >>> 0;
  return () => {
    t += 0x6d2b79f5;
    let r = Math.imul(t ^ (t >>> 15), 1 | t);
    r ^= r + Math.imul(r ^ (r >>> 7), 61 | r);
    return ((r ^ (r >>> 14)) >>> 0) / 4294967296;
  };
}

function createSnapshot(worldWidth: number, organismCount: number, foodCount: number) {
  const rand = mulberry32(42);

  const occupancy = [];
  for (let r = 0; r < worldWidth; r += 1) {
    for (let q = 0; q < worldWidth; q += 1) {
      if (rand() < 0.14) {
        occupancy.push({ q, r, occupant: { type: 'Wall' } });
      }
    }
  }

  const organisms = Array.from({ length: organismCount }, (_, idx) => ({
    id: idx,
    species_id: idx % 120,
    q: Math.floor(rand() * worldWidth),
    r: Math.floor(rand() * worldWidth),
    generation: 1,
    age_turns: 5,
    facing: FACINGS[idx % FACINGS.length],
    energy: 15 + rand() * 90,
    consumptions_count: 0,
    reproductions_count: 0,
  }));

  const foods = Array.from({ length: foodCount }, (_, idx) => ({
    id: idx,
    q: Math.floor(rand() * worldWidth),
    r: Math.floor(rand() * worldWidth),
    energy: 20,
  }));

  return {
    turn: 1,
    rng_seed: 42,
    config: {
      world_width: worldWidth,
      steps_per_second: 24,
      num_organisms: organismCount,
      periodic_injection_interval_turns: 10,
      periodic_injection_count: 5,
      food_energy: 20,
      move_action_energy_cost: 1,
      action_temperature: 1,
      neuron_metabolism_cost: 0,
      plant_growth_speed: 1,
      food_regrowth_interval: 10,
      food_fertility_noise_scale: 1,
      food_fertility_exponent: 1,
      food_fertility_floor: 0,
      terrain_noise_scale: 1,
      terrain_threshold: 0.5,
      max_organism_age: 1000,
      speciation_threshold: 1,
      global_mutation_rate_modifier: 1,
      seed_genome_config: {
        num_neurons: 8,
        num_synapses: 16,
        spatial_prior_sigma: 1,
        vision_distance: 4,
        starting_energy: 50,
        age_of_maturity: 5,
        hebb_eta_gain: 0.1,
        eligibility_retention: 0.9,
        synapse_prune_threshold: 0.01,
        mutation_rate_age_of_maturity: 0.1,
        mutation_rate_vision_distance: 0.1,
        mutation_rate_inter_bias: 0.1,
        mutation_rate_inter_update_rate: 0.1,
        mutation_rate_action_bias: 0.1,
        mutation_rate_eligibility_retention: 0.1,
        mutation_rate_synapse_prune_threshold: 0.1,
        mutation_rate_neuron_location: 0.1,
        mutation_rate_synapse_weight_perturbation: 0.1,
        mutation_rate_add_neuron_split_edge: 0.1,
      },
    },
    organisms,
    foods,
    metrics: {
      turns: 1,
      organisms: organismCount,
      synapse_ops_last_turn: 0,
      actions_applied_last_turn: 0,
      consumptions_last_turn: 0,
      predations_last_turn: 0,
      total_consumptions: 0,
      reproductions_last_turn: 0,
      starvations_last_turn: 0,
      total_species_created: 1,
      species_counts: {},
    },
    occupancy,
  };
}

function advanceSnapshot(snapshot: any, tick: number) {
  const worldWidth = snapshot.config.world_width;
  for (let i = 0; i < snapshot.organisms.length; i += 1) {
    const organism = snapshot.organisms[i];
    if ((i + tick) % 3 !== 0) continue;
    organism.q = (organism.q + 1) % worldWidth;
    organism.r = (organism.r + ((i + tick) % 2 === 0 ? 1 : worldWidth - 1)) % worldWidth;
    organism.facing = FACINGS[(i + tick) % FACINGS.length];
    organism.energy = Math.max(1, organism.energy + ((i + tick) % 5 === 0 ? 1 : -0.5));
  }

  for (let i = 0; i < snapshot.foods.length; i += 1) {
    if ((i + tick) % 12 !== 0) continue;
    const food = snapshot.foods[i];
    food.q = (food.q + 1) % worldWidth;
  }

  snapshot.turn += 1;
}

function createCanvas(width: number, height: number) {
  return {
    width,
    height,
    getContext: () => new NoopContext(),
  };
}

function runPerTickRendererBenchmark(label: string, renderer: Function, initialSnapshot: any, tickCount: number) {
  const snapshot = structuredClone(initialSnapshot);
  const canvas = createCanvas(1400, 1400);
  const ctx = canvas.getContext('2d');
  const viewport = { zoom: 1, panX: 0, panY: 0 };

  for (let i = 0; i < 30; i += 1) {
    renderer(ctx, canvas, snapshot, null, viewport, { organisms: true, plants: true });
    advanceSnapshot(snapshot, i);
  }

  const startMs = performance.now();
  for (let tick = 0; tick < tickCount; tick += 1) {
    renderer(ctx, canvas, snapshot, null, viewport, { organisms: true, plants: true });
    advanceSnapshot(snapshot, tick);
  }
  const elapsedMs = performance.now() - startMs;

  return {
    label,
    elapsedMs,
    ticks: tickCount,
    draws: tickCount,
    msPerTick: elapsedMs / tickCount,
  };
}

function runPlaybackBenchmark(
  label: string,
  renderer: Function,
  initialSnapshot: any,
  durationSeconds: number,
  tickRate: number,
  drawRate: number,
) {
  const snapshot = structuredClone(initialSnapshot);
  const canvas = createCanvas(1400, 1400);
  const ctx = canvas.getContext('2d');
  const viewport = { zoom: 1, panX: 0, panY: 0 };

  const ticks = Math.floor(durationSeconds * tickRate);
  const draws = Math.floor(durationSeconds * drawRate);

  let nextTickAt = 0;
  let nextDrawAt = 0;
  const tickDt = 1000 / tickRate;
  const drawDt = 1000 / drawRate;

  const startMs = performance.now();
  for (let tick = 0, draw = 0; tick < ticks || draw < draws; ) {
    if (draw >= draws || (tick < ticks && nextTickAt <= nextDrawAt)) {
      advanceSnapshot(snapshot, tick);
      tick += 1;
      nextTickAt += tickDt;
      continue;
    }

    renderer(ctx, canvas, snapshot, null, viewport, { organisms: true, plants: true });
    draw += 1;
    nextDrawAt += drawDt;
  }
  const elapsedMs = performance.now() - startMs;

  return {
    label,
    elapsedMs,
    ticks,
    draws,
    msPerTick: elapsedMs / ticks,
  };
}

function printResult(result: any) {
  console.log(
    [
      `${result.label}:`,
      `elapsed=${result.elapsedMs.toFixed(2)}ms`,
      `ticks=${result.ticks}`,
      `draws=${result.draws}`,
      `ms_per_tick=${result.msPerTick.toFixed(4)}`,
    ].join(' '),
  );
}

function main() {
  (globalThis as any).OffscreenCanvas = MockOffscreenCanvas;

  const worldWidth = 192;
  const organismCount = 12000;
  const foodCount = 16000;
  const tickCount = 900;

  console.log(
    `World benchmark config: world=${worldWidth}x${worldWidth}, organisms=${organismCount}, foods=${foodCount}, ticks=${tickCount}`,
  );

  const snapshot = createSnapshot(worldWidth, organismCount, foodCount);

  const baselinePerTick = runPerTickRendererBenchmark(
    'baseline_renderer_per_tick',
    renderWorldBaseline,
    snapshot,
    tickCount,
  );
  const optimizedPerTick = runPerTickRendererBenchmark(
    'optimized_renderer_per_tick',
    renderWorld,
    snapshot,
    tickCount,
  );

  const baselinePlayback = runPlaybackBenchmark(
    'baseline_playback_60fps',
    renderWorldBaseline,
    snapshot,
    45,
    24,
    60,
  );
  const optimizedPlayback = runPlaybackBenchmark(
    'optimized_playback_tick_driven',
    renderWorld,
    snapshot,
    45,
    24,
    24,
  );

  printResult(baselinePerTick);
  printResult(optimizedPerTick);
  printResult(baselinePlayback);
  printResult(optimizedPlayback);

  const rendererSpeedup = baselinePerTick.elapsedMs / optimizedPerTick.elapsedMs;
  const playbackSpeedup = baselinePlayback.elapsedMs / optimizedPlayback.elapsedMs;
  const overallSpeedup = Math.min(rendererSpeedup, playbackSpeedup);

  console.log(
    `Speedups: renderer=${rendererSpeedup.toFixed(2)}x playback=${playbackSpeedup.toFixed(2)}x overall=${overallSpeedup.toFixed(2)}x`,
  );

  if (overallSpeedup < 2) {
    throw new Error(`Expected >=2.0x overall speedup, got ${overallSpeedup.toFixed(2)}x`);
  }
}

main();
