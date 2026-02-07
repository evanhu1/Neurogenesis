type OrganismId = { 0: number } | number;

type Envelope<T> = {
  protocol_version: number;
  payload: T;
};

type SurvivalRule = {
  CenterBandX?: { min_fraction: number; max_fraction: number };
  center_band_x?: { min_fraction: number; max_fraction: number };
  min_fraction?: number;
  max_fraction?: number;
};

type WorldConfig = {
  columns: number;
  rows: number;
  steps_per_epoch: number;
  steps_per_second: number;
  num_organisms: number;
  num_neurons: number;
  max_num_neurons: number;
  num_synapses: number;
  vision_depth: number;
  action_potential_length: number;
  mutation_chance: number;
  mutation_magnitude: number;
  unfit_kill_probability: number;
  offspring_fill_ratio: number;
  survival_rule: SurvivalRule;
};

type SynapseEdge = { post_neuron_id: number | { 0: number }; weight: number };

type NeuronState = {
  neuron_id: number | { 0: number };
  neuron_type: 'Sensory' | 'Inter' | 'Action' | string;
  is_inverted: boolean;
  action_potential_threshold: number;
  resting_potential: number;
  potential: number;
  incoming_current: number;
  potential_decay_rate: number;
  action_potential_length: number;
  action_potential_time: number | null;
  parent_ids: Array<number | { 0: number }>;
};

type SensoryNeuronState = {
  neuron: NeuronState;
  receptor_type: string;
  synapses: SynapseEdge[];
};

type InterNeuronState = {
  neuron: NeuronState;
  synapses: SynapseEdge[];
};

type ActionNeuronState = {
  neuron: NeuronState;
  action_type: string;
  is_active: boolean;
};

type BrainState = {
  sensory: SensoryNeuronState[];
  inter: InterNeuronState[];
  action: ActionNeuronState[];
  synapse_count: number;
};

type OrganismState = {
  id: OrganismId;
  x: number;
  y: number;
  brain: BrainState;
};

type MetricsSnapshot = {
  ticks: number;
  epochs: number;
  survivors_last_epoch: number;
  organisms: number;
  synapse_ops_last_tick: number;
  actions_applied_last_tick: number;
};

type WorldSnapshot = {
  epoch: number;
  tick_in_epoch: number;
  rng_seed: number;
  config: WorldConfig;
  organisms: OrganismState[];
  occupancy: Array<{ x: number; y: number; organism_ids: OrganismId[] }>;
  metrics: MetricsSnapshot;
};

type SessionMetadata = {
  id: string;
  created_at_unix_ms: number;
  config: WorldConfig;
};

type CreateSessionResponse = {
  metadata: SessionMetadata;
  snapshot: WorldSnapshot;
};

type TickDelta = {
  tick_in_epoch: number;
  epoch: number;
  moves: Array<{ id: OrganismId; from: [number, number]; to: [number, number] }>;
  metrics: MetricsSnapshot;
};

type ServerEvent = {
  type:
    | 'StateSnapshot'
    | 'TickDelta'
    | 'EpochCompleted'
    | 'FocusBrain'
    | 'Metrics'
    | 'Error';
  data: unknown;
};

const apiBase = import.meta.env.VITE_API_BASE ?? 'http://127.0.0.1:8080';
const wsBase = apiBase.replace('http://', 'ws://').replace('https://', 'wss://');

const sessionMetaEl = document.querySelector('#session-meta') as HTMLDivElement;
const metricsEl = document.querySelector('#metrics') as HTMLDivElement;
const focusMetaEl = document.querySelector('#focus-meta') as HTMLDivElement;

const worldCanvas = document.querySelector('#world-canvas') as HTMLCanvasElement;
const worldCtx = worldCanvas.getContext('2d');
if (!worldCtx) throw new Error('world canvas 2d context unavailable');

const brainCanvas = document.querySelector('#brain-canvas') as HTMLCanvasElement;
const brainCtx = brainCanvas.getContext('2d');
if (!brainCtx) throw new Error('brain canvas 2d context unavailable');

const createBtn = document.querySelector('#btn-create') as HTMLButtonElement;
const resetBtn = document.querySelector('#btn-reset') as HTMLButtonElement;
const startBtn = document.querySelector('#btn-start') as HTMLButtonElement;
const pauseBtn = document.querySelector('#btn-pause') as HTMLButtonElement;
const stepBtn = document.querySelector('#btn-step') as HTMLButtonElement;
const epochBtn = document.querySelector('#btn-epoch') as HTMLButtonElement;
const scatterBtn = document.querySelector('#btn-scatter') as HTMLButtonElement;
const survivorsBtn = document.querySelector('#btn-survivors') as HTMLButtonElement;

let session: SessionMetadata | null = null;
let snapshot: WorldSnapshot | null = null;
let ws: WebSocket | null = null;
let focusedOrganismId: number | null = null;
let focusedBrain: BrainState | null = null;

const DEFAULT_CONFIG: WorldConfig = {
  columns: 20,
  rows: 20,
  steps_per_epoch: 20,
  steps_per_second: 5,
  num_organisms: 500,
  num_neurons: 2,
  max_num_neurons: 20,
  num_synapses: 4,
  vision_depth: 3,
  action_potential_length: 1,
  mutation_chance: 0.04,
  mutation_magnitude: 1,
  unfit_kill_probability: 0.95,
  offspring_fill_ratio: 0.2,
  survival_rule: {
    CenterBandX: { min_fraction: 0.25, max_fraction: 0.75 },
  },
};

function unwrapId(id: OrganismId | number | { 0: number }): number {
  if (typeof id === 'number') return id;
  return id[0];
}

async function request<T>(path: string, method: string, body?: unknown): Promise<T> {
  const response = await fetch(`${apiBase}${path}`, {
    method,
    headers: {
      'Content-Type': 'application/json',
    },
    body: body ? JSON.stringify(body) : undefined,
  });

  const json = await response.json();
  if (!response.ok) {
    throw new Error(json?.payload?.message ?? 'request failed');
  }

  return (json as Envelope<T>).payload;
}

async function createSession() {
  const payload = await request<CreateSessionResponse>('/v1/sessions', 'POST', {
    config: DEFAULT_CONFIG,
    seed: Math.floor(Date.now() / 1000),
  });

  session = payload.metadata;
  snapshot = payload.snapshot;
  focusedOrganismId = null;
  focusedBrain = null;
  sessionMetaEl.textContent = `session=${session.id}\ncreated=${new Date(session.created_at_unix_ms).toISOString()}`;

  connectWs(session.id);
  updateMetrics();
  renderBrain();
}

function connectWs(id: string) {
  if (ws) ws.close();
  ws = new WebSocket(`${wsBase}/v1/sessions/${id}/stream`);

  ws.onmessage = async (evt) => {
    try {
      const envelope = JSON.parse(String(evt.data)) as Envelope<ServerEvent>;
      handleServerEvent(envelope.payload);
    } catch (err) {
      console.error('ws parse error', err);
    }
  };

  ws.onclose = () => {
    console.log('ws closed');
  };
}

function handleServerEvent(event: ServerEvent) {
  switch (event.type) {
    case 'StateSnapshot': {
      snapshot = event.data as WorldSnapshot;
      if (focusedOrganismId !== null) {
        const organism = snapshot.organisms.find((o) => unwrapId(o.id) === focusedOrganismId);
        focusedBrain = organism?.brain ?? null;
      }
      updateMetrics();
      renderBrain();
      break;
    }
    case 'TickDelta': {
      const delta = event.data as TickDelta;
      applyTickDelta(delta);
      updateMetrics();
      break;
    }
    case 'EpochCompleted': {
      if (!session) return;
      void request<WorldSnapshot>(`/v1/sessions/${session.id}/state`, 'GET').then((state) => {
        snapshot = state;
        if (focusedOrganismId !== null) {
          const org = snapshot.organisms.find((o) => unwrapId(o.id) === focusedOrganismId);
          focusedBrain = org?.brain ?? null;
        }
        updateMetrics();
        renderBrain();
      });
      break;
    }
    case 'FocusBrain': {
      const org = event.data as OrganismState;
      focusedOrganismId = unwrapId(org.id);
      focusedBrain = org.brain;
      focusMetaEl.textContent = `focused organism: ${focusedOrganismId}`;
      renderBrain();
      break;
    }
    case 'Metrics': {
      if (snapshot) {
        snapshot.metrics = event.data as MetricsSnapshot;
      }
      updateMetrics();
      break;
    }
    case 'Error': {
      console.error('server event error', event.data);
      break;
    }
    default:
      break;
  }
}

function applyTickDelta(delta: TickDelta) {
  if (!snapshot) return;

  snapshot.tick_in_epoch = delta.tick_in_epoch;
  snapshot.epoch = delta.epoch;
  snapshot.metrics = delta.metrics;

  const index = new Map<number, OrganismState>();
  for (const org of snapshot.organisms) index.set(unwrapId(org.id), org);

  for (const move of delta.moves) {
    const id = unwrapId(move.id);
    const org = index.get(id);
    if (!org) continue;
    org.x = move.to[0];
    org.y = move.to[1];
  }
}

function sendCommand(command: unknown) {
  if (!ws || ws.readyState !== WebSocket.OPEN) return;
  ws.send(JSON.stringify({ protocol_version: 1, payload: command }));
}

function updateMetrics() {
  if (!snapshot) {
    metricsEl.textContent = 'No metrics';
    return;
  }
  metricsEl.textContent = [
    `epoch=${snapshot.epoch}`,
    `tick=${snapshot.tick_in_epoch}`,
    `organisms=${snapshot.metrics.organisms}`,
    `survivors_last_epoch=${snapshot.metrics.survivors_last_epoch}`,
    `synapse_ops_last_tick=${snapshot.metrics.synapse_ops_last_tick}`,
    `actions_last_tick=${snapshot.metrics.actions_applied_last_tick}`,
  ].join('\n');
}

function renderWorld() {
  const ctx = worldCtx;
  const width = worldCanvas.width;
  const height = worldCanvas.height;

  ctx.clearRect(0, 0, width, height);

  if (!snapshot) {
    ctx.fillStyle = '#1b2638';
    ctx.font = '20px IBM Plex Sans';
    ctx.fillText('Create a session to begin', 24, 40);
    return;
  }

  const cols = snapshot.config.columns;
  const rows = snapshot.config.rows;
  const cell = Math.min(width / cols, height / rows);

  for (let y = 0; y < rows; y += 1) {
    for (let x = 0; x < cols; x += 1) {
      ctx.fillStyle = (x + y) % 2 === 0 ? '#d7dde8' : '#e3e8f0';
      ctx.fillRect(x * cell, (rows - 1 - y) * cell, cell, cell);
    }
  }

  const survival = snapshot.config.survival_rule as SurvivalRule;
  const centerBand = (survival.CenterBandX ?? survival.center_band_x) ?? {
    min_fraction: survival.min_fraction ?? 0.25,
    max_fraction: survival.max_fraction ?? 0.75,
  };
  const left = centerBand.min_fraction * cols * cell;
  const right = centerBand.max_fraction * cols * cell;
  ctx.fillStyle = 'rgba(17, 103, 177, 0.12)';
  ctx.fillRect(left, 0, right - left, rows * cell);

  const countsByCell = new Map<string, number>();
  for (const org of snapshot.organisms) {
    const key = `${org.x},${org.y}`;
    countsByCell.set(key, (countsByCell.get(key) ?? 0) + 1);
  }

  const drawnInCell = new Map<string, number>();
  for (const org of snapshot.organisms) {
    const id = unwrapId(org.id);
    const key = `${org.x},${org.y}`;
    const idx = drawnInCell.get(key) ?? 0;
    drawnInCell.set(key, idx + 1);

    const total = countsByCell.get(key) ?? 1;
    const baseX = org.x * cell + cell / 2;
    const baseY = (rows - 1 - org.y) * cell + cell / 2;
    const offset = total > 1 ? ((idx - (total - 1) / 2) * cell) / 5 : 0;

    ctx.beginPath();
    ctx.arc(baseX + offset, baseY + offset, Math.max(2, cell * 0.18), 0, Math.PI * 2);
    ctx.fillStyle = id === focusedOrganismId ? '#0d3f73' : '#1dd679';
    ctx.fill();
  }
}

function renderBrain() {
  const ctx = brainCtx;
  const width = brainCanvas.width;
  const height = brainCanvas.height;
  ctx.clearRect(0, 0, width, height);

  if (!focusedBrain) {
    ctx.fillStyle = '#1b2638';
    ctx.font = '15px IBM Plex Sans';
    ctx.fillText('No focused organism', 16, 30);
    return;
  }

  const sensory = focusedBrain.sensory;
  const inter = focusedBrain.inter;
  const action = focusedBrain.action;

  const layers: Array<Array<{ id: number; label: string; type: string; potential: number }>> = [];
  layers.push(
    sensory.map((n) => ({
      id: unwrapId(n.neuron.neuron_id),
      label: n.receptor_type,
      type: 'sensory',
      potential: n.neuron.potential,
    })),
  );

  const interColumns = Math.max(1, Math.ceil(inter.length / 8));
  for (let c = 0; c < interColumns; c += 1) {
    const start = c * 8;
    const slice = inter.slice(start, start + 8);
    layers.push(
      slice.map((n) => ({
        id: unwrapId(n.neuron.neuron_id),
        label: `I${unwrapId(n.neuron.neuron_id)}`,
        type: 'inter',
        potential: n.neuron.potential,
      })),
    );
  }

  layers.push(
    action.map((n) => ({
      id: unwrapId(n.neuron.neuron_id),
      label: n.action_type,
      type: 'action',
      potential: n.neuron.potential,
    })),
  );

  const xGap = width / Math.max(2, layers.length);
  const positions = new Map<number, { x: number; y: number; type: string; potential: number; label: string }>();

  layers.forEach((layer, layerIndex) => {
    const yGap = height / (layer.length + 1);
    layer.forEach((node, idx) => {
      const x = 20 + layerIndex * xGap;
      const y = (idx + 1) * yGap;
      positions.set(node.id, { x, y, type: node.type, potential: node.potential, label: node.label });
    });
  });

  ctx.lineWidth = 1;
  for (const neuron of sensory) {
    const pre = positions.get(unwrapId(neuron.neuron.neuron_id));
    if (!pre) continue;
    for (const syn of neuron.synapses) {
      const post = positions.get(unwrapId(syn.post_neuron_id));
      if (!post) continue;
      ctx.strokeStyle = syn.weight >= 0 ? 'rgba(17,103,177,0.6)' : 'rgba(177,28,59,0.7)';
      ctx.lineWidth = Math.max(0.5, Math.abs(syn.weight) / 8 * 2);
      ctx.beginPath();
      ctx.moveTo(pre.x, pre.y);
      ctx.lineTo(post.x, post.y);
      ctx.stroke();
    }
  }

  for (const neuron of inter) {
    const pre = positions.get(unwrapId(neuron.neuron.neuron_id));
    if (!pre) continue;
    for (const syn of neuron.synapses) {
      const post = positions.get(unwrapId(syn.post_neuron_id));
      if (!post) continue;
      ctx.strokeStyle = syn.weight >= 0 ? 'rgba(17,103,177,0.55)' : 'rgba(177,28,59,0.65)';
      ctx.lineWidth = Math.max(0.5, Math.abs(syn.weight) / 8 * 2);
      ctx.beginPath();
      ctx.moveTo(pre.x, pre.y);
      ctx.lineTo(post.x, post.y);
      ctx.stroke();
    }
  }

  for (const [id, node] of positions) {
    ctx.beginPath();
    ctx.fillStyle = node.type === 'sensory' ? '#1167b1' : node.type === 'inter' ? '#1f9aa8' : '#16a34a';
    ctx.arc(node.x, node.y, 10, 0, Math.PI * 2);
    ctx.fill();

    ctx.fillStyle = '#10233f';
    ctx.font = '11px IBM Plex Sans';
    ctx.fillText(node.label, node.x + 12, node.y + 4);
    ctx.fillStyle = '#43556f';
    ctx.fillText(node.potential.toFixed(1), node.x + 12, node.y + 16);

    if (focusedOrganismId !== null && id === focusedOrganismId) {
      // no-op marker reservation for future id highlighting
    }
  }
}

function setupCanvasClick() {
  worldCanvas.addEventListener('click', async (evt) => {
    if (!snapshot || !session) return;

    const rect = worldCanvas.getBoundingClientRect();
    const xPx = ((evt.clientX - rect.left) / rect.width) * worldCanvas.width;
    const yPx = ((evt.clientY - rect.top) / rect.height) * worldCanvas.height;

    const cell = Math.min(
      worldCanvas.width / snapshot.config.columns,
      worldCanvas.height / snapshot.config.rows,
    );
    const gridX = Math.floor(xPx / cell);
    const gridYFromTop = Math.floor(yPx / cell);
    const gridY = snapshot.config.rows - 1 - gridYFromTop;

    const candidates = snapshot.organisms.filter((org) => org.x === gridX && org.y === gridY);
    if (candidates.length === 0) return;

    const chosen = candidates[0];
    focusedOrganismId = unwrapId(chosen.id);
    focusedBrain = chosen.brain;
    focusMetaEl.textContent = `focused organism: ${focusedOrganismId} at (${gridX}, ${gridY})`;
    renderBrain();

    await request(`/v1/sessions/${session.id}/focus`, 'POST', {
      organism_id: focusedOrganismId,
    });
  });
}

function setupControls() {
  createBtn.addEventListener('click', () => {
    void createSession();
  });

  resetBtn.addEventListener('click', () => {
    if (!session) return;
    void request<WorldSnapshot>(`/v1/sessions/${session.id}/reset`, 'POST', { seed: null }).then(
      (next) => {
        snapshot = next;
        focusedOrganismId = null;
        focusedBrain = null;
        renderBrain();
        updateMetrics();
      },
    );
  });

  startBtn.addEventListener('click', () => {
    sendCommand({ type: 'Start', data: { ticks_per_second: 10 } });
  });

  pauseBtn.addEventListener('click', () => {
    sendCommand({ type: 'Pause' });
  });

  stepBtn.addEventListener('click', () => {
    sendCommand({ type: 'Step', data: { count: 1 } });
  });

  epochBtn.addEventListener('click', () => {
    sendCommand({ type: 'Epoch', data: { count: 1 } });
  });

  scatterBtn.addEventListener('click', () => {
    if (!session) return;
    void request<WorldSnapshot>(`/v1/sessions/${session.id}/scatter`, 'POST').then((next) => {
      snapshot = next;
      updateMetrics();
    });
  });

  survivorsBtn.addEventListener('click', () => {
    if (!session) return;
    void request<WorldSnapshot>(`/v1/sessions/${session.id}/survivors/process`, 'POST').then(
      (next) => {
        snapshot = next;
        updateMetrics();
      },
    );
  });
}

function renderLoop() {
  renderWorld();
  requestAnimationFrame(renderLoop);
}

setupControls();
setupCanvasClick();
renderLoop();
void createSession();
