// API layer for the new-substrate sim-server (lean REST + JSON). Types mirror
// the Rust wire shapes (sim-hexworld RenderSnapshot / PopulationStats and
// sim-substrate Genome / BrainNet).

const BASE = import.meta.env.VITE_API_BASE ?? 'http://127.0.0.1:8080';

export interface RenderOrganism {
  id: number;
  q: number;
  r: number;
  energy: number;
  health: number;
  color: [number, number, number];
}

export interface RenderSnapshot {
  turn: number;
  width: number;
  organisms: RenderOrganism[];
  food: [number, number][];
  walls: [number, number][];
  spikes: [number, number][];
  extinct: boolean;
}

export interface PopulationStats {
  turn: number;
  alive: number;
  total_ever: number;
  extinct_at: number | null;
  mean_energy: number;
  mean_neurons: number;
  mean_edges: number;
  mean_generation: number;
  max_generation: number;
}

// --- genome / brain (structural; enough to summarize + draw) ---
export interface CppnNode {
  id: number;
  kind: string;
  activation: string;
  bias: number;
}
export interface CppnConn {
  innovation: number;
  from: number;
  to: number;
  weight: number;
  enabled: boolean;
}
export interface Genome {
  cppn: { nodes: CppnNode[]; conns: CppnConn[] };
  header: {
    plasticity: Record<string, number>;
    lifecycle: Record<string, number>;
    mutation_rates: Record<string, number>;
    morphology: number[];
  };
}
export interface BrainNeuron {
  kind: 'Input' | 'Hidden' | 'Output';
  bias: number;
  alpha: number;
}
export interface BrainEdge {
  from: number;
  to: number;
  weight: number;
  plasticity_scale: number;
}
export interface BrainNet {
  neurons: BrainNeuron[];
  edges: BrainEdge[];
  input_count: number;
  hidden_count: number;
  output_count: number;
}
export interface OrganismDetail {
  id: number;
  energy: number;
  health: number;
  age_turns: number;
  generation: number;
  morphology: number[];
  genome: Genome;
  brain: BrainNet;
}

export interface ChampionEntry {
  quality: number;
  descriptor: { values: number[] };
  genome: Genome;
}
export interface Champions {
  schema_version: number;
  coverage: number;
  qd_score: number;
  entries: ChampionEntry[];
}

async function getJson<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE}${path}`);
  if (!res.ok) throw new Error(`${path} -> ${res.status}`);
  return (await res.json()) as T;
}
async function post(path: string): Promise<unknown> {
  const res = await fetch(`${BASE}${path}`, { method: 'POST' });
  if (!res.ok) throw new Error(`${path} -> ${res.status}`);
  return res.json();
}

export const api = {
  snapshot: () => getJson<RenderSnapshot>('/api/snapshot'),
  state: () => getJson<PopulationStats>('/api/state'),
  organism: (id: number) => getJson<OrganismDetail>(`/api/organism/${id}`),
  champions: () => getJson<Champions>('/api/champions'),
  saveChampion: (id: number) => post(`/api/champions/${id}`),
  control: (cmd: 'play' | 'pause' | 'step') => post(`/api/control/${cmd}`),
};
