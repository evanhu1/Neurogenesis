// Typed client for the file-based sim-server. Every world is addressed by name;
// mutating calls advance the `<name>.bin` file on disk. `openStream` wraps the
// live-animation WebSocket.

import type {
  ApiChampionPoolResponse,
  ApiOrganismDetail,
  ApiStreamFrame,
  ApiWorldResponse,
  ApiWorldSnapshot,
  EcoView,
  FindResult,
  GenomeView,
  LineageView,
  PillarsView,
  TimeseriesData,
} from '../../../types';
import { apiBase, wsBase } from '../constants';

async function request<T>(path: string, method: string, body?: unknown): Promise<T> {
  const response = await fetch(`${apiBase}${path}`, {
    method,
    headers: { 'Content-Type': 'application/json' },
    body: body === undefined ? undefined : JSON.stringify(body),
  });

  const text = await response.text();
  let parsed: unknown = null;
  if (text) {
    try {
      parsed = JSON.parse(text);
    } catch {
      parsed = null;
    }
  }

  if (!response.ok) {
    let message: string | null = null;
    const record = parsed && typeof parsed === 'object' ? (parsed as Record<string, unknown>) : null;
    if (record && typeof record.message === 'string') message = record.message;
    if (!message && text.trim()) message = text.trim();
    const error = new Error(message ?? `request failed (${response.status})`) as Error & {
      status?: number;
    };
    error.status = response.status;
    throw error;
  }

  if (parsed !== null) return parsed as T;
  throw new Error('request succeeded but response was not JSON');
}

export type NewWorldParams = {
  name?: string;
  seed?: number;
  config?: string;
  set?: string[];
  scale?: [number, number];
  threads?: number;
  report_every?: number;
};

export const worldClient = {
  listWorlds: () => request<string[]>('/worlds', 'GET'),
  createWorld: (params: NewWorldParams) => request<ApiWorldResponse>('/worlds', 'POST', params),
  getSnapshot: (name: string) =>
    request<ApiWorldSnapshot>(`/worlds/${encodeURIComponent(name)}/snapshot`, 'GET'),
  step: (name: string, count: number) =>
    request<ApiWorldResponse>(`/worlds/${encodeURIComponent(name)}/step`, 'POST', { count }),
  runTo: (name: string, turn: number) =>
    request<ApiWorldResponse>(`/worlds/${encodeURIComponent(name)}/run-to`, 'POST', { turn }),
  getOrganism: (name: string, id: number) =>
    request<ApiOrganismDetail>(
      `/worlds/${encodeURIComponent(name)}/organism/${id}`,
      'GET',
    ),

  // --- Research reads (CLI-parity), surfaced in the cockpit panels ----------
  getPillars: (name: string) =>
    request<PillarsView>(`/worlds/${encodeURIComponent(name)}/pillars`, 'GET'),
  getEco: (name: string) => request<EcoView>(`/worlds/${encodeURIComponent(name)}/eco`, 'GET'),
  getLineage: (name: string) =>
    request<LineageView>(`/worlds/${encodeURIComponent(name)}/lineage`, 'GET'),
  getGenome: (name: string) =>
    request<GenomeView>(`/worlds/${encodeURIComponent(name)}/genome`, 'GET'),
  getTimeseries: (name: string, cols?: string[], last?: number) => {
    const params = new URLSearchParams();
    if (cols && cols.length > 0) params.set('cols', cols.join(','));
    if (last != null) params.set('last', String(last));
    const query = params.toString();
    return request<TimeseriesData>(
      `/worlds/${encodeURIComponent(name)}/timeseries${query ? `?${query}` : ''}`,
      'GET',
    );
  },
  find: (name: string, expr: string, limit?: number, fields?: string[]) => {
    const params = new URLSearchParams({ expr });
    if (limit != null) params.set('limit', String(limit));
    if (fields && fields.length > 0) params.set('fields', fields.join(','));
    return request<FindResult>(
      `/worlds/${encodeURIComponent(name)}/find?${params.toString()}`,
      'GET',
    );
  },
  getChampions: () => request<ApiChampionPoolResponse>('/champions', 'GET'),
  saveChampions: (name: string) =>
    request<ApiChampionPoolResponse>(`/worlds/${encodeURIComponent(name)}/champions`, 'POST'),
  deleteChampion: (index: number) =>
    request<ApiChampionPoolResponse>(`/champions/${index}`, 'DELETE'),
  clearChampions: () => request<ApiChampionPoolResponse>('/champions', 'DELETE'),
};

export type StreamHandlers = {
  onFrame: (frame: ApiStreamFrame) => void;
  onClose: () => void;
};

/// Open the live-animation WebSocket for `name` at `tps` ticks/second (0 = as
/// fast as possible). Returns the socket; close it to pause (the server persists
/// the world on disconnect).
export function openStream(name: string, tps: number, handlers: StreamHandlers): WebSocket {
  const socket = new WebSocket(
    `${wsBase}/worlds/${encodeURIComponent(name)}/stream?tps=${tps}`,
  );
  socket.onmessage = (evt) => {
    try {
      handlers.onFrame(JSON.parse(String(evt.data)) as ApiStreamFrame);
    } catch (err) {
      console.error('stream frame parse error', err);
    }
  };
  socket.onclose = () => handlers.onClose();
  return socket;
}
