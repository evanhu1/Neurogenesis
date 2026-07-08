import { useCallback, useEffect, useRef, useState } from 'react';
import { api, type Champions, type PopulationStats, type RenderSnapshot } from './api';

export interface PopulationPoint {
  turn: number;
  alive: number;
  food: number;
}

// Central data hook: polls the REST server for the world snapshot + stats +
// champions, tracks the selected organism, exposes play/pause/step controls, and
// accumulates a short population-over-time history for the sparkline (the old
// server streamed deltas; option (2) keeps it REST-poll only).
export function useWorld() {
  const [snapshot, setSnapshot] = useState<RenderSnapshot | null>(null);
  const [stats, setStats] = useState<PopulationStats | null>(null);
  const [champions, setChampions] = useState<Champions | null>(null);
  const [selectedId, setSelectedId] = useState<number | null>(null);
  const [running, setRunning] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const history = useRef<PopulationPoint[]>([]);
  const [historyTick, setHistoryTick] = useState(0);

  const poll = useCallback(async () => {
    try {
      const [snap, st] = await Promise.all([api.snapshot(), api.state()]);
      setSnapshot(snap);
      setStats(st);
      setError(null);
      const h = history.current;
      const last = h[h.length - 1];
      if (!last || last.turn !== st.turn) {
        h.push({ turn: st.turn, alive: st.alive, food: snap.food.length });
        if (h.length > 400) h.shift();
        setHistoryTick((t) => t + 1);
      }
    } catch (e) {
      setError(String(e));
    }
  }, []);

  useEffect(() => {
    poll();
    const h = setInterval(poll, 200);
    return () => clearInterval(h);
  }, [poll]);

  useEffect(() => {
    const h = setInterval(() => api.champions().then(setChampions).catch(() => {}), 1500);
    return () => clearInterval(h);
  }, []);

  const control = useCallback(async (cmd: 'play' | 'pause' | 'step') => {
    try {
      await api.control(cmd);
      if (cmd === 'play') setRunning(true);
      if (cmd === 'pause') setRunning(false);
      if (cmd === 'step') poll();
    } catch {
      /* ignore transient control errors */
    }
  }, [poll]);

  return {
    snapshot,
    stats,
    champions,
    selectedId,
    setSelectedId,
    running,
    error,
    control,
    history: history.current,
    historyTick,
  };
}

export type World = ReturnType<typeof useWorld>;
