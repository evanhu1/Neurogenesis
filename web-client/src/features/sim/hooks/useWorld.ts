import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { normalizeChampionPoolResponse, normalizeOrganismDetail, normalizeTickDelta, normalizeWorldSnapshot } from '../../../protocol';
import { WorldRenderer } from '../../../rendering/WorldRenderer';
import type {
  ChampionPoolEntry,
  LiveMetricsData,
  OrganismState,
  WorldOrganismState,
  WorldSnapshot,
} from '../../../types';
import { openStream, worldClient } from '../api/worldClient';
import { SPEED_LEVELS } from '../constants';
import { clearPersistedWorldName, loadPersistedWorldName, persistWorldName } from '../storage';
import { captureError } from './captureError';

const DEFAULT_SPEED_LEVEL_INDEX = 1;
const MAX_SPECIES_HISTORY_POINTS = 2048;
const OVERVIEW_SAMPLE_INTERVAL_MS = 200;
const CHAMPION_POOL_REFRESH_MS = 5_000;

export type SpeciesPopulationPoint = {
  turn: number;
  speciesCounts: Record<string, number>;
};

function speciesCountsEqual(a: Record<string, number>, b: Record<string, number>): boolean {
  const keysA = Object.keys(a);
  if (keysA.length !== Object.keys(b).length) return false;
  return keysA.every((key) => a[key] === b[key]);
}

function upsertSpeciesPopulationHistory(
  previous: SpeciesPopulationPoint[],
  point: SpeciesPopulationPoint,
): SpeciesPopulationPoint[] {
  const latest = previous[previous.length - 1];
  if (!latest || point.turn < latest.turn) return [point];
  if (point.turn === latest.turn) {
    if (speciesCountsEqual(latest.speciesCounts, point.speciesCounts)) return previous;
    const next = previous.slice();
    next[next.length - 1] = point;
    return next;
  }
  const trimmed =
    previous.length >= MAX_SPECIES_HISTORY_POINTS
      ? previous.slice(previous.length - MAX_SPECIES_HISTORY_POINTS + 1)
      : previous;
  return trimmed.concat(point);
}

function parseSeedInput(seedInput?: string): { seed: number | null; error: string | null } {
  const trimmed = seedInput?.trim();
  if (!trimmed) return { seed: null, error: null };
  if (!/^\d+$/.test(trimmed)) return { seed: null, error: 'Seed must be a non-negative integer' };
  const parsed = Number.parseInt(trimmed, 10);
  if (!Number.isSafeInteger(parsed)) {
    return { seed: null, error: `Seed must be <= ${Number.MAX_SAFE_INTEGER.toLocaleString()}` };
  }
  return { seed: parsed, error: null };
}

export type WorldController = {
  worldName: string | null;
  snapshot: WorldSnapshot | null;
  liveMetrics: LiveMetricsData | null;
  championPool: ChampionPoolEntry[];
  speciesPopulationHistory: SpeciesPopulationPoint[];
  focusedOrganismId: number | null;
  focusedOrganism: OrganismState | null;
  activeActionNeuronId: number | null;
  isRunning: boolean;
  isStepPending: boolean;
  speedLevels: readonly number[];
  speedLevelIndex: number;
  /** The turn of the world's last persisted state (create / step / pause). */
  savedTurn: number | null;
  /** Bumps whenever the world file is saved, so cockpit reads know to refetch. */
  revision: number;
  errorText: string | null;
  registerRenderer: (renderer: WorldRenderer | null) => void;
  createWorld: (seedInput?: string) => Promise<void>;
  saveChampions: () => Promise<void>;
  toggleRun: () => void;
  setSpeedLevelIndex: (levelIndex: number) => void;
  step: (count: number) => void;
  focusOrganism: (organism: WorldOrganismState) => void;
  focusOrganismById: (id: number) => void;
  defocusOrganism: () => void;
  deleteChampionPoolEntry: (index: number) => Promise<void>;
  clearChampionPool: () => Promise<void>;
};

export function useWorld(): WorldController {
  const [snapshot, setSnapshot] = useState<WorldSnapshot | null>(null);
  const [liveMetrics, setLiveMetrics] = useState<LiveMetricsData | null>(null);
  const [championPool, setChampionPool] = useState<ChampionPoolEntry[]>([]);
  const [speciesPopulationHistory, setSpeciesPopulationHistory] = useState<SpeciesPopulationPoint[]>(
    [],
  );
  const [focusedOrganismId, setFocusedOrganismId] = useState<number | null>(null);
  const [focusedOrganism, setFocusedOrganism] = useState<OrganismState | null>(null);
  const [activeActionNeuronId, setActiveActionNeuronId] = useState<number | null>(null);
  const [isRunning, setIsRunning] = useState(false);
  const [isStepPending, setIsStepPending] = useState(false);
  const [speedLevelIndex, setSpeedLevelIndexState] = useState(DEFAULT_SPEED_LEVEL_INDEX);
  const [errorText, setErrorText] = useState<string | null>(null);
  const [worldName, setWorldName] = useState<string | null>(null);
  const [savedTurn, setSavedTurn] = useState<number | null>(null);
  const [revision, setRevision] = useState(0);

  const rendererRef = useRef<WorldRenderer | null>(null);
  const latestSnapshotRef = useRef<WorldSnapshot | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const worldNameRef = useRef<string | null>(null);
  const isRunningRef = useRef(false);
  const speedIndexRef = useRef(DEFAULT_SPEED_LEVEL_INDEX);
  const focusedOrganismIdRef = useRef<number | null>(null);
  const initStartedRef = useRef(false);

  const setError = useCallback((err: unknown, fallback: string) => {
    captureError(setErrorText, err, fallback);
  }, []);

  const recordSpeciesHistory = useCallback((snap: WorldSnapshot) => {
    setSpeciesPopulationHistory((previous) =>
      upsertSpeciesPopulationHistory(previous, {
        turn: snap.turn,
        speciesCounts: { ...snap.metrics.species_counts },
      }),
    );
  }, []);

  const pushOverviewFrom = useCallback(
    (snap: WorldSnapshot) => {
      setSnapshot(snap);
      setLiveMetrics({ turn: snap.turn, metrics: snap.metrics });
      recordSpeciesHistory(snap);
    },
    [recordSpeciesHistory],
  );

  // Mark the world file as persisted at `turn` so the cockpit reads refetch.
  // Fires only on saves (create / step / pause), never per stream tick.
  const markSaved = useCallback((turn: number) => {
    setSavedTurn(turn);
    setRevision((r) => r + 1);
  }, []);

  // Apply a full snapshot to the renderer + React overview state (static feed:
  // load / create / step / run-to). These all correspond to a server-side save.
  const commitSnapshot = useCallback(
    (snap: WorldSnapshot) => {
      latestSnapshotRef.current = snap;
      rendererRef.current?.setWorld(snap);
      pushOverviewFrom(snap);
      markSaved(snap.turn);
    },
    [markSaved, pushOverviewFrom],
  );

  // Read the renderer's in-memory snapshot into React overview state (during a
  // live run the renderer, not React, holds the per-tick world).
  const syncFromRenderer = useCallback(() => {
    const snap = rendererRef.current?.getSnapshot() ?? null;
    if (!snap) return;
    latestSnapshotRef.current = snap;
    pushOverviewFrom(snap);
    // Defocus if the focused organism has died out of the world.
    const focusId = focusedOrganismIdRef.current;
    if (focusId != null && !snap.organisms.some((o) => o.id === focusId)) {
      focusedOrganismIdRef.current = null;
      rendererRef.current?.setFocusedOrganismId(null);
      setFocusedOrganismId(null);
      setFocusedOrganism(null);
      setActiveActionNeuronId(null);
    }
  }, [pushOverviewFrom]);

  // Fetch the focused organism's full detail (brain + genome) from the file.
  // KNOWN LIMITATION: during a live run these reads hit the on-disk world, not
  // the resident stream sim, so the inspector's brain reflects the last
  // persisted state and only refreshes on select / step / pause — the canvas
  // still animates live via deltas.
  const refreshFocusedDetail = useCallback(async () => {
    const name = worldNameRef.current;
    const id = focusedOrganismIdRef.current;
    if (!name || id == null) return;
    try {
      const detail = normalizeOrganismDetail(await worldClient.getOrganism(name, id));
      if (focusedOrganismIdRef.current !== id) return;
      setFocusedOrganism(detail.organism);
      setActiveActionNeuronId(detail.active_action_neuron_id);
    } catch {
      // Organism gone (died) or transient failure: drop focus quietly.
      if (focusedOrganismIdRef.current === id) {
        focusedOrganismIdRef.current = null;
        rendererRef.current?.setFocusedOrganismId(null);
        setFocusedOrganismId(null);
        setFocusedOrganism(null);
        setActiveActionNeuronId(null);
      }
    }
  }, []);

  const refreshChampionPool = useCallback(async () => {
    try {
      const pool = normalizeChampionPoolResponse(await worldClient.getChampions());
      setChampionPool(pool.entries);
    } catch (err) {
      setError(err, 'Failed to load champion pool');
    }
  }, [setError]);

  const registerRenderer = useCallback((renderer: WorldRenderer | null) => {
    rendererRef.current = renderer;
    if (renderer && latestSnapshotRef.current) {
      renderer.setWorld(latestSnapshotRef.current);
      renderer.setFocusedOrganismId(focusedOrganismIdRef.current);
    }
  }, []);

  // --- Streaming ------------------------------------------------------------

  const startStream = useCallback(
    (name: string, tps: number) => {
      const socket = openStream(name, tps, {
        onFrame: (frame) => {
          const renderer = rendererRef.current;
          if (!renderer) return;
          if (frame.type === 'StateSnapshot') {
            const snap = normalizeWorldSnapshot(frame.data);
            latestSnapshotRef.current = snap;
            renderer.setWorld(snap);
          } else {
            renderer.applyDelta(normalizeTickDelta(frame.data));
            latestSnapshotRef.current = renderer.getSnapshot();
          }
        },
        onClose: () => {
          if (wsRef.current !== socket) return; // superseded by a speed-change reopen
          wsRef.current = null;
          isRunningRef.current = false;
          setIsRunning(false);
          syncFromRenderer();
          void refreshFocusedDetail();
        },
      });
      wsRef.current = socket;
      isRunningRef.current = true;
      setIsRunning(true);
    },
    [refreshFocusedDetail, syncFromRenderer],
  );

  const toggleRun = useCallback(() => {
    const name = worldNameRef.current;
    if (!name) return;
    if (isRunningRef.current) {
      wsRef.current?.close();
    } else {
      startStream(name, SPEED_LEVELS[speedIndexRef.current]);
    }
  }, [startStream]);

  const setSpeedLevelIndex = useCallback(
    (levelIndex: number) => {
      const next = Math.max(0, Math.min(SPEED_LEVELS.length - 1, levelIndex));
      speedIndexRef.current = next;
      setSpeedLevelIndexState(next);
      // Re-open the stream at the new rate if we're mid-run.
      const name = worldNameRef.current;
      if (isRunningRef.current && name) {
        const previous = wsRef.current;
        startStream(name, SPEED_LEVELS[next]);
        previous?.close(); // its onClose no-ops: wsRef now points at the new socket
      }
    },
    [startStream],
  );

  // --- World lifecycle ------------------------------------------------------

  const adoptWorld = useCallback(
    (name: string, snap: WorldSnapshot) => {
      worldNameRef.current = name;
      setWorldName(name);
      persistWorldName(name);
      // reset focus + history for the new world
      focusedOrganismIdRef.current = null;
      setFocusedOrganismId(null);
      setFocusedOrganism(null);
      setActiveActionNeuronId(null);
      setSpeciesPopulationHistory([]);
      commitSnapshot(snap);
    },
    [commitSnapshot],
  );

  const createWorld = useCallback(
    async (seedInput?: string) => {
      const { seed, error } = parseSeedInput(seedInput);
      if (error) {
        setErrorText(error);
        return;
      }
      wsRef.current?.close();
      try {
        setErrorText(null);
        const resp = await worldClient.createWorld({
          seed: seed ?? Math.floor(Date.now() / 1000),
        });
        adoptWorld(resp.name, normalizeWorldSnapshot(resp.snapshot));
        await refreshChampionPool();
      } catch (err) {
        setError(err, 'Failed to create world');
      }
    },
    [adoptWorld, refreshChampionPool, setError],
  );

  const step = useCallback(
    (count: number) => {
      const name = worldNameRef.current;
      const requested = Math.floor(count);
      if (!name || isRunningRef.current || isStepPending || !Number.isFinite(requested)) return;
      setIsStepPending(true);
      void (async () => {
        try {
          const resp = await worldClient.step(name, Math.max(1, requested));
          commitSnapshot(normalizeWorldSnapshot(resp.snapshot));
          await refreshFocusedDetail();
        } catch (err) {
          setError(err, 'Failed to step world');
        } finally {
          setIsStepPending(false);
        }
      })();
    },
    [commitSnapshot, isStepPending, refreshFocusedDetail, setError],
  );

  // --- Focus ----------------------------------------------------------------

  const focusOrganism = useCallback(
    (organism: WorldOrganismState) => {
      focusedOrganismIdRef.current = organism.id;
      setFocusedOrganismId(organism.id);
      rendererRef.current?.setFocusedOrganismId(organism.id);
      void refreshFocusedDetail();
    },
    [refreshFocusedDetail],
  );

  // Focus an organism by id (e.g. clicking a `find`-result row) and pan to it.
  const focusOrganismById = useCallback(
    (id: number) => {
      focusedOrganismIdRef.current = id;
      setFocusedOrganismId(id);
      rendererRef.current?.setFocusedOrganismId(id);
      rendererRef.current?.panToOrganism(id);
      void refreshFocusedDetail();
    },
    [refreshFocusedDetail],
  );

  const defocusOrganism = useCallback(() => {
    focusedOrganismIdRef.current = null;
    rendererRef.current?.setFocusedOrganismId(null);
    setFocusedOrganismId(null);
    setFocusedOrganism(null);
    setActiveActionNeuronId(null);
  }, []);

  // --- Champions ------------------------------------------------------------

  const saveChampions = useCallback(async () => {
    const name = worldNameRef.current;
    if (!name) return;
    try {
      setErrorText(null);
      const pool = normalizeChampionPoolResponse(await worldClient.saveChampions(name));
      setChampionPool(pool.entries);
    } catch (err) {
      setError(err, 'Failed to save champions');
    }
  }, [setError]);

  const deleteChampionPoolEntry = useCallback(
    async (index: number) => {
      try {
        const pool = normalizeChampionPoolResponse(await worldClient.deleteChampion(index));
        setChampionPool(pool.entries);
      } catch (err) {
        setError(err, 'Failed to delete champion');
      }
    },
    [setError],
  );

  const clearChampionPool = useCallback(async () => {
    try {
      const pool = normalizeChampionPoolResponse(await worldClient.clearChampions());
      setChampionPool(pool.entries);
    } catch (err) {
      setError(err, 'Failed to clear champion pool');
    }
  }, [setError]);

  // --- Effects --------------------------------------------------------------

  // Load a persisted world (or create a fresh one) on mount.
  useEffect(() => {
    // Run once, even under React 18 StrictMode's double-invoke, so we don't race
    // two `createWorld` calls and leave a stray world file on disk.
    if (initStartedRef.current) return;
    initStartedRef.current = true;
    void (async () => {
      const persisted = loadPersistedWorldName();
      if (persisted) {
        try {
          const view = await worldClient.getSnapshot(persisted);
          adoptWorld(persisted, normalizeWorldSnapshot(view));
          await refreshChampionPool();
          return;
        } catch {
          clearPersistedWorldName();
        }
      }
      await createWorld();
    })();
    return () => {
      wsRef.current?.close();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Sample the renderer's live world into overview/chart state while running.
  useEffect(() => {
    if (!isRunning) return;
    const id = window.setInterval(syncFromRenderer, OVERVIEW_SAMPLE_INTERVAL_MS);
    return () => window.clearInterval(id);
  }, [isRunning, syncFromRenderer]);

  // Periodic champion-pool refresh.
  useEffect(() => {
    const id = window.setInterval(() => void refreshChampionPool(), CHAMPION_POOL_REFRESH_MS);
    return () => window.clearInterval(id);
  }, [refreshChampionPool]);

  return useMemo(
    () => ({
      worldName,
      snapshot,
      liveMetrics,
      championPool,
      speciesPopulationHistory,
      focusedOrganismId,
      focusedOrganism,
      activeActionNeuronId,
      isRunning,
      isStepPending,
      speedLevels: SPEED_LEVELS,
      speedLevelIndex,
      savedTurn,
      revision,
      errorText,
      registerRenderer,
      createWorld,
      saveChampions,
      toggleRun,
      setSpeedLevelIndex,
      step,
      focusOrganism,
      focusOrganismById,
      defocusOrganism,
      deleteChampionPoolEntry,
      clearChampionPool,
    }),
    [
      worldName,
      snapshot,
      liveMetrics,
      championPool,
      speciesPopulationHistory,
      focusedOrganismId,
      focusedOrganism,
      activeActionNeuronId,
      isRunning,
      isStepPending,
      speedLevelIndex,
      savedTurn,
      revision,
      errorText,
      registerRenderer,
      createWorld,
      saveChampions,
      toggleRun,
      setSpeedLevelIndex,
      step,
      focusOrganism,
      focusOrganismById,
      defocusOrganism,
      deleteChampionPoolEntry,
      clearChampionPool,
    ],
  );
}
