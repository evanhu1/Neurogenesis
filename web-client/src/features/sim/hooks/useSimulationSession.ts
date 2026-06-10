import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  applyTickDelta,
  normalizeChampionPoolResponse,
  normalizeCreateSessionResponse,
  normalizeFocusBrainData,
  normalizeLiveMetricsData,
  normalizeTickDelta,
  normalizeWorldSnapshot,
} from '../../../protocol';
import type {
  ApiChampionPoolResponse,
  ApiCreateSessionResponse,
  ApiServerEvent,
  ApiWorldSnapshot,
  ChampionPoolEntry,
  LiveMetricsData,
  OrganismState,
  SessionMetadata,
  StepProgressData,
  StreamMode,
  WorldOrganismState,
  WorldSnapshot,
} from '../../../types';
import { createSimHttpClient } from '../api/simHttpClient';
import { apiBase } from '../constants';
import { clearPersistedSessionId, loadPersistedSessionId, persistSessionId } from '../storage';
import { captureError } from './captureError';
import { useSimulationConnection } from './useSimulationConnection';
import { useSimulationControls } from './useSimulationControls';
import { NO_FOCUS_TURN, useSimulationFocus } from './useSimulationFocus';

export type SpeciesPopulationPoint = {
  turn: number;
  speciesCounts: Record<string, number>;
};

const MAX_SPECIES_HISTORY_POINTS = 2048;
const FOCUS_POLL_INTERVAL_MS = 100;
const CHAMPION_POOL_REFRESH_MS = 5_000;

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
  if (!latest || point.turn < latest.turn) {
    return [point];
  }

  if (point.turn === latest.turn) {
    if (speciesCountsEqual(latest.speciesCounts, point.speciesCounts)) {
      return previous;
    }
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

export type SimulationSessionState = {
  session: SessionMetadata | null;
  snapshot: WorldSnapshot | null;
  liveMetrics: LiveMetricsData | null;
  championPool: ChampionPoolEntry[];
  speciesPopulationHistory: SpeciesPopulationPoint[];
  focusedOrganismId: number | null;
  focusedOrganism: OrganismState | null;
  activeActionNeuronId: number | null;
  isRunning: boolean;
  isStepPending: boolean;
  stepProgress: StepProgressData | null;
  speedLevels: readonly number[];
  speedLevelIndex: number;
  streamMode: StreamMode;
  isFastMode: boolean;
  errorText: string | null;
  createSession: (seedInput?: string) => Promise<void>;
  saveChampions: () => Promise<void>;
  toggleRun: () => void;
  toggleFastRun: () => void;
  setSpeedLevelIndex: (levelIndex: number) => void;
  step: (count: number) => void;
  focusOrganism: (organism: WorldOrganismState) => void;
  defocusOrganism: () => void;
  deleteChampionPoolEntry: (index: number) => Promise<void>;
  clearChampionPool: () => Promise<void>;
};

function parseSessionSeedInput(seedInput?: string): { seed: number | null; error: string | null } {
  const trimmed = seedInput?.trim();
  if (!trimmed) {
    return { seed: null, error: null };
  }
  if (!/^\d+$/.test(trimmed)) {
    return { seed: null, error: 'Seed must be a non-negative integer' };
  }
  const parsed = Number.parseInt(trimmed, 10);
  if (!Number.isSafeInteger(parsed)) {
    return {
      seed: null,
      error: `Seed must be <= ${Number.MAX_SAFE_INTEGER.toLocaleString()} in the web client`,
    };
  }
  return { seed: parsed, error: null };
}

export function useSimulationSession(): SimulationSessionState {
  const [session, setSession] = useState<SessionMetadata | null>(null);
  const [snapshot, setSnapshot] = useState<WorldSnapshot | null>(null);
  const [liveMetrics, setLiveMetrics] = useState<LiveMetricsData | null>(null);
  const [speciesPopulationHistory, setSpeciesPopulationHistory] = useState<
    SpeciesPopulationPoint[]
  >([]);
  const [championPool, setChampionPool] = useState<ChampionPoolEntry[]>([]);
  const [errorText, setErrorText] = useState<string | null>(null);
  const latestSnapshotTurnRef = useRef<number>(NO_FOCUS_TURN);
  const snapshotRef = useRef<WorldSnapshot | null>(snapshot);
  const liveMetricsRef = useRef<LiveMetricsData | null>(liveMetrics);
  const request = useMemo(() => createSimHttpClient(apiBase), []);
  const sendCommandRef = useRef<(command: unknown) => boolean>(() => false);
  const sendCommand = useCallback((command: unknown): boolean => {
    return sendCommandRef.current(command);
  }, []);

  const {
    isRunning,
    isStepPending,
    stepProgress,
    speedLevels,
    speedLevelIndex,
    streamMode,
    clearPendingStep,
    handleSocketClose,
    handleStepProgress,
    syncSessionState,
    toggleRun,
    toggleFastRun,
    setSpeedLevelIndex,
    step,
  } = useSimulationControls(sendCommand);
  const {
    focusedOrganismId,
    focusedOrganism,
    activeActionNeuronId,
    focusedOrganismIdRef,
    nextFocusPollAtMsRef,
    handleFocusBrain,
    focusOrganism,
    defocusOrganism,
    resetFocusState,
  } = useSimulationFocus({ snapshot, session, request, setErrorText });

  const recordSpeciesHistory = useCallback(
    (turn: number, speciesCounts: Record<string, number>) => {
      setSpeciesPopulationHistory((previous) =>
        upsertSpeciesPopulationHistory(previous, { turn, speciesCounts: { ...speciesCounts } }),
      );
    },
    [],
  );

  const commitSnapshot = useCallback(
    (nextSnapshot: WorldSnapshot) => {
      const nextMetrics = { turn: nextSnapshot.turn, metrics: nextSnapshot.metrics };
      latestSnapshotTurnRef.current = nextSnapshot.turn;
      snapshotRef.current = nextSnapshot;
      liveMetricsRef.current = nextMetrics;
      setSnapshot(nextSnapshot);
      setLiveMetrics(nextMetrics);
      recordSpeciesHistory(nextSnapshot.turn, nextSnapshot.metrics.species_counts);
    },
    [recordSpeciesHistory],
  );

  const refreshFocusedBrain = useCallback(
    (throttled: boolean) => {
      const trackedFocusedId = focusedOrganismIdRef.current;
      if (trackedFocusedId === null) return;
      if (throttled) {
        const now = Date.now();
        if (now < nextFocusPollAtMsRef.current) return;
        nextFocusPollAtMsRef.current = now + FOCUS_POLL_INTERVAL_MS;
      }
      sendCommand({ type: 'SetFocus', data: { organism_id: trackedFocusedId } });
    },
    [focusedOrganismIdRef, nextFocusPollAtMsRef, sendCommand],
  );

  const handleServerEvent = useCallback(
    (event: ApiServerEvent) => {
      switch (event.type) {
        case 'StateSnapshot': {
          const nextSnapshot = normalizeWorldSnapshot(
            event.data,
            liveMetricsRef.current?.metrics.total_species_created ?? 0,
          );
          if (nextSnapshot.turn <= latestSnapshotTurnRef.current) {
            break;
          }
          clearPendingStep();
          commitSnapshot(nextSnapshot);
          refreshFocusedBrain(false);
          break;
        }
        case 'TickDelta': {
          const previousSnapshot = snapshotRef.current;
          latestSnapshotTurnRef.current = event.data.turn;
          clearPendingStep();
          if (!previousSnapshot) {
            break;
          }
          commitSnapshot(applyTickDelta(previousSnapshot, normalizeTickDelta(event.data)));
          refreshFocusedBrain(true);
          break;
        }
        case 'StepProgress': {
          handleStepProgress(event.data);
          break;
        }
        case 'FocusBrain': {
          handleFocusBrain(normalizeFocusBrainData(event.data), latestSnapshotTurnRef.current);
          break;
        }
        case 'Metrics': {
          clearPendingStep();
          const nextMetrics = normalizeLiveMetricsData(
            event.data,
            liveMetricsRef.current?.metrics.total_species_created ?? 0,
          );
          liveMetricsRef.current = nextMetrics;
          latestSnapshotTurnRef.current = Math.max(latestSnapshotTurnRef.current, nextMetrics.turn);
          setLiveMetrics(nextMetrics);
          recordSpeciesHistory(nextMetrics.turn, nextMetrics.metrics.species_counts);
          break;
        }
        case 'Error': {
          clearPendingStep();
          setErrorText(event.data.message || 'Simulation server reported an error');
          break;
        }
      }
    },
    [
      clearPendingStep,
      commitSnapshot,
      handleFocusBrain,
      handleStepProgress,
      recordSpeciesHistory,
      refreshFocusedBrain,
    ],
  );

  const { connectWs, sendCommand: sendSocketCommand } = useSimulationConnection({
    onServerEvent: handleServerEvent,
    onSocketClose: handleSocketClose,
  });
  sendCommandRef.current = sendSocketCommand;

  const applyLoadedSession = useCallback(
    (metadata: SessionMetadata, loadedSnapshot: WorldSnapshot) => {
      setErrorText(null);
      setSession(metadata);
      setSpeciesPopulationHistory([]);
      commitSnapshot(loadedSnapshot);
      resetFocusState();
      syncSessionState(metadata.running, metadata.ticks_per_second, metadata.stream_mode);
      persistSessionId(metadata.id);
      connectWs(metadata.id);
    },
    [commitSnapshot, connectWs, resetFocusState, syncSessionState],
  );

  const refreshChampionPool = useCallback(async () => {
    const payload = await request<ApiChampionPoolResponse>('/v1/champion-pool', 'GET');
    setChampionPool(normalizeChampionPoolResponse(payload).entries);
  }, [request]);

  const runChampionPoolAction = useCallback(
    async (action: () => Promise<unknown>, failureMessage: string) => {
      try {
        setErrorText(null);
        await action();
        await refreshChampionPool();
      } catch (err) {
        captureError(setErrorText, err, failureMessage);
      }
    },
    [refreshChampionPool],
  );

  const deleteChampionPoolEntry = useCallback(
    (index: number) =>
      runChampionPoolAction(
        () => request(`/v1/champion-pool/${index}`, 'DELETE'),
        'Failed to delete champion genome',
      ),
    [request, runChampionPoolAction],
  );

  const clearChampionPool = useCallback(
    () =>
      runChampionPoolAction(
        () => request('/v1/champion-pool', 'DELETE'),
        'Failed to clear champion pool',
      ),
    [request, runChampionPoolAction],
  );

  const saveChampions = useCallback(async () => {
    if (!session) return;
    await runChampionPoolAction(
      () => request(`/v1/sessions/${session.id}/champions`, 'POST'),
      'Failed to save champions',
    );
  }, [request, runChampionPoolAction, session]);

  const createSession = useCallback(
    async (seedInput?: string) => {
      const { seed, error } = parseSessionSeedInput(seedInput);
      if (error) {
        setErrorText(error);
        return;
      }
      try {
        setErrorText(null);
        const payload = await request<ApiCreateSessionResponse>('/v1/sessions', 'POST', {
          seed: seed ?? Math.floor(Date.now() / 1000),
        });
        const normalized = normalizeCreateSessionResponse(payload);
        applyLoadedSession(normalized.metadata, normalized.snapshot);
        await refreshChampionPool();
      } catch (err) {
        captureError(setErrorText, err, 'Failed to create session');
      }
    },
    [applyLoadedSession, refreshChampionPool, request],
  );

  const restoreSession = useCallback(
    async (sessionId: string): Promise<boolean> => {
      try {
        const [metadata, restoredSnapshot] = await Promise.all([
          request<SessionMetadata>(`/v1/sessions/${sessionId}`, 'GET'),
          request<ApiWorldSnapshot>(`/v1/sessions/${sessionId}/state`, 'GET'),
        ]);
        applyLoadedSession(metadata, normalizeWorldSnapshot(restoredSnapshot));
        return true;
      } catch {
        clearPersistedSessionId();
        return false;
      }
    },
    [applyLoadedSession, request],
  );

  useEffect(() => {
    let cancelled = false;

    const restoreOrCreateSession = async () => {
      const persistedSessionId = loadPersistedSessionId();
      if (persistedSessionId) {
        const restored = await restoreSession(persistedSessionId);
        if (restored || cancelled) return;
      }
      if (!cancelled) {
        await createSession();
      }
    };

    void restoreOrCreateSession();
    return () => {
      cancelled = true;
    };
  }, [createSession, restoreSession]);

  // Metrics-only mode streams no snapshots or focus data: drop focus on entry
  // (the inspector would freeze on stale state), and re-fetch the world on exit
  // (the canvas would otherwise keep showing the pre-fast-run world).
  const isFastMode = isRunning && streamMode === 'metrics_only';
  const wasFastModeRef = useRef(false);
  useEffect(() => {
    if (isFastMode) {
      wasFastModeRef.current = true;
      resetFocusState();
      return;
    }
    if (!wasFastModeRef.current || !session) return;
    wasFastModeRef.current = false;
    void request<ApiWorldSnapshot>(`/v1/sessions/${session.id}/state`, 'GET')
      .then((payload) => {
        const nextSnapshot = normalizeWorldSnapshot(
          payload,
          liveMetricsRef.current?.metrics.total_species_created ?? 0,
        );
        if (nextSnapshot.turn >= latestSnapshotTurnRef.current) {
          commitSnapshot(nextSnapshot);
        }
      })
      .catch((err: unknown) => {
        captureError(setErrorText, err, 'Failed to refresh world after fast run');
      });
  }, [commitSnapshot, isFastMode, request, resetFocusState, session]);

  useEffect(() => {
    const refresh = () =>
      void refreshChampionPool().catch((err: unknown) => {
        captureError(setErrorText, err, 'Failed to load champion pool');
      });
    refresh();
    const intervalId = window.setInterval(refresh, CHAMPION_POOL_REFRESH_MS);
    return () => {
      window.clearInterval(intervalId);
    };
  }, [refreshChampionPool]);

  return {
    session,
    snapshot,
    liveMetrics,
    championPool,
    speciesPopulationHistory,
    focusedOrganismId,
    focusedOrganism,
    activeActionNeuronId,
    isRunning,
    isStepPending,
    stepProgress,
    speedLevels,
    speedLevelIndex,
    streamMode,
    isFastMode,
    errorText,
    createSession,
    saveChampions,
    toggleRun,
    toggleFastRun,
    setSpeedLevelIndex,
    step,
    focusOrganism,
    defocusOrganism,
    deleteChampionPoolEntry,
    clearChampionPool,
  };
}
