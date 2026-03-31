import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  applyTickDelta,
  normalizeChampionPoolResponse,
  normalizeCreateSessionResponse,
  normalizeFocusBrainData,
  normalizeTickDelta,
  normalizeWorldSnapshot,
} from '../../../protocol';
import type {
  ApiChampionPoolResponse,
  ApiCreateSessionResponse,
  ApiServerEvent,
  ApiWorldSnapshot,
  ChampionPoolEntry,
  OrganismState,
  SessionMetadata,
  StepProgressData,
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

function cloneSpeciesCounts(speciesCounts: Record<string, number>): Record<string, number> {
  return { ...speciesCounts };
}

function speciesCountsEqual(a: Record<string, number>, b: Record<string, number>): boolean {
  const keysA = Object.keys(a);
  const keysB = Object.keys(b);
  if (keysA.length !== keysB.length) return false;
  for (const key of keysA) {
    if (a[key] !== b[key]) return false;
  }
  return true;
}

function parseStepProgressData(data: unknown): StepProgressData | null {
  if (!data || typeof data !== 'object') return null;
  const candidate = data as Partial<StepProgressData>;
  const requested = Number.isFinite(candidate.requested_count)
    ? Math.max(1, Math.floor(candidate.requested_count as number))
    : null;
  const completed = Number.isFinite(candidate.completed_count)
    ? Math.max(0, Math.floor(candidate.completed_count as number))
    : null;
  if (requested === null || completed === null) return null;
  return {
    requested_count: requested,
    completed_count: Math.min(completed, requested),
  };
}

function upsertSpeciesPopulationHistory(
  previous: SpeciesPopulationPoint[],
  point: SpeciesPopulationPoint,
): SpeciesPopulationPoint[] {
  const latest = previous[previous.length - 1];
  if (!latest) {
    return [point];
  }

  if (point.turn < latest.turn) {
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
  errorText: string | null;
  createSession: (seedInput?: string) => Promise<void>;
  saveChampions: () => Promise<void>;
  toggleRun: () => void;
  setSpeedLevelIndex: (levelIndex: number) => void;
  step: (count: number) => void;
  focusOrganism: (organism: WorldOrganismState) => void;
  defocusOrganism: () => void;
  refreshChampionPool: () => Promise<void>;
  deleteChampionPoolEntry: (index: number) => Promise<void>;
  clearChampionPool: () => Promise<void>;
};

function parseSessionSeedInput(seedInput?: string): { seed: number | null; error: string | null } {
  if (seedInput === undefined) {
    return { seed: null, error: null };
  }
  const trimmed = seedInput.trim();
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
  const [speciesPopulationHistory, setSpeciesPopulationHistory] = useState<
    SpeciesPopulationPoint[]
  >([]);
  const [championPool, setChampionPool] = useState<ChampionPoolEntry[]>([]);
  const [errorText, setErrorText] = useState<string | null>(null);
  const latestSnapshotTurnRef = useRef<number>(NO_FOCUS_TURN);
  const snapshotRef = useRef<WorldSnapshot | null>(snapshot);
  const request = useMemo(() => createSimHttpClient(apiBase), []);
  const sendCommandRef = useRef<(command: unknown) => boolean>(() => false);
  snapshotRef.current = snapshot;
  const sendCommand = useCallback((command: unknown): boolean => {
    return sendCommandRef.current(command);
  }, []);

  const {
    isRunning,
    isStepPending,
    stepProgress,
    speedLevels,
    speedLevelIndex,
    clearPendingStep,
    handleSocketClose,
    handleStepProgress,
    syncSessionState,
    toggleRun,
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

  const handleServerEvent = useCallback(
    (event: ApiServerEvent) => {
      switch (event.type) {
        case 'StateSnapshot': {
          const nextSnapshot = normalizeWorldSnapshot(event.data);
          if (nextSnapshot.turn <= latestSnapshotTurnRef.current) {
            break;
          }
          latestSnapshotTurnRef.current = nextSnapshot.turn;
          snapshotRef.current = nextSnapshot;
          clearPendingStep();
          setSnapshot(nextSnapshot);
          setSpeciesPopulationHistory((previous) =>
            upsertSpeciesPopulationHistory(previous, {
              turn: nextSnapshot.turn,
              speciesCounts: cloneSpeciesCounts(nextSnapshot.metrics.species_counts),
            }),
          );
          const trackedFocusedId = focusedOrganismIdRef.current;
          if (trackedFocusedId !== null) {
            sendCommand({ type: 'SetFocus', data: { organism_id: trackedFocusedId } });
          }
          break;
        }
        case 'TickDelta': {
          const delta = normalizeTickDelta(event.data);
          latestSnapshotTurnRef.current = delta.turn;
          clearPendingStep();
          const previousSnapshot = snapshotRef.current;
          if (!previousSnapshot) {
            break;
          }
          const nextSnapshot = applyTickDelta(previousSnapshot, delta);
          snapshotRef.current = nextSnapshot;
          setSnapshot(nextSnapshot);
          setSpeciesPopulationHistory((previous) =>
            upsertSpeciesPopulationHistory(previous, {
              turn: nextSnapshot.turn,
              speciesCounts: cloneSpeciesCounts(nextSnapshot.metrics.species_counts),
            }),
          );

          const trackedFocusedId = focusedOrganismIdRef.current;
          if (trackedFocusedId !== null) {
            const now = Date.now();
            if (now >= nextFocusPollAtMsRef.current) {
              nextFocusPollAtMsRef.current = now + FOCUS_POLL_INTERVAL_MS;
              sendCommand({ type: 'SetFocus', data: { organism_id: trackedFocusedId } });
            }
          }
          break;
        }
        case 'StepProgress': {
          const progress = parseStepProgressData(event.data);
          if (progress) {
            handleStepProgress(progress);
          }
          break;
        }
        case 'FocusBrain': {
          handleFocusBrain(normalizeFocusBrainData(event.data), latestSnapshotTurnRef.current);
          break;
        }
        case 'Metrics': {
          break;
        }
        case 'Error': {
          clearPendingStep();
          const message = event.data.message || 'Simulation server reported an error';
          setErrorText(message);
          break;
        }
        default:
          break;
      }
    },
    [
      clearPendingStep,
      focusedOrganismIdRef,
      handleFocusBrain,
      handleStepProgress,
      nextFocusPollAtMsRef,
      sendCommand,
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
      latestSnapshotTurnRef.current = loadedSnapshot.turn;
      snapshotRef.current = loadedSnapshot;
      setSnapshot(loadedSnapshot);
      resetFocusState(true);
      syncSessionState(metadata.running, metadata.ticks_per_second);
      setSpeciesPopulationHistory([
        {
          turn: loadedSnapshot.turn,
          speciesCounts: cloneSpeciesCounts(loadedSnapshot.metrics.species_counts),
        },
      ]);
      persistSessionId(metadata.id);
      connectWs(metadata.id);
    },
    [connectWs, resetFocusState, syncSessionState],
  );

  const refreshChampionPool = useCallback(async () => {
    const payload = await request<ApiChampionPoolResponse>('/v1/champion-pool', 'GET');
    setChampionPool(normalizeChampionPoolResponse(payload).entries);
  }, [request]);

  const deleteChampionPoolEntry = useCallback(
    async (index: number) => {
      try {
        setErrorText(null);
        await request<ApiChampionPoolResponse>(`/v1/champion-pool/${index}`, 'DELETE');
        await refreshChampionPool();
      } catch (err) {
        captureError(setErrorText, err, 'Failed to delete champion genome');
      }
    },
    [refreshChampionPool, request],
  );

  const clearChampionPool = useCallback(async () => {
    try {
      setErrorText(null);
      await request<ApiChampionPoolResponse>('/v1/champion-pool', 'DELETE');
      await refreshChampionPool();
    } catch (err) {
      captureError(setErrorText, err, 'Failed to clear champion pool');
    }
  }, [refreshChampionPool, request]);

  const createSession = useCallback(
    async (seedInput?: string) => {
      const { seed, error } = parseSessionSeedInput(seedInput);
      if (error) {
        setErrorText(error);
        return;
      }
      const sessionSeed = seed ?? Math.floor(Date.now() / 1000);
      try {
        setErrorText(null);
        const payload = await request<ApiCreateSessionResponse>('/v1/sessions', 'POST', {
          seed: sessionSeed,
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

  const saveChampions = useCallback(async () => {
    if (!session) return;
    try {
      setErrorText(null);
      await request<ApiChampionPoolResponse>(`/v1/sessions/${session.id}/champions`, 'POST');
      await refreshChampionPool();
    } catch (err) {
      captureError(setErrorText, err, 'Failed to save champions');
    }
  }, [refreshChampionPool, request, session]);

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

  useEffect(() => {
    void refreshChampionPool().catch((err: unknown) => {
      captureError(setErrorText, err, 'Failed to load champion pool');
    });
  }, [refreshChampionPool]);

  useEffect(() => {
    const intervalId = window.setInterval(() => {
      void refreshChampionPool().catch((err: unknown) => {
        captureError(setErrorText, err, 'Failed to refresh champion pool');
      });
    }, 5_000);
    return () => {
      window.clearInterval(intervalId);
    };
  }, [refreshChampionPool]);

  return {
    session,
    snapshot,
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
    errorText,
    createSession,
    saveChampions,
    toggleRun,
    setSpeedLevelIndex,
    step,
    focusOrganism,
    defocusOrganism,
    refreshChampionPool,
    deleteChampionPoolEntry,
    clearChampionPool,
  };
}
