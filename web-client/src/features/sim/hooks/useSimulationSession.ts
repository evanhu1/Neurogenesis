import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  applyTickDelta,
  normalizeCreateSessionResponse,
  normalizeFocusBrainData,
  normalizeTickDelta,
  normalizeWorldSnapshot,
} from '../../../protocol';
import type {
  ApiCreateSessionResponse,
  ApiServerEvent,
  ApiWorldSnapshot,
  ArchivedWorldSummary,
  BatchRunStatusResponse,
  OrganismState,
  SessionMetadata,
  StepProgressData,
  WorldOrganismState,
  WorldSnapshot,
} from '../../../types';
import { createSimHttpClient } from '../api/simHttpClient';
import { apiBase } from '../constants';
import { clearPersistedSessionId, loadPersistedSessionId, persistSessionId } from '../storage';
import { useSimulationArchive } from './useSimulationArchive';
import { useSimulationConnection } from './useSimulationConnection';
import { useSimulationControls } from './useSimulationControls';
import { NO_FOCUS_TURN, useSimulationFocus } from './useSimulationFocus';

export type SpeciesPopulationPoint = {
  turn: number;
  speciesCounts: Record<string, number>;
};

const MAX_SPECIES_HISTORY_POINTS = 2048;
const FOCUS_POLL_INTERVAL_MS = 100;

function normalizeSpeciesCounts(speciesCounts: Record<string, number>): Record<string, number> {
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
  batchRunStatus: BatchRunStatusResponse | null;
  archivedWorlds: ArchivedWorldSummary[];
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
  resetSession: (seedInput?: string) => void;
  toggleRun: () => void;
  setSpeedLevelIndex: (levelIndex: number) => void;
  step: (count: number) => void;
  focusOrganism: (organism: WorldOrganismState) => void;
  defocusOrganism: () => void;
  saveCurrentWorld: () => Promise<void>;
  deleteArchivedWorld: (worldId: string) => Promise<void>;
  deleteAllArchivedWorlds: () => Promise<void>;
  startBatchRun: (worldCount: number, ticksPerWorld: number) => Promise<void>;
  loadArchivedWorld: (worldId: string) => Promise<void>;
  refreshArchivedWorlds: () => Promise<void>;
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
  const [errorText, setErrorText] = useState<string | null>(null);
  const latestSnapshotTurnRef = useRef<number>(NO_FOCUS_TURN);
  const snapshotRef = useRef<WorldSnapshot | null>(snapshot);
  const request = useMemo(() => createSimHttpClient(apiBase), []);
  const sendCommandRef = useRef<(command: unknown) => boolean>(() => false);
  snapshotRef.current = snapshot;
  const sendCommand = useCallback((command: unknown): boolean => {
    return sendCommandRef.current(command);
  }, []);

  const controls = useSimulationControls(sendCommand);
  const focus = useSimulationFocus({ snapshot, session, request, setErrorText });

  const handleServerEvent = useCallback(
    (event: ApiServerEvent) => {
      switch (event.type) {
        case 'StateSnapshot': {
          const nextSnapshot = normalizeWorldSnapshot(event.data);
          latestSnapshotTurnRef.current = nextSnapshot.turn;
          snapshotRef.current = nextSnapshot;
          controls.clearPendingStep();
          setSnapshot(nextSnapshot);
          setSpeciesPopulationHistory((previous) =>
            upsertSpeciesPopulationHistory(previous, {
              turn: nextSnapshot.turn,
              speciesCounts: normalizeSpeciesCounts(nextSnapshot.metrics.species_counts),
            }),
          );
          const trackedFocusedId = focus.focusedOrganismIdRef.current;
          if (trackedFocusedId !== null) {
            sendCommand({ type: 'SetFocus', data: { organism_id: trackedFocusedId } });
          }
          break;
        }
        case 'TickDelta': {
          const delta = normalizeTickDelta(event.data);
          latestSnapshotTurnRef.current = delta.turn;
          controls.clearPendingStep();
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
              speciesCounts: normalizeSpeciesCounts(nextSnapshot.metrics.species_counts),
            }),
          );

          const trackedFocusedId = focus.focusedOrganismIdRef.current;
          if (trackedFocusedId !== null) {
            const now = Date.now();
            if (now >= focus.nextFocusPollAtMsRef.current) {
              focus.nextFocusPollAtMsRef.current = now + FOCUS_POLL_INTERVAL_MS;
              sendCommand({ type: 'SetFocus', data: { organism_id: trackedFocusedId } });
            }
          }
          break;
        }
        case 'StepProgress': {
          const progress = parseStepProgressData(event.data);
          if (progress) {
            controls.handleStepProgress(progress);
          }
          break;
        }
        case 'FocusBrain': {
          focus.handleFocusBrain(normalizeFocusBrainData(event.data), latestSnapshotTurnRef.current);
          break;
        }
        case 'Metrics': {
          break;
        }
        case 'Error': {
          controls.clearPendingStep();
          const message = event.data.message || 'Simulation server reported an error';
          setErrorText(message);
          break;
        }
        default:
          break;
      }
    },
    [controls, focus, sendCommand],
  );

  const connection = useSimulationConnection({
    onServerEvent: handleServerEvent,
    onSocketClose: controls.handleSocketClose,
  });
  sendCommandRef.current = connection.sendCommand;

  const applyLoadedSession = useCallback(
    (metadata: SessionMetadata, loadedSnapshot: WorldSnapshot) => {
      setErrorText(null);
      setSession(metadata);
      latestSnapshotTurnRef.current = loadedSnapshot.turn;
      snapshotRef.current = loadedSnapshot;
      setSnapshot(loadedSnapshot);
      focus.resetFocusState(true);
      controls.syncSessionState(metadata.running, metadata.ticks_per_second);
      setSpeciesPopulationHistory([
        {
          turn: loadedSnapshot.turn,
          speciesCounts: normalizeSpeciesCounts(loadedSnapshot.metrics.species_counts),
        },
      ]);
      persistSessionId(metadata.id);
      connection.connectWs(metadata.id);
    },
    [connection, controls, focus],
  );

  const archive = useSimulationArchive({
    request,
    session,
    applyLoadedSession,
    setErrorText,
  });

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
      } catch (err) {
        setErrorText(err instanceof Error ? err.message : 'Failed to create session');
      }
    },
    [applyLoadedSession, request],
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

  const resetSession = useCallback(
    (seedInput?: string) => {
      if (!session) return;
      const { seed, error } = parseSessionSeedInput(seedInput);
      if (error) {
        setErrorText(error);
        return;
      }
      setErrorText(null);
      if (controls.isRunning) {
        sendCommand({ type: 'Pause' });
      }
      void request<ApiWorldSnapshot>(`/v1/sessions/${session.id}/reset`, 'POST', { seed })
        .then((nextSnapshot) => {
          const normalized = normalizeWorldSnapshot(nextSnapshot);
          latestSnapshotTurnRef.current = normalized.turn;
          snapshotRef.current = normalized;
          controls.syncSessionState(false, 0);
          setSnapshot(normalized);
          setSpeciesPopulationHistory([
            {
              turn: normalized.turn,
              speciesCounts: normalizeSpeciesCounts(normalized.metrics.species_counts),
            },
          ]);
          focus.resetFocusState(true);
        })
        .catch((err) => {
          setErrorText(err instanceof Error ? err.message : 'Failed to reset session');
        });
    },
    [controls, focus, request, sendCommand, session],
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

  return {
    session,
    snapshot,
    batchRunStatus: archive.batchRunStatus,
    archivedWorlds: archive.archivedWorlds,
    speciesPopulationHistory,
    focusedOrganismId: focus.focusedOrganismId,
    focusedOrganism: focus.focusedOrganism,
    activeActionNeuronId: focus.activeActionNeuronId,
    isRunning: controls.isRunning,
    isStepPending: controls.isStepPending,
    stepProgress: controls.stepProgress,
    speedLevels: controls.speedLevels,
    speedLevelIndex: controls.speedLevelIndex,
    errorText,
    createSession,
    resetSession,
    toggleRun: controls.toggleRun,
    setSpeedLevelIndex: controls.setSpeedLevelIndex,
    step: controls.step,
    focusOrganism: focus.focusOrganism,
    defocusOrganism: focus.defocusOrganism,
    saveCurrentWorld: archive.saveCurrentWorld,
    deleteArchivedWorld: archive.deleteArchivedWorld,
    deleteAllArchivedWorlds: archive.deleteAllArchivedWorlds,
    startBatchRun: archive.startBatchRun,
    loadArchivedWorld: archive.loadArchivedWorld,
    refreshArchivedWorlds: archive.refreshArchivedWorlds,
  };
}
