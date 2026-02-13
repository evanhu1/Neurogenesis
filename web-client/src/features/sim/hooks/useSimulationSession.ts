import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { applyTickDelta, findOrganism, unwrapId } from '../../../protocol';
import type {
  ArchivedWorldSummary,
  BatchRunStatusResponse,
  CreateBatchRunResponse,
  CreateSessionResponse,
  FocusBrainData,
  ListArchivedWorldsResponse,
  OrganismState,
  ServerEvent,
  SessionMetadata,
  StepProgressData,
  TickDelta,
  WorldOrganismState,
  WorldSnapshot,
} from '../../../types';
import { connectSimulationWs, sendSimulationCommand } from '../api/simWsClient';
import { createSimHttpClient } from '../api/simHttpClient';
import { apiBase, SPEED_LEVELS, wsBase } from '../constants';
import { clearPersistedSessionId, loadPersistedSessionId, persistSessionId } from '../storage';

export type SpeciesPopulationPoint = {
  turn: number;
  speciesCounts: Record<string, number>;
};

const MAX_SPECIES_HISTORY_POINTS = 2048;
const FOCUS_POLL_INTERVAL_MS = 100;
const BATCH_RUN_POLL_INTERVAL_MS = 500;

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

function nearestSpeedLevelIndex(ticksPerSecond: number): number {
  if (!Number.isFinite(ticksPerSecond) || ticksPerSecond <= 0) return 0;
  let bestIndex = 0;
  let bestDistance = Math.abs(SPEED_LEVELS[0] - ticksPerSecond);
  for (let i = 1; i < SPEED_LEVELS.length; i += 1) {
    const distance = Math.abs(SPEED_LEVELS[i] - ticksPerSecond);
    if (distance < bestDistance) {
      bestDistance = distance;
      bestIndex = i;
    }
  }
  return bestIndex;
}

function syncFocusedOrganismFromWorld(
  focused: OrganismState | null,
  worldOrganism: WorldOrganismState,
): OrganismState | null {
  if (!focused) return focused;
  if (unwrapId(focused.id) !== unwrapId(worldOrganism.id)) return focused;
  return {
    ...focused,
    species_id: worldOrganism.species_id,
    q: worldOrganism.q,
    r: worldOrganism.r,
    age_turns: worldOrganism.age_turns,
    facing: worldOrganism.facing,
    energy: worldOrganism.energy,
    consumptions_count: worldOrganism.consumptions_count,
    reproductions_count: worldOrganism.reproductions_count,
  };
}

export type SimulationSessionState = {
  session: SessionMetadata | null;
  snapshot: WorldSnapshot | null;
  batchRunStatus: BatchRunStatusResponse | null;
  archivedWorlds: ArchivedWorldSummary[];
  speciesPopulationHistory: SpeciesPopulationPoint[];
  focusedOrganismId: number | null;
  focusedOrganism: OrganismState | null;
  activeNeuronIds: Set<number> | null;
  isRunning: boolean;
  isStepPending: boolean;
  stepProgress: StepProgressData | null;
  speedLevels: readonly number[];
  speedLevelIndex: number;
  errorText: string | null;
  createSession: () => Promise<void>;
  resetSession: () => void;
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

function nextRandomUniverseSeed(): number {
  if (typeof crypto !== 'undefined' && typeof crypto.getRandomValues === 'function') {
    const value = new Uint32Array(1);
    crypto.getRandomValues(value);
    return value[0];
  }
  return Math.floor(Math.random() * 0x1_0000_0000);
}

export function useSimulationSession(): SimulationSessionState {
  const [session, setSession] = useState<SessionMetadata | null>(null);
  const [snapshot, setSnapshot] = useState<WorldSnapshot | null>(null);
  const [activeBatchRunId, setActiveBatchRunId] = useState<string | null>(null);
  const [batchRunStatus, setBatchRunStatus] = useState<BatchRunStatusResponse | null>(null);
  const [archivedWorlds, setArchivedWorlds] = useState<ArchivedWorldSummary[]>([]);
  const [speciesPopulationHistory, setSpeciesPopulationHistory] = useState<
    SpeciesPopulationPoint[]
  >([]);
  const [focusedOrganismId, setFocusedOrganismId] = useState<number | null>(null);
  const [focusedOrganism, setFocusedOrganism] = useState<OrganismState | null>(null);
  const [activeNeuronIds, setActiveNeuronIds] = useState<Set<number> | null>(null);
  const [isRunning, setIsRunning] = useState(false);
  const [isStepPending, setIsStepPending] = useState(false);
  const [stepProgress, setStepProgress] = useState<StepProgressData | null>(null);
  const [speedLevelIndex, setSpeedLevelIndex] = useState(1);
  const [errorText, setErrorText] = useState<string | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const focusedOrganismIdRef = useRef<number | null>(null);
  const nextFocusPollAtMsRef = useRef(0);
  const request = useMemo(() => createSimHttpClient(apiBase), []);
  const setFocusedOrganismIdTracked = useCallback((organismId: number | null) => {
    focusedOrganismIdRef.current = organismId;
    nextFocusPollAtMsRef.current = 0;
    setFocusedOrganismId(organismId);
  }, []);

  const handleServerEvent = useCallback((event: ServerEvent) => {
    switch (event.type) {
      case 'StateSnapshot': {
        const nextSnapshot = event.data as WorldSnapshot;
        setIsStepPending(false);
        setStepProgress(null);
        setSnapshot(nextSnapshot);
        setSpeciesPopulationHistory((previous) =>
          upsertSpeciesPopulationHistory(previous, {
            turn: nextSnapshot.turn,
            speciesCounts: normalizeSpeciesCounts(nextSnapshot.metrics.species_counts),
          }),
        );
        const trackedFocusedId = focusedOrganismIdRef.current;
        if (trackedFocusedId !== null) {
          sendSimulationCommand(wsRef.current, {
            type: 'SetFocus',
            data: { organism_id: trackedFocusedId },
          });
        }
        break;
      }
      case 'TickDelta': {
        const delta = event.data as TickDelta;
        setIsStepPending(false);
        setStepProgress(null);
        setSpeciesPopulationHistory((previous) =>
          upsertSpeciesPopulationHistory(previous, {
            turn: delta.turn,
            speciesCounts: normalizeSpeciesCounts(delta.metrics.species_counts),
          }),
        );

        setSnapshot((prev) => {
          if (!prev) return prev;
          return applyTickDelta(prev, delta);
        });

        const trackedFocusedId = focusedOrganismIdRef.current;
        if (trackedFocusedId !== null) {
          const now = Date.now();
          if (now >= nextFocusPollAtMsRef.current) {
            nextFocusPollAtMsRef.current = now + FOCUS_POLL_INTERVAL_MS;
            sendSimulationCommand(wsRef.current, {
              type: 'SetFocus',
              data: { organism_id: trackedFocusedId },
            });
          }
        }
        break;
      }
      case 'StepProgress': {
        const progress = parseStepProgressData(event.data);
        if (!progress) {
          break;
        }
        setStepProgress(progress);
        setIsStepPending(progress.completed_count < progress.requested_count);
        break;
      }
      case 'FocusBrain': {
        const { organism, active_neuron_ids } = event.data as FocusBrainData;
        const organismId = unwrapId(organism.id);
        setFocusedOrganismIdTracked(organismId);
        setFocusedOrganism(organism);
        setActiveNeuronIds(new Set(active_neuron_ids.map((id) => unwrapId(id))));
        break;
      }
      case 'Metrics': {
        break;
      }
      case 'Error': {
        setIsStepPending(false);
        setStepProgress(null);
        const message =
          typeof event.data === 'string' ? event.data : 'Simulation server reported an error';
        setErrorText(message);
        break;
      }
      default:
        break;
    }
  }, [setFocusedOrganismIdTracked]);

  const connectWs = useCallback(
    (sessionId: string) => {
      wsRef.current?.close();
      let nextSocket: WebSocket;
      nextSocket = connectSimulationWs(
        wsBase,
        sessionId,
        handleServerEvent,
        () => {
          setIsRunning(false);
          setStepProgress(null);
          if (wsRef.current === nextSocket) {
            wsRef.current = null;
          }
        },
      );
      wsRef.current = nextSocket;
    },
    [handleServerEvent],
  );

  const applyLoadedSession = useCallback(
    (metadata: SessionMetadata, loadedSnapshot: WorldSnapshot) => {
      setErrorText(null);
      setSession(metadata);
      setSnapshot(loadedSnapshot);
      setFocusedOrganismIdTracked(null);
      setFocusedOrganism(null);
      setActiveNeuronIds(null);
      setIsRunning(metadata.running);
      setIsStepPending(false);
      setStepProgress(null);
      setSpeedLevelIndex(nearestSpeedLevelIndex(metadata.ticks_per_second));
      setSpeciesPopulationHistory([
        {
          turn: loadedSnapshot.turn,
          speciesCounts: normalizeSpeciesCounts(loadedSnapshot.metrics.species_counts),
        },
      ]);
      persistSessionId(metadata.id);
      connectWs(metadata.id);
    },
    [connectWs, setFocusedOrganismIdTracked],
  );

  const createSession = useCallback(async () => {
    try {
      setErrorText(null);
      const payload = await request<CreateSessionResponse>('/v1/sessions', 'POST', {
        seed: Math.floor(Date.now() / 1000),
      });
      applyLoadedSession(payload.metadata, payload.snapshot);
    } catch (err) {
      setErrorText(err instanceof Error ? err.message : 'Failed to create session');
    }
  }, [applyLoadedSession, request]);

  const refreshArchivedWorlds = useCallback(async () => {
    const payload = await request<ListArchivedWorldsResponse>('/v1/worlds', 'GET');
    setArchivedWorlds(payload.worlds);
  }, [request]);

  const loadArchivedWorld = useCallback(
    async (worldId: string) => {
      try {
        setErrorText(null);
        const payload = await request<CreateSessionResponse>(`/v1/worlds/${worldId}/sessions`, 'POST');
        applyLoadedSession(payload.metadata, payload.snapshot);
      } catch (err) {
        setErrorText(err instanceof Error ? err.message : 'Failed to load archived world');
      }
    },
    [applyLoadedSession, request],
  );

  const saveCurrentWorld = useCallback(async () => {
    if (!session) return;
    try {
      setErrorText(null);
      await request<ArchivedWorldSummary>(`/v1/sessions/${session.id}/archive`, 'POST');
      await refreshArchivedWorlds();
    } catch (err) {
      setErrorText(err instanceof Error ? err.message : 'Failed to save current world');
    }
  }, [refreshArchivedWorlds, request, session]);

  const deleteArchivedWorld = useCallback(
    async (worldId: string) => {
      try {
        setErrorText(null);
        await request<ArchivedWorldSummary>(`/v1/worlds/${worldId}`, 'DELETE');
        await refreshArchivedWorlds();
        setBatchRunStatus((current) => {
          if (!current) return current;
          return {
            ...current,
            worlds: current.worlds.filter((world) => world.world_id !== worldId),
          };
        });
      } catch (err) {
        setErrorText(err instanceof Error ? err.message : 'Failed to delete archived world');
      }
    },
    [refreshArchivedWorlds, request],
  );

  const deleteAllArchivedWorlds = useCallback(async () => {
    if (archivedWorlds.length === 0) return;
    setErrorText(null);

    const worldIds = archivedWorlds.map((world) => world.world_id);
    const deletedWorldIds = new Set<string>();
    const deleteResults = await Promise.allSettled(
      worldIds.map(async (worldId) => {
        await request<ArchivedWorldSummary>(`/v1/worlds/${worldId}`, 'DELETE');
        deletedWorldIds.add(worldId);
      }),
    );

    await refreshArchivedWorlds();
    setBatchRunStatus((current) => {
      if (!current || deletedWorldIds.size === 0) return current;
      return {
        ...current,
        worlds: current.worlds.filter((world) => !deletedWorldIds.has(world.world_id)),
      };
    });

    const failedDeletes = deleteResults.reduce((count, result) => {
      return result.status === 'rejected' ? count + 1 : count;
    }, 0);
    if (failedDeletes > 0) {
      const suffix = failedDeletes === 1 ? '' : 's';
      setErrorText(`Failed to delete ${failedDeletes} archived world${suffix}`);
    }
  }, [archivedWorlds, refreshArchivedWorlds, request]);

  const startBatchRun = useCallback(
    async (worldCount: number, ticksPerWorld: number) => {
      const normalizedWorldCount = Math.max(1, Math.floor(worldCount));
      const normalizedTicksPerWorld = Math.max(1, Math.floor(ticksPerWorld));
      const normalizedUniverseSeed = nextRandomUniverseSeed();

      try {
        setErrorText(null);
        const payload = await request<CreateBatchRunResponse>(
          '/v1/world-runs',
          'POST',
          {
            world_count: normalizedWorldCount,
            ticks_per_world: normalizedTicksPerWorld,
            universe_seed: normalizedUniverseSeed,
          },
        );
        setBatchRunStatus(null);
        setActiveBatchRunId(payload.run_id);
      } catch (err) {
        setErrorText(err instanceof Error ? err.message : 'Failed to start world batch run');
      }
    },
    [request],
  );

  const restoreSession = useCallback(
    async (sessionId: string): Promise<boolean> => {
      try {
        const [metadata, restoredSnapshot] = await Promise.all([
          request<SessionMetadata>(`/v1/sessions/${sessionId}`, 'GET'),
          request<WorldSnapshot>(`/v1/sessions/${sessionId}/state`, 'GET'),
        ]);
        applyLoadedSession(metadata, restoredSnapshot);
        return true;
      } catch {
        clearPersistedSessionId();
        return false;
      }
    },
    [applyLoadedSession, request],
  );

  const sendCommand = useCallback((command: unknown): boolean => {
    return sendSimulationCommand(wsRef.current, command);
  }, []);

  const setSpeedLevel = useCallback(
    (levelIndex: number) => {
      const nextLevel = Math.max(0, Math.min(SPEED_LEVELS.length - 1, levelIndex));
      setSpeedLevelIndex(nextLevel);
      if (isRunning) {
        sendCommand({ type: 'Start', data: { ticks_per_second: SPEED_LEVELS[nextLevel] } });
      }
    },
    [isRunning, sendCommand],
  );

  const toggleRun = useCallback(() => {
    setIsRunning((currentlyRunning) => {
      const nextCommand = currentlyRunning
        ? { type: 'Pause' }
        : { type: 'Start', data: { ticks_per_second: SPEED_LEVELS[speedLevelIndex] } };
      return sendCommand(nextCommand) ? !currentlyRunning : currentlyRunning;
    });
  }, [sendCommand, speedLevelIndex]);

  const step = useCallback(
    (count: number) => {
      const requestedCount = Math.floor(count);
      if (!Number.isFinite(requestedCount)) return;
      const sent = sendCommand({ type: 'Step', data: { count: Math.max(1, requestedCount) } });
      if (sent) {
        setIsStepPending(true);
        setStepProgress(
          requestedCount > 1
            ? {
                requested_count: requestedCount,
                completed_count: 0,
              }
            : null,
        );
      }
    },
    [sendCommand],
  );

  const resetSession = useCallback(() => {
    if (!session) return;
    void request<WorldSnapshot>(`/v1/sessions/${session.id}/reset`, 'POST', { seed: null })
      .then((nextSnapshot) => {
        setIsStepPending(false);
        setStepProgress(null);
        setSnapshot(nextSnapshot);
        setSpeciesPopulationHistory([
          {
            turn: nextSnapshot.turn,
            speciesCounts: normalizeSpeciesCounts(nextSnapshot.metrics.species_counts),
          },
        ]);
        setFocusedOrganismIdTracked(null);
        setFocusedOrganism(null);
        setActiveNeuronIds(null);
      })
      .catch((err) => {
        setErrorText(err instanceof Error ? err.message : 'Failed to reset session');
      });
  }, [request, session, setFocusedOrganismIdTracked]);

  const focusOrganism = useCallback(
    (organism: WorldOrganismState) => {
      const organismId = unwrapId(organism.id);
      setFocusedOrganismIdTracked(organismId);
      setFocusedOrganism((current) =>
        current && unwrapId(current.id) === organismId ? current : null,
      );
      setActiveNeuronIds(null);
      if (!session) return;
      void request(`/v1/sessions/${session.id}/focus`, 'POST', {
        organism_id: organismId,
      }).catch((err) => {
        setErrorText(err instanceof Error ? err.message : 'Failed to focus organism');
      });
    },
    [request, session, setFocusedOrganismIdTracked],
  );

  const defocusOrganism = useCallback(() => {
    setFocusedOrganismIdTracked(null);
    setFocusedOrganism(null);
    setActiveNeuronIds(null);
  }, [setFocusedOrganismIdTracked]);

  useEffect(() => {
    if (!snapshot || focusedOrganismId === null) {
      setFocusedOrganism(null);
      setActiveNeuronIds(null);
      return;
    }

    const worldFocusedOrganism = findOrganism(snapshot, focusedOrganismId);
    if (!worldFocusedOrganism) {
      setFocusedOrganismIdTracked(null);
      setFocusedOrganism(null);
      setActiveNeuronIds(null);
      return;
    }
    setFocusedOrganism((current) => syncFocusedOrganismFromWorld(current, worldFocusedOrganism));
  }, [snapshot, focusedOrganismId, setFocusedOrganismIdTracked]);

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
    void refreshArchivedWorlds().catch((err: unknown) => {
      if (err instanceof Error) {
        setErrorText(err.message);
      }
    });
  }, [refreshArchivedWorlds]);

  useEffect(() => {
    if (!activeBatchRunId) return;
    let cancelled = false;
    let timerId: number | null = null;

    const poll = async () => {
      try {
        const status = await request<BatchRunStatusResponse>(`/v1/world-runs/${activeBatchRunId}`, 'GET');
        if (cancelled) return;
        setBatchRunStatus(status);

        if (status.status === 'Running') {
          timerId = window.setTimeout(() => {
            void poll();
          }, BATCH_RUN_POLL_INTERVAL_MS);
          return;
        }

        setActiveBatchRunId(null);
        if (status.error) {
          setErrorText(status.error);
        }
        void refreshArchivedWorlds().catch((err: unknown) => {
          if (err instanceof Error) {
            setErrorText(err.message);
          }
        });
      } catch (err) {
        if (cancelled) return;
        setActiveBatchRunId(null);
        setErrorText(err instanceof Error ? err.message : 'Failed to fetch world run status');
      }
    };

    void poll();

    return () => {
      cancelled = true;
      if (timerId !== null) {
        window.clearTimeout(timerId);
      }
    };
  }, [activeBatchRunId, refreshArchivedWorlds, request]);

  useEffect(() => {
    return () => {
      wsRef.current?.close();
    };
  }, []);

  return {
    session,
    snapshot,
    batchRunStatus,
    archivedWorlds,
    speciesPopulationHistory,
    focusedOrganismId,
    focusedOrganism,
    activeNeuronIds,
    isRunning,
    isStepPending,
    stepProgress,
    speedLevels: SPEED_LEVELS,
    speedLevelIndex,
    errorText,
    createSession,
    resetSession,
    toggleRun,
    setSpeedLevelIndex: setSpeedLevel,
    step,
    focusOrganism,
    defocusOrganism,
    saveCurrentWorld,
    deleteArchivedWorld,
    deleteAllArchivedWorlds,
    startBatchRun,
    loadArchivedWorld,
    refreshArchivedWorlds,
  };
}
