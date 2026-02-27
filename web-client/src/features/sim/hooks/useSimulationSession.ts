import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  applyTickDelta,
  findOrganism,
  normalizeBatchRunStatusResponse,
  normalizeCreateSessionResponse,
  normalizeFocusBrainData,
  normalizeListArchivedWorldsResponse,
  normalizeTickDelta,
  normalizeWorldSnapshot,
  unwrapId,
} from '../../../protocol';
import type {
  ApiBatchRunStatusResponse,
  ApiCreateSessionResponse,
  ApiListArchivedWorldsResponse,
  ApiServerEvent,
  ApiWorldSnapshot,
  ArchivedWorldSummary,
  BatchRunStatusResponse,
  CreateBatchRunResponse,
  FocusBrainData,
  OrganismState,
  SessionMetadata,
  StepProgressData,
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
const NO_FOCUS_TURN = -1;
const DEFAULT_SPEED_LEVEL_INDEX = 1;

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

function speedLevelIndexForTicksPerSecond(ticksPerSecond: number): number {
  return ticksPerSecond > 0
    ? nearestSpeedLevelIndex(ticksPerSecond)
    : DEFAULT_SPEED_LEVEL_INDEX;
}

function mergeFocusedOrganismWithWorld(
  focused: OrganismState,
  worldOrganism: WorldOrganismState,
): OrganismState {
  return {
    ...focused,
    species_id: worldOrganism.species_id,
    q: worldOrganism.q,
    r: worldOrganism.r,
    generation: worldOrganism.generation,
    age_turns: worldOrganism.age_turns,
    facing: worldOrganism.facing,
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

function nextRandomUniverseSeed(): number {
  if (typeof crypto !== 'undefined' && typeof crypto.getRandomValues === 'function') {
    const value = new Uint32Array(1);
    crypto.getRandomValues(value);
    return value[0];
  }
  return Math.floor(Math.random() * 0x1_0000_0000);
}

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
  const [activeBatchRunId, setActiveBatchRunId] = useState<string | null>(null);
  const [batchRunStatus, setBatchRunStatus] = useState<BatchRunStatusResponse | null>(null);
  const [archivedWorlds, setArchivedWorlds] = useState<ArchivedWorldSummary[]>([]);
  const [speciesPopulationHistory, setSpeciesPopulationHistory] = useState<
    SpeciesPopulationPoint[]
  >([]);
  const [focusedOrganismId, setFocusedOrganismId] = useState<number | null>(null);
  const [focusedOrganismDetails, setFocusedOrganismDetails] = useState<OrganismState | null>(null);
  const [focusedOrganismTurn, setFocusedOrganismTurn] = useState<number>(NO_FOCUS_TURN);
  const [activeActionNeuronId, setActiveActionNeuronId] = useState<number | null>(null);
  const [isRunning, setIsRunning] = useState(false);
  const [isStepPending, setIsStepPending] = useState(false);
  const [stepProgress, setStepProgress] = useState<StepProgressData | null>(null);
  const [speedLevelIndex, setSpeedLevelIndex] = useState(DEFAULT_SPEED_LEVEL_INDEX);
  const [errorText, setErrorText] = useState<string | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const focusedOrganismIdRef = useRef<number | null>(null);
  const latestSnapshotTurnRef = useRef<number>(NO_FOCUS_TURN);
  const latestFocusedTurnRef = useRef<number>(NO_FOCUS_TURN);
  const nextFocusPollAtMsRef = useRef(0);
  const request = useMemo(() => createSimHttpClient(apiBase), []);
  const setFocusedOrganismIdTracked = useCallback((organismId: number | null, resetPoll = false) => {
    const changed = focusedOrganismIdRef.current !== organismId;
    focusedOrganismIdRef.current = organismId;
    if (changed) {
      latestFocusedTurnRef.current = NO_FOCUS_TURN;
    }
    if (resetPoll) {
      nextFocusPollAtMsRef.current = 0;
    }
    setFocusedOrganismId(organismId);
  }, []);

  const handleServerEvent = useCallback((event: ApiServerEvent) => {
    switch (event.type) {
      case 'StateSnapshot': {
        const nextSnapshot = normalizeWorldSnapshot(event.data);
        latestSnapshotTurnRef.current = nextSnapshot.turn;
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
        const delta = normalizeTickDelta(event.data);
        latestSnapshotTurnRef.current = delta.turn;
        setIsStepPending(false);
        setStepProgress(null);
        setSnapshot((prev) => {
          if (!prev) return prev;
          const nextSnapshot = applyTickDelta(prev, delta);
          setSpeciesPopulationHistory((previous) =>
            upsertSpeciesPopulationHistory(previous, {
              turn: nextSnapshot.turn,
              speciesCounts: normalizeSpeciesCounts(nextSnapshot.metrics.species_counts),
            }),
          );
          return nextSnapshot;
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
        const { turn, organism, active_action_neuron_id }: FocusBrainData =
          normalizeFocusBrainData(event.data);
        const organismId = unwrapId(organism.id);
        const trackedFocusedId = focusedOrganismIdRef.current;
        if (trackedFocusedId === null || trackedFocusedId !== organismId) {
          break;
        }
        const minimumAcceptedTurn = Math.max(
          latestSnapshotTurnRef.current,
          latestFocusedTurnRef.current,
        );
        if (turn < minimumAcceptedTurn) {
          break;
        }
        latestFocusedTurnRef.current = turn;
        setFocusedOrganismDetails(organism);
        setFocusedOrganismTurn(turn);
        setActiveActionNeuronId(active_action_neuron_id);
        break;
      }
      case 'Metrics': {
        break;
      }
      case 'Error': {
        setIsStepPending(false);
        setStepProgress(null);
        const message = event.data.message || 'Simulation server reported an error';
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

  const resetTransientUiState = useCallback((running: boolean, ticksPerSecond: number) => {
    setIsRunning(running);
    setIsStepPending(false);
    setStepProgress(null);
    setSpeedLevelIndex(speedLevelIndexForTicksPerSecond(ticksPerSecond));
  }, []);

  const applyLoadedSession = useCallback(
    (metadata: SessionMetadata, loadedSnapshot: WorldSnapshot) => {
      setErrorText(null);
      setSession(metadata);
      latestSnapshotTurnRef.current = loadedSnapshot.turn;
      latestFocusedTurnRef.current = NO_FOCUS_TURN;
      setSnapshot(loadedSnapshot);
      setFocusedOrganismIdTracked(null, true);
      setFocusedOrganismDetails(null);
      setFocusedOrganismTurn(NO_FOCUS_TURN);
      setActiveActionNeuronId(null);
      resetTransientUiState(metadata.running, metadata.ticks_per_second);
      setSpeciesPopulationHistory([
        {
          turn: loadedSnapshot.turn,
          speciesCounts: normalizeSpeciesCounts(loadedSnapshot.metrics.species_counts),
        },
      ]);
      persistSessionId(metadata.id);
      connectWs(metadata.id);
    },
    [connectWs, resetTransientUiState, setFocusedOrganismIdTracked],
  );

  const createSession = useCallback(async (seedInput?: string) => {
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
  }, [applyLoadedSession, request]);

  const refreshArchivedWorlds = useCallback(async () => {
    const payload = await request<ApiListArchivedWorldsResponse>('/v1/worlds', 'GET');
    setArchivedWorlds(normalizeListArchivedWorldsResponse(payload).worlds);
  }, [request]);

  const loadArchivedWorld = useCallback(
    async (worldId: string) => {
      try {
        setErrorText(null);
        const payload = await request<ApiCreateSessionResponse>(`/v1/worlds/${worldId}/sessions`, 'POST');
        const normalized = normalizeCreateSessionResponse(payload);
        applyLoadedSession(normalized.metadata, normalized.snapshot);
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

  const resetSession = useCallback((seedInput?: string) => {
    if (!session) return;
    const { seed, error } = parseSessionSeedInput(seedInput);
    if (error) {
      setErrorText(error);
      return;
    }
    setErrorText(null);
    if (isRunning) {
      sendCommand({ type: 'Pause' });
    }
    void request<ApiWorldSnapshot>(`/v1/sessions/${session.id}/reset`, 'POST', { seed })
      .then((nextSnapshot) => {
        const normalized = normalizeWorldSnapshot(nextSnapshot);
        latestSnapshotTurnRef.current = normalized.turn;
        latestFocusedTurnRef.current = NO_FOCUS_TURN;
        resetTransientUiState(false, 0);
        setSnapshot(normalized);
        setSpeciesPopulationHistory([
          {
            turn: normalized.turn,
            speciesCounts: normalizeSpeciesCounts(normalized.metrics.species_counts),
          },
        ]);
        setFocusedOrganismIdTracked(null, true);
        setFocusedOrganismDetails(null);
        setFocusedOrganismTurn(NO_FOCUS_TURN);
        setActiveActionNeuronId(null);
      })
      .catch((err) => {
        setErrorText(err instanceof Error ? err.message : 'Failed to reset session');
      });
  }, [isRunning, request, resetTransientUiState, sendCommand, session, setFocusedOrganismIdTracked]);

  const focusOrganism = useCallback(
    (organism: WorldOrganismState) => {
      const organismId = unwrapId(organism.id);
      const isSameFocus = focusedOrganismIdRef.current === organismId;
      setFocusedOrganismIdTracked(organismId, true);
      if (!isSameFocus) {
        setFocusedOrganismDetails(null);
        setFocusedOrganismTurn(NO_FOCUS_TURN);
        setActiveActionNeuronId(null);
      }
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
    setFocusedOrganismIdTracked(null, true);
    setFocusedOrganismDetails(null);
    setFocusedOrganismTurn(NO_FOCUS_TURN);
    setActiveActionNeuronId(null);
  }, [setFocusedOrganismIdTracked]);

  const focusedWorldOrganism = useMemo(() => {
    if (!snapshot || focusedOrganismId === null) {
      return null;
    }
    return findOrganism(snapshot, focusedOrganismId);
  }, [snapshot, focusedOrganismId]);

  const focusedOrganism = useMemo(() => {
    if (!focusedOrganismDetails || focusedOrganismId === null) {
      return null;
    }
    if (unwrapId(focusedOrganismDetails.id) !== focusedOrganismId) {
      return null;
    }
    if (
      snapshot &&
      focusedWorldOrganism &&
      snapshot.turn >= focusedOrganismTurn
    ) {
      return mergeFocusedOrganismWithWorld(focusedOrganismDetails, focusedWorldOrganism);
    }
    return focusedOrganismDetails;
  }, [
    focusedOrganismDetails,
    focusedOrganismId,
    focusedOrganismTurn,
    focusedWorldOrganism,
    snapshot,
  ]);

  useEffect(() => {
    if (!snapshot || focusedOrganismId === null || focusedWorldOrganism) {
      return;
    }
    setFocusedOrganismIdTracked(null, true);
    setFocusedOrganismDetails(null);
    setFocusedOrganismTurn(NO_FOCUS_TURN);
    setActiveActionNeuronId(null);
  }, [focusedOrganismId, focusedWorldOrganism, setFocusedOrganismIdTracked, snapshot]);

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
        const status = await request<ApiBatchRunStatusResponse>(`/v1/world-runs/${activeBatchRunId}`, 'GET');
        if (cancelled) return;
        const normalized = normalizeBatchRunStatusResponse(status);
        setBatchRunStatus(normalized);

        if (normalized.status === 'Running') {
          timerId = window.setTimeout(() => {
            void poll();
          }, BATCH_RUN_POLL_INTERVAL_MS);
          return;
        }

        setActiveBatchRunId(null);
        if (normalized.error) {
          setErrorText(normalized.error);
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
    activeActionNeuronId,
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
