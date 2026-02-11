import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { applyTickDelta, findOrganism, unwrapId } from '../../../protocol';
import { DEFAULT_CONFIG } from '../../../types';
import type {
  CreateSessionResponse,
  FocusBrainData,
  MetricsSnapshot,
  OrganismState,
  ServerEvent,
  SessionMetadata,
  TickDelta,
  WorldSnapshot,
} from '../../../types';
import { connectSimulationWs, sendSimulationCommand } from '../api/simWsClient';
import { createSimHttpClient } from '../api/simHttpClient';
import { apiBase, SPEED_LEVELS, wsBase } from '../constants';
import { clearPersistedSessionId, loadPersistedSessionId, persistSessionId } from '../storage';

type DeadCellFlashState = { turn: number; cells: Array<{ q: number; r: number }> } | null;
type BornCellFlashState = { turn: number; cells: Array<{ q: number; r: number }> } | null;

export type SpeciesPopulationPoint = {
  turn: number;
  speciesCounts: Record<string, number>;
};

function normalizeSpeciesCounts(speciesCounts: Record<string, number>): Record<string, number> {
  return Object.fromEntries(
    Object.entries(speciesCounts).sort(([speciesA], [speciesB]) => Number(speciesA) - Number(speciesB)),
  );
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
    const next = previous.slice();
    next[next.length - 1] = point;
    return next;
  }

  return previous.concat(point);
}

export type SimulationSessionState = {
  session: SessionMetadata | null;
  snapshot: WorldSnapshot | null;
  speciesPopulationHistory: SpeciesPopulationPoint[];
  focusedOrganismId: number | null;
  focusedOrganism: OrganismState | null;
  activeNeuronIds: Set<number> | null;
  isRunning: boolean;
  speedLevels: readonly number[];
  speedLevelIndex: number;
  errorText: string | null;
  deadFlashCells: Array<{ q: number; r: number }> | null;
  bornFlashCells: Array<{ q: number; r: number }> | null;
  createSession: () => Promise<void>;
  resetSession: () => void;
  toggleRun: () => void;
  setSpeedLevelIndex: (levelIndex: number) => void;
  step: (count: number) => void;
  focusOrganism: (organism: OrganismState) => void;
  defocusOrganism: () => void;
};

export function useSimulationSession(): SimulationSessionState {
  const [session, setSession] = useState<SessionMetadata | null>(null);
  const [snapshot, setSnapshot] = useState<WorldSnapshot | null>(null);
  const [speciesPopulationHistory, setSpeciesPopulationHistory] = useState<
    SpeciesPopulationPoint[]
  >([]);
  const [focusedOrganismId, setFocusedOrganismId] = useState<number | null>(null);
  const [focusedOrganism, setFocusedOrganism] = useState<OrganismState | null>(null);
  const [activeNeuronIds, setActiveNeuronIds] = useState<Set<number> | null>(null);
  const [isRunning, setIsRunning] = useState(false);
  const [speedLevelIndex, setSpeedLevelIndex] = useState(1);
  const [errorText, setErrorText] = useState<string | null>(null);
  const [deadCellFlashState, setDeadCellFlashState] = useState<DeadCellFlashState>(null);
  const [bornCellFlashState, setBornCellFlashState] = useState<BornCellFlashState>(null);

  const wsRef = useRef<WebSocket | null>(null);
  const request = useMemo(() => createSimHttpClient(apiBase), []);

  const handleServerEvent = useCallback((event: ServerEvent) => {
    switch (event.type) {
      case 'StateSnapshot': {
        const nextSnapshot = event.data as WorldSnapshot;
        setDeadCellFlashState((prev) =>
          prev !== null && prev.turn !== nextSnapshot.turn ? null : prev,
        );
        setBornCellFlashState((prev) =>
          prev !== null && prev.turn !== nextSnapshot.turn ? null : prev,
        );
        setSnapshot(nextSnapshot);
        setSpeciesPopulationHistory((previous) =>
          upsertSpeciesPopulationHistory(previous, {
            turn: nextSnapshot.turn,
            speciesCounts: normalizeSpeciesCounts(nextSnapshot.metrics.species_counts),
          }),
        );
        break;
      }
      case 'TickDelta': {
        const delta = event.data as TickDelta;
        setSpeciesPopulationHistory((previous) =>
          upsertSpeciesPopulationHistory(previous, {
            turn: delta.turn,
            speciesCounts: normalizeSpeciesCounts(delta.metrics.species_counts),
          }),
        );
        setSnapshot((prev) => {
          if (!prev) return prev;

          const removedPositions = Array.isArray(delta.removed_positions)
            ? delta.removed_positions
            : [];

          if (removedPositions.length > 0) {
            const seenCells = new Set<string>();
            const cells: Array<{ q: number; r: number }> = [];
            for (const entry of removedPositions) {
              const key = `${entry.q},${entry.r}`;
              if (seenCells.has(key)) continue;
              seenCells.add(key);
              cells.push({ q: entry.q, r: entry.r });
            }
            setDeadCellFlashState(cells.length > 0 ? { turn: delta.turn, cells } : null);
          } else {
            setDeadCellFlashState((currentFlash) =>
              currentFlash !== null && delta.turn > currentFlash.turn ? null : currentFlash,
            );
          }

          if (delta.spawned.length > 0) {
            const seenCells = new Set<string>();
            const cells: Array<{ q: number; r: number }> = [];
            for (const spawned of delta.spawned) {
              const key = `${spawned.q},${spawned.r}`;
              if (seenCells.has(key)) continue;
              seenCells.add(key);
              cells.push({ q: spawned.q, r: spawned.r });
            }
            setBornCellFlashState(cells.length > 0 ? { turn: delta.turn, cells } : null);
          } else {
            setBornCellFlashState((currentFlash) =>
              currentFlash !== null && delta.turn > currentFlash.turn ? null : currentFlash,
            );
          }

          return applyTickDelta(prev, delta);
        });
        break;
      }
      case 'FocusBrain': {
        const { organism, active_neuron_ids } = event.data as FocusBrainData;
        const organismId = unwrapId(organism.id);
        setFocusedOrganismId(organismId);
        setFocusedOrganism(organism);
        setActiveNeuronIds(new Set(active_neuron_ids));
        break;
      }
      case 'Metrics': {
        const nextMetrics = event.data as MetricsSnapshot;
        setSnapshot((prev) => (prev ? { ...prev, metrics: nextMetrics } : prev));
        setSpeciesPopulationHistory((previous) => {
          const latest = previous[previous.length - 1];
          if (!latest) return previous;
          return upsertSpeciesPopulationHistory(previous, {
            turn: latest.turn,
            speciesCounts: normalizeSpeciesCounts(nextMetrics.species_counts),
          });
        });
        break;
      }
      case 'Error': {
        const message =
          typeof event.data === 'string' ? event.data : 'Simulation server reported an error';
        setErrorText(message);
        break;
      }
      default:
        break;
    }
  }, []);

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
      setFocusedOrganismId(null);
      setFocusedOrganism(null);
      setActiveNeuronIds(null);
      setIsRunning(false);
      setDeadCellFlashState(null);
      setBornCellFlashState(null);
      setSpeciesPopulationHistory([
        {
          turn: loadedSnapshot.turn,
          speciesCounts: normalizeSpeciesCounts(loadedSnapshot.metrics.species_counts),
        },
      ]);
      persistSessionId(metadata.id);
      connectWs(metadata.id);
    },
    [connectWs],
  );

  const createSession = useCallback(async () => {
    try {
      setErrorText(null);
      const payload = await request<CreateSessionResponse>('/v1/sessions', 'POST', {
        config: DEFAULT_CONFIG,
        seed: Math.floor(Date.now() / 1000),
      });
      applyLoadedSession(payload.metadata, payload.snapshot);
    } catch (err) {
      setErrorText(err instanceof Error ? err.message : 'Failed to create session');
    }
  }, [applyLoadedSession, request]);

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
      sendCommand({ type: 'Step', data: { count } });
    },
    [sendCommand],
  );

  const resetSession = useCallback(() => {
    if (!session) return;
    void request<WorldSnapshot>(`/v1/sessions/${session.id}/reset`, 'POST', { seed: null })
      .then((nextSnapshot) => {
        setSnapshot(nextSnapshot);
        setSpeciesPopulationHistory([
          {
            turn: nextSnapshot.turn,
            speciesCounts: normalizeSpeciesCounts(nextSnapshot.metrics.species_counts),
          },
        ]);
        setFocusedOrganismId(null);
        setFocusedOrganism(null);
        setActiveNeuronIds(null);
        setDeadCellFlashState(null);
        setBornCellFlashState(null);
      })
      .catch((err) => {
        setErrorText(err instanceof Error ? err.message : 'Failed to reset session');
      });
  }, [request, session]);

  const focusOrganism = useCallback(
    (organism: OrganismState) => {
      const organismId = unwrapId(organism.id);
      setFocusedOrganismId(organismId);
      setFocusedOrganism(organism);
      if (!session) return;
      void request(`/v1/sessions/${session.id}/focus`, 'POST', {
        organism_id: organismId,
      }).catch((err) => {
        setErrorText(err instanceof Error ? err.message : 'Failed to focus organism');
      });
    },
    [request, session],
  );

  const defocusOrganism = useCallback(() => {
    setFocusedOrganismId(null);
    setFocusedOrganism(null);
    setActiveNeuronIds(null);
  }, []);

  useEffect(() => {
    if (!snapshot || focusedOrganismId === null) {
      setFocusedOrganism(null);
      setActiveNeuronIds(null);
      return;
    }

    const nextFocusedOrganism = findOrganism(snapshot, focusedOrganismId);
    if (!nextFocusedOrganism) {
      setFocusedOrganismId(null);
      setFocusedOrganism(null);
      setActiveNeuronIds(null);
      return;
    }
    setFocusedOrganism(nextFocusedOrganism);
  }, [snapshot, focusedOrganismId]);

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
    return () => {
      wsRef.current?.close();
    };
  }, []);

  const deadFlashCells = useMemo(() => {
    if (!snapshot) return null;
    if (!deadCellFlashState) return null;
    if (deadCellFlashState.turn !== snapshot.turn) return null;
    return deadCellFlashState.cells;
  }, [deadCellFlashState, snapshot]);

  const bornFlashCells = useMemo(() => {
    if (!snapshot) return null;
    if (!bornCellFlashState) return null;
    if (bornCellFlashState.turn !== snapshot.turn) return null;
    return bornCellFlashState.cells;
  }, [bornCellFlashState, snapshot]);

  return {
    session,
    snapshot,
    speciesPopulationHistory,
    focusedOrganismId,
    focusedOrganism,
    activeNeuronIds,
    isRunning,
    speedLevels: SPEED_LEVELS,
    speedLevelIndex,
    errorText,
    deadFlashCells,
    bornFlashCells,
    createSession,
    resetSession,
    toggleRun,
    setSpeedLevelIndex: setSpeedLevel,
    step,
    focusOrganism,
    defocusOrganism,
  };
}
