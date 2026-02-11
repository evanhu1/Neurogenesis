import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { applyTickDelta, findOrganism, unwrapId } from '../../../protocol';
import { DEFAULT_CONFIG } from '../../../types';
import type {
  CreateSessionResponse,
  FocusBrainData,
  OrganismState,
  ServerEvent,
  SessionMetadata,
  TickDelta,
  WorldOrganismState,
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
  focusOrganism: (organism: WorldOrganismState) => void;
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

        // Compute flash cells OUTSIDE the setSnapshot updater to avoid
        // calling setState inside a setState updater (causes cascading renders).
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
      case 'FocusBrain': {
        const { organism, active_neuron_ids } = event.data as FocusBrainData;
        const organismId = unwrapId(organism.id);
        setFocusedOrganismIdTracked(organismId);
        setFocusedOrganism(organism);
        setActiveNeuronIds(new Set(active_neuron_ids));
        break;
      }
      case 'Metrics': {
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
    [connectWs, setFocusedOrganismIdTracked],
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
        setFocusedOrganismIdTracked(null);
        setFocusedOrganism(null);
        setActiveNeuronIds(null);
        setDeadCellFlashState(null);
        setBornCellFlashState(null);
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
