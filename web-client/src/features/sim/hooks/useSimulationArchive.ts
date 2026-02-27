import { useCallback, useEffect, useState } from 'react';
import {
  normalizeBatchRunStatusResponse,
  normalizeCreateSessionResponse,
  normalizeListArchivedWorldsResponse,
} from '../../../protocol';
import type {
  ApiBatchRunStatusResponse,
  ApiCreateSessionResponse,
  ApiListArchivedWorldsResponse,
  ArchivedWorldSummary,
  BatchRunStatusResponse,
  CreateBatchRunResponse,
  SessionMetadata,
  WorldSnapshot,
} from '../../../types';
import type { SimRequestFn } from '../api/simHttpClient';
import { captureError } from './captureError';

const BATCH_RUN_POLL_INTERVAL_MS = 500;

function nextRandomUniverseSeed(): number {
  if (typeof crypto !== 'undefined' && typeof crypto.getRandomValues === 'function') {
    const value = new Uint32Array(1);
    crypto.getRandomValues(value);
    return value[0];
  }
  return Math.floor(Math.random() * 0x1_0000_0000);
}

type UseSimulationArchiveArgs = {
  request: SimRequestFn;
  session: SessionMetadata | null;
  applyLoadedSession: (metadata: SessionMetadata, snapshot: WorldSnapshot) => void;
  setErrorText: (message: string | null) => void;
};

export function useSimulationArchive({
  request,
  session,
  applyLoadedSession,
  setErrorText,
}: UseSimulationArchiveArgs) {
  const [activeBatchRunId, setActiveBatchRunId] = useState<string | null>(null);
  const [batchRunStatus, setBatchRunStatus] = useState<BatchRunStatusResponse | null>(null);
  const [archivedWorlds, setArchivedWorlds] = useState<ArchivedWorldSummary[]>([]);

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
        captureError(setErrorText, err, 'Failed to load archived world');
      }
    },
    [applyLoadedSession, request, setErrorText],
  );

  const saveCurrentWorld = useCallback(async () => {
    if (!session) return;
    try {
      setErrorText(null);
      await request<ArchivedWorldSummary>(`/v1/sessions/${session.id}/archive`, 'POST');
      await refreshArchivedWorlds();
    } catch (err) {
      captureError(setErrorText, err, 'Failed to save current world');
    }
  }, [refreshArchivedWorlds, request, session, setErrorText]);

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
        captureError(setErrorText, err, 'Failed to delete archived world');
      }
    },
    [refreshArchivedWorlds, request, setErrorText],
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
  }, [archivedWorlds, refreshArchivedWorlds, request, setErrorText]);

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
        captureError(setErrorText, err, 'Failed to start world batch run');
      }
    },
    [request, setErrorText],
  );

  useEffect(() => {
    void refreshArchivedWorlds().catch((err: unknown) => {
      captureError(setErrorText, err, 'Failed to load archived worlds');
    });
  }, [refreshArchivedWorlds, setErrorText]);

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
          captureError(setErrorText, err, 'Failed to load archived worlds');
        });
      } catch (err) {
        if (cancelled) return;
        setActiveBatchRunId(null);
        captureError(setErrorText, err, 'Failed to fetch world run status');
      }
    };

    void poll();

    return () => {
      cancelled = true;
      if (timerId !== null) {
        window.clearTimeout(timerId);
      }
    };
  }, [activeBatchRunId, refreshArchivedWorlds, request, setErrorText]);

  return {
    archivedWorlds,
    batchRunStatus,
    refreshArchivedWorlds,
    loadArchivedWorld,
    saveCurrentWorld,
    deleteArchivedWorld,
    deleteAllArchivedWorlds,
    startBatchRun,
  };
}
