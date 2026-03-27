import { useCallback, useEffect, useState } from 'react';
import {
  normalizeCreateSessionResponse,
  normalizeListArchivedWorldsResponse,
} from '../../../protocol';
import type {
  ApiCreateSessionResponse,
  ApiListArchivedWorldsResponse,
  ArchivedWorldSummary,
  SessionMetadata,
  WorldSnapshot,
} from '../../../types';
import type { SimRequestFn } from '../api/simHttpClient';
import { captureError } from './captureError';

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
    const deleteResults = await Promise.allSettled(
      worldIds.map(async (worldId) => {
        await request<ArchivedWorldSummary>(`/v1/worlds/${worldId}`, 'DELETE');
      }),
    );

    await refreshArchivedWorlds();

    const failedDeletes = deleteResults.reduce((count, result) => {
      return result.status === 'rejected' ? count + 1 : count;
    }, 0);
    if (failedDeletes > 0) {
      const suffix = failedDeletes === 1 ? '' : 's';
      setErrorText(`Failed to delete ${failedDeletes} archived world${suffix}`);
    }
  }, [archivedWorlds, refreshArchivedWorlds, request, setErrorText]);

  useEffect(() => {
    void refreshArchivedWorlds().catch((err: unknown) => {
      captureError(setErrorText, err, 'Failed to load archived worlds');
    });
  }, [refreshArchivedWorlds, setErrorText]);

  return {
    archivedWorlds,
    refreshArchivedWorlds,
    loadArchivedWorld,
    saveCurrentWorld,
    deleteArchivedWorld,
    deleteAllArchivedWorlds,
  };
}
