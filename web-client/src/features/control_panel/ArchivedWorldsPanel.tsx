import { useMemo, useState } from 'react';
import type { ArchivedWorldSummary } from '../../types';
import { ControlButton } from './ControlButton';

type ArchivedWorldSortMode = 'Newest' | 'OrganismsAlive';

type ArchivedWorldsPanelProps = {
  archivedWorlds: ArchivedWorldSummary[];
  onLoadArchivedWorld: (worldId: string) => void;
  onDeleteArchivedWorld: (worldId: string) => void;
  onDeleteAllArchivedWorlds: () => void;
};

export function ArchivedWorldsPanel({
  archivedWorlds,
  onLoadArchivedWorld,
  onDeleteArchivedWorld,
  onDeleteAllArchivedWorlds,
}: ArchivedWorldsPanelProps) {
  const [archivedWorldSortMode, setArchivedWorldSortMode] =
    useState<ArchivedWorldSortMode>('Newest');

  const sortedArchivedWorlds = useMemo(() => {
    const worlds = [...archivedWorlds];
    worlds.sort((a, b) => {
      if (archivedWorldSortMode === 'OrganismsAlive' && b.organisms_alive !== a.organisms_alive) {
        return b.organisms_alive - a.organisms_alive;
      }
      if (b.created_at_unix_ms !== a.created_at_unix_ms) {
        return b.created_at_unix_ms - a.created_at_unix_ms;
      }
      return a.world_id.localeCompare(b.world_id);
    });
    return worlds;
  }, [archivedWorldSortMode, archivedWorlds]);

  if (archivedWorlds.length === 0) {
    return null;
  }

  return (
    <div className="mt-2 rounded-lg bg-white/70 px-2 py-2">
      <div className="mb-2 flex items-center justify-between">
        <div className="text-xs font-semibold uppercase tracking-wide text-ink/75">
          Archived Worlds
        </div>
        <div className="flex items-center gap-2">
          <label className="text-[11px] font-semibold uppercase tracking-wide text-ink/70">
            Sort
            <select
              value={archivedWorldSortMode}
              onChange={(evt) => setArchivedWorldSortMode(evt.target.value as ArchivedWorldSortMode)}
              className="ml-1 rounded-md border border-accent/30 bg-white px-1.5 py-1 font-mono text-[11px] text-ink outline-none ring-accent/20 focus:ring-2"
            >
              <option value="Newest">Newest</option>
              <option value="OrganismsAlive">Alive Organisms</option>
            </select>
          </label>
          <button
            onClick={onDeleteAllArchivedWorlds}
            className="rounded-md border border-rose-300 bg-rose-100 px-2 py-1 text-[11px] font-semibold text-rose-700 transition hover:bg-rose-200"
          >
            Delete All
          </button>
        </div>
      </div>
      <div className="max-h-72 space-y-2 overflow-y-auto pr-1">
        {sortedArchivedWorlds.map((world) => (
          <div
            key={world.world_id}
            className="flex items-center justify-between rounded-md bg-slate-100/80 px-2 py-1"
          >
            <div className="font-mono text-[11px] text-ink/80">
              <div>{new Date(world.created_at_unix_ms).toLocaleString()}</div>
              <div>
                org={world.organisms_alive} species={world.species_alive}
              </div>
            </div>
            <div className="flex items-center gap-1">
              <ControlButton label="Load" onClick={() => onLoadArchivedWorld(world.world_id)} />
              <ControlButton label="Delete" onClick={() => onDeleteArchivedWorld(world.world_id)} />
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
