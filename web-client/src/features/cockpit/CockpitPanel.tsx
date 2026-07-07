// The research cockpit: the right-hand column. A tab strip over the shared read
// surface the agent's CLI uses — organism inspector plus pillars / ecology /
// lineage / genome / timeseries / find. Read tabs reflect the world file's
// last-persisted state (they refetch when the file is saved, i.e. on
// create / step / pause), with a manual refresh and an "as of turn N" marker.

import { useEffect, useRef, useState } from 'react';
import type { WorldController } from '../sim/hooks/useWorld';
import { InspectorContent } from '../inspector_panel/InspectorPanel';
import { EcologyTab, FindTab, GenomeTab, LineageTab, PillarsTab, SeriesTab } from './tabs';

type TabId = 'inspect' | 'pillars' | 'ecology' | 'lineage' | 'genome' | 'series' | 'find';

const TABS: { id: TabId; label: string }[] = [
  { id: 'inspect', label: 'Inspect' },
  { id: 'pillars', label: 'Pillars' },
  { id: 'ecology', label: 'Ecology' },
  { id: 'lineage', label: 'Lineage' },
  { id: 'genome', label: 'Genome' },
  { id: 'series', label: 'Series' },
  { id: 'find', label: 'Find' },
];

export function CockpitPanel({ world }: { world: WorldController }) {
  const [tab, setTab] = useState<TabId>('pillars');
  const [manualNonce, setManualNonce] = useState(0);
  const readRevision = world.revision + manualNonce;

  // Jump to the inspector when an organism is newly selected (e.g. canvas click
  // or a find-row click).
  const prevFocusRef = useRef<number | null>(null);
  useEffect(() => {
    if (world.focusedOrganismId != null && world.focusedOrganismId !== prevFocusRef.current) {
      setTab('inspect');
    }
    prevFocusRef.current = world.focusedOrganismId;
  }, [world.focusedOrganismId]);

  const tabProps = { worldName: world.worldName, revision: readRevision, active: true };

  return (
    <aside className="flex h-full flex-col overflow-hidden rounded-2xl border border-line bg-panel/90 shadow-panel backdrop-blur">
      <header className="shrink-0 border-b border-line px-2 pt-2">
        <div className="flex flex-wrap gap-1">
          {TABS.map((t) => (
            <button
              key={t.id}
              onClick={() => setTab(t.id)}
              className={`rounded-md px-2 py-1 text-[10px] font-semibold uppercase tracking-[0.12em] transition ${
                tab === t.id
                  ? 'bg-accent/15 text-ink/80'
                  : 'text-ink/35 hover:bg-surface/50 hover:text-ink/60'
              }`}
            >
              {t.label}
            </button>
          ))}
        </div>
        {tab !== 'inspect' && tab !== 'find' && (
          <div className="flex items-center justify-between px-1 py-1 text-[10px] text-ink/30">
            <span className="font-mono">
              as of turn {world.savedTurn?.toLocaleString() ?? '—'}
              {world.isRunning ? ' · running' : ''}
            </span>
            <button
              onClick={() => setManualNonce((n) => n + 1)}
              className="rounded p-0.5 text-ink/30 transition hover:bg-surface/60 hover:text-ink/60"
              aria-label="Refresh"
              title="Refresh"
            >
              <svg viewBox="0 0 16 16" className="h-3.5 w-3.5 fill-current">
                <path d="M8 3V1L5 4l3 3V5a3 3 0 1 1-3 3H3a5 5 0 1 0 5-5Z" />
              </svg>
            </button>
          </div>
        )}
      </header>

      <div className="min-h-0 flex-1 overflow-auto">
        {tab === 'inspect' && (
          <InspectorContent
            focusedOrganism={world.focusedOrganism}
            activeActionNeuronId={world.activeActionNeuronId}
            onDefocus={world.defocusOrganism}
          />
        )}
        {tab === 'pillars' && <PillarsTab {...tabProps} />}
        {tab === 'ecology' && <EcologyTab {...tabProps} />}
        {tab === 'lineage' && <LineageTab {...tabProps} />}
        {tab === 'genome' && <GenomeTab {...tabProps} />}
        {tab === 'series' && <SeriesTab {...tabProps} />}
        {tab === 'find' && <FindTab {...tabProps} onFocusOrganism={world.focusOrganismById} />}
      </div>
    </aside>
  );
}
