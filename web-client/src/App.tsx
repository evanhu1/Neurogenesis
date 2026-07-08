import { useState } from 'react';
import { useWorld } from './useWorld';
import { HexWorld } from './HexWorld';
import { ControlPanel } from './ControlPanel';
import { Inspector } from './Inspector';
import { Champions } from './Champions';

// NeuroGenesis lab — a 3-column layout on the new REST backend (option 2: full
// snapshot polling, no server-side streaming). Left: controls + telemetry.
// Center: the real hex world (pan/zoom/click). Right: a tabbed inspector /
// Quality-Diversity champion archive.
export default function App() {
  const world = useWorld();
  const [tab, setTab] = useState<'inspector' | 'champions'>('inspector');

  return (
    <div className="h-screen bg-page p-3 font-sans text-ink">
      <div className="mx-auto grid h-full max-w-[1760px] gap-3 xl:grid-cols-[320px_minmax(480px,1fr)_400px]">
        <ControlPanel world={world} />

        <main className="relative h-full overflow-hidden rounded-2xl border border-line bg-water shadow-panel">
          <HexWorld
            snapshot={world.snapshot}
            selectedId={world.selectedId}
            onSelect={(id) => {
              world.setSelectedId(id);
              setTab('inspector');
            }}
          />
        </main>

        <aside className="flex h-full flex-col overflow-hidden rounded-2xl border border-line bg-panel shadow-panel">
          <div className="flex border-b border-line">
            {(['inspector', 'champions'] as const).map((t) => (
              <button
                key={t}
                onClick={() => setTab(t)}
                className={`flex-1 px-3 py-2.5 text-sm font-medium capitalize transition ${
                  tab === t ? 'bg-surface text-ink' : 'text-ink/55 hover:bg-void'
                }`}
              >
                {t}
                {t === 'champions' && world.champions ? ` (${world.champions.coverage})` : ''}
              </button>
            ))}
          </div>
          <div className="min-h-0 flex-1">
            {tab === 'inspector' ? (
              <Inspector id={world.selectedId} />
            ) : (
              <Champions data={world.champions} />
            )}
          </div>
        </aside>
      </div>
    </div>
  );
}
