import { CockpitPanel } from './features/cockpit/CockpitPanel';
import { ControlPanel } from './features/control_panel/ControlPanel';
import { useWorld } from './features/sim/hooks/useWorld';
import { WorldCanvas } from './features/world/components/WorldCanvas';

export default function App() {
  const world = useWorld();

  return (
    <div className="h-screen bg-page p-3 text-ink">
      <div className="mx-auto grid h-full max-w-[1760px] gap-3 xl:grid-cols-[320px_minmax(480px,1fr)_400px]">
        <ControlPanel world={world} />

        <main className="relative h-full overflow-hidden rounded-2xl border border-line bg-water shadow-panel">
          <WorldCanvas onRenderer={world.registerRenderer} onOrganismSelect={world.focusOrganism} />
        </main>

        <CockpitPanel world={world} />
      </div>
    </div>
  );
}
