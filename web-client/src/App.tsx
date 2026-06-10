import { useRef } from 'react';
import { ControlPanel } from './features/control_panel/ControlPanel';
import { InspectorPanel } from './features/inspector_panel/InspectorPanel';
import { useSimulationSession } from './features/sim/hooks/useSimulationSession';
import { WorldCanvas } from './features/world/components/WorldCanvas';

function FastModeWorldPanel() {
  return (
    <div className="flex h-full w-full items-center justify-center px-6 text-center">
      <div className="max-w-md">
        <div className="mx-auto flex h-20 w-20 items-center justify-center rounded-2xl border border-accent/20 bg-accent/10">
          <svg aria-hidden="true" viewBox="0 0 24 24" className="h-10 w-10 fill-accent/70">
            <path d="M4 5.5v13l8-6.5-8-6.5Z" />
            <path d="M12 5.5v13l8-6.5-8-6.5Z" />
          </svg>
        </div>
        <div className="mt-5 text-xs font-semibold uppercase tracking-[0.34em] text-ink/35">
          Metrics-Only Fast Run
        </div>
        <h2 className="mt-3 text-2xl font-semibold tracking-tight text-ink">
          World rendering paused
        </h2>
        <p className="mt-3 text-sm leading-6 text-ink/50">
          Only the tick counter and species population stream stay live while the simulation runs
          at full speed.
        </p>
      </div>
    </div>
  );
}

export default function App() {
  const simulation = useSimulationSession();
  const panToHexRef = useRef<((q: number, r: number) => void) | null>(null);

  return (
    <div className="h-screen bg-page p-3 text-ink">
      <div className="mx-auto grid h-full max-w-[1760px] gap-3 xl:grid-cols-[320px_minmax(480px,1fr)_400px]">
        <ControlPanel simulation={simulation} panToHexRef={panToHexRef} />

        <main className="relative h-full overflow-hidden rounded-2xl border border-white/5 bg-void shadow-panel">
          {simulation.isFastMode ? (
            <FastModeWorldPanel />
          ) : (
            <WorldCanvas
              key={simulation.session?.id ?? 'no-session'}
              snapshot={simulation.snapshot}
              focusedOrganismId={simulation.focusedOrganismId}
              onOrganismSelect={simulation.focusOrganism}
              panToHexRef={panToHexRef}
            />
          )}
        </main>

        <InspectorPanel
          focusedOrganism={simulation.focusedOrganism}
          activeActionNeuronId={simulation.activeActionNeuronId}
          onDefocus={simulation.defocusOrganism}
        />
      </div>
    </div>
  );
}
