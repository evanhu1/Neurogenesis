import { useMemo, useRef } from 'react';
import { ControlPanel } from './features/control_panel/ControlPanel';
import { InspectorPanel } from './features/inspector_panel/InspectorPanel';
import { useSimulationSession } from './features/sim/hooks/useSimulationSession';
import { formatRuntimeMetrics } from './features/sim/selectors';
import { WorldCanvas } from './features/world/components/WorldCanvas';

function FastModeWorldPanel() {
  return (
    <div className="flex h-full w-full items-center justify-center rounded-2xl border border-accent/15 bg-[radial-gradient(circle_at_top,_rgba(255,255,255,0.14),_transparent_42%),linear-gradient(180deg,_rgba(15,23,42,0.94),_rgba(15,23,42,0.82))] px-6 text-center text-white shadow-panel">
      <div className="max-w-md">
        <div className="mx-auto flex h-20 w-20 items-center justify-center rounded-3xl border border-white/15 bg-white/10 shadow-2xl">
          <svg aria-hidden="true" viewBox="0 0 24 24" className="h-10 w-10 fill-current">
            <path d="M4 5.5v13l8-6.5-8-6.5Z" />
            <path d="M12 5.5v13l8-6.5-8-6.5Z" />
          </svg>
        </div>
        <div className="mt-5 text-xs font-semibold uppercase tracking-[0.34em] text-white/65">
          Metrics-Only Fast Run
        </div>
        <h2 className="mt-3 text-2xl font-semibold tracking-tight">
          World rendering paused
        </h2>
        <p className="mt-3 text-sm leading-6 text-white/72">
          The canvas and viewport hooks are fully unmounted in fast mode. Only the tick counter and
          species population stream stay live.
        </p>
      </div>
    </div>
  );
}

export default function App() {
  const simulation = useSimulationSession();
  const panToHexRef = useRef<((q: number, r: number) => void) | null>(null);

  const metricsText = useMemo(
    () => formatRuntimeMetrics(simulation.liveMetrics),
    [simulation.liveMetrics],
  );
  const focusedSpeciesId = useMemo(
    () => simulation.focusedOrganism?.species_id ?? null,
    [simulation.focusedOrganism],
  );

  return (
    <div className="h-screen bg-page px-4 py-4 text-ink sm:px-6 lg:px-8">
      <div className="mx-auto grid h-full max-w-[1720px] gap-4 xl:grid-cols-[320px_minmax(480px,1fr)_450px]">
        <ControlPanel
          overview={{
            metricsText,
          }}
          championPool={{
            entries: simulation.championPool,
            onDeleteEntry: (index) => void simulation.deleteChampionPoolEntry(index),
            onClearAll: () => void simulation.clearChampionPool(),
          }}
          species={{
            history: simulation.speciesPopulationHistory,
            focusedSpeciesId,
            isFastMode: simulation.isFastMode,
            snapshot: simulation.snapshot,
            onFocusOrganism: simulation.focusOrganism,
            panToHexRef,
          }}
          controls={{
            snapshot: simulation.snapshot,
            isRunning: simulation.isRunning,
            isStepPending: simulation.isStepPending,
            stepProgress: simulation.stepProgress,
            speedLevelIndex: simulation.speedLevelIndex,
            speedLevelCount: simulation.speedLevels.length,
            streamMode: simulation.streamMode,
            isFastMode: simulation.isFastMode,
            onNewSession: (seedInput) => void simulation.createSession(seedInput),
            onSaveChampions: () => void simulation.saveChampions(),
            onToggleRun: simulation.toggleRun,
            onToggleFastRun: simulation.toggleFastRun,
            onSpeedLevelChange: simulation.setSpeedLevelIndex,
            onStep: simulation.step,
          }}
          errorText={simulation.errorText}
        />

        <main className="flex h-full items-center justify-center overflow-hidden rounded-2xl border border-accent/15 bg-panel/70 p-3 shadow-panel">
          {simulation.isFastMode ? (
            <FastModeWorldPanel />
          ) : (
            <WorldCanvas
              key={simulation.session?.id ?? 'no-session'}
              snapshot={simulation.snapshot}
              focusedOrganismId={simulation.focusedOrganismId}
              showFastOverlay={false}
              onOrganismSelect={simulation.focusOrganism}
              panToHexRef={panToHexRef}
            />
          )}
        </main>

        <InspectorPanel
          focusedOrganism={simulation.focusedOrganism}
          focusedBrain={simulation.focusedOrganism?.brain ?? null}
          activeActionNeuronId={simulation.activeActionNeuronId}
          onDefocus={simulation.defocusOrganism}
        />
      </div>
    </div>
  );
}
