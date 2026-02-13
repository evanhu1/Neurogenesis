import { useMemo, useRef } from 'react';
import { unwrapId } from './protocol';
import { ControlPanel } from './features/control_panel/ControlPanel';
import { InspectorPanel } from './features/inspector_panel/InspectorPanel';
import { useSimulationSession } from './features/sim/hooks/useSimulationSession';
import {
  formatFocusMeta,
  formatMetrics,
  formatSessionMeta,
} from './features/sim/selectors';
import { WorldCanvas } from './features/world/components/WorldCanvas';

export default function App() {
  const simulation = useSimulationSession();
  const panToHexRef = useRef<((q: number, r: number) => void) | null>(null);

  const sessionMeta = useMemo(() => formatSessionMeta(simulation.session), [simulation.session]);
  const metricsText = useMemo(() => formatMetrics(simulation.snapshot), [simulation.snapshot]);
  const focusMetaText = useMemo(
    () => formatFocusMeta(simulation.focusedOrganismId, simulation.focusedOrganism),
    [simulation.focusedOrganism, simulation.focusedOrganismId],
  );
  const focusedSpeciesId = useMemo(
    () =>
      simulation.focusedOrganism ? String(unwrapId(simulation.focusedOrganism.species_id)) : null,
    [simulation.focusedOrganism],
  );

  return (
    <div className="h-screen bg-page px-4 py-4 text-ink sm:px-6 lg:px-8">
      <div className="mx-auto grid h-full max-w-[1720px] gap-4 xl:grid-cols-[320px_minmax(480px,1fr)_450px]">
        <ControlPanel
          sessionMeta={sessionMeta}
          speciesPopulationHistory={simulation.speciesPopulationHistory}
          focusedSpeciesId={focusedSpeciesId}
          snapshot={simulation.snapshot}
          metricsText={metricsText}
          errorText={simulation.errorText}
          batchRunStatus={simulation.batchRunStatus}
          archivedWorlds={simulation.archivedWorlds}
          isRunning={simulation.isRunning}
          isStepPending={simulation.isStepPending}
          stepProgress={simulation.stepProgress}
          speedLevelIndex={simulation.speedLevelIndex}
          speedLevelCount={simulation.speedLevels.length}
          onNewSession={() => void simulation.createSession()}
          onReset={simulation.resetSession}
          onToggleRun={simulation.toggleRun}
          onSpeedLevelChange={simulation.setSpeedLevelIndex}
          onStep={simulation.step}
          onFocusOrganism={simulation.focusOrganism}
          onSaveCurrentWorld={() => void simulation.saveCurrentWorld()}
          onDeleteArchivedWorld={(worldId) => void simulation.deleteArchivedWorld(worldId)}
          onDeleteAllArchivedWorlds={() => void simulation.deleteAllArchivedWorlds()}
          onStartBatchRun={(worldCount, ticksPerWorld) =>
            void simulation.startBatchRun(worldCount, ticksPerWorld)
          }
          onLoadArchivedWorld={(worldId) => void simulation.loadArchivedWorld(worldId)}
          panToHexRef={panToHexRef}
        />

        <main className="flex h-full items-center justify-center overflow-hidden rounded-2xl border border-accent/15 bg-panel/70 p-3 shadow-panel">
          <WorldCanvas
            key={simulation.session?.id ?? 'no-session'}
            snapshot={simulation.snapshot}
            focusedOrganismId={simulation.focusedOrganismId}
            onOrganismSelect={simulation.focusOrganism}
            panToHexRef={panToHexRef}
          />
        </main>

        <InspectorPanel
          focusMetaText={focusMetaText}
          focusedOrganism={simulation.focusedOrganism}
          focusedBrain={simulation.focusedOrganism?.brain ?? null}
          activeNeuronIds={simulation.activeNeuronIds}
          onDefocus={simulation.defocusOrganism}
        />
      </div>
    </div>
  );
}
