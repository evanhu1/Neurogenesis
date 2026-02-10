import { useMemo } from 'react';
import { ControlPanel } from './features/layout/components/ControlPanel';
import { InspectorPanel } from './features/layout/components/InspectorPanel';
import { useSimulationSession } from './features/sim/hooks/useSimulationSession';
import {
  formatFitnessStats,
  formatFocusedStats,
  formatFocusMeta,
  formatMetrics,
  formatSessionMeta,
} from './features/sim/selectors';
import { WorldCanvas } from './features/world/components/WorldCanvas';

export default function App() {
  const simulation = useSimulationSession();

  const sessionMeta = useMemo(() => formatSessionMeta(simulation.session), [simulation.session]);
  const metricsText = useMemo(() => formatMetrics(simulation.snapshot), [simulation.snapshot]);
  const fitnessStatsText = useMemo(
    () => formatFitnessStats(simulation.snapshot),
    [simulation.snapshot],
  );
  const focusMetaText = useMemo(
    () => formatFocusMeta(simulation.focusedOrganismId, simulation.focusedOrganism),
    [simulation.focusedOrganism, simulation.focusedOrganismId],
  );
  const focusedStatsText = useMemo(
    () => formatFocusedStats(simulation.focusedOrganism),
    [simulation.focusedOrganism],
  );

  return (
    <div className="h-screen bg-page px-4 py-4 text-ink sm:px-6 lg:px-8">
      <div className="mx-auto grid h-full max-w-[1720px] gap-4 xl:grid-cols-[320px_minmax(480px,1fr)_460px]">
        <ControlPanel
          sessionMeta={sessionMeta}
          fitnessStatsText={fitnessStatsText}
          metricsText={metricsText}
          errorText={simulation.errorText}
          isRunning={simulation.isRunning}
          speedLevelIndex={simulation.speedLevelIndex}
          speedLevelCount={simulation.speedLevels.length}
          onNewSession={() => void simulation.createSession()}
          onReset={simulation.resetSession}
          onToggleRun={simulation.toggleRun}
          onSpeedLevelChange={simulation.setSpeedLevelIndex}
          onStep={simulation.step}
        />

        <main className="flex h-full items-center justify-center overflow-hidden rounded-2xl border border-accent/15 bg-panel/70 p-3 shadow-panel">
          <WorldCanvas
            key={simulation.session?.id ?? 'no-session'}
            snapshot={simulation.snapshot}
            focusedOrganismId={simulation.focusedOrganismId}
            deadFlashCells={simulation.deadFlashCells}
            bornFlashCells={simulation.bornFlashCells}
            onOrganismSelect={simulation.focusOrganism}
          />
        </main>

        <InspectorPanel
          focusMetaText={focusMetaText}
          focusedStatsText={focusedStatsText}
          focusedBrain={simulation.focusedOrganism?.brain ?? null}
        />
      </div>
    </div>
  );
}
