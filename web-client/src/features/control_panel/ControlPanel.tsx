import { useCallback, useRef, type RefObject } from 'react';
import type { SimulationSessionState } from '../sim/hooks/useSimulationSession';
import { ChampionPoolPanel } from './ChampionPoolPanel';
import { SessionOverviewPanel } from './SessionOverviewPanel';
import { SimulationControlsPanel } from './SimulationControlsPanel';
import { SpeciesPopulationChart } from './SpeciesPopulationChart';

type ControlPanelProps = {
  simulation: SimulationSessionState;
  panToHexRef: RefObject<((q: number, r: number) => void) | null>;
};

export function ControlPanel({ simulation, panToHexRef }: ControlPanelProps) {
  const { snapshot, isFastMode, focusOrganism } = simulation;
  const speciesCycleRef = useRef<{ speciesId: number; index: number } | null>(null);

  // Clicking a species cycles focus through its members, highest energy first.
  const onSpeciesClick = useCallback(
    (speciesId: number) => {
      if (isFastMode || !snapshot) return;
      const candidates = snapshot.organisms
        .filter((organism) => organism.species_id === speciesId)
        .sort((a, b) => b.energy - a.energy);
      if (candidates.length === 0) return;

      const previous = speciesCycleRef.current;
      const index =
        previous && previous.speciesId === speciesId ? (previous.index + 1) % candidates.length : 0;

      speciesCycleRef.current = { speciesId, index };
      const organism = candidates[index];
      focusOrganism(organism);
      panToHexRef.current?.(organism.q, organism.r);
    },
    [focusOrganism, isFastMode, panToHexRef, snapshot],
  );

  return (
    <aside className="h-full overflow-auto rounded-2xl border border-line bg-panel/90 p-3.5 shadow-panel backdrop-blur">
      <SessionOverviewPanel
        liveMetrics={simulation.liveMetrics}
        isRunning={simulation.isRunning}
        isFastMode={simulation.isFastMode}
      />

      <div className="mt-3">
        <h3 className="text-[10px] font-semibold uppercase tracking-[0.18em] text-ink/35">
          Species Population
        </h3>
        <SpeciesPopulationChart
          history={simulation.speciesPopulationHistory}
          focusedSpeciesId={simulation.focusedOrganism?.species_id ?? null}
          onSpeciesClick={onSpeciesClick}
        />
      </div>

      <SimulationControlsPanel simulation={simulation} />

      {simulation.errorText ? (
        <div className="mt-3 rounded-lg border border-rose-500/20 bg-rose-500/10 px-2.5 py-1.5 font-mono text-[11px] text-rose-300">
          {simulation.errorText}
        </div>
      ) : null}

      <ChampionPoolPanel
        entries={simulation.championPool}
        onDeleteEntry={(index) => void simulation.deleteChampionPoolEntry(index)}
        onClearAll={() => void simulation.clearChampionPool()}
      />
    </aside>
  );
}
