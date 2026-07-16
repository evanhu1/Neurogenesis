import { useCallback, useRef } from 'react';
import type { WorldController } from '../sim/hooks/useWorld';
import { SessionOverviewPanel } from './SessionOverviewPanel';
import { SimulationControlsPanel } from './SimulationControlsPanel';
import { SpeciesPopulationChart } from './SpeciesPopulationChart';

type ControlPanelProps = {
  world: WorldController;
};

export function ControlPanel({ world }: ControlPanelProps) {
  const { snapshot, focusOrganism } = world;
  const speciesCycleRef = useRef<{ speciesId: number; index: number } | null>(null);

  // Clicking a species cycles focus through its members, highest energy first.
  // Focusing an organism auto-pans the renderer to it.
  const onSpeciesClick = useCallback(
    (speciesId: number) => {
      if (!snapshot) return;
      const candidates = snapshot.organisms
        .filter((organism) => organism.species_id === speciesId)
        .sort((a, b) => b.energy - a.energy);
      if (candidates.length === 0) return;

      const previous = speciesCycleRef.current;
      const index =
        previous && previous.speciesId === speciesId ? (previous.index + 1) % candidates.length : 0;

      speciesCycleRef.current = { speciesId, index };
      focusOrganism(candidates[index]);
    },
    [focusOrganism, snapshot],
  );

  return (
    <aside className="h-full overflow-auto rounded-2xl border border-line bg-panel/90 p-3.5 shadow-panel backdrop-blur">
      <SessionOverviewPanel liveMetrics={world.liveMetrics} isRunning={world.isRunning} />

      <div className="mt-3">
        <h3 className="text-[10px] font-semibold uppercase tracking-[0.18em] text-ink/35">
          Species Population
        </h3>
        <SpeciesPopulationChart
          history={world.speciesPopulationHistory}
          focusedSpeciesId={world.focusedOrganism?.species_id ?? null}
          onSpeciesClick={onSpeciesClick}
        />
      </div>

      <SimulationControlsPanel world={world} />

      {world.errorText ? (
        <div className="mt-3 rounded-lg border border-rose-500/20 bg-rose-500/10 px-2.5 py-1.5 font-mono text-[11px] text-rose-300">
          {world.errorText}
        </div>
      ) : null}
    </aside>
  );
}
