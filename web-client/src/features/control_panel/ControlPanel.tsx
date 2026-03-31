import { useCallback, useRef, type MutableRefObject } from 'react';
import type { ChampionPoolEntry, StepProgressData, WorldOrganismState, WorldSnapshot } from '../../types';
import type { SpeciesPopulationPoint } from '../sim/hooks/useSimulationSession';
import { ChampionPoolPanel } from './ChampionPoolPanel';
import { SessionOverviewPanel } from './SessionOverviewPanel';
import { SimulationControlsPanel } from './SimulationControlsPanel';
import { SpeciesPopulationChart } from './SpeciesPopulationChart';

type ControlPanelProps = {
  overview: {
    sessionMeta: string;
    metricsText: string;
  };
  species: {
    history: SpeciesPopulationPoint[];
    focusedSpeciesId: number | null;
    snapshot: WorldSnapshot | null;
    onFocusOrganism: (organism: WorldOrganismState) => void;
    panToHexRef: MutableRefObject<((q: number, r: number) => void) | null>;
  };
  championPool: {
    entries: ChampionPoolEntry[];
    onDeleteEntry: (index: number) => void;
    onClearAll: () => void;
  };
  controls: {
    snapshot: WorldSnapshot | null;
    isRunning: boolean;
    isStepPending: boolean;
    stepProgress: StepProgressData | null;
    speedLevelIndex: number;
    speedLevelCount: number;
    onNewSession: (seedInput: string) => void;
    onSaveChampions: () => void;
    onToggleRun: () => void;
    onSpeedLevelChange: (levelIndex: number) => void;
    onStep: (count: number) => void;
  };
  errorText: string | null;
};

export function ControlPanel({
  overview,
  species,
  championPool,
  controls,
  errorText,
}: ControlPanelProps) {
  const speciesCycleRef = useRef<{ speciesId: string; index: number } | null>(null);

  const onSpeciesClick = useCallback(
    (speciesId: number) => {
      if (!species.snapshot) return;
      const candidates = species.snapshot.organisms
        .filter((organism) => organism.species_id === speciesId)
        .sort((a, b) => b.energy - a.energy);
      if (candidates.length === 0) return;

      const previous = speciesCycleRef.current;
      const speciesIdKey = String(speciesId);
      const index =
        previous && previous.speciesId === speciesIdKey ? (previous.index + 1) % candidates.length : 0;

      speciesCycleRef.current = { speciesId: speciesIdKey, index };
      const organism = candidates[index];
      species.onFocusOrganism(organism);
      species.panToHexRef.current?.(organism.q, organism.r);
    },
    [species],
  );

  return (
    <aside className="h-full overflow-auto rounded-2xl border border-accent/15 bg-panel/95 p-4 shadow-panel">
      <SessionOverviewPanel sessionMeta={overview.sessionMeta} metricsText={overview.metricsText} />

      <h3 className="mt-3 text-sm font-semibold uppercase tracking-wide text-ink/80">
        Species Population
      </h3>
      <SpeciesPopulationChart
        history={species.history}
        focusedSpeciesId={species.focusedSpeciesId}
        onSpeciesClick={onSpeciesClick}
      />

      <SimulationControlsPanel
        snapshot={controls.snapshot}
        isRunning={controls.isRunning}
        isStepPending={controls.isStepPending}
        stepProgress={controls.stepProgress}
        speedLevelIndex={controls.speedLevelIndex}
        speedLevelCount={controls.speedLevelCount}
        onNewSession={controls.onNewSession}
        onSaveChampions={controls.onSaveChampions}
        onToggleRun={controls.onToggleRun}
        onSpeedLevelChange={controls.onSpeedLevelChange}
        onStep={controls.onStep}
      />

      {errorText ? (
        <div className="mt-3 rounded-xl border border-rose-300 bg-rose-50 px-3 py-2 font-mono text-xs text-rose-700">
          {errorText}
        </div>
      ) : null}

      <ChampionPoolPanel
        entries={championPool.entries}
        onDeleteEntry={championPool.onDeleteEntry}
        onClearAll={championPool.onClearAll}
      />
    </aside>
  );
}
