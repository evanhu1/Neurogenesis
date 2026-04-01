import { useCallback, useRef, type MutableRefObject } from 'react';
import type {
  ChampionPoolEntry,
  StepProgressData,
  StreamMode,
  WorldOrganismState,
  WorldSnapshot,
} from '../../types';
import type { SpeciesPopulationPoint } from '../sim/hooks/useSimulationSession';
import { ChampionPoolPanel } from './ChampionPoolPanel';
import { SessionOverviewPanel } from './SessionOverviewPanel';
import { SimulationControlsPanel } from './SimulationControlsPanel';
import { SpeciesPopulationChart } from './SpeciesPopulationChart';

type ControlPanelProps = {
  overview: {
    metricsText: string;
  };
  species: {
    history: SpeciesPopulationPoint[];
    focusedSpeciesId: number | null;
    isFastMode: boolean;
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
    streamMode: StreamMode;
    isFastMode: boolean;
    onNewSession: (seedInput: string) => void;
    onSaveChampions: () => void;
    onToggleRun: () => void;
    onToggleFastRun: () => void;
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
      if (species.isFastMode) return;
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
    <aside className="h-full overflow-auto rounded-xl bg-panel p-3">
      <SessionOverviewPanel metricsText={overview.metricsText} />

      <h3 className="mt-2 text-[10px] font-semibold uppercase tracking-[0.16em] text-ink/40">
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
        streamMode={controls.streamMode}
        isFastMode={controls.isFastMode}
        onNewSession={controls.onNewSession}
        onSaveChampions={controls.onSaveChampions}
        onToggleRun={controls.onToggleRun}
        onToggleFastRun={controls.onToggleFastRun}
        onSpeedLevelChange={controls.onSpeedLevelChange}
        onStep={controls.onStep}
      />

      {errorText ? (
        <div className="mt-2 rounded bg-rose-500/10 px-2 py-1.5 font-mono text-[11px] text-rose-400">
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
