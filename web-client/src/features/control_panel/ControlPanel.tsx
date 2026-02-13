import { useCallback, useMemo, useState, type ChangeEvent, type MutableRefObject } from 'react';
import { unwrapId } from '../../protocol';
import { colorForSpeciesId } from '../../speciesColor';
import type {
  ArchivedWorldSummary,
  BatchRunStatusResponse,
  StepProgressData,
  WorldOrganismState,
  WorldSnapshot,
} from '../../types';
import type { SpeciesPopulationPoint } from '../sim/hooks/useSimulationSession';

type ControlPanelProps = {
  sessionMeta: string;
  speciesPopulationHistory: SpeciesPopulationPoint[];
  focusedSpeciesId: string | null;
  snapshot: WorldSnapshot | null;
  metricsText: string;
  errorText: string | null;
  batchRunStatus: BatchRunStatusResponse | null;
  archivedWorlds: ArchivedWorldSummary[];
  isRunning: boolean;
  isStepPending: boolean;
  stepProgress: StepProgressData | null;
  speedLevelIndex: number;
  speedLevelCount: number;
  onNewSession: () => void;
  onReset: () => void;
  onToggleRun: () => void;
  onSpeedLevelChange: (levelIndex: number) => void;
  onStep: (count: number) => void;
  onFocusOrganism: (organism: WorldOrganismState) => void;
  onSaveCurrentWorld: () => void;
  onDeleteArchivedWorld: (worldId: string) => void;
  onDeleteAllArchivedWorlds: () => void;
  onStartBatchRun: (worldCount: number, ticksPerWorld: number) => void;
  onLoadArchivedWorld: (worldId: string) => void;
  panToHexRef: MutableRefObject<((q: number, r: number) => void) | null>;
};

type ArchivedWorldSortMode = 'Newest' | 'OrganismsAlive';

export function ControlPanel({
  sessionMeta,
  speciesPopulationHistory,
  focusedSpeciesId,
  snapshot,
  metricsText,
  errorText,
  batchRunStatus,
  archivedWorlds,
  isRunning,
  isStepPending,
  stepProgress,
  speedLevelIndex,
  speedLevelCount,
  onNewSession,
  onReset,
  onToggleRun,
  onSpeedLevelChange,
  onStep,
  onFocusOrganism,
  onSaveCurrentWorld,
  onDeleteArchivedWorld,
  onDeleteAllArchivedWorlds,
  onStartBatchRun,
  onLoadArchivedWorld,
  panToHexRef,
}: ControlPanelProps) {
  const [skipCountInput, setSkipCountInput] = useState('1000');
  const [worldCountInput, setWorldCountInput] = useState('8');
  const [ticksInput, setTicksInput] = useState('1000');
  const [archivedWorldSortMode, setArchivedWorldSortMode] =
    useState<ArchivedWorldSortMode>('Newest');
  const isBatchRunning = batchRunStatus?.status === 'Running';
  const skipProgress = useMemo(() => {
    if (!stepProgress || stepProgress.requested_count <= 1) return null;
    const requested = Math.max(1, stepProgress.requested_count);
    const completed = Math.min(requested, Math.max(0, stepProgress.completed_count));
    const percent = Math.round((completed / requested) * 100);
    return {
      requested,
      completed,
      percent,
    };
  }, [stepProgress]);

  const handleSpeedLevelInput = (evt: ChangeEvent<HTMLInputElement>) => {
    const rawLevel = Number.parseInt(evt.target.value, 10);
    if (!Number.isFinite(rawLevel)) return;
    onSpeedLevelChange(rawLevel);
  };

  const handleSkipCountInput = useCallback((evt: ChangeEvent<HTMLInputElement>) => {
    setSkipCountInput(evt.target.value);
  }, []);

  const runSkip = useCallback(() => {
    if (isStepPending) return;
    const parsed = Number.parseInt(skipCountInput, 10);
    if (!Number.isFinite(parsed)) return;
    const count = Math.max(1, Math.min(parsed, 1_000_000_000));
    onStep(count);
    setSkipCountInput(String(count));
  }, [isStepPending, onStep, skipCountInput]);

  const batchProgress = useMemo(() => {
    if (!batchRunStatus) return null;
    const total = Math.max(1, batchRunStatus.total_worlds);
    const completed = Math.max(0, Math.min(total, batchRunStatus.completed_worlds));
    const percent = Math.round((completed / total) * 100);
    return { total, completed, percent };
  }, [batchRunStatus]);

  const runtimeStatsText = useMemo(() => {
    const visibleMetricPrefixes = ['turn=', 'organisms=', 'species_alive='];
    const filtered = metricsText
      .split('\n')
      .filter((line) => visibleMetricPrefixes.some((prefix) => line.startsWith(prefix)));
    return filtered.length > 0 ? filtered.join('\n') : 'No metrics';
  }, [metricsText]);

  const sortedArchivedWorlds = useMemo(() => {
    const worlds = [...archivedWorlds];
    worlds.sort((a, b) => {
      if (archivedWorldSortMode === 'OrganismsAlive') {
        if (b.organisms_alive !== a.organisms_alive) {
          return b.organisms_alive - a.organisms_alive;
        }
      }
      if (b.created_at_unix_ms !== a.created_at_unix_ms) {
        return b.created_at_unix_ms - a.created_at_unix_ms;
      }
      return a.world_id.localeCompare(b.world_id);
    });
    return worlds;
  }, [archivedWorldSortMode, archivedWorlds]);

  const runBatch = useCallback(() => {
    if (isBatchRunning) return;
    const parsedWorldCount = Number.parseInt(worldCountInput, 10);
    const parsedTicks = Number.parseInt(ticksInput, 10);
    if (!Number.isFinite(parsedWorldCount) || !Number.isFinite(parsedTicks)) return;
    const worldCount = Math.max(1, parsedWorldCount);
    const ticks = Math.max(1, parsedTicks);
    onStartBatchRun(worldCount, ticks);
    setWorldCountInput(String(worldCount));
    setTicksInput(String(ticks));
  }, [isBatchRunning, onStartBatchRun, ticksInput, worldCountInput]);

  return (
    <aside className="h-full overflow-auto rounded-2xl border border-accent/15 bg-panel/95 p-4 shadow-panel">
      <h1 className="text-2xl font-semibold tracking-tight">Neurogenesis</h1>
      <pre className="mt-3 whitespace-pre-wrap rounded-xl bg-slate-100/80 p-3 font-mono text-xs">
        {runtimeStatsText}
      </pre>
      <h3 className="mt-3 text-sm font-semibold uppercase tracking-wide text-ink/80">
        Species Population
      </h3>
      <SpeciesPopulationChart
        history={speciesPopulationHistory}
        focusedSpeciesId={focusedSpeciesId}
        onSpeciesClick={(speciesId) => {
          if (!snapshot) return;
          const candidates = snapshot.organisms.filter(
            (o) => String(unwrapId(o.species_id)) === speciesId,
          );
          if (candidates.length === 0) return;
          const organism = candidates[Math.floor(Math.random() * candidates.length)];
          onFocusOrganism(organism);
          panToHexRef.current?.(organism.q, organism.r);
        }}
      />

      <div className="mt-3 flex flex-wrap gap-2">
        <ControlButton label="New Session" onClick={onNewSession} />
        <ControlButton label="Reset" onClick={onReset} />
        <ControlButton label="Save World" onClick={onSaveCurrentWorld} disabled={!snapshot} />
        <div className="flex w-full items-center gap-2 rounded-lg bg-white/70 px-2 py-1">
          <ControlButton
            label={isRunning ? 'Stop' : 'Start'}
            onClick={onToggleRun}
            disabled={isStepPending}
          />
          <span className="text-xs" role="img" aria-label="turtle">🐢</span>
          <input
            aria-label="Simulation speed"
            type="range"
            min={0}
            max={speedLevelCount - 1}
            step={1}
            value={speedLevelIndex}
            onChange={handleSpeedLevelInput}
            className="h-1.5 min-w-0 flex-1 cursor-pointer accent-accent"
          />
          <span className="text-xs" role="img" aria-label="rabbit">🐇</span>
        </div>
      </div>
      <div className="mt-2 flex gap-2">
        <ControlButton label="Step 1" onClick={() => onStep(1)} disabled={isStepPending} />
        <ControlButton label="Step 10" onClick={() => onStep(10)} disabled={isStepPending} />
        <ControlButton label="Step 100" onClick={() => onStep(100)} disabled={isStepPending} />
      </div>
      <div className="mt-2 flex items-center gap-2 rounded-lg bg-white/70 px-2 py-2">
        <span className="text-xs font-semibold uppercase tracking-wide text-ink/75">Skip X</span>
        <input
          aria-label="Skip step count"
          type="number"
          min={1}
          step={1}
          value={skipCountInput}
          disabled={isStepPending}
          onChange={handleSkipCountInput}
          onKeyDown={(evt) => {
            if (evt.key === 'Enter') {
              runSkip();
            }
          }}
          className="w-28 rounded-md border border-accent/30 bg-white px-2 py-1 font-mono text-sm text-ink outline-none ring-accent/20 focus:ring-2 disabled:cursor-not-allowed disabled:opacity-50 disabled:grayscale"
        />
        <ControlButton label="Skip" onClick={runSkip} disabled={isStepPending} />
      </div>
      {skipProgress ? (
        <div className="mt-2 rounded-lg bg-white/70 px-2 py-2">
          <div className="mb-1 flex items-center justify-between font-mono text-[11px] text-ink/75">
            <span>Skipping {skipProgress.requested.toLocaleString()} ticks</span>
            <span>
              {skipProgress.completed.toLocaleString()} / {skipProgress.requested.toLocaleString()}{' '}
              ({skipProgress.percent}%)
            </span>
          </div>
          <div className="h-2 overflow-hidden rounded-full bg-accent/20">
            <div
              className="h-full rounded-full bg-accent transition-[width] duration-200 ease-out"
              style={{ width: `${skipProgress.percent}%` }}
            />
          </div>
        </div>
      ) : null}

      <h3 className="mt-3 text-sm font-semibold uppercase tracking-wide text-ink/80">World Batch</h3>
      <div className="mt-2 space-y-2 rounded-lg bg-white/70 px-2 py-2">
        <div className="grid grid-cols-2 gap-2">
          <label className="text-[11px] text-ink/70">
            N
            <input
              type="number"
              min={1}
              step={1}
              value={worldCountInput}
              disabled={isBatchRunning}
              onChange={(evt) => setWorldCountInput(evt.target.value)}
              className="mt-1 w-full rounded-md border border-accent/30 bg-white px-2 py-1 font-mono text-sm text-ink outline-none ring-accent/20 focus:ring-2 disabled:cursor-not-allowed disabled:opacity-50 disabled:grayscale"
            />
          </label>
          <label className="text-[11px] text-ink/70">
            Ticks
            <input
              type="number"
              min={1}
              step={1}
              value={ticksInput}
              disabled={isBatchRunning}
              onChange={(evt) => setTicksInput(evt.target.value)}
              className="mt-1 w-full rounded-md border border-accent/30 bg-white px-2 py-1 font-mono text-sm text-ink outline-none ring-accent/20 focus:ring-2 disabled:cursor-not-allowed disabled:opacity-50 disabled:grayscale"
            />
          </label>
        </div>
        <ControlButton
          label={isBatchRunning ? 'Running...' : 'Run Worlds'}
          onClick={runBatch}
          disabled={isBatchRunning}
        />
        {batchProgress ? (
          <div className="rounded-lg bg-accent/10 px-2 py-2">
            <div className="mb-1 flex items-center justify-between font-mono text-[11px] text-ink/80">
              <span>Status: {batchRunStatus?.status ?? 'Unknown'}</span>
              <span>
                {batchProgress.completed.toLocaleString()} / {batchProgress.total.toLocaleString()} (
                {batchProgress.percent}%)
              </span>
            </div>
            <div className="h-2 overflow-hidden rounded-full bg-accent/20">
              <div
                className="h-full rounded-full bg-accent transition-[width] duration-200 ease-out"
                style={{ width: `${batchProgress.percent}%` }}
              />
            </div>
          </div>
        ) : null}
      </div>

      {batchRunStatus?.status === 'Completed' && batchRunStatus.aggregate ? (
        <div className="mt-2 rounded-lg border border-accent/25 bg-accent/10 px-3 py-2">
          <div className="text-xs font-semibold uppercase tracking-wide text-ink/80">
            Batch Aggregate
          </div>
          <div className="mt-2 grid grid-cols-2 gap-2 font-mono text-xs">
            <div>Total organisms: {batchRunStatus.aggregate.total_organisms_alive.toLocaleString()}</div>
            <div>Total species: {batchRunStatus.aggregate.total_species_alive.toLocaleString()}</div>
            <div>Mean organisms: {batchRunStatus.aggregate.mean_organisms_alive.toFixed(2)}</div>
            <div>Mean species: {batchRunStatus.aggregate.mean_species_alive.toFixed(2)}</div>
          </div>
        </div>
      ) : null}

      {archivedWorlds.length ? (
        <div className="mt-2 rounded-lg bg-white/70 px-2 py-2">
          <div className="mb-2 flex items-center justify-between">
            <div className="text-xs font-semibold uppercase tracking-wide text-ink/75">
              Archived Worlds
            </div>
            <div className="flex items-center gap-2">
              <label className="text-[11px] font-semibold uppercase tracking-wide text-ink/70">
                Sort
                <select
                  value={archivedWorldSortMode}
                  onChange={(evt) =>
                    setArchivedWorldSortMode(evt.target.value as ArchivedWorldSortMode)
                  }
                  className="ml-1 rounded-md border border-accent/30 bg-white px-1.5 py-1 font-mono text-[11px] text-ink outline-none ring-accent/20 focus:ring-2"
                >
                  <option value="Newest">Newest</option>
                  <option value="OrganismsAlive">Alive Organisms</option>
                </select>
              </label>
              <button
                onClick={onDeleteAllArchivedWorlds}
                className="rounded-md border border-rose-300 bg-rose-100 px-2 py-1 text-[11px] font-semibold text-rose-700 transition hover:bg-rose-200"
              >
                Delete All
              </button>
            </div>
          </div>
          <div className="max-h-72 space-y-2 overflow-y-auto pr-1">
            {sortedArchivedWorlds.map((world) => (
              <div
                key={world.world_id}
                className="flex items-center justify-between rounded-md bg-slate-100/80 px-2 py-1"
              >
                <div className="font-mono text-[11px] text-ink/80">
                  <div>{new Date(world.created_at_unix_ms).toLocaleString()}</div>
                  <div>
                    org={world.organisms_alive} species={world.species_alive}
                  </div>
                </div>
                <div className="flex items-center gap-1">
                  <ControlButton label="Load" onClick={() => onLoadArchivedWorld(world.world_id)} />
                  <ControlButton
                    label="Delete"
                    onClick={() => onDeleteArchivedWorld(world.world_id)}
                  />
                </div>
              </div>
            ))}
          </div>
        </div>
      ) : null}

      {errorText ? (
        <div className="mt-3 rounded-xl border border-rose-300 bg-rose-50 px-3 py-2 font-mono text-xs text-rose-700">
          {errorText}
        </div>
      ) : null}
    </aside>
  );
}

type ControlButtonProps = {
  label: string;
  onClick: () => void;
  disabled?: boolean;
};

function ControlButton({ label, onClick, disabled = false }: ControlButtonProps) {
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      className={`rounded-lg bg-accent px-3 py-2 text-sm font-semibold text-white transition ${
        disabled ? 'cursor-not-allowed opacity-50 grayscale' : 'hover:brightness-110'
      }`}
    >
      {label}
    </button>
  );
}

const MAX_DISPLAY_POINTS = 300;
const MAX_VISIBLE_SPECIES = 10;
const SPECIES_ZERO_GRACE_POINTS = 100;

function pointPeakSpeciesCount(point: SpeciesPopulationPoint): number {
  let max = 0;
  for (const speciesId in point.speciesCounts) {
    const count = point.speciesCounts[speciesId];
    if (count > max) {
      max = count;
    }
  }
  return max;
}

function downsampleHistoryPreserveSpikes(history: SpeciesPopulationPoint[]): SpeciesPopulationPoint[] {
  if (history.length <= MAX_DISPLAY_POINTS) return history;

  const first = history[0];
  const last = history[history.length - 1];
  const middle = history.slice(1, history.length - 1);
  const middleBudget = MAX_DISPLAY_POINTS - 2;
  if (middleBudget <= 0 || middle.length === 0) return [first, last];

  const bucketCount = Math.ceil(middleBudget / 2);
  const sampledMiddle: SpeciesPopulationPoint[] = [];

  for (let bucket = 0; bucket < bucketCount; bucket += 1) {
    const start = Math.floor((bucket * middle.length) / bucketCount);
    const end = Math.floor(((bucket + 1) * middle.length) / bucketCount);
    if (start >= end) continue;

    let minIdx = start;
    let maxIdx = start;
    let minValue = pointPeakSpeciesCount(middle[start]);
    let maxValue = minValue;

    for (let i = start + 1; i < end; i += 1) {
      const value = pointPeakSpeciesCount(middle[i]);
      if (value < minValue) {
        minValue = value;
        minIdx = i;
      }
      if (value > maxValue) {
        maxValue = value;
        maxIdx = i;
      }
    }

    if (minIdx === maxIdx) {
      sampledMiddle.push(middle[minIdx]);
    } else if (minIdx < maxIdx) {
      sampledMiddle.push(middle[minIdx], middle[maxIdx]);
    } else {
      sampledMiddle.push(middle[maxIdx], middle[minIdx]);
    }
  }

  const sampled = [first, ...sampledMiddle, last];
  if (sampled.length <= MAX_DISPLAY_POINTS) return sampled;
  return sampled.slice(0, MAX_DISPLAY_POINTS - 1).concat(last);
}

type ChartSeries = {
  id: string;
  label: string;
  color: string;
  latestCount: number;
  counts: number[];
};

type ChartData = {
  displayPoints: SpeciesPopulationPoint[];
  series: ChartSeries[];
  maxCount: number;
  turnStart: number;
  turnEnd: number;
};

function rankSpeciesIdsByCount(speciesCounts: Record<string, number>): string[] {
  return Object.entries(speciesCounts)
    .filter(([, count]) => count > 0)
    .sort((a, b) => b[1] - a[1] || Number(a[0]) - Number(b[0]))
    .map(([speciesId]) => speciesId);
}

function computeStableVisibleSpeciesIds(
  history: SpeciesPopulationPoint[],
  pinnedSpeciesIds: string[],
  focusedSpeciesId: string | null,
): string[] {
  const protectedSpeciesSet = new Set<string>(pinnedSpeciesIds);
  if (focusedSpeciesId) {
    protectedSpeciesSet.add(focusedSpeciesId);
  }

  const visibleSpeciesIds: string[] = [];
  const zeroStreakBySpecies = new Map<string, number>();

  for (const point of history) {
    for (const speciesId of visibleSpeciesIds) {
      const count = point.speciesCounts[speciesId] ?? 0;
      zeroStreakBySpecies.set(
        speciesId,
        count > 0 ? 0 : (zeroStreakBySpecies.get(speciesId) ?? 0) + 1,
      );
    }

    for (let idx = visibleSpeciesIds.length - 1; idx >= 0; idx -= 1) {
      const speciesId = visibleSpeciesIds[idx];
      if (protectedSpeciesSet.has(speciesId)) continue;
      if ((zeroStreakBySpecies.get(speciesId) ?? 0) >= SPECIES_ZERO_GRACE_POINTS) {
        visibleSpeciesIds.splice(idx, 1);
        zeroStreakBySpecies.delete(speciesId);
      }
    }

    for (const speciesId of protectedSpeciesSet) {
      if (!visibleSpeciesIds.includes(speciesId)) {
        visibleSpeciesIds.push(speciesId);
        zeroStreakBySpecies.set(speciesId, 0);
      }
    }

    const rankedSpeciesIds = rankSpeciesIdsByCount(point.speciesCounts);
    for (const speciesId of rankedSpeciesIds) {
      if (visibleSpeciesIds.includes(speciesId)) continue;

      if (visibleSpeciesIds.length < MAX_VISIBLE_SPECIES) {
        visibleSpeciesIds.push(speciesId);
        zeroStreakBySpecies.set(speciesId, 0);
        continue;
      }

      let replacementIndex = -1;
      let replacementStreak = SPECIES_ZERO_GRACE_POINTS;
      for (let idx = 0; idx < visibleSpeciesIds.length; idx += 1) {
        const candidateId = visibleSpeciesIds[idx];
        if (protectedSpeciesSet.has(candidateId)) continue;
        const streak = zeroStreakBySpecies.get(candidateId) ?? 0;
        if (streak >= replacementStreak) {
          replacementStreak = streak;
          replacementIndex = idx;
        }
      }

      if (replacementIndex === -1) continue;
      zeroStreakBySpecies.delete(visibleSpeciesIds[replacementIndex]);
      visibleSpeciesIds[replacementIndex] = speciesId;
      zeroStreakBySpecies.set(speciesId, 0);
    }
  }

  const latest = history[history.length - 1];
  return visibleSpeciesIds.sort((a, b) => {
    const countDiff = (latest.speciesCounts[b] ?? 0) - (latest.speciesCounts[a] ?? 0);
    if (countDiff !== 0) return countDiff;
    return Number(a) - Number(b);
  });
}

function computeChartData(
  history: SpeciesPopulationPoint[],
  pinnedSpeciesIds: string[],
  focusedSpeciesId: string | null,
): ChartData | null {
  if (history.length === 0) return null;

  const displayPoints = downsampleHistoryPreserveSpikes(history);
  const latest = history[history.length - 1];
  const visibleSpeciesIds = computeStableVisibleSpeciesIds(history, pinnedSpeciesIds, focusedSpeciesId);

  if (visibleSpeciesIds.length === 0) return null;

  const countsBySeries = new Map<string, number[]>();
  for (const speciesId of visibleSpeciesIds) {
    countsBySeries.set(speciesId, []);
  }
  let maxCount = 1;

  for (const point of displayPoints) {
    for (const speciesId of visibleSpeciesIds) {
      const count = point.speciesCounts[speciesId] ?? 0;
      countsBySeries.get(speciesId)?.push(count);
      if (count > maxCount) maxCount = count;
    }
  }

  const series: ChartSeries[] = visibleSpeciesIds.map((speciesId) => ({
    id: speciesId,
    label: speciesId,
    color: colorForSpeciesId(speciesId),
    latestCount: latest.speciesCounts[speciesId] ?? 0,
    counts: countsBySeries.get(speciesId) ?? [],
  }));

  return {
    displayPoints,
    series,
    maxCount,
    turnStart: displayPoints[0].turn,
    turnEnd: displayPoints[displayPoints.length - 1].turn,
  };
}

function SpeciesPopulationChart({
  history,
  focusedSpeciesId,
  onSpeciesClick,
}: {
  history: SpeciesPopulationPoint[];
  focusedSpeciesId: string | null;
  onSpeciesClick: (speciesId: string) => void;
}) {
  const [pinnedSpeciesIds, setPinnedSpeciesIds] = useState<string[]>([]);
  const chartData = useMemo(
    () => computeChartData(history, pinnedSpeciesIds, focusedSpeciesId),
    [focusedSpeciesId, history, pinnedSpeciesIds],
  );

  const onSpeciesSeriesClick = useCallback(
    (speciesId: string) => {
      setPinnedSpeciesIds((previous) =>
        previous.includes(speciesId) ? previous : previous.concat(speciesId),
      );
      onSpeciesClick(speciesId);
    },
    [onSpeciesClick],
  );

  if (!chartData) {
    return (
      <div className="mt-2 rounded-xl bg-slate-100/80 p-3 font-mono text-xs text-ink/70">
        No species population history yet
      </div>
    );
  }

  const { displayPoints, series, maxCount, turnStart, turnEnd } = chartData;

  const chartWidth = 300;
  const chartHeight = 180;
  const padLeft = 36;
  const padRight = 12;
  const padTop = 10;
  const padBottom = 24;
  const innerWidth = chartWidth - padLeft - padRight;
  const innerHeight = chartHeight - padTop - padBottom;
  const turnSpan = Math.max(1, turnEnd - turnStart);

  return (
    <div className="mt-2 rounded-xl bg-slate-100/80 p-3">
      <svg viewBox={`0 0 ${chartWidth} ${chartHeight}`} className="h-44 w-full rounded-lg bg-white/70">
        <title>Species population over turns</title>
        <line
          x1={padLeft}
          y1={padTop}
          x2={padLeft}
          y2={chartHeight - padBottom}
          stroke="#94a3b8"
          strokeWidth={1}
        />
        <line
          x1={padLeft}
          y1={chartHeight - padBottom}
          x2={chartWidth - padRight}
          y2={chartHeight - padBottom}
          stroke="#94a3b8"
          strokeWidth={1}
        />
        {series.map((item) => {
          const points = item.counts
            .map((count, idx) => {
              const point = displayPoints[idx];
              const x =
                displayPoints.length === 1
                  ? padLeft + innerWidth / 2
                  : padLeft + ((point.turn - turnStart) / turnSpan) * innerWidth;
              const y = padTop + ((maxCount - count) / maxCount) * innerHeight;
              return `${x.toFixed(2)},${y.toFixed(2)}`;
            })
            .join(' ');

          return (
            <polyline
              key={item.id}
              points={points}
              fill="none"
              stroke={item.color}
              strokeWidth={item.id === focusedSpeciesId ? 3 : 2.25}
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          );
        })}
        <text x={4} y={padTop + 8} className="fill-slate-500 text-[10px] font-mono">
          {maxCount}
        </text>
        <text
          x={padLeft}
          y={chartHeight - 6}
          className="fill-slate-500 text-[10px] font-mono"
          textAnchor="start"
        >
          {turnStart}
        </text>
        <text
          x={chartWidth - padRight}
          y={chartHeight - 6}
          className="fill-slate-500 text-[10px] font-mono"
          textAnchor="end"
        >
          {turnEnd}
        </text>
      </svg>
      <div className="mt-2 flex gap-2 overflow-x-auto scrollbar-none font-mono text-[11px] text-ink/80">
        {series
          .slice()
          .sort((a, b) => b.latestCount - a.latestCount)
          .map((item) => (
            <button
              key={item.id}
              onClick={() => {
                onSpeciesSeriesClick(item.id);
              }}
              className={`flex shrink-0 items-center gap-1 rounded bg-white/70 px-2 py-1 transition hover:bg-slate-200 ${item.id === focusedSpeciesId ? 'ring-1 ring-ink/40' : ''
                }`}
            >
              <span
                className="inline-block h-2.5 w-2.5 rounded-full"
                style={{ backgroundColor: item.color }}
              />
              <span>{`${item.label}: ${item.latestCount}`}</span>
            </button>
          ))}
      </div>
    </div>
  );
}
