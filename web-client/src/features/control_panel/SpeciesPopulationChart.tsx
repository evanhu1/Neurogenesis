import { useCallback, useMemo, useState } from 'react';
import { colorForSpeciesId } from '../../speciesColor';
import type { SpeciesPopulationPoint } from '../sim/hooks/useSimulationSession';

const MAX_DISPLAY_POINTS = 300;
const MAX_VISIBLE_SPECIES = 10;
const SPECIES_ZERO_GRACE_POINTS = 100;

type ChartSeries = {
  id: number;
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

    for (let index = start + 1; index < end; index += 1) {
      const value = pointPeakSpeciesCount(middle[index]);
      if (value < minValue) {
        minValue = value;
        minIdx = index;
      }
      if (value > maxValue) {
        maxValue = value;
        maxIdx = index;
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

function rankSpeciesIdsByCount(speciesCounts: Record<string, number>): number[] {
  return Object.entries(speciesCounts)
    .filter(([, count]) => count > 0)
    .sort((a, b) => b[1] - a[1] || Number(a[0]) - Number(b[0]))
    .map(([speciesId]) => Number(speciesId));
}

function speciesCountAt(point: SpeciesPopulationPoint, speciesId: number): number {
  return point.speciesCounts[String(speciesId)] ?? 0;
}

function computeStableVisibleSpeciesIds(
  history: SpeciesPopulationPoint[],
  pinnedSpeciesIds: number[],
  focusedSpeciesId: number | null,
): number[] {
  const protectedSpeciesSet = new Set<number>(pinnedSpeciesIds);
  if (focusedSpeciesId !== null) {
    protectedSpeciesSet.add(focusedSpeciesId);
  }

  const visibleSpeciesIds: number[] = [];
  const visibleSpeciesSet = new Set<number>();
  const zeroStreakBySpecies = new Map<number, number>();

  for (const point of history) {
    for (const speciesId of visibleSpeciesIds) {
      const count = speciesCountAt(point, speciesId);
      zeroStreakBySpecies.set(
        speciesId,
        count > 0 ? 0 : (zeroStreakBySpecies.get(speciesId) ?? 0) + 1,
      );
    }

    for (let index = visibleSpeciesIds.length - 1; index >= 0; index -= 1) {
      const speciesId = visibleSpeciesIds[index];
      if (protectedSpeciesSet.has(speciesId)) continue;
      if ((zeroStreakBySpecies.get(speciesId) ?? 0) >= SPECIES_ZERO_GRACE_POINTS) {
        visibleSpeciesIds.splice(index, 1);
        visibleSpeciesSet.delete(speciesId);
        zeroStreakBySpecies.delete(speciesId);
      }
    }

    for (const speciesId of protectedSpeciesSet) {
      if (!visibleSpeciesSet.has(speciesId)) {
        visibleSpeciesIds.push(speciesId);
        visibleSpeciesSet.add(speciesId);
        zeroStreakBySpecies.set(speciesId, 0);
      }
    }

    const rankedSpeciesIds = rankSpeciesIdsByCount(point.speciesCounts);
    for (const speciesId of rankedSpeciesIds) {
      if (visibleSpeciesSet.has(speciesId)) continue;

      if (visibleSpeciesIds.length < MAX_VISIBLE_SPECIES) {
        visibleSpeciesIds.push(speciesId);
        visibleSpeciesSet.add(speciesId);
        zeroStreakBySpecies.set(speciesId, 0);
        continue;
      }

      let replacementIndex = -1;
      let replacementStreak = SPECIES_ZERO_GRACE_POINTS;
      for (let index = 0; index < visibleSpeciesIds.length; index += 1) {
        const candidateId = visibleSpeciesIds[index];
        if (protectedSpeciesSet.has(candidateId)) continue;
        const streak = zeroStreakBySpecies.get(candidateId) ?? 0;
        if (streak >= replacementStreak) {
          replacementStreak = streak;
          replacementIndex = index;
        }
      }

      if (replacementIndex === -1) continue;
      const replacedSpeciesId = visibleSpeciesIds[replacementIndex];
      zeroStreakBySpecies.delete(replacedSpeciesId);
      visibleSpeciesSet.delete(replacedSpeciesId);
      visibleSpeciesIds[replacementIndex] = speciesId;
      visibleSpeciesSet.add(speciesId);
      zeroStreakBySpecies.set(speciesId, 0);
    }
  }

  const latest = history[history.length - 1];
  return visibleSpeciesIds.sort((a, b) => {
    const countDiff = speciesCountAt(latest, b) - speciesCountAt(latest, a);
    if (countDiff !== 0) return countDiff;
    return a - b;
  });
}

function computeChartData(
  history: SpeciesPopulationPoint[],
  pinnedSpeciesIds: number[],
  focusedSpeciesId: number | null,
): ChartData | null {
  if (history.length === 0) return null;

  const displayPoints = downsampleHistoryPreserveSpikes(history);
  const latest = history[history.length - 1];
  const visibleSpeciesIds = computeStableVisibleSpeciesIds(history, pinnedSpeciesIds, focusedSpeciesId);

  if (visibleSpeciesIds.length === 0) return null;

  const countsBySeries = new Map<number, number[]>();
  for (const speciesId of visibleSpeciesIds) {
    countsBySeries.set(speciesId, []);
  }
  let maxCount = 1;

  for (const point of displayPoints) {
    for (const speciesId of visibleSpeciesIds) {
      const count = speciesCountAt(point, speciesId);
      countsBySeries.get(speciesId)?.push(count);
      if (count > maxCount) maxCount = count;
    }
  }

  const series: ChartSeries[] = visibleSpeciesIds.map((speciesId) => ({
    id: speciesId,
    label: String(speciesId),
    color: colorForSpeciesId(String(speciesId)),
    latestCount: speciesCountAt(latest, speciesId),
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

type SpeciesPopulationChartProps = {
  history: SpeciesPopulationPoint[];
  focusedSpeciesId: number | null;
  onSpeciesClick: (speciesId: number) => void;
};

export function SpeciesPopulationChart({
  history,
  focusedSpeciesId,
  onSpeciesClick,
}: SpeciesPopulationChartProps) {
  const [pinnedSpeciesIds, setPinnedSpeciesIds] = useState<number[]>([]);
  const chartData = useMemo(
    () => computeChartData(history, pinnedSpeciesIds, focusedSpeciesId),
    [focusedSpeciesId, history, pinnedSpeciesIds],
  );

  const onSpeciesSeriesClick = useCallback(
    (speciesId: number) => {
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
            .map((count, index) => {
              const point = displayPoints[index];
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
              className={`flex shrink-0 items-center gap-1 rounded bg-white/70 px-2 py-1 transition hover:bg-slate-200 ${
                item.id === focusedSpeciesId ? 'ring-1 ring-ink/40' : ''
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
