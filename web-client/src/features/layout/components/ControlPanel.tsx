import type { ChangeEvent, MutableRefObject } from 'react';
import { unwrapId } from '../../../protocol';
import type { OrganismState, WorldSnapshot } from '../../../types';
import type { SpeciesPopulationPoint } from '../../sim/hooks/useSimulationSession';

type ControlPanelProps = {
  sessionMeta: string;
  speciesPopulationHistory: SpeciesPopulationPoint[];
  snapshot: WorldSnapshot | null;
  metricsText: string;
  errorText: string | null;
  isRunning: boolean;
  speedLevelIndex: number;
  speedLevelCount: number;
  onNewSession: () => void;
  onReset: () => void;
  onToggleRun: () => void;
  onSpeedLevelChange: (levelIndex: number) => void;
  onStep: (count: number) => void;
  onFocusOrganism: (organism: OrganismState) => void;
  panToHexRef: MutableRefObject<((q: number, r: number) => void) | null>;
};

export function ControlPanel({
  sessionMeta,
  speciesPopulationHistory,
  snapshot,
  metricsText,
  errorText,
  isRunning,
  speedLevelIndex,
  speedLevelCount,
  onNewSession,
  onReset,
  onToggleRun,
  onSpeedLevelChange,
  onStep,
  onFocusOrganism,
  panToHexRef,
}: ControlPanelProps) {
  const handleSpeedLevelInput = (evt: ChangeEvent<HTMLInputElement>) => {
    const rawLevel = Number.parseInt(evt.target.value, 10);
    if (!Number.isFinite(rawLevel)) return;
    onSpeedLevelChange(rawLevel);
  };

  return (
    <aside className="h-full overflow-auto rounded-2xl border border-accent/15 bg-panel/95 p-4 shadow-panel">
      <h1 className="text-2xl font-semibold tracking-tight">NeuroGenesis</h1>
      <pre className="mt-3 whitespace-pre-wrap rounded-xl bg-slate-100/80 p-3 font-mono text-xs">
        {sessionMeta}
      </pre>

      <div className="mt-3 flex flex-wrap gap-2">
        <ControlButton label="New Session" onClick={onNewSession} />
        <ControlButton label="Reset" onClick={onReset} />
        <div className="flex items-center gap-2 rounded-lg border border-accent/20 bg-white/70 px-2 py-1">
          <ControlButton label={isRunning ? 'Pause' : 'Start'} onClick={onToggleRun} />
          <div className="flex items-center gap-2">
            <div className="flex h-12 w-6 items-center justify-center">
              <input
                aria-label="Simulation speed"
                type="range"
                min={0}
                max={speedLevelCount - 1}
                step={1}
                value={speedLevelIndex}
                onChange={handleSpeedLevelInput}
                className="h-2 w-24 -rotate-90 cursor-pointer accent-accent"
              />
            </div>
            <div className="flex h-24 flex-col justify-between">
              <div className="flex items-center gap-1 text-sm leading-none">
                <span className="inline-block h-0.5 w-2 rounded-full bg-slate-500" />
                <span role="img" aria-label="rabbit">
                  🐇
                </span>
              </div>
              <div className="flex items-center gap-1 text-sm leading-none">
                <span className="inline-block h-0.5 w-2 rounded-full bg-slate-500" />
              </div>
              <div className="flex items-center gap-1 text-sm leading-none">
                <span className="inline-block h-0.5 w-2 rounded-full bg-slate-500" />
                <span role="img" aria-label="turtle">
                  🐢
                </span>
              </div>
            </div>
          </div>
        </div>
        <ControlButton label="Step 1" onClick={() => onStep(1)} />
        <ControlButton label="Step 10" onClick={() => onStep(10)} />
        <ControlButton label="Step 100" onClick={() => onStep(100)} />
      </div>

      <h3 className="mt-3 text-sm font-semibold uppercase tracking-wide text-ink/80">
        Species Population
      </h3>
      <SpeciesPopulationChart
        history={speciesPopulationHistory}
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

      <h3 className="mt-3 text-sm font-semibold uppercase tracking-wide text-ink/80">Runtime Metrics</h3>
      <pre className="mt-3 whitespace-pre-wrap rounded-xl bg-slate-100/80 p-3 font-mono text-xs">
        {metricsText}
      </pre>

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
};

function ControlButton({ label, onClick }: ControlButtonProps) {
  return (
    <button
      onClick={onClick}
      className="rounded-lg bg-accent px-3 py-2 text-sm font-semibold text-white transition hover:brightness-110"
    >
      {label}
    </button>
  );
}

const CHART_COLORS = ['#0f766e', '#1d4ed8', '#b45309', '#dc2626', '#7e22ce', '#be185d'];

function SpeciesPopulationChart({
  history,
  onSpeciesClick,
}: {
  history: SpeciesPopulationPoint[];
  onSpeciesClick: (speciesId: string) => void;
}) {
  if (history.length === 0) {
    return (
      <div className="mt-2 rounded-xl bg-slate-100/80 p-3 font-mono text-xs text-ink/70">
        No species population history yet
      </div>
    );
  }

  const speciesIds = Array.from(
    history.reduce((set, point) => {
      for (const speciesId of Object.keys(point.speciesCounts)) {
        set.add(speciesId);
      }
      return set;
    }, new Set<string>()),
  ).sort((a, b) => Number(a) - Number(b));

  if (speciesIds.length === 0) {
    return (
      <div className="mt-2 rounded-xl bg-slate-100/80 p-3 font-mono text-xs text-ink/70">
        No species data available
      </div>
    );
  }

  const chartWidth = 300;
  const chartHeight = 180;
  const padLeft = 36;
  const padRight = 12;
  const padTop = 10;
  const padBottom = 24;
  const innerWidth = chartWidth - padLeft - padRight;
  const innerHeight = chartHeight - padTop - padBottom;

  const maxCount = Math.max(
    1,
    ...history.flatMap((point) =>
      speciesIds.map((speciesId) => point.speciesCounts[speciesId] ?? 0),
    ),
  );

  const turnStart = history[0]?.turn ?? 0;
  const turnEnd = history[history.length - 1]?.turn ?? 0;
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
        {speciesIds.map((speciesId, speciesIdx) => {
          const points = history
            .map((point) => {
              const x =
                history.length === 1
                  ? padLeft + innerWidth / 2
                  : padLeft + ((point.turn - turnStart) / turnSpan) * innerWidth;
              const count = point.speciesCounts[speciesId] ?? 0;
              const y = padTop + ((maxCount - count) / maxCount) * innerHeight;
              return `${x.toFixed(2)},${y.toFixed(2)}`;
            })
            .join(' ');

          return (
            <polyline
              key={speciesId}
              points={points}
              fill="none"
              stroke={CHART_COLORS[speciesIdx % CHART_COLORS.length]}
              strokeWidth={2.25}
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
      <div className="mt-2 flex gap-2 overflow-x-auto font-mono text-[11px] text-ink/80">
        {speciesIds.map((speciesId, speciesIdx) => {
          const latest = history[history.length - 1];
          const count = latest?.speciesCounts[speciesId] ?? 0;
          return (
            <button
              key={speciesId}
              onClick={() => onSpeciesClick(speciesId)}
              className="flex shrink-0 items-center gap-1 rounded bg-white/70 px-2 py-1 transition hover:bg-slate-200"
            >
              <span
                className="inline-block h-2.5 w-2.5 rounded-full"
                style={{ backgroundColor: CHART_COLORS[speciesIdx % CHART_COLORS.length] }}
              />
              <span>{`${speciesId}: ${count}`}</span>
            </button>
          );
        })}
      </div>
    </div>
  );
}
